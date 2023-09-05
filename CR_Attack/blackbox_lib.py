import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tools import get_RS_CR, sample_noise

class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''

    def __init__(self, orig_input, eps, step_size, use_grad=True, p=1):
        '''
        Initialize the attacker step with a given perturbation magnitude.
        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
        '''
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad
        self.p = p

    def project(self, x):
        '''
        Given an input x, project it back into the feasible set
        Args:
            ch.tensor x : the input to project back into the feasible set.
        Returns:
            A `ch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{x' \in S} \|x' - x\|_2
        '''
        raise NotImplementedError

    def step(self, x, g):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p`
        norms).
        Parameters:
            g (ch.tensor): the raw gradient
        Returns:
            The new input, a ch.tensor for the next step.
        '''
        raise NotImplementedError

    def random_perturb(self, x):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

    def to_image(self, x):
        '''
        Given an input (which may be in an alternative parameterization),
        convert it to a valid image (this is implemented as the identity
        function by default as most of the time we use the pixel
        parameterization, but for alternative parameterizations this functino
        must be overriden).
        '''
        return x


class LpStep(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:
    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """

    def project(self, x):
        diff = x - self.orig_input
        if self.p != np.inf:
            diff = diff.renorm(p=self.p, dim=0, maxnorm=self.eps)
        return torch.clamp(self.orig_input + diff, 0, 1)

    def step(self, x, g):
        l = len(x.shape) - 1
        if self.p != np.inf:
            g_norm = torch.norm(g.contiguous().view(g.shape[0], -1),
                                self.p,
                                dim=1)
            g_norm = g_norm.view(-1, *([1] * l))
            scaled_g = g / (g_norm + 1e-10)
        else:
            scaled_g = torch.sign(g)
        return x + scaled_g * self.step_size

    def random_perturb(self, x):
        l = len(x.shape) - 1
        rp = torch.randn_like(x)
        rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1] * l))
        return torch.clamp(x + self.eps * rp / (rp_norm + 1e-10), 0, 1)


class BlackBoxStep_WOCR(
        AttackerStep
):  # BGD with two queries per round, xt is perturbation, not image

    def __init__(self,
                 orig_input,
                 step_size,
                 delta,
                 alpha,
                 scale=2,
                 numclasses=21,
                 eps=1,
                 use_grad=False,
                 p=1,
                 cuda=True,
                 output=False):
        super(BlackBoxStep_WOCR, self).__init__(orig_input, eps, step_size,
                                                use_grad, p)
        self.delta = delta
        self.alpha = alpha
        self.numcls = numclasses
        self.cuda = cuda
        self.SCALE = scale
        self.output = output
        self.upscale = nn.Upsample(size=(orig_input.size(2),
                                         orig_input.size(3)))
        self.lowshape = [
            orig_input.size(0),
            orig_input.size(1),
            int(orig_input.size(2) // self.SCALE),
            int(orig_input.size(3) // self.SCALE)
        ]
        self.low_orig = F.interpolate(self.orig_input, size=self.lowshape[2:])
        self.N = self.low_orig.numel()
        if self.cuda:
            self.xt = torch.zeros(self.N).cuda()
        else:
            self.xt = torch.zeros(self.N)
        self.bestloss = 0
        self.best_output = 0

    def upscale(self, v):
        pass

    def downscale(self, v):
        pass

    def project2(self, x, x0):
        diff = x.view(1, -1)
        diff = diff.renorm(p=self.p, dim=0, maxnorm=self.eps)
        diff = diff.reshape(x0.shape)
        return torch.clamp(x0 + diff, 0, 1)

    def project3(self, x, x0):
        diff = x.view(1, -1)
        if self.p != np.inf:
            diff = diff.renorm(p=self.p, dim=0, maxnorm=self.eps)
        else:
            diff = torch.sign(diff) * self.eps
        diff = diff.reshape(x0.shape)

    def project(self, diff, radius=1.):
        diff = diff.view(-1)
        ret = torch.clamp(
            radius * self.low_orig.contiguous().view(-1) + diff, 0,
            radius) - radius * self.low_orig.contiguous().view(-1)
        return ret

    def randomUnitVector(self):  # Euclidean norm
        vec = torch.randn(self.N)
        return vec / vec.norm()

    def BGD_step(self, model, loss_func, label):
        with torch.no_grad():
            if self.cuda:
                u = self.randomUnitVector().cuda()
            else:
                u = self.randomUnitVector()
            q1 = self.upscale(
                (self.xt + self.delta * u).reshape(self.lowshape))
            adversarial_example1 = self.project2(q1, self.orig_input)
            adversarial_example1 = adversarial_example1.float()
            q2 = self.upscale(
                (self.xt - self.delta * u).reshape(self.lowshape))
            adversarial_example2 = self.project2(q2, self.orig_input)
            adversarial_example2 = adversarial_example2.float()
            q3 = self.upscale(self.xt.reshape(self.lowshape))
            adversarial_example3 = self.project2(q3, self.orig_input)
            adversarial_example3 = adversarial_example3.float()
            output = model(
                torch.cat((adversarial_example1, adversarial_example2,
                           adversarial_example3),
                          dim=0))
            loss1 = loss_func(output[0:1], label).mean()
            loss2 = loss_func(output[1:2], label).mean()
            loss3 = loss_func(output[2:3], label).mean()
            sign = -1
            coef = self.N / self.delta
            gradEst = coef * sign * (loss1.item() - loss2.item()) * u
            update = self.xt - self.step_size * gradEst
            self.xt = update  #self.project(update,1-self.alpha)
            if loss3 > self.bestloss:
                self.bestloss = loss3
                self.best_output = output[2:3].clone().cpu().detach()
            return self.xt


class BlackBoxStep_CR(
        AttackerStep
):  # BGD with two queries per round, xt is perturbation, not image

    def __init__(self,
                 orig_input,
                 step_size,
                 delta,
                 alpha,
                 scale=2,
                 numclasses=21,
                 eps=1,
                 use_grad=False,
                 p=1,
                 cuda=True,
                 refresh=100,
                 output=False):
        super(BlackBoxStep_CR, self).__init__(orig_input, eps, step_size,
                                              use_grad, p)
        self.delta = delta
        self.alpha = alpha
        self.numcls = numclasses
        self.cuda = cuda
        self.SCALE = scale
        self.output = output
        self.refresh = refresh
        self.upscale = nn.Upsample(size=(orig_input.size(2),
                                         orig_input.size(3)))
        self.lowshape = [
            orig_input.size(0),
            orig_input.size(1),
            int(orig_input.size(2) // self.SCALE),
            int(orig_input.size(3) // self.SCALE)
        ]
        self.low_orig = F.interpolate(self.orig_input, size=self.lowshape[2:])
        self.N = self.low_orig.numel()
        if self.cuda:
            self.xt = torch.zeros(self.N).cuda()
        else:
            self.xt = torch.zeros(self.N)
        self.bestloss = 0
        self.best_output = 0
        self.weight = 0
        self.stepcnt = 0

    def upscale(self, v):
        pass

    def downscale(self, v):
        pass

    def project2(self, x, x0):
        diff = x.view(1, -1)
        diff = diff.renorm(p=self.p, dim=0, maxnorm=self.eps)
        diff = diff.reshape(x0.shape)
        return torch.clamp(x0 + diff, 0, 1)

    def project(self, diff, radius=1.):
        diff = diff.view(-1)
        ret = torch.clamp(
            radius * self.low_orig.contiguous().view(-1) + diff, 0,
            radius) - radius * self.low_orig.contiguous().view(-1)
        return ret

    def randomUnitVector(self):  # Euclidean norm
        vec = torch.randn(self.N)
        return vec / vec.norm()

    def BGD_step(self, model, loss_func, label):
        with torch.no_grad():
            if self.cuda:
                u = self.randomUnitVector().cuda()
            else:
                u = self.randomUnitVector()
            q1 = self.upscale(
                (self.xt + self.delta * u).reshape(self.lowshape))
            adversarial_example1 = self.project2(q1, self.orig_input)
            adversarial_example1 = adversarial_example1.float()
            q2 = self.upscale(
                (self.xt - self.delta * u).reshape(self.lowshape))
            adversarial_example2 = self.project2(q2, self.orig_input)
            adversarial_example2 = adversarial_example2.float()
            q3 = self.upscale(self.xt.reshape(self.lowshape))
            adversarial_example3 = self.project2(q3, self.orig_input)
            adversarial_example3 = adversarial_example3.float()
            if self.stepcnt % self.refresh == 0:
                avg = sample_noise(adversarial_example3,
                                   label,
                                   num=16,
                                   sigma=0.001,
                                   batch_size=8,
                                   model=model)
                CR = get_RS_CR(avg.unsqueeze(0), label)
                self.weight = 1 / (1 + torch.exp(2 * CR - 4))
            output = model(
                torch.cat((adversarial_example1, adversarial_example2,
                           adversarial_example3),
                          dim=0))
            loss1 = loss_func(output[0:1], label)
            loss1 = torch.mul(loss1, self.weight).mean()
            loss2 = loss_func(output[1:2], label)
            loss2 = torch.mul(loss2, self.weight).mean()
            loss3 = loss_func(output[2:3], label).mean()
            loss4 = loss3.mean()
            loss3 = torch.mul(loss3, self.weight).mean()
            sign = -1
            coef = self.N / self.delta
            gradEst = coef * sign * (loss1.item() - loss2.item()) * u
            if self.p == 1 or self.p == np.inf:
                gradEst = torch.sign(gradEst)
            update = self.xt - self.step_size * gradEst
            self.xt = update  #self.project(update,1-self.alpha)
            if loss4.item() > self.bestloss:
                self.bestloss = loss4.item()
                self.best_output = output[2:3].clone().cpu().detach()
            self.stepcnt += 1
            return self.xt
