import torch
import numpy as np
from blackbox_lib import LpStep
from tools import get_RS_CR, sample_noise


def getnorm(norm):
    if norm == '2':
        norm = 2
    elif norm == '1':
        norm = 1
    elif norm == 'inf':
        norm = np.inf
    return norm


def get_adv_examples_targeted(data,
                              target,
                              target2,
                              model,
                              lossfunc,
                              norm,
                              eps,
                              step_size,
                              iterations,
                              N=20,
                              SIGMA=0.01,
                              BATCH=10,
                              a=2,
                              b=4,
                              INTEV=1):
    m = -1  # for targeted -1
    iterator = range(iterations)
    x = data.clone()

    step = LpStep(x, eps, step_size, True, getnorm(norm))
    i = 0
    for i in iterator:
        if i % INTEV == 0:  # update CR
            with torch.no_grad():
                avg = sample_noise(x,
                                   target,
                                   num=N,
                                   sigma=SIGMA,
                                   batch_size=BATCH,
                                   model=model)
                CR = get_RS_CR(avg.unsqueeze(0), target)
                weight = 1 / (1 + torch.exp(a * CR - b))

        x = x.clone().detach().requires_grad_(True)
        output = model(x)
        losses = lossfunc(output, target2)
        losses = torch.mul(losses, weight)
        loss = torch.mean(losses)

        if step.use_grad:
            grad, = torch.autograd.grad(m * loss, [x])
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)
    return x


def get_adv_examples_PGD(
    data,
    target,
    model,
    lossfunc,
    norm,
    eps,
    step_size,
    iterations,
):
    m = 1  # for untargeted 1
    iterator = range(iterations)
    x = data.clone()
    step = LpStep(x, eps, step_size, True, getnorm(norm))
    i = 0
    for i in iterator:
        x = x.clone().detach().requires_grad_(True)
        output = model(x)
        losses = lossfunc(output, target)
        loss = torch.mean(losses)
        if step.use_grad:
            grad, = torch.autograd.grad(m * loss, [x])
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)
    return x


def get_adv_examples_CRPGD(
    data,
    target,
    model,
    lossfunc,
    norm,
    eps,
    step_size,
    iterations,
    N=8,
    SIGMA=0.001,
    BATCH=8,
    a=2,
    b=4,
    INTEV=5,
):
    m = 1  # for untargeted 1
    iterator = range(iterations)
    x = data.clone()
    step = LpStep(x, eps, step_size, True, getnorm(norm))
    for i in iterator:
        if i % INTEV == 0:  #update CR
            with torch.no_grad():
                avg = sample_noise(x,
                                   target,
                                   num=N,
                                   sigma=SIGMA,
                                   batch_size=BATCH,
                                   model=model)
                CR = get_RS_CR(avg.unsqueeze(0), target)
                weight = 1 / (1 + torch.exp(a * CR - b))

        x = x.clone().detach().requires_grad_(True)
        output = model(x)
        losses = lossfunc(output, target)
        loss = torch.mul(losses, weight).mean()

        if step.use_grad:
            grad, = torch.autograd.grad(m * loss, [x])
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)
    return x