import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from math import ceil


def get_RS_CR(output, y):
    m = Normal(torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())
    out = output.permute(0, 2, 3, 1)
    values, indices = torch.topk(out, 2)
    vl = values.permute(3, 0, 1, 2)
    vl = torch.clamp(vl, 1e-8, 0.999998)
    vl = m.icdf(vl[0].detach())
    _, predict = torch.max(output.data, 1)
    ans = F.relu(vl)
    return ans


def sample_noise(x, target, num, sigma, batch_size, model):
    """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
    avg = 0
    NUM = num
    with torch.no_grad():
        for _ in range(ceil(num / batch_size)):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size
            if len(x.shape) == 4:
                batch = x.repeat((this_batch_size, 1, 1, 1))
            noise = torch.randn_like(batch, device='cuda') * sigma
            batch_ = batch + noise
            batch_ = torch.clamp(batch_, 0, 1)
            output = model(batch_)
            output = F.softmax(output, dim=1)
            avg = avg + torch.sum(output, dim=0)
    avg = avg / NUM
    return avg
