import abc

import torch
import torch.nn as nn
import torch.optim as optim


class Schedule(nn.Module, abc.ABC):

    def __init__(self):
        super(Schedule, self).__init__()

    @abc.abstractmethod
    def alpha(self, t):
        pass

    @abc.abstractmethod
    def sigma(self, t):
        pass

    def beta(self, t):
        # Note: Only works for variance-preserving noise schedules
        return -2 * self.alpha_dot(t) / self.alpha(t)

    def lambda_(self, t):  # log(signal-to-noise ratio)
        return torch.log(self.alpha(t) ** 2 / self.sigma(t) ** 2)

    def alpha_dot(self, t):
        t = t.detach().clone().requires_grad_(True)
        alpha = self.alpha(t)
        alpha.backward(torch.ones_like(alpha))
        return t.grad

    def sigma_dot(self, t):
        t = t.detach().clone().requires_grad_(True)
        sigma = self.sigma(t)
        sigma.backward(torch.ones_like(sigma))
        return t.grad

    def lambda_dot(self, t):
        t = t.detach().clone().requires_grad_(True)
        lambda_ = self.lambda_(t)
        lambda_.backward(torch.ones_like(lambda_))
        return t.grad


class VariancePreserving(Schedule):

    def __init__(self, s=0.008):
        super(VariancePreserving, self).__init__()
        self.s = torch.tensor(s).float()

    def alpha(self, t):
        s = self.s
        num = torch.cos((t + s) / (1 + s) * torch.pi/2)
        den = torch.cos((    s) / (1 + s) * torch.pi/2)
        num = torch.maximum(torch.full_like(num, 1e-5), num)
        den = torch.maximum(torch.full_like(den, 1e-5), den)
        return torch.sqrt(num / den)

    def sigma(self, t):
        return torch.sqrt(1 - self.alpha(t) ** 2)


class VarianceExploding(Schedule):

    def __init__(self, sigma_max=100.0):
        super(VarianceExploding, self).__init__()
        self.sigma_max = torch.tensor(sigma_max).float()

    def alpha(self, t):
        return torch.ones_like(t)

    def sigma(self, t):
        return t * self.sigma_max + 1e-5


class FlowMatching(Schedule):

    def __init__(self, s=0.999):
        super(FlowMatching, self).__init__()
        self.s = torch.tensor(s).float()

    def alpha(self, t):
        t = t * self.s + (1-self.s)/2
        return 1 - t

    def sigma(self, t):
        return 1 - self.alpha(t)