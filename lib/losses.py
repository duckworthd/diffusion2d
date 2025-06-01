
import torch
import torch.nn as nn
import torch.optim as optim


class Epsilon(nn.Module):

    def __init__(self, sched, param):
        super(Epsilon, self).__init__()
        self.sched = sched
        self.param = param

    def forward(self, x0, t):
        epsilon = torch.randn(*x0.shape)
        z_t = self.sched.alpha(t) * x0 + self.sched.sigma(t) * epsilon
        epsilon_hat = self.param.epsilon(z_t, t)
        return torch.mean(l2_error(epsilon, epsilon_hat))


class X0(nn.Module):

    def __init__(self, sched, param):
        super(X0, self).__init__()
        self.sched = sched
        self.param = param

    def forward(self, x0, t):
        epsilon = torch.randn(*x0.shape)
        z_t = self.sched.alpha(t) * x0 + self.sched.sigma(t) * epsilon
        x0_hat = self.param.x0(z_t, t)
        return torch.mean(l2_error(x0, x0_hat))


class V(nn.Module):

    def __init__(self, sched, param):
        super(V, self).__init__()
        self.sched = sched
        self.param = param

    def forward(self, x0, t):
        epsilon = torch.randn(*x0.shape)
        z_t = self.sched.alpha(t) * x0 + self.sched.sigma(t) * epsilon
        v_hat = self.param.v(z_t, t)
        v = self.sched.alpha(t) * epsilon - self.sched.sigma(t) * x0
        return torch.mean(l2_error(v, v_hat))


class U(nn.Module):

    def __init__(self, sched, param):
        super(U, self).__init__()
        self.sched = sched
        self.param = param

    def forward(self, x0, t):
        epsilon = torch.randn(*x0.shape)
        z_t = self.sched.alpha(t) * x0 + self.sched.sigma(t) * epsilon
        u_hat = self.param.u(z_t, t)
        u = epsilon - x0  # # Note: Only works for conditional flow noise schedule
        return torch.mean(l2_error(u, u_hat))


def l2_error(x, x_hat):
    err = torch.square(x - x_hat)
    axes = [i for i in range(x.ndim) if i > 0]
    return torch.mean(err, axis=axes)