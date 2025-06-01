
import torch
import torch.nn as nn
import torch.optim as optim


class DDIM(nn.Module):

    def __init__(self, param, schedule, shape):
        super(DDIM, self).__init__()
        self.param = param
        self.schedule = schedule
        self.shape = shape
        
    def sample(self, num_examples=64, num_steps=200):
        k = num_examples
        param = self.param
        alpha = self.schedule.alpha
        sigma = self.schedule.sigma
        
        x_t = torch.randn(k, *self.shape)
        ts = torch.linspace(1.0, 0.0, num_steps + 1)
        ts, ss = ts[0:-1], ts[1:]
        for s, t in zip(ss, ts):
            t_ = torch.full((k, 1), t)
            x0_hat = param.x0(x_t, t_)
            epsilon = (x_t - alpha(t) * x0_hat) / sigma(t)
            x_s = alpha(s) * x0_hat + sigma(s) * epsilon
            x_t = x_s
        return x_t


class Heun(nn.Module):

    def __init__(self, param, schedule, shape):
        super(Heun, self).__init__()
        self.param = param
        self.schedule = schedule
        self.shape = shape
        
    def sample(self, num_examples=64, num_steps=200):
        k = num_examples
        alpha = self.schedule.alpha
        sigma = self.schedule.sigma

        def epsilon(x_t, t):
            t = torch.full((k, 1), t)
            return self.param.epsilon(x_t, t)

        def dx_dt(x_t, t):
            t = torch.full((k, 1), t)
            return self.param.dx_dt(x_t, t)
        
        x_t = torch.randn(k, *self.shape)
        ts = torch.linspace(1.0, 0.0, num_steps + 1)
        ts, ss = ts[0:-1], ts[1:]
        for s, t in zip(ss, ts):
            d_t = dx_dt(x_t, t)
            x_s = x_t + (s - t) * d_t
            if s > 0:
                d_s = dx_dt(x_s, s)
                x_s = x_t + (s - t) * 0.5 * (d_t + d_s)
            else:
                x_s = (x_t - sigma(t) * epsilon(x_t, t)) / alpha(t)
            x_t = x_s
        return x_t


class FlowMatching(nn.Module):

    def __init__(self, param, schedule, shape):
        super(FlowMatching, self).__init__()
        self.param = param
        self.schedule = schedule
        self.shape = shape
        
    def sample(self, num_examples=64, num_steps=200):
        k = num_examples
        alpha = self.schedule.alpha
        sigma = self.schedule.sigma

        def u(x_t, t):
            t = torch.full((k, 1), t)
            return self.param.dx_dt(x_t, t)
        
        x_t = torch.randn(k, *self.shape)
        ts = torch.linspace(1.0, 0.0, num_steps + 1)
        ts, ss = ts[0:-1], ts[1:]
        for s, t in zip(ss, ts):
            x_s = x_t + (s - t) * u(x_t, t)
            x_t = x_s
        return x_t