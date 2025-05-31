r"""

Each of x_0, \epsilon, v, and u can be derived from the other, but changing the
regression target results in a different weight. For example, if the target is
x_0, this is equivalent to weighting the loss like so,

  E[                                             (     x_0 -      \hat{x}_0)^2 ]
= E[ (\sigma(t)^2 / \alpha(t)^2)                 (\epsilon - \hat{\epsilon})^2 ]
= E[ (\sigma(t)^2 / (\alpha(t)^2 + \sigma(t)^2)) (       v -        \hat{v})^2 ]

This implies that a noise schedule tuned for one regression target needs to be
adjusted when changing the regression target.

https://sander.ai/2024/06/14/noise-schedules.html

The relationship between x_0, \epsilon, v, and u is given by,

  z_t      = \alpha(t) x_0 + \sigma(t) \epsilon
  x_0      = \alpha(t)^{-1} (z_t - \sigma(t) \epsilon)
  \epsilon = \sigma(t)^{-1} (z_t - \alpha(t)      x_0)
  v        = \alpha(t) \epsilon - \sigma(t) x_0
  u        =           \epsilon -           x_0

The score is given by,

    \nabla_{z_t} \log q(z_t | x) 
  = \nabla_{z_t} (- (2 \sigma^2)^{-1} ||z_t - \alpha(t) x_0||^2)
  =              (-    \sigma^{-2)      (z_t - \alpha(t) x_0)
  =              (-    \sigma^{-2)      (\alpha(t) x_0 + \sigma(t) \epsilon - \alpha(t) x_0)
  =              (-    \sigma^{-2)      (                \sigma(t) \epsilon                )
  =              (-    \sigma^{-1)      (                          \epsilon                )

"""
import abc

import torch
import torch.nn as nn
import torch.optim as optim


class Parameterization(nn.Module, abc.ABC):

    def __init__(self, schedule):
        super(Parameterization, self).__init__()
        self.schedule = schedule

    @abc.abstractmethod
    def x0(self, x, t):
        pass

    @abc.abstractmethod
    def epsilon(self, x, t):
        pass
        
    def v(self, x, t):
        # Target in "Progressive distillation ..."
        alpha = self.schedule.alpha(t)
        sigma = self.schedule.sigma(t)
        return alpha * self.epsilon(x, t) - sigma * self.x0(x, t)

    def u(self, x, t):
        # Target in "Scaling Rectified Flow ..."
        return self.epsilon(x, t) - self.x0(x, t)


class Epsilon(Parameterization):

    def __init__(self, model, schedule):
        super(Epsilon, self).__init__(schedule)
        self.model = model

    def epsilon(self, x, t):
        # Target in "Denoising Diffusion ..."
        return self.model(x, t)

    def x0(self, x, t):
        # Target in "Elucidating the design space ..."
        epsilon = self.epsilon(x, t)
        alpha = self.schedule.alpha(t)
        sigma = self.schedule.sigma(t)
        return (x - sigma * epsilon) / alpha


class X0(Parameterization):

    def __init__(self, model, schedule):
        super(X0, self).__init__(schedule)
        self.model = model
        self.schedule = schedule

    def epsilon(self, x, t):
        x0 = self.x0(x, t)
        alpha = self.schedule.alpha(t)
        sigma = self.schedule.sigma(t)
        return (xt - alpha * x0) / sigma

    def x0(self, x, t):
        return self.model(x, t)
