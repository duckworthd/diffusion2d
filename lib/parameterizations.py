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

from . import schedules


class Parameterization(nn.Module, abc.ABC):

    def __init__(self, schedule):
        super(Parameterization, self).__init__()
        self.schedule = schedule

    @abc.abstractmethod
    def x0(self, x, t):
        # Target in "Elucidating the design space ..."
        pass

    @abc.abstractmethod
    def epsilon(self, x, t):
        # Target in "Denoising Diffusion ..."
        pass
        
    def v(self, x, t):
        # Target in "Progressive distillation ..."
        alpha = self.schedule.alpha
        sigma = self.schedule.sigma
        return alpha(t) * self.epsilon(x, t) - sigma(t) * self.x0(x, t)

    def u(self, x, t):
        # Target in "Scaling Rectified Flow ..." and "Flow Matching for ..."
        # This is the dx/dt when using conditional flow matching's noise schedule.
        return self.epsilon(x, t) - self.x0(x, t)

    def score(self, x, t):
        return -1 * self.epsilon(x, t) / self.schedule.sigma(t)

    def dx_dt(self, x, t):
        # Note: The following is functionally correct but numerically unstable.
        s = self.schedule
        w_x = s.alpha_dot(t) / s.alpha(t)
        w_epsilon = s.sigma(t) * (
            s.sigma_dot(t) / s.sigma(t) 
            - s.alpha_dot(t) * s.sigma(t) / (s.alpha(t) ** 2)
        )
        return w_x * x + w_epsilon * self.epsilon(x, t)


class Epsilon(Parameterization):

    def __init__(self, model, schedule):
        super(Epsilon, self).__init__(schedule)
        self.model = model

    def epsilon(self, x, t):
        return self.model(x, t)

    def x0(self, x, t):
        epsilon = self.epsilon(x, t)
        alpha = self.schedule.alpha
        sigma = self.schedule.sigma
        return (x - sigma(t) * epsilon) / alpha(t)

    def dx_dt(self, x, t):
        if not isinstance(self.schedule, schedules.VariancePreserving):
            return super(Epsilon, self).dx_dt(x, t)
        # ODE from "Score-Based Generative Modeling ..."
        # Note: This is dx/dt when using a variance-preserving noise schedule.
        s = self.schedule
        return -0.5 * s.beta(t) * x - 0.5 * s.beta(t) * self.score(x, t)


class X0(Parameterization):

    def __init__(self, model, schedule):
        super(X0, self).__init__(schedule)
        self.model = model
        self.schedule = schedule

    def epsilon(self, x, t):
        s = self.schedule
        x0 = self.x0(x, t)
        return (x - s.alpha(t) * x0) / s.sigma(t)

    def x0(self, x, t):
        return self.model(x, t)

    def dx_dt(self, x, t):
        if not isinstance(self.schedule, schedules.VarianceExploding):
            return super(X0, self).dx_dt(x, t)
        # ODE from "Elucidating ..."
        # Note: This is dx/dt when alpha(t) = 1.0 and sigma(t) increases.
        s = self.schedule
        return self.epsilon(x, t) * s.sigma_dot(t)


class U(Parameterization):

    def __init__(self, model, schedule):
        super(U, self).__init__(schedule)
        self.model = model
        self.schedule = schedule

    def x0(self, x, t):
        sigma = self.schedule.sigma
        return x - sigma(t) * self.u(x, t)

    def epsilon(self, x, t):
        alpha = self.schedule.alpha
        return x + alpha(t) * self.u(x, t)

    def u(self, x, t):
        return self.model(x, t)

    def dx_dt(self, x, t):
        if not isinstance(self.schedule, schedules.FlowMatching):
            return super(U, self).dx_dt(x, t)
        # ODE from "Flow Matching for ..."
        # Note: This is dx/dt when alpha(t) = 1 - t and sigma(t) = 1.0
        return self.u(x, t)
