from typing import Callable

import torch

from pykeops.torch import Vi, Vj
from torch.autograd import grad


def GaussKernel(sigma):
    x, y, b = Vi(0, 3), Vj(1, 3), Vj(2, 3)
    gamma = 1 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp()
    return (K * b).sum_reduction(axis=1)


def GaussLinKernel(sigma):
    x, y, u, v, b = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3), Vj(4, 1)
    gamma = 1 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp() * (u * v).sum() ** 2
    return (K * b).sum_reduction(axis=1)


class HamiltonianSystem:
    def __init__(self, K: Callable):
        self.K = K

        # Hamiltonia
        self.base = lambda p, q: 0.5 * (p * self.K(q, q, p)).sum()

    def __call__(self, p: torch.tensor, q: torch.tensor):
        Gp, Gq = grad(self.base(p, q), (p, q), create_graph=True)
        return -Gq, Gp


class RalstonIntegrator:
    def __init__(self, ode_system: Callable):
        self.ode_system = ode_system

    def __call__(self, x0: torch.tensor, nt: int = 10, deltat: float = 1.0):
        x = tuple(map(lambda x: x.clone(), x0))
        dt = deltat / nt
        l = [x]
        for _ in range(nt):
            xdot = self.ode_system(*x)
            xi = tuple(map(lambda x, xdot: x + (2 * dt / 3) * xdot, x, xdot))
            xdoti = self.ode_system(*xi)
            x = tuple(
                map(
                    lambda x, xdot, xdoti: x + (0.25 * dt) * (xdot + 3 * xdoti),
                    x,
                    xdot,
                    xdoti,
                )
            )
            l.append(x)
        return l

    def shoot(self, p0: torch.tensor, q0: torch.tensor, nt: int = 10, deltat: float = 1.0):
        return self((p0, q0), nt, deltat)

    @classmethod
    def with_gauss_kernel(cls, sigma: torch.tensor):
        K = GaussKernel(sigma)
        H = HamiltonianSystem(K)
        return cls(H)
