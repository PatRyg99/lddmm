from typing import Callable, Dict

import torch

from src.lddmm.utils import GaussLinKernel, RalstonIntegrator


class VarifoldDataLoss:
    def __init__(self, FS: torch.tensor, VT: torch.tensor, FT: torch.tensor, sigma: torch.tensor):
        self.loss = self._prepare(FS, VT, FT, GaussLinKernel(sigma=sigma))

    def _prepare(self, FS: torch.tensor, VT: torch.tensor, FT: torch.tensor, K: torch.tensor):
        def get_center_length_normal(F, V):
            V0, V1, V2 = (
                V.index_select(0, F[:, 0]),
                V.index_select(0, F[:, 1]),
                V.index_select(0, F[:, 2]),
            )
            centers, normals = (V0 + V1 + V2) / 3, 0.5 * torch.cross(V1 - V0, V2 - V0)
            length = (normals**2).sum(dim=1)[:, None].sqrt()
            return centers, length, normals / length

        CT, LT, NTn = get_center_length_normal(FT, VT)
        cst = (LT * K(CT, CT, NTn, NTn, LT)).sum()

        def loss(VS):
            CS, LS, NSn = get_center_length_normal(FS, VS)
            return cst + (LS * K(CS, CS, NSn, NSn, LS)).sum() - 2 * (LS * K(CS, CT, NSn, NTn, LT)).sum()

        return loss

    def __call__(self, q: torch.tensor):
        return self.loss(q)

class LDDMMLoss:
    def __init__(self, dataloss: Callable, sigma: torch.tensor, timestep: int = -1, gamma: float = 0):
        self.sigma = sigma
        self.timestep = timestep
        self.gamma = gamma
        self.dataloss = dataloss

        self.integrator = RalstonIntegrator.with_gauss_kernel(sigma)

    def __call__(self, p0: torch.tensor, q0: torch.tensor):
        _, q = self.integrator.shoot(p0, q0)[self.timestep]
        return self.gamma * self.integrator.ode_system.base(p0, q0) + self.dataloss(q)

class MultiLDDMMLoss:
    def __init__(self, datalosses: Dict[int, LDDMMLoss], sigma: torch.tensor, gamma: float = 0):
        self.datalosses = datalosses
        self.sigma = sigma
        self.gamma = gamma

        self.integrator = RalstonIntegrator.with_gauss_kernel(sigma)

    def __call__(self, p0: torch.tensor, q0: torch.tensor, nt: int = 10):
        shots = self.integrator.shoot(p0, q0, nt=nt)
        dataloss_mean = torch.stack([dataloss(shots[t][1]) for t, dataloss in self.datalosses.items()]).mean()
        return self.gamma * self.integrator.ode_system.base(p0, q0) + dataloss_mean