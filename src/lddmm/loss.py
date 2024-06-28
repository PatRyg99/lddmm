import torch
from src.lddmm.utils import GaussKernel, GaussLinKernel, Shooting, Hamiltonian


class LDDMMLoss:
    def __init__(self, FS: torch.tensor, VT: torch.tensor, FT: torch.tensor, sigma: torch.tensor, gamma: float = 0):
        self.sigma = sigma
        self.gamma = gamma
        self.Kv = GaussKernel(sigma=sigma)
        self.dataloss = self._init_dataloss(FS, VT, FT, GaussLinKernel(sigma=sigma))

    def _init_dataloss(self, FS: torch.tensor, VT: torch.tensor, FT: torch.tensor, K: torch.tensor):
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

    def __call__(self, p0: torch.tensor, q0: torch.tensor):
        _, q = Shooting(p0, q0, self.Kv)[-1]
        return self.gamma * Hamiltonian(self.Kv)(p0, q0) + self.dataloss(q)
