import os

from typing import Callable

import pyvista as pv
import torch

from torch.types import _dtype
from tqdm.auto import tqdm

from src.lddmm.loss import LDDMMLoss, VarifoldDataLoss


class LDDMMRegistrator:
    def __init__(
        self,
        source_mesh: pv.PolyData,
        target_mesh: pv.PolyData,
        sigma: float,
        dataloss: Callable = VarifoldDataLoss,
        dtype: _dtype = torch.float32,
        device: str = "cpu",
    ):
        self.source_mesh = source_mesh
        self.target_mesh = target_mesh
        self.sigma = sigma
        self.dataloss = dataloss
        self.dtype = dtype
        self.device = device

        self._reset()

    def _reset(self):
        VS, FS = self._prepare_mesh(self.source_mesh, dtype=self.dtype, device=self.device)
        VT, FT = self._prepare_mesh(self.target_mesh, dtype=self.dtype, device=self.device)
        sigma = torch.tensor([self.sigma], dtype=self.dtype, device=self.device)

        self.q0 = VS.clone().requires_grad_(True)
        self.p0 = torch.zeros(self.q0.shape, dtype=self.dtype, device=self.device, requires_grad=True)
        self.optimizer = torch.optim.LBFGS([self.p0], max_eval=10, max_iter=10)
        self.loss = LDDMMLoss(self.dataloss(FS, VT, FT, sigma), sigma)

    def _prepare_mesh(self, mesh: pv.PolyData, dtype: _dtype = torch.FloatType, device: str = "cpu"):
        V, F = mesh.points, mesh.regular_faces
        V = torch.tensor(V, dtype=dtype, device=device)
        F = torch.tensor(F, dtype=torch.long, device=device)

        return V, F

    def optimize(self, iters: int):
        with tqdm(total=iters) as pbar:

            def closure():
                self.optimizer.zero_grad()
                L = self.loss(self.p0, self.q0)
                L.backward()
                pbar.set_postfix({"loss": L.item()})
                return L

            for _ in range(iters):
                self.optimizer.step(closure)
                pbar.update(1)

    def shoot(self, nt: int = 10, deltat: float = 1.0):
        return self.loss.integrator.shoot(self.p0, self.q0, nt=nt, deltat=deltat)

    def export_shoot(self, output_dir: str, nt: int = 10, deltat: float = 1.0):
        listpq = self.shoot(nt, deltat)

        os.makedirs(output_dir, exist_ok=True)
        for i, VS in enumerate(listpq):
            verts = VS[1].detach().cpu().numpy()
            poly = pv.PolyData.from_regular_faces(verts, faces=self.source_mesh.regular_faces)
            poly.save(os.path.join(output_dir, f"shoot_{i}.vtp"))
