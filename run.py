import os, glob
from typing import List

import pyvista as pv
import pandas as pd
from src.lddmm.registrator import LDDMMRegistrator

class LongitudinalSample:
    def __init__(self, root_dir: str, name: str):
        self.root_dir = root_dir
        self.name = name
        self.data = self._assemble_dataframe()

        self.registrator = None

    def _assemble_dataframe(self):
        def _parse_date(filepath: str):
            raw_date = filepath.split("/")[-2].split("_")[1]
            return pd.to_datetime(f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}")

        data = pd.DataFrame(
            [
                (_parse_date(file), file) 
                for file in sorted(glob.glob(os.path.join(self.root_dir, f"{self.name}*", "full.vtp")))
            ],
            columns=["date", "path"]
        )
        data["days"] = (data["date"] - data.iloc[0]["date"]).dt.days
        return data
    
    def fit_LMMDD(
        self, 
        source_t: int = 0, 
        target_ts: List[int] = [-1], 
        centralize: bool = True,
        optimizer_iters: int = 10, 
        ode_steps: int = 10, 
        sigma: float = 20,
        device: str = "cpu"
    ):
        # Load source mesh
        source_days, source_file = self.data.iloc[source_t][["days", "path"]]
        source_mesh = pv.read(source_file)

        # Load target meshes
        target_dict = {
            self.data.iloc[i]["days"] - source_days: pv.read(self.data.iloc[i]["path"]) for i in target_ts
        }
        target_dict = {
            round(ode_steps * t / max(target_dict.keys())): mesh for t, mesh in target_dict.items()
        }

        # Rigid centralization
        if centralize:
            source_mesh.points -= source_mesh.points.mean(axis=0)
            
            for k in target_dict.keys():
                target_dict[k].points -= target_dict[k].points.mean(axis=0)

        # Optimize
        self.registrator = LDDMMRegistrator(source_mesh, target_dict, sigma=sigma, device=device)
        self.registrator.optimize(optimizer_iters, ode_steps)

    def infer_LMMDD(self, output_dir: str, steps: int = 15, deltat: int = 1.0):
        self.registrator.export_shoot(output_dir, nt=steps, deltat=deltat)


root_dir = "/home/rygielpt/data/lddmm/cases"
patient_id = "P20"

sample = LongitudinalSample(root_dir, patient_id)
sample.fit_LMMDD(0, [-2, -1], device="cuda")