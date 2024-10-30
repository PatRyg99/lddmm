# LDDMM with PyTorch and KeOps
Large deformation diffeomorphic metric mapping (LDDMM) in PyTorch with KeOps library.  
More friendly wrapper based on this tutorial:  
https://www.kernel-operations.io/keops/_auto_tutorials/surface_registration/plot_LDDMM_Surface.html

## Installation
Navigate to main direction and run
```
pip install .
```
Now you can import `lddmm` and enjoy.

Alternatively, you can install this without cloning with:

```
pip install git+https://github.com/PatRyg99/lddmm
```

## Getting started

```python
import pyvista as pv
from lddmm.registrator import LDDMMRegistrator

source_mesh = pv.read("<source-mesh-path>")
target_mesh = pv.read("<target-mesh-path>")

# Rigid regsitration
source_mesh.points -= source_mesh.points.mean(axis=0)
target_mesh.points -= target_mesh.points.mean(axis=0)

# Define and run LDDMM registration
registrator = LDDMMRegistrator(source_mesh, target_mesh, sigma=20, device="cpu")
registrator.optimize(2)
```

Obtaining interpolated deformation timepoints:
```python
# Get raw shooting results
shots = registrator.shoot(nt=15)
```

```python
# Get shooting results and save as .vtp files
registrator.export_shoot("shots", nt=15)
```
