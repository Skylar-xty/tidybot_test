import trimesh
import numpy as np

# half extents: (Lx/2, Ly/2, Lz/2)
box = trimesh.creation.box(extents=[0.6, 0.6, 0.2])  # 60x60x20 cm base

trimesh.exchange.export.export_mesh(box, "meshes/base/base.stl")