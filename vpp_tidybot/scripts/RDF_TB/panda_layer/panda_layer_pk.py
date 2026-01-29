# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the RDF project.
# Copyright (c) 2023 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yimming.li@idiap.ch>
# -----------------------------------------------------------------------------

# panda layer implementation using pytorch kinematics
import torch
import trimesh
import glob
import os
import numpy as np
import pytorch_kinematics as pk
import copy

CUR_PATH = os.path.dirname(os.path.realpath(__file__))


def save_to_mesh(vertices, faces, output_mesh_path=None):
    assert output_mesh_path is not None
    with open(output_mesh_path, "w") as fp:
        for vert in vertices:
            fp.write("v %f %f %f\n" % (vert[0], vert[1], vert[2]))
        for face in faces + 1:
            fp.write("f %d %d %d\n" % (face[0], face[1], face[2]))
    print("Output mesh saved to:", os.path.abspath(output_mesh_path))


class PandaLayer(torch.nn.Module):
    def __init__(
        self,
        device="cpu",
        mesh_path="/Users/stav.42/f_lab/tidybot_model/iiwa_description/meshes/iiwa7/collision/*.stl",
    ):
        super().__init__()
        self.device = device

        # Your mobile-base + IIWA URDF
        urdf_path = "/Users/stav.42/f_lab/tidybot_model/model/stanford_tidybot/base_fix.urdf"
        self.urdf_path = urdf_path

        # Mesh path (we'll override this from __main__)
        self.mesh_path = mesh_path

        # Build kinematic chain from URDF
        # end_link_name: last link in the chain
        # root_link_name: base link of the chain (mobile base)
        self.chain = pk.build_serial_chain_from_urdf(
            open(self.urdf_path).read(),
            end_link_name="iiwa_link_7",
            root_link_name="mobile_base",
        ).to(dtype=torch.float32, device=self.device)

        # Joint limits
        joint_lim = torch.tensor(self.chain.get_joint_limits(), dtype=torch.float32)
        self.theta_min = joint_lim[:, 0].to(self.device)
        self.theta_max = joint_lim[:, 1].to(self.device)

        self.theta_mid = (self.theta_min + self.theta_max) / 2.0
        self.theta_min_soft = (self.theta_min - self.theta_mid) * 0.8 + self.theta_mid
        self.theta_max_soft = (self.theta_max - self.theta_mid) * 0.8 + self.theta_mid
        self.dof = len(self.theta_min)

        # Load all link meshes
        self.meshes = self.load_meshes()

    def load_meshes(self):
        """
        Load STL meshes and name them so that keys match URDF link names:
          - link_0.stl -> "iiwa_link_0"
          - ...
          - link_7.stl -> "iiwa_link_7"
          - mobile_base.stl -> "mobile_base"
        """
        mesh_files = glob.glob(self.mesh_path)
        mesh_files = [f for f in mesh_files if os.path.isfile(f)]

        meshes = {}
        for mesh_file in mesh_files:
            fname = os.path.basename(mesh_file)[:-4]  # strip ".stl"
            # Map filename to URDF link name
            if fname.startswith("link_"):
                # iiwa arm link
                name = "iiwa_" + fname  # e.g. "iiwa_link_0"
            else:
                # mobile_base.stl -> "mobile_base"
                name = fname

            mesh = trimesh.load(mesh_file, force="mesh")
            meshes[name] = mesh

        print("Loaded meshes with keys:", list(meshes.keys()))
        return meshes

    def forward_kinematics(self, theta):
        """
        Run FK and return a dict: link_name -> 4x4 transform
        with link_name matching the URDF names
        (mobile_base, iiwa_link_0, ..., iiwa_link_7).
        """
        ret = self.chain.forward_kinematics(theta, end_only=False)
        transformations = {}
        for frame_name, frame in ret.items():
            trans_mat = frame.get_matrix()  # (B, 4, 4)
            # frame_name is already something like "mobile_base", "iiwa_link_0", ...
            transformations[frame_name] = trans_mat
        return transformations

    def theta2mesh(self, theta):
        """
        Take joint angles theta: (B, 7)
        and return a list of trimesh meshes with all vertices transformed to world.
        """
        trans = self.forward_kinematics(theta)
        robot_mesh = []

        for name, mesh in self.meshes.items():
            if name not in trans:
                # Mesh has no corresponding transform in the chain (e.g. stray file)
                print(f"[theta2mesh] No transform for mesh '{name}', skipping.")
                continue

            T = trans[name].squeeze(0)  # (4, 4) for batch_size = 1

            # Homogeneous coordinates for vertices
            vertices = torch.from_numpy(mesh.vertices).to(self.device, dtype=torch.float32)
            ones = torch.ones((vertices.shape[0], 1), device=self.device, dtype=torch.float32)
            vertices_h = torch.cat([vertices, ones], dim=-1).t()  # (4, N)

            transformed_vertices = (T @ vertices_h).t()[:, :3].detach().cpu().numpy()

            mesh_copy = copy.deepcopy(mesh)
            mesh_copy.vertices = transformed_vertices
            robot_mesh.append(mesh_copy)

        return robot_mesh


if __name__ == "__main__":
    # ---- Device selection: MPS if available, else CPU ----
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # ---- Instantiate PandaLayer with your collision meshes ----
    panda = PandaLayer(
        device=device,
        mesh_path="/Users/stav.42/f_lab/tidybot_model/iiwa_description/meshes/iiwa7/collision/*.stl",
    ).to(device)

    # Example joint configuration (7 DOF)
    theta = torch.tensor(
        [0.0, -0.5, 0.0, 1.0, 0.0, 1.2, 0.2],   # [j1..j7] in radians
        dtype=torch.float32,
        device=device,
    ).reshape(1, 7)

    # Just to inspect available frames
    trans = panda.forward_kinematics(theta)
    print("FK frames:", list(trans.keys()))


    print("FK frames:", list(trans.keys()))
    print("Mesh keys:", list(panda.meshes.keys()))
    # Build robot mesh in world frame
    robot_mesh_list = panda.theta2mesh(theta)

    # Visualize with trimesh
    scene = trimesh.Scene()
    for m in robot_mesh_list:
        scene.add_geometry(m)
    scene.show()
