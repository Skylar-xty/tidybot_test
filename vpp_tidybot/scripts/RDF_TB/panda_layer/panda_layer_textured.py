# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the RDF project.
# Copyright (c) 2023 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yimming.li@idiap.ch>
# -----------------------------------------------------------------------------

# PandaLayer re-implemented for your mobile_base + IIWA URDF using pytorch_kinematics

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
    """
    This is now a generic 'robot layer' for your mobile_base + IIWA arm.

    - FK is done via pytorch_kinematics from your URDF
    - Meshes are loaded from your iiwa_description collision/visual folder
    - get_transformations_each_link(pose, theta) returns world transforms for each link
    - get_forward_robot_mesh(pose, theta) returns per-link trimesh meshes in world frame
    """

    def __init__(
        self,
        device="cpu",
        urdf_path=None,
        mesh_path=None,
    ):
        super().__init__()

        # ---- Device handling ----
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # ---- Paths ----
        # Default to your URDF if not provided
        if urdf_path is None:
            # Build an absolute path relative to this file so the code works
            # regardless of the current working directory when imported/run.
            urdf_path = os.path.abspath(
                os.path.join(CUR_PATH, "..", "..", "..", "models", "urdf", "tidybot", "base_fix.urdf")
            )
        self.urdf_path = urdf_path

        # Default to your IIWA meshes if not provided
        if mesh_path is None:
            mesh_path = os.path.abspath(
                os.path.join(CUR_PATH, "..", "..", "..", "models", "urdf", "tidybot", "iiwa_description", "meshes", "iiwa7", "collision", "*.stl")
            )
        self.mesh_path = mesh_path

        # ---- Build kinematic chain from URDF ----
        # root_link_name = "mobile_base", end_link_name = "iiwa_link_7"
        with open(self.urdf_path, "r") as f:
            urdf_str = f.read()

        self.chain = pk.build_serial_chain_from_urdf(
            urdf_str,
            end_link_name="iiwa_link_7",
            root_link_name="mobile_base",
        ).to(dtype=torch.float32, device=self.device)

        # link order we care about (must match how your SDF model is built)
        # 0: mobile_base, 1..8: iiwa_link_0..7
        self.link_names = ["mobile_base"] + [f"iiwa_link_{i}" for i in range(8)]

        # ---- Joint limits ----
        joint_lim = torch.tensor(self.chain.get_joint_limits(), dtype=torch.float32, device=self.device)
        self.theta_min = joint_lim[:, 0]
        self.theta_max = joint_lim[:, 1]

        self.theta_mid = (self.theta_min + self.theta_max) / 2.0
        self.theta_min_soft = (self.theta_min - self.theta_mid) * 0.8 + self.theta_mid
        self.theta_max_soft = (self.theta_max - self.theta_mid) * 0.8 + self.theta_mid
        self.dof = len(self.theta_min)

        # ---- Load link meshes ----
        self.meshes = self._load_meshes()

    # -------------------------------------------------------------------------
    # Mesh loading
    # -------------------------------------------------------------------------
    def _load_meshes(self):
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

            if fname.startswith("link_"):
                # map "link_0" -> "iiwa_link_0"
                name = "iiwa_" + fname
            else:
                # "mobile_base" -> "mobile_base"
                name = fname

            mesh = trimesh.load(mesh_file, force="mesh")
            meshes[name] = mesh

        print("Loaded meshes with keys:", list(meshes.keys()))
        return meshes

    # -------------------------------------------------------------------------
    # FK helpers
    # -------------------------------------------------------------------------
    def forward_kinematics(self, theta, end_only=False):
        """
        Thin wrapper around pytorch_kinematics forward_kinematics.

        theta: (B,7) joint angles
        returns dict: link_name -> Frame (B,4,4) when end_only=False
        """
        return self.chain.forward_kinematics(theta, end_only=end_only)

    def get_transformations_each_link(self, pose, theta):
        """
        Returns world transforms for each link we care about.

        pose:  (B,4,4)  world -> mobile_base transform
        theta: (B,7)    joint angles

        returns: list of length 9
          [T_world_mobile_base, T_world_iiwa_link_0, ..., T_world_iiwa_link_7]
        where each element is (B,4,4)
        """
        # FK in mobile_base frame
        fk = self.chain.forward_kinematics(theta, end_only=False)

        transforms = []
        for name in self.link_names:
            T_link = fk[name].get_matrix()  # (B,4,4) in mobile_base frame
            # world -> link = (world->mobile_base) @ (mobile_base->link)
            T_world = torch.matmul(pose, T_link)
            transforms.append(T_world)

        return transforms

    # -------------------------------------------------------------------------
    # Mesh helpers
    # -------------------------------------------------------------------------
    def theta2mesh(self, theta):
        """
        Convenience: assume pose = identity (world == mobile_base),
        return list of meshes transformed to world frame for batch_size=1.

        theta: (1,7)
        """
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        B = theta.shape[0]
        assert B == 1, "theta2mesh currently only supports batch_size=1"

        # pose = identity
        pose = torch.eye(4, dtype=torch.float32, device=self.device).unsqueeze(0)
        batch_meshes = self.get_forward_robot_mesh(pose, theta)
        return batch_meshes[0]

    def get_forward_robot_mesh(self, pose, theta):
        """
        Build transformed trimesh meshes for each link in the batch.

        pose:  (B,4,4) world -> mobile_base
        theta: (B,7)

        returns: list of length B
          each element is a list of trimesh.Trimesh for each link that has a mesh
        """
        transforms = self.get_transformations_each_link(pose, theta)
        batch_size = pose.shape[0]

        out = []
        for b in range(batch_size):
            meshes_this = []
            for name, T in zip(self.link_names, transforms):
                if name not in self.meshes:
                    continue  # no mesh for this link
                mesh = copy.deepcopy(self.meshes[name])

                # vertices -> tensor (N,3)
                verts = torch.from_numpy(mesh.vertices).to(self.device, dtype=torch.float32)
                ones = torch.ones((verts.shape[0], 1), device=self.device, dtype=torch.float32)
                verts_h = torch.cat([verts, ones], dim=-1).t()  # (4,N)

                T_b = T[b]  # (4,4)
                verts_world = (T_b @ verts_h).t()[:, :3].detach().cpu().numpy()
                mesh.vertices = verts_world
                meshes_this.append(mesh)
            out.append(meshes_this)

        return out


if __name__ == "__main__":
    # Pick a reasonable default device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    panda = PandaLayer(
        device=device,
        urdf_path="G:/My Drive/PENN/vpp_tidybot_test/vpp_tidybot/models/urdf/tidybot/base_move.urdf",
        mesh_path="G:/My Drive/PENN/vpp_tidybot_test/vpp_tidybot/models/urdf/tidybot/iiwa7/collision/*.stl",
        # urdf_path="/Users/stav.42/f_lab/tidybot_model/model/stanford_tidybot/base_fix.urdf",
        # mesh_path="/Users/stav.42/f_lab/tidybot_model/iiwa_description/meshes/iiwa7/visual/*.stl",
    )

    # example joint config
    theta = torch.tensor(
        [0.0, -0.5, 0.0, 1.0, 0.0, 1.2, 0.2],
        dtype=torch.float32,
        device=panda.device,
    ).reshape(1, 7)

    # world -> mobile_base = identity for now
    pose = torch.eye(4, dtype=torch.float32, device=panda.device).unsqueeze(0)

    # FK check
    fk = panda.forward_kinematics(theta, end_only=False)
    print("FK frames:", list(fk.keys()))

    # Build and visualize meshes
    batch_meshes = panda.get_forward_robot_mesh(pose, theta)
    robot_mesh_list = batch_meshes[0]

    scene = trimesh.Scene()
    for m in robot_mesh_list:
        scene.add_geometry(m)
    scene.show()

class PandaLayerMovableBase(torch.nn.Module):
    def __init__(
        self,
        device="cpu",
        urdf_path=None,
        mesh_path=None,
    ):
        super().__init__()
        
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # ---- Paths ----
        if urdf_path is None:
            # Ensure this points to your NEW urdf with prismatic joints
            urdf_path = "G:/My Drive/PENN/vpp_tidybot_test/vpp_tidybot/models/urdf/tidybot/base_move.urdf"
            # urdf_path = "/Users/stav.42/f_lab/tidybot_model/model/stanford_tidybot/base_move.urdf"
        self.urdf_path = urdf_path

        if mesh_path is None:
            mesh_path = "G:/My Drive/PENN/vpp_tidybot_test/vpp_tidybot/models/urdf/tidybot/iiwa7/collision/*.stl"
            # mesh_path = "/Users/stav.42/f_lab/tidybot_model/iiwa_description/meshes/iiwa7/collision/*.stl"
        self.mesh_path = mesh_path

        # ---- Build kinematic chain ----
        # CRITICAL CHANGE: Start from "world" to include the prismatic base joints
        with open(self.urdf_path, "r") as f:
            urdf_str = f.read()

        self.chain = pk.build_serial_chain_from_urdf(
            urdf_str,
            end_link_name="iiwa_link_7",
            root_link_name="world", 
        ).to(dtype=torch.float32, device=self.device)

        # List of links we actually want to visualize/collide
        # Note: 'world' and 'world_x_link' are virtual, so we usually don't visualize them
        self.link_names = ["mobile_base"] + [f"iiwa_link_{i}" for i in range(8)]

        # ---- Joint limits ----
        # The chain now contains 9 joints: [prismatic_x, prismatic_y, joint_1 ... joint_7]
        joint_lim = torch.tensor(self.chain.get_joint_limits(), dtype=torch.float32, device=self.device)
        self.theta_min = joint_lim[:, 0]
        self.theta_max = joint_lim[:, 1]
        
        # Override Base limits manually if needed (URDF usually has arbitrary large limits for base)
        # self.theta_min[:2] = -10.0
        # self.theta_max[:2] = 10.0

        self.dof = len(self.theta_min)  # Should be 9
        print(f"Robot Loaded. DOF: {self.dof}. Joint names: {self.chain.get_joint_parameter_names()}")

        self.meshes = self._load_meshes()

    def _load_meshes(self):
        mesh_files = glob.glob(self.mesh_path)
        mesh_files = [f for f in mesh_files if os.path.isfile(f)]
        meshes = {}
        for mesh_file in mesh_files:
            fname = os.path.basename(mesh_file)[:-4]
            if fname.startswith("link_"):
                name = "iiwa_" + fname
            else:
                name = fname
            mesh = trimesh.load(mesh_file, force="mesh")
            meshes[name] = mesh
        return meshes

    # -------------------------------------------------------------------------
    # Forward Kinematics
    # -------------------------------------------------------------------------
    def get_transformations_each_link(self, theta):
        """
        theta: (B, 9) -> [base_x, base_y, q1, q2, q3, q4, q5, q6, q7]
        """
        # Because we built the chain from "world", we just pass the full 9-DOF vector
        # The chain handles the prismatic joints automatically.
        fk = self.chain.forward_kinematics(theta, end_only=False)

        transforms = []
        for name in self.link_names:
            if name in fk:
                transforms.append(fk[name].get_matrix())
            else:
                # Fallback if a link name implies identity or isn't in chain
                # (Shouldn't happen with correct URDF)
                B = theta.shape[0]
                transforms.append(torch.eye(4, device=self.device).unsqueeze(0).repeat(B, 1, 1))

        return transforms

    def get_forward_robot_mesh(self, theta):
        """
        theta: (B, 9)
        """
        transforms = self.get_transformations_each_link(theta)
        batch_size = theta.shape[0]

        out = []
        for b in range(batch_size):
            meshes_this = []
            for name, T in zip(self.link_names, transforms):
                if name not in self.meshes:
                    continue
                mesh = copy.deepcopy(self.meshes[name])
                
                # Vertices in local frame
                verts = torch.from_numpy(mesh.vertices).to(self.device, dtype=torch.float32)
                ones = torch.ones((verts.shape[0], 1), device=self.device, dtype=torch.float32)
                verts_h = torch.cat([verts, ones], dim=-1).t()

                # Transform to world
                T_b = T[b]
                verts_world = (T_b @ verts_h).t()[:, :3].detach().cpu().numpy()
                mesh.vertices = verts_world
                meshes_this.append(mesh)
            out.append(meshes_this)

        return out

if __name__ == "__main__":
    # Pick a reasonable default device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    panda = PandaLayer(
        device=device,
        urdf_path="G:/My Drive/PENN/vpp_tidybot_test/vpp_tidybot/models/urdf/tidybot/base_move.urdf",
        mesh_path="G:/My Drive/PENN/vpp_tidybot_test/vpp_tidybot/models/urdf/tidybot/iiwa7/collision/*.stl",
        # urdf_path="/Users/stav.42/f_lab/tidybot_model/model/stanford_tidybot/base_fix.urdf",
        # mesh_path="/Users/stav.42/f_lab/tidybot_model/iiwa_description/meshes/iiwa7/visual/*.stl",
    )

    # example joint config
    theta = torch.tensor(
        [0.0, -0.5, 0.0, 1.0, 0.0, 1.2, 0.2],
        dtype=torch.float32,
        device=panda.device,
    ).reshape(1, 7)

    # world -> mobile_base = identity for now
    pose = torch.eye(4, dtype=torch.float32, device=panda.device).unsqueeze(0)

    # FK check
    fk = panda.forward_kinematics(theta, end_only=False)
    print("FK frames:", list(fk.keys()))

    # Build and visualize meshes
    batch_meshes = panda.get_forward_robot_mesh(pose, theta)
    robot_mesh_list = batch_meshes[0]

    scene = trimesh.Scene()
    for m in robot_mesh_list:
        scene.add_geometry(m)
    scene.show()
