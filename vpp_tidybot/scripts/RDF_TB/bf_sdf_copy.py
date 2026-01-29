# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the RDF project.
# Copyright (c) 2023 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yimming.li@idiap.ch>
# -----------------------------------------------------------------------------

import torch
import os
import numpy as np
np.set_printoptions(threshold=np.inf)
import glob
import trimesh
try:
    from . import utils as utils
except ImportError:
    import utils as utils
import mesh_to_sdf
import skimage
from RDF_TB.panda_layer.panda_layer_textured import PandaLayerMovableBase
import argparse

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


class BPSDF():
    def __init__(self, n_func, domain_min, domain_max, robot, device):
        self.n_func = n_func
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.device = device
        self.robot = robot
        self.model_path = os.path.join(CUR_DIR, 'models')

    # -------------------------------------------------------------------------
    # Bernstein basis utils
    # -------------------------------------------------------------------------
    def binomial_coefficient(self, n, k):
        n = torch.tensor(float(n), device=self.device, dtype=torch.float32)
        k = k.to(dtype=torch.float32)
        return torch.exp(
            torch.lgamma(n + 1)
            - torch.lgamma(k + 1)
            - torch.lgamma(n - k + 1)
        )

    def build_bernstein_t(self, t, use_derivative=False):
        t = torch.clamp(t, min=1e-4, max=1 - 1e-4)
        n = self.n_func - 1
        i = torch.arange(self.n_func, device=self.device)

        comb = self.binomial_coefficient(n, i)  # float32
        phi = comb * (1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** i
        if not use_derivative:
            return phi.float(), None
        else:
            dphi = -comb * (n - i) * (1 - t).unsqueeze(-1) ** (n - i - 1) * t.unsqueeze(-1) ** i \
                   + comb * i * (1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** (i - 1)
            dphi = torch.clamp(dphi, min=-1e4, max=1e4)
            return phi.float(), dphi.float()

    def build_basis_function_from_points(self, p, use_derivative=False):
        N = len(p)
        p = ((p - self.domain_min) / (self.domain_max - self.domain_min)).reshape(-1)
        phi, d_phi = self.build_bernstein_t(p, use_derivative)
        phi = phi.reshape(N, 3, self.n_func)

        phi_x = phi[:, 0, :]
        phi_y = phi[:, 1, :]
        phi_z = phi[:, 2, :]

        phi_xy = torch.einsum("ij,ik->ijk", phi_x, phi_y).view(-1, self.n_func ** 2)
        phi_xyz = torch.einsum("ij,ik->ijk", phi_xy, phi_z).view(-1, self.n_func ** 3)

        if not use_derivative:
            return phi_xyz, None
        else:
            d_phi = d_phi.reshape(N, 3, self.n_func)
            d_phi_x_1D = d_phi[:, 0, :]
            d_phi_y_1D = d_phi[:, 1, :]
            d_phi_z_1D = d_phi[:, 2, :]

            d_phi_x = torch.einsum(
                "ij,ik->ijk",
                torch.einsum("ij,ik->ijk", d_phi_x_1D, phi_y).view(-1, self.n_func ** 2),
                phi_z,
            ).view(-1, self.n_func ** 3)
            d_phi_y = torch.einsum(
                "ij,ik->ijk",
                torch.einsum("ij,ik->ijk", phi_x, d_phi_y_1D).view(-1, self.n_func ** 2),
                phi_z,
            ).view(-1, self.n_func ** 3)
            d_phi_z = torch.einsum(
                "ij,ik->ijk",
                phi_xy,
                d_phi_z_1D,
            ).view(-1, self.n_func ** 3)

            d_phi_xyz = torch.cat(
                (d_phi_x.unsqueeze(-1), d_phi_y.unsqueeze(-1), d_phi_z.unsqueeze(-1)),
                dim=-1,
            )
            return phi_xyz, d_phi_xyz

    # -------------------------------------------------------------------------
    # *** FIXED ***  Training Bernstein-SDF for each link
    # -------------------------------------------------------------------------
    def train_bf_sdf(self, epoches=200):
        """
        Train one Bernstein polynomial SDF per link.

        IMPORTANT: The order of entries in mesh_dict (index 0..8) is:
          0: mobile_base
          1: link_0
          2: link_1
          ...
          8: link_7

        This matches PandaLayer.link_names:
          ["mobile_base", "iiwa_link_0", ..., "iiwa_link_7"]

        So index i in the trained model corresponds to transform i from
        PandaLayer.get_transformations_each_link().
        """

        # All collision meshes
        mesh_path = "/Users/stav.42/f_lab/tidybot_model/iiwa_description/meshes/iiwa7/collision/*.stl"
        mesh_files = glob.glob(mesh_path)

        # Map "basename without .stl" -> full path
        mesh_name_to_file = {
            os.path.basename(f).split('.')[0]: f
            for f in mesh_files
        }

        # Desired order of links in the model
        ordered_mesh_names = ["mobile_base"] + [f"link_{i}" for i in range(8)]

        mesh_dict = {}

        for i, mesh_name in enumerate(ordered_mesh_names):
            mf = mesh_name_to_file[mesh_name]
            print(mesh_name)

            # Load mesh
            mesh = trimesh.load(mf)

            # Offset/scale in ORIGINAL mesh space
            offset_np = np.array(mesh.bounding_box.centroid, dtype=np.float32)
            scale_np = np.max(np.linalg.norm(mesh.vertices - offset_np, axis=1)).astype(np.float32)

            # For training, we assume you already generated SDF samples in the
            # same canonical space (unit sphere) and saved them here:
            data_path = os.path.join(
                CUR_DIR, "data", "sdf_points", f"voxel_128_{mesh_name}.npy"
            )
            data = np.load(data_path, allow_pickle=True).item()

            point_near_data = data['near_points']
            sdf_near_data = data['near_sdf']
            point_random_data = data['random_points']
            sdf_random_data = data['random_sdf']

            # small hack from original code
            sdf_random_data[sdf_random_data < -1] = -sdf_random_data[sdf_random_data < -1]

            # Initialize posterior parameters
            wb = torch.zeros(self.n_func ** 3, dtype=torch.float32, device=self.device)
            B = torch.eye(self.n_func ** 3, dtype=torch.float32, device=self.device) / 1e-4

            for _ in range(epoches):
                # sample near-surface points
                choice_near = np.random.choice(len(point_near_data), 1024, replace=False)
                p_near = torch.from_numpy(point_near_data[choice_near]).float().to(self.device)
                sdf_near = torch.from_numpy(sdf_near_data[choice_near]).float().to(self.device)

                # sample random points
                choice_random = np.random.choice(len(point_random_data), 256, replace=False)
                p_random = torch.from_numpy(point_random_data[choice_random]).float().to(self.device)
                sdf_random = torch.from_numpy(sdf_random_data[choice_random]).float().to(self.device)

                p = torch.cat([p_near, p_random], dim=0)
                sdf = torch.cat([sdf_near, sdf_random], dim=0)

                phi_xyz, _ = self.build_basis_function_from_points(
                    p.float().to(self.device),
                    use_derivative=False,
                )

                I = torch.eye(len(p), dtype=torch.float32, device=self.device)
                K = B @ phi_xyz.T @ torch.linalg.inv(I + (phi_xyz @ B @ phi_xyz.T))
                B = B - K @ phi_xyz @ B
                delta_wb = K @ (sdf - phi_xyz @ wb).squeeze()
                wb = wb + delta_wb

            print(f"mesh name {mesh_name} finished!")
            mesh_dict[i] = {
                "mesh_name": mesh_name,
                "weights": wb.cpu(),  # keep on CPU
                "offset": torch.from_numpy(offset_np.copy()).float(),  # avoid non-writable warning
                "scale": float(scale_np),
            }

        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        out_path = os.path.join(self.model_path, f"BP_{self.n_func}.pt")
        torch.save(mesh_dict, out_path)
        print(f"{out_path} model saved!")

    # -------------------------------------------------------------------------
    # SDF -> mesh reconstruction (per link)
    # -------------------------------------------------------------------------
    def sdf_to_mesh(self, model, nbData, use_derivative=False):
        verts_list, faces_list, mesh_name_list = [], [], []
        for i, k in enumerate(model.keys()):
            mesh_dict = model[k]
            mesh_name = mesh_dict['mesh_name']
            print(f'{mesh_name}')
            mesh_name_list.append(mesh_name)
            weights = mesh_dict['weights'].float().to(self.device)

            domain = torch.linspace(
                self.domain_min,
                self.domain_max,
                nbData,
                device=self.device,
                dtype=torch.float32
            )
            grid_x, grid_y, grid_z = torch.meshgrid(domain, domain, domain, indexing='ij')
            grid_x, grid_y, grid_z = grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), grid_z.reshape(-1, 1)
            p = torch.cat([grid_x, grid_y, grid_z], dim=1).float().to(self.device)

            p_split = torch.split(p, 10000, dim=0)
            d_list = []
            for p_s in p_split:
                phi_p, _ = self.build_basis_function_from_points(p_s, use_derivative)
                d_s = phi_p @ weights
                d_list.append(d_s)
            d = torch.cat(d_list, dim=0)

            d_grid = d.view(nbData, nbData, nbData).detach().cpu().numpy()
            spacing = np.array([(self.domain_max - self.domain_min) / nbData] * 3)
            verts, faces, normals, values = skimage.measure.marching_cubes(
                d_grid, level=0.0, spacing=spacing
            )
            verts = verts - [1, 1, 1]
            verts_list.append(verts)
            faces_list.append(faces)
        return verts_list, faces_list, mesh_name_list

    def create_surface_mesh(self, model, nbData, vis=False, save_mesh_name=None):
        verts_list, faces_list, mesh_name_list = self.sdf_to_mesh(model, nbData)
        vis = False
        # print("Reconstructed meshes:")
        print("Mesh names:", mesh_name_list)
        for verts, faces, mesh_name in zip(verts_list, faces_list, mesh_name_list):
            rec_mesh = trimesh.Trimesh(verts, faces)
            if vis:
                rec_mesh.show(viewer='gl')
                print("Current mesh:", mesh_name)
            if save_mesh_name is not None:
                save_path = os.path.join(CUR_DIR, "output_meshes")
                if os.path.exists(save_path) is False:
                    os.mkdir(save_path)
                # print("Current mesh:", mesh_name)
                print("Saving to:", os.path.join(save_path, f"{save_mesh_name}_{mesh_name}.stl"))
                trimesh.exchange.export.export_mesh(
                    rec_mesh,
                    os.path.join(save_path, f"{save_mesh_name}_{mesh_name}.stl")
                )

    # -------------------------------------------------------------------------
    # Whole-body SDF
    # -------------------------------------------------------------------------
    def get_whole_body_sdf_batch(
        self,
        x,
        pose,
        theta,
        model,
        use_derivative=True,
        used_links=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        return_index=False
    ):
        B = len(theta)
        N = len(x)
        K = len(used_links)

        # offsets
        offset = torch.cat(
            [model[i]['offset'].unsqueeze(0) for i in used_links],
            dim=0
        )
        offset = offset.float().to(self.device)
        offset = offset.unsqueeze(0).expand(B, K, 3).reshape(B * K, 3)

        # scales
        scale = torch.tensor(
            [model[i]['scale'] for i in used_links],
            dtype=torch.float32,
            device=self.device,
        )
        scale = scale.unsqueeze(0).expand(B, K).reshape(B * K)

        # FK transforms (world -> link)
        trans_list = self.robot.get_transformations_each_link(pose, theta)
        fk_trans = torch.cat(
            [t.unsqueeze(1) for t in trans_list],
            dim=1
        )[:, used_links, :, :].reshape(-1, 4, 4)  # (B*K,4,4)

        # transform query points into each link frame
        x_robot_frame_batch = utils.transform_points(
            x.float(),
            torch.linalg.inv(fk_trans).float(),
            device=self.device
        )  # (B*K,N,3)

        x_robot_frame_batch_scaled = x_robot_frame_batch - offset.unsqueeze(1)
        x_robot_frame_batch_scaled = x_robot_frame_batch_scaled / scale.unsqueeze(-1).unsqueeze(-1)

        x_bounded = torch.where(
            x_robot_frame_batch_scaled > 1.0 - 1e-2,
            torch.tensor(1.0 - 1e-2, device=self.device),
            x_robot_frame_batch_scaled
        )
        x_bounded = torch.where(
            x_bounded < -1.0 + 1e-2,
            torch.tensor(-1.0 + 1e-2, device=self.device),
            x_bounded
        )
        res_x = x_robot_frame_batch_scaled - x_bounded

        if not use_derivative:
            phi, _ = self.build_basis_function_from_points(
                x_bounded.reshape(B * K * N, 3),
                use_derivative=False
            )
            phi = phi.reshape(B, K, N, -1).transpose(0, 1).reshape(K, B * N, -1)

            weights_near = torch.cat(
                [model[i]['weights'].float().unsqueeze(0) for i in used_links],
                dim=0
            ).to(self.device)

            sdf = torch.einsum('ijk,ik->ij', phi, weights_near).reshape(K, B, N).transpose(0, 1).reshape(B * K, N)
            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(B, K, N)
            sdf = sdf * scale.reshape(B, K).unsqueeze(-1)
            sdf_value, idx = sdf.min(dim=1)
            if return_index:
                return sdf_value, None, idx
            return sdf_value, None
        else:
            phi, dphi = self.build_basis_function_from_points(
                x_bounded.reshape(B * K * N, 3),
                use_derivative=True
            )
            phi_cat = torch.cat([phi.unsqueeze(-1), dphi], dim=-1)
            phi_cat = phi_cat.reshape(B, K, N, -1, 4).transpose(0, 1).reshape(K, B * N, -1, 4)

            weights_near = torch.cat(
                [model[i]['weights'].float().unsqueeze(0) for i in used_links],
                dim=0
            ).to(self.device)

            output = torch.einsum('ijkl,ik->ijl', phi_cat, weights_near).reshape(K, B, N, 4).transpose(0, 1).reshape(B * K, N, 4)
            sdf = output[:, :, 0]
            gradient = output[:, :, 1:]

            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(B, K, N)
            sdf = sdf * scale.reshape(B, K).unsqueeze(-1)
            sdf_value, idx = sdf.min(dim=1)

            gradient = res_x + torch.nn.functional.normalize(gradient, dim=-1)
            gradient = torch.nn.functional.normalize(gradient, dim=-1).float()

            fk_rotation = fk_trans[:, :3, :3]
            gradient_base_frame = torch.einsum(
                'ijk,ikl->ijl',
                fk_rotation,
                gradient.transpose(1, 2)
            ).transpose(1, 2).reshape(B, K, N, 3)

            idx_grad = idx.unsqueeze(1).unsqueeze(-1).expand(B, K, N, 3)
            gradient_value = torch.gather(gradient_base_frame, 1, idx_grad)[:, 0, :, :]

            if return_index:
                return sdf_value, gradient_value, idx
            return sdf_value, gradient_value


    def get_whole_body_sdf_batch_base(
        self,
        x,
        pose,
        theta,
        model,
        use_derivative=True,
        used_links=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        return_index=False
    ):
        B = len(theta)
        N = len(x)
        K = len(used_links)

        # offsets
        offset = torch.cat(
            [model[i]['offset'].unsqueeze(0) for i in used_links],
            dim=0
        )
        offset = offset.float().to(self.device)
        offset = offset.unsqueeze(0).expand(B, K, 3).reshape(B * K, 3)

        # scales
        scale = torch.tensor(
            [model[i]['scale'] for i in used_links],
            dtype=torch.float32,
            device=self.device,
        )
        scale = scale.unsqueeze(0).expand(B, K).reshape(B * K)

        # FK transforms (world -> link)
        trans_list = self.robot.get_transformations_each_link(theta)
        fk_trans = torch.cat(
            [t.unsqueeze(1) for t in trans_list],
            dim=1
        )[:, used_links, :, :].reshape(-1, 4, 4)  # (B*K,4,4)

        # transform query points into each link frame
        x_robot_frame_batch = utils.transform_points(
            x.float(),
            torch.linalg.inv(fk_trans).float(),
            device=self.device
        )  # (B*K,N,3)

        x_robot_frame_batch_scaled = x_robot_frame_batch - offset.unsqueeze(1)
        x_robot_frame_batch_scaled = x_robot_frame_batch_scaled / scale.unsqueeze(-1).unsqueeze(-1)

        x_bounded = torch.where(
            x_robot_frame_batch_scaled > 1.0 - 1e-2,
            torch.tensor(1.0 - 1e-2, device=self.device),
            x_robot_frame_batch_scaled
        )
        x_bounded = torch.where(
            x_bounded < -1.0 + 1e-2,
            torch.tensor(-1.0 + 1e-2, device=self.device),
            x_bounded
        )
        res_x = x_robot_frame_batch_scaled - x_bounded

        if not use_derivative:
            phi, _ = self.build_basis_function_from_points(
                x_bounded.reshape(B * K * N, 3),
                use_derivative=False
            )
            phi = phi.reshape(B, K, N, -1).transpose(0, 1).reshape(K, B * N, -1)

            weights_near = torch.cat(
                [model[i]['weights'].float().unsqueeze(0) for i in used_links],
                dim=0
            ).to(self.device)

            sdf = torch.einsum('ijk,ik->ij', phi, weights_near).reshape(K, B, N).transpose(0, 1).reshape(B * K, N)
            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(B, K, N)
            sdf = sdf * scale.reshape(B, K).unsqueeze(-1)
            sdf_value, idx = sdf.min(dim=1)
            if return_index:
                return sdf_value, None, idx
            return sdf_value, None
        else:
            phi, dphi = self.build_basis_function_from_points(
                x_bounded.reshape(B * K * N, 3),
                use_derivative=True
            )
            phi_cat = torch.cat([phi.unsqueeze(-1), dphi], dim=-1)
            phi_cat = phi_cat.reshape(B, K, N, -1, 4).transpose(0, 1).reshape(K, B * N, -1, 4)

            weights_near = torch.cat(
                [model[i]['weights'].float().unsqueeze(0) for i in used_links],
                dim=0
            ).to(self.device)

            output = torch.einsum('ijkl,ik->ijl', phi_cat, weights_near).reshape(K, B, N, 4).transpose(0, 1).reshape(B * K, N, 4)
            sdf = output[:, :, 0]
            gradient = output[:, :, 1:]

            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(B, K, N)
            sdf = sdf * scale.reshape(B, K).unsqueeze(-1)
            sdf_value, idx = sdf.min(dim=1)

            gradient = res_x + torch.nn.functional.normalize(gradient, dim=-1)
            gradient = torch.nn.functional.normalize(gradient, dim=-1).float()

            fk_rotation = fk_trans[:, :3, :3]
            gradient_base_frame = torch.einsum(
                'ijk,ikl->ijl',
                fk_rotation,
                gradient.transpose(1, 2)
            ).transpose(1, 2).reshape(B, K, N, 3)

            idx_grad = idx.unsqueeze(1).unsqueeze(-1).expand(B, K, N, 3)
            gradient_value = torch.gather(gradient_base_frame, 1, idx_grad)[:, 0, :, :]

            if return_index:
                return sdf_value, gradient_value, idx
            return sdf_value, gradient_value

    # -------------------------------------------------------------------------
    # Joints gradients helpers (unchanged)
    # -------------------------------------------------------------------------
    def get_whole_body_sdf_with_joints_grad_batch(
        self,
        x,
        pose,
        theta,
        model,
        used_links=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    ):
        delta = 0.001
        B = theta.shape[0]
        theta = theta.unsqueeze(1)
        d_theta = (theta.expand(B, 7, 7) + torch.eye(7, device=self.device).unsqueeze(0).expand(B, 7, 7) * delta).reshape(B, -1, 7)
        theta = torch.cat([theta, d_theta], dim=1).reshape(B * 8, 7)
        pose = pose.unsqueeze(1).expand(B, 8, 4, 4).reshape(B * 8, 4, 4)
        sdf, _ = self.get_whole_body_sdf_batch(
            x,
            pose,
            theta,
            model,
            use_derivative=False,
            used_links=used_links
        )
        sdf = sdf.reshape(B, 8, -1)
        d_sdf = (sdf[:, 1:, :] - sdf[:, :1, :]) / delta
        return sdf[:, 0, :], d_sdf.transpose(1, 2)

    def get_whole_body_sdf_with_joints_grad_batch_base(
        self,
        x,
        pose,
        theta,
        model,
        used_links=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    ):
        delta = 0.001
        B = theta.shape[0]
        theta = theta.unsqueeze(1)
        d_theta = (theta.expand(B, 9, 9) + torch.eye(9, device=self.device).unsqueeze(0).expand(B, 9, 9) * delta).reshape(B, -1, 9)
        theta = torch.cat([theta, d_theta], dim=1).reshape(B * 10, 9)
        pose = pose.unsqueeze(1).expand(B, 10, 4, 4).reshape(B * 10, 4, 4)
        sdf, _ = self.get_whole_body_sdf_batch_base(
            x,
            pose,
            theta,
            model,
            use_derivative=False,
            used_links=used_links
        )
        sdf = sdf.reshape(B, 10, -1)
        d_sdf = (sdf[:, 1:, :] - sdf[:, :1, :]) / delta
        return sdf[:, 0, :], d_sdf.transpose(1, 2)


    def get_whole_body_normal_with_joints_grad_batch(
        self,
        x,
        pose,
        theta,
        model,
        used_links=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    ):
        delta = 0.001
        B = theta.shape[0]
        theta = theta.unsqueeze(1)
        d_theta = (theta.expand(B, 7, 7) + torch.eye(7, device=self.device).unsqueeze(0).expand(B, 7, 7) * delta).reshape(B, -1, 7)
        theta = torch.cat([theta, d_theta], dim=1).reshape(B * 8, 7)
        pose = pose.unsqueeze(1).expand(B, 8, 4, 4).reshape(B * 8, 4, 4)
        sdf, normal = self.get_whole_body_sdf_batch(
            x,
            pose,
            theta,
            model,
            use_derivative=True,
            used_links=used_links
        )
        normal = normal.reshape(B, 8, -1, 3).transpose(1, 2)
        return normal  # (B,N,8,3)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--domain_max', default=1.0, type=float)
    parser.add_argument('--domain_min', default=-1.0, type=float)
    parser.add_argument('--n_func', default=8, type=int)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    # Resolve device nicely for CPU / CUDA / MPS
    if args.device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        if args.device == 'mps':
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                print("MPS not available, falling back to CPU")
                device = torch.device('cpu')
        elif args.device == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                print("CUDA not available, falling back to CPU")
                device = torch.device('cpu')
        else:
            device = torch.device(args.device)

    print(f"Using device: {device}")

    panda = PandaLayer(device)
    bp_sdf = BPSDF(args.n_func, args.domain_min, args.domain_max, panda, device)

    print("Bernstein polynomial SDF with n_func =", args.n_func)
    # train model (if requested)
    if args.train:
        bp_sdf.train_bf_sdf()

    # load trained model
    model_path = f'models/BP_{args.n_func}.pt'
    model = torch.load(model_path, map_location='cpu', weights_only=False)

    for k in model.keys():
        if 'weights' in model[k]:
            model[k]['weights'] = model[k]['weights'].float()
        if 'offset' in model[k]:
            model[k]['offset'] = model[k]['offset'].float()

    # visualize per-link reconstruction
    bp_sdf.create_surface_mesh(model, nbData=128, vis=True, save_mesh_name=f'BP_{args.n_func}')

    # whole-body SDF sanity check
    theta = torch.tensor(
        [0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4],
        dtype=torch.float32,
        device=device
    ).reshape(-1, 7)
    pose = torch.from_numpy(np.identity(4, dtype=np.float32))
    pose = pose.to(device).reshape(-1, 4, 4).expand(len(theta), 4, 4)
    trans_list = panda.get_transformations_each_link(pose, theta)
    utils.visualize_reconstructed_whole_body(model, trans_list, tag=f'BP_{args.n_func}')

    x = torch.rand(128, 3, device=device) * 2.0 - 1.0
    theta = torch.rand(2, 7, device=device, dtype=torch.float32)
    pose = torch.from_numpy(np.identity(4, dtype=np.float32))
    pose = pose.to(device).unsqueeze(0).expand(len(theta), 4, 4)
    sdf, gradient = bp_sdf.get_whole_body_sdf_batch(x, pose, theta, model, use_derivative=True)
    print('sdf:', sdf.shape, 'gradient:', gradient.shape)
    sdf, joint_grad = bp_sdf.get_whole_body_sdf_with_joints_grad_batch(x, pose, theta, model)
    print('sdf:', sdf.shape, 'joint gradient:', joint_grad.shape)
