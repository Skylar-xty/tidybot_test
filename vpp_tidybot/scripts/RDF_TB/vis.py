# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the RDF project.
# Copyright (c) 2023 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

import torch
import os
from panda_layer.panda_layer_textured import PandaLayer
import bf_sdf
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import utils
import argparse
import skimage

def visualize_sdf_shells(panda, bp_sdf, model, pose, theta, device,
                         nbData=96,
                         levels=(0.0, 0.015, 0.03, 0.045, 0.06)):
    """
    Visualize multiple whole-body SDF isosurfaces φ(x) = level around the robot.

    levels are in *meters* (because get_whole_body_sdf_batch multiplies back by the link scales).
    """

    # 1) Build a regular 3D grid in the Bernstein domain
    domain = torch.linspace(
        bp_sdf.domain_min,
        bp_sdf.domain_max,
        nbData,
        device=device,
        dtype=torch.float32,
    )

    grid_x, grid_y, grid_z = torch.meshgrid(domain, domain, domain, indexing="ij")
    points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)  # (nbData^3, 3)

    # 2) Evaluate whole-body SDF on this grid in chunks to avoid memory blowup
    sdf_chunks = []
    with torch.no_grad():
        for pts_chunk in torch.split(points, 50000, dim=0):
            sdf_chunk, _ = bp_sdf.get_whole_body_sdf_batch(
                pts_chunk,
                pose,
                theta,
                model,
                use_derivative=False,
            )
            # sdf_chunk shape: (B, N_chunk). We have B = 1 here.
            sdf_chunks.append(sdf_chunk[0].detach().cpu())

    sdf = torch.cat(sdf_chunks, dim=0).reshape(nbData, nbData, nbData).numpy().astype("float32")

    # 3) Build a trimesh scene with the real Panda mesh
    scene = trimesh.Scene()
    robot_mesh = panda.get_forward_robot_mesh(pose, theta)[0]
    robot_mesh = np.sum(robot_mesh)  # merge list of meshes from PandaLayer
    scene.add_geometry(robot_mesh)

    # 4) For each SDF level, run marching cubes and add a translucent colored shell mesh
    spacing = np.array(
        [(bp_sdf.domain_max - bp_sdf.domain_min) / nbData] * 3,
        dtype=np.float32,
    )

    # use a matplotlib colormap for pretty colors
    cmap = plt.cm.rainbow(np.linspace(0.0, 1.0, len(levels)))

    for lvl, col in zip(levels, cmap):
        # lvl is the iso-value in meters; marching_cubes will find φ(x) = lvl
        verts, faces, normals, values = skimage.measure.marching_cubes(
            sdf,
            level=float(lvl),
            spacing=spacing,
        )

        # recenter from [0, 2] to roughly [-1, 1] like the original sdf_to_mesh
        verts = verts - np.array([1.0, 1.0, 1.0], dtype=np.float32)

        # RGBA color: take RGB from col, set alpha for translucency
        rgba = np.array([col[0], col[1], col[2], 0.3], dtype=np.float32)  # alpha = 0.3
        vcols = (rgba * 255).astype(np.uint8)
        vcols = np.repeat(vcols[None, :], verts.shape[0], axis=0)

        shell_mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_colors=vcols,
        )
        scene.add_geometry(shell_mesh)

    scene.show()


def plot_2D_panda_sdf(pose, theta, bp_sdf, nbData, model, device):
    domain_0 = torch.linspace(-1.0, 1.0, nbData, device=device)
    domain_1 = torch.linspace(-1.0, 1.0, nbData, device=device)
    grid_x, grid_y = torch.meshgrid(domain_0, domain_1, indexing='ij')

    p1 = torch.stack(
        [grid_x.reshape(-1),
         grid_y.reshape(-1),
         torch.zeros_like(grid_x.reshape(-1))],
        dim=1,
    )
    p2 = torch.stack(
        [
            torch.zeros_like(grid_x.reshape(-1)),
            grid_x.reshape(-1) * 0.4,
            grid_y.reshape(-1) * 0.4 + 0.375,
        ],
        dim=1,
    )
    p3 = torch.stack(
        [
            grid_x.reshape(-1) * 0.4 + 0.2,
            torch.zeros_like(grid_x.reshape(-1)),
            grid_y.reshape(-1) * 0.4 + 0.375,
        ],
        dim=1,
    )

    grid_x_np, grid_y_np = grid_x.detach().cpu().numpy(), grid_y.detach().cpu().numpy()

    # YoZ plane
    plt.figure(figsize=(10, 10))
    plt.rc('font', size=25)
    p2_split = torch.split(p2, 1000, dim=0)
    sdf_list, grad_list = [], []
    for p_2 in p2_split:
        sdf_split, grad_split = bp_sdf.get_whole_body_sdf_batch(
            p_2,
            pose,
            theta,
            model,
            use_derivative=True,
        )
        sdf_list.append(sdf_split.squeeze())
        grad_list.append(grad_split.squeeze())
    sdf = torch.cat(sdf_list, dim=0)
    ana_grad = torch.cat(grad_list, dim=0)

    p2_np = p2.detach().cpu().numpy()
    sdf_np = sdf.squeeze().reshape(nbData, nbData).detach().cpu().numpy()

    ct1 = plt.contour(
        grid_x_np * 0.4,
        grid_y_np * 0.4 + 0.375,
        sdf_np,
        levels=12,
    )
    plt.clabel(ct1, inline=False, fontsize=10)

    ana_grad_2d = -torch.nn.functional.normalize(ana_grad[:, [1, 2]], dim=-1) * 0.01
    p2_3d = p2_np.reshape(nbData, nbData, 3)
    ana_grad_3d = ana_grad_2d.reshape(nbData, nbData, 2)

    plt.quiver(
        p2_3d[0:-1:4, 0:-1:4, 1],
        p2_3d[0:-1:4, 0:-1:4, 2],
        ana_grad_3d[0:-1:4, 0:-1:4, 0].detach().cpu().numpy(),
        ana_grad_3d[0:-1:4, 0:-1:4, 1].detach().cpu().numpy(),
        scale=0.5,
        color=[0.1, 0.1, 0.1],
    )
    plt.title('YoZ')
    plt.show()

    # XoZ plane
    plt.figure(figsize=(10, 10))
    plt.rc('font', size=25)
    p3_split = torch.split(p3, 1000, dim=0)
    sdf_list, grad_list = [], []
    for p_3 in p3_split:
        sdf_split, grad_split = bp_sdf.get_whole_body_sdf_batch(
            p_3,
            pose,
            theta,
            model,
            use_derivative=True,
        )
        sdf_list.append(sdf_split.squeeze())
        grad_list.append(grad_split.squeeze())
    sdf = torch.cat(sdf_list, dim=0)
    ana_grad = torch.cat(grad_list, dim=0)

    p3_np = p3.detach().cpu().numpy()
    sdf_np = sdf.squeeze().reshape(nbData, nbData).detach().cpu().numpy()

    ct1 = plt.contour(
        grid_x_np * 0.4 + 0.2,
        grid_y_np * 0.4 + 0.375,
        sdf_np,
        levels=12,
    )
    plt.clabel(ct1, inline=False, fontsize=10)

    ana_grad_2d = -torch.nn.functional.normalize(ana_grad[:, [0, 2]], dim=-1) * 0.01
    p3_3d = p3_np.reshape(nbData, nbData, 3)
    ana_grad_3d = ana_grad_2d.reshape(nbData, nbData, 2)

    plt.quiver(
        p3_3d[0:-1:4, 0:-1:4, 0],
        p3_3d[0:-1:4, 0:-1:4, 2],
        ana_grad_3d[0:-1:4, 0:-1:4, 0].detach().cpu().numpy(),
        ana_grad_3d[0:-1:4, 0:-1:4, 1].detach().cpu().numpy(),
        scale=0.5,
        color=[0.1, 0.1, 0.1],
    )
    plt.title('XoZ')
    plt.show()


def plot_3D_panda_with_gradient(pose, theta, bp_sdf, model, device):
    # forward kinematics meshes
    robot_mesh = panda.get_forward_robot_mesh(pose, theta)[0]
    robot_mesh = np.sum(robot_mesh)

    surface_points = robot_mesh.vertices
    scene = trimesh.Scene()
    scene.add_geometry(robot_mesh)

    choice = np.random.choice(len(surface_points), 1024, replace=False)
    surface_points = surface_points[choice]
    p = torch.from_numpy(surface_points).float().to(device)

    ball_query = trimesh.creation.uv_sphere(1).vertices
    choice_ball = np.random.choice(len(ball_query), 1024, replace=False)
    ball_query = ball_query[choice_ball]

    p = p + torch.from_numpy(ball_query).float().to(device) * 0.5

    sdf, ana_grad = bp_sdf.get_whole_body_sdf_batch(
        p,
        pose,
        theta,
        model,
        use_derivative=True,
        used_links=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    )
    sdf = sdf.squeeze().detach().cpu().numpy()
    ana_grad = ana_grad.squeeze().detach().cpu().numpy()

    pts = p.detach().cpu().numpy()
    colors = np.zeros_like(pts, dtype=object)
    colors[:, 0] = np.abs(sdf) * 400

    for i in range(len(pts)):
        dg = ana_grad[i]
        if dg.sum() == 0:
            continue
        c = colors[i]
        m = utils.create_arrow(-dg, pts[i], vec_length=0.05, color=c)
        scene.add_geometry(m)

    scene.show()


def generate_panda_mesh_sdf_points(max_dist=0.10):
    import glob
    import mesh_to_sdf

    mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/panda_layer/meshes/voxel_128/*"
    mesh_path = "/Users/stav.42/f_lab/tidybot_model/iiwa_description/meshes/iiwa7/collision/*.stl"
    mesh_files = glob.glob(mesh_path)
    mesh_files = sorted(mesh_files)[1:]  # except finger
    mesh_dict = {}

    for i, mf in enumerate(mesh_files):
        mesh_name = mf.split('/')[-1].split('.')[0]
        print(mesh_name)
        mesh = trimesh.load(mf)
        mesh_dict[i] = {}
        mesh_dict[i]['mesh_name'] = mesh_name

        vert = mesh.vertices
        points = vert + np.random.uniform(-max_dist, max_dist, size=vert.shape)
        sdf = mesh_to_sdf.mesh_to_sdf(
            mesh,
            points,
            surface_point_method='scan',
            sign_method='normal',
            bounding_radius=None,
            scan_count=100,
            scan_resolution=400,
            sample_point_count=10000000,
            normal_sample_count=100,
        )
        mesh_dict[i]['points'] = points
        mesh_dict[i]['sdf'] = sdf
    np.save('data/panda_mesh_sdf.npy', mesh_dict)


def vis_panda_sdf(pose, theta, device):
    data = np.load('data/panda_mesh_sdf.npy', allow_pickle=True).item()
    trans = panda.get_transformations_each_link(pose, theta)
    pts = []
    for i, k in enumerate(data.keys()):
        points = data[k]['points']
        sdf = data[k]['sdf']
        print(points.shape, sdf.shape)
        choice = (sdf < 0.05) * (sdf > 0.045)
        points = points[choice]
        sdf = sdf[choice]

        sample = np.random.choice(len(points), 128, replace=True)
        points, sdf = points[sample], sdf[sample]

        points = torch.from_numpy(points).float().to(device)
        ones = torch.ones([len(points), 1], device=device).float()
        points = torch.cat([points, ones], dim=-1)
        t = trans[i].squeeze()
        print(points.shape, t.shape)

        trans_points = (t @ points.t()).t()[:, :3]
        pts.append(trans_points)
    pts = torch.cat(pts, dim=0).detach().cpu().numpy()
    print(pts.shape)

    scene = trimesh.Scene()
    robot_mesh = panda.get_forward_robot_mesh(pose, theta)[0]
    robot_mesh = np.sum(robot_mesh)
    scene.add_geometry(robot_mesh)
    pc = trimesh.PointCloud(pts, colors=[255, 0, 0])
    scene.add_geometry(pc)
    scene.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--domain_max', default=1.0, type=float)
    parser.add_argument('--domain_min', default=-
1.0, type=float)
    parser.add_argument('--n_func', default=8, type=int)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    # resolve device like we did in bf_sdf.py
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
    bp = bf_sdf.BPSDF(args.n_func, args.domain_min, args.domain_max, panda, device)

    # load model safely on CPU, then fix dtypes
    model_path = f'models/BP_{args.n_func}.pt'
    model = torch.load(model_path, map_location='cpu', weights_only=False)

    for k in model.keys():
        if 'weights' in model[k]:
            model[k]['weights'] = model[k]['weights'].float()
        if 'offset' in model[k]:
            model[k]['offset'] = model[k]['offset'].float()

    # initial robot configuration
    theta = torch.tensor(
        [0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4],
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 7)

    # make a 4x4 identity matrix as float32 and move it to device
    pose = torch.from_numpy(np.identity(4, dtype=np.float32))  # tensor on CPU
    pose = pose.to(device).unsqueeze(0).expand(len(theta), 4, 4)  # (B,4,4)

    # # 2D SDF visualization (optional)
    # plot_2D_panda_sdf(pose, theta, bp, nbData=80, model=model, device=device)

    # # 3D SDF with gradient visualization
    # plot_3D_panda_with_gradient(pose, theta, bp, model=model, device=device)

    visualize_sdf_shells(
        panda=panda,
        bp_sdf=bp,
        model=model,
        pose=pose,
        theta=theta,
        device=device,
        nbData=96,
        levels=(0.0, 0.015, 0.03, 0.045, 0.06)  # tweak these until it looks nice
    )