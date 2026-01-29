
# -----------------------------------------------------------------------------
# This file is part of the RDF project.
# Copyright (c) 2023 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

import trimesh
import glob
import os
import numpy as np
import mesh_to_sdf
# import skimage
import pyrender
# import torch

# mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/panda_layer/meshes/voxel_128/*.stl"

# For the iiwa robot
mesh_path = "/Users/stav.42/f_lab/tidybot_model/iiwa_description/meshes/iiwa7/collision/*.stl"


mesh_files = glob.glob(mesh_path)
mesh_files = sorted(mesh_files)

for index, mf in enumerate(mesh_files):
    if index < 4:
        continue
    # break
    mesh_name = mf.split('/')[-1].split('.')[0]
    print(mesh_name)
    mesh = trimesh.load(mf)
    mesh = mesh_to_sdf.scale_to_unit_sphere(mesh)

    center = mesh.bounding_box.centroid
    scale = np.max(np.linalg.norm(mesh.vertices-center, axis=1))

    print(f"Mesh center: {center}, scale: {scale}")
    print(f"Mesh bounding box extents: {mesh.bounding_box.extents}")

    # sample points near surface (as same as deepSDF)
    near_points, near_sdf = mesh_to_sdf.sample_sdf_near_surface(mesh, 
                                                      number_of_points = 500000, 
                                                      surface_point_method='scan', 
                                                      sign_method='normal', 
                                                      scan_count=100, 
                                                      scan_resolution=400, 
                                                      sample_point_count=10000000, 
                                                      normal_sample_count=100, 
                                                      min_size=0.015, 
                                                      return_gradients=False)
    
    # Analyze the distribution of SDF values for the near values as well as the 
    # distribution of the points chosen
    
    # Print some values of nearly sampled points and the corresponding SDF values
    print("Some near points and their SDF values:")
    for i in range(20):
        print(f"Point: {near_points[i]}, SDF: {near_sdf[i]}")
    
    
    # # sample points randomly within the bounding box [-1,1]
    random_points = np.random.rand(500000,3)*2.0-1.0
    random_sdf = mesh_to_sdf.mesh_to_sdf(mesh, 
                                     random_points, 
                                     surface_point_method='scan', 
                                     sign_method='depth', 
                                     bounding_radius=None, 
                                     scan_count=100, 
                                     scan_resolution=400, 
                                     sample_point_count=10000000, 
                                     normal_sample_count=100) 
    

    print("Some random points and their SDF values:")
    for i in range(20):
        print(f"Point: {random_points[i]}, SDF: {random_sdf[i]}")

    # save data
    data = {
        'near_points': near_points,
        'near_sdf': near_sdf,
        'random_points': random_points,
        'random_sdf': random_sdf,
        'center': center,
        'scale': scale
    }
    save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),f'data/sdf_points')
    if os.path.exists(save_path) is not True:
        os.mkdir(save_path)
    np.save(os.path.join(save_path,f'voxel_128_{mesh_name}.npy'), data)

    # # for visualization
    data = np.load(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),f'data/sdf_points/voxel_128_{mesh_name}.npy')), allow_pickle=True).item()
    random_points = data['random_points']
    random_sdf = data['random_sdf']
    near_points = data['near_points']
    near_sdf = data['near_sdf']
    colors = np.zeros(random_points.shape)
    colors[random_sdf < 0, 2] = 1
    colors[random_sdf > 0, 0] = 1
    near_colors = np.zeros(near_points.shape)
    near_colors[near_sdf < 0, 2] = 1
    near_colors[near_sdf > 0, 0] = 1
    cloud = pyrender.Mesh.from_points(random_points, colors=colors)
    cloud_near = pyrender.Mesh.from_points(near_points, colors=near_colors)
    scene = pyrender.Scene()
    scene.add(cloud)
    scene.add(cloud_near)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

    # break
## Now generate the same kind of data for the cuboidal rectangular box which is the mobile base

# ------------------ MOBILE BASE BOX SDF GENERATION ------------------

# Physical dimensions of the base (whatever you use in URDF, etc.)
# Let's say the base is 0.6 m x 0.4 m x 0.2 m, just as an example.
# Only the *ratios* matter if you later scale to unit sphere.
# base_extents = np.array([0.6, 0.6, 0.2])  # [Lx, Ly, Lz]

# # Create a trimesh box centered at origin with given extents
# base_mesh = trimesh.creation.box(extents=base_extents)

# # Make it consistent with other meshes: scale to unit sphere
# base_mesh = mesh_to_sdf.scale_to_unit_sphere(base_mesh)

# # Compute center and scale exactly like you do for the robot links
# center = base_mesh.bounding_box.centroid
# scale = np.max(np.linalg.norm(base_mesh.vertices - center, axis=1))

# print("MOBILE BASE BOX")
# print(f"Mesh center: {center}, scale: {scale}")
# print(f"Mesh bounding box extents: {base_mesh.bounding_box.extents}")

# # Sample near-surface SDF points
# near_points, near_sdf = mesh_to_sdf.sample_sdf_near_surface(
#     base_mesh,
#     number_of_points=500000,
#     surface_point_method='scan',
#     sign_method='normal',
#     scan_count=100,
#     scan_resolution=400,
#     sample_point_count=10000000,
#     normal_sample_count=100,
#     min_size=0.015,
#     return_gradients=False,
# )

# print("Some near points and their SDF values (mobile base):")
# for i in range(20):
#     print(f"Point: {near_points[i]}, SDF: {near_sdf[i]}")

# # Sample random points in [-1, 1]^3 like before
# random_points = np.random.rand(500000, 3) * 2.0 - 1.0
# random_sdf = mesh_to_sdf.mesh_to_sdf(
#     base_mesh,
#     random_points,
#     surface_point_method='scan',
#     sign_method='normal',
#     bounding_radius=None,
#     scan_count=100,
#     scan_resolution=400,
#     sample_point_count=10000000,
#     normal_sample_count=100,
# )

# print("Some random points and their SDF values (mobile base):")
# for i in range(20):
#     print(f"Point: {random_points[i]}, SDF: {random_sdf[i]}")

# # Save exactly the same structure as for the other links
# data = {
#     'near_points': near_points,
#     'near_sdf': near_sdf,
#     'random_points': random_points,
#     'random_sdf': random_sdf,
#     'center': center,
#     'scale': scale,
# }

# save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/sdf_points')
# if not os.path.exists(save_path):
#     os.mkdir(save_path)

# np.save(os.path.join(save_path, 'voxel_128_mobile_base.npy'), data)

# # Visualization (optional, same as above)
# data = np.load(
#     os.path.join(save_path, 'voxel_128_mobile_base.npy'),
#     allow_pickle=True
# ).item()

# random_points = data['random_points']
# random_sdf = data['random_sdf']
# near_points = data['near_points']
# near_sdf = data['near_sdf']

# colors = np.zeros(random_points.shape)
# colors[random_sdf < 0, 2] = 1   # inside -> blue
# colors[random_sdf > 0, 0] = 1   # outside -> red

# near_colors = np.zeros(near_points.shape)
# near_colors[near_sdf < 0, 2] = 1
# near_colors[near_sdf > 0, 0] = 1

# cloud = pyrender.Mesh.from_points(random_points, colors=colors)
# cloud_near = pyrender.Mesh.from_points(near_points, colors=near_colors)

# scene = pyrender.Scene()
# scene.add(cloud)
# scene.add(cloud_near)
# viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
