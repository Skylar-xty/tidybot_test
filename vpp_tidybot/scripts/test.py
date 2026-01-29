# main_sdf_eval_pybullet.py
#
# Evaluate Bernstein SDF vs PyBullet collision distance
# at random points using a fixed Panda configuration.

import os
import sys
import time
import math

import numpy as np
import pandas as pd

import pybullet as p
import pybullet_data

# Your helper modules (adjust paths if needed)
sys.path.append('./src')
import functions
from Panda import Panda
from rdf import query_sdf   # your wrapper around BPSDF / query_sdf_batch

SEED = 42
np.random.seed(SEED)


if __name__ == "__main__":
    os.makedirs("./output", exist_ok=True)

    # ------------- Basic simulation setup -------------
    # If Panda() already connects to PyBullet and loads URDF, you might NOT
    # need the following connect+setAdditionalSearchPath, but it's harmless.
    # p.connect(p.GUI)
    # p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # p.setGravity(0, 0, -9.81)

    # Small step size (not that important since we keep robot fixed)
    stepsize = 2e-3
    robot = Panda(stepsize)
    robot.setControlMode("torque")

    # Let things settle a bit
    for _ in range(50):
        p.stepSimulation()
        time.sleep(stepsize)

    # ------------- Fix robot in a default joint pose -------------
    # If Panda spawns in its home pose already, we can just read it:
    q, qd = robot.getJointStates()
    q = np.array(q, dtype=np.float32)
    qd = np.array(qd, dtype=np.float32)

    # If you want hard home pose (e.g. all zeros) instead, you can do:
    # q = np.zeros(7, dtype=np.float32)
    # robot.resetJoints(q.tolist())   # assuming such a method exists
    # qd = np.zeros(7, dtype=np.float32)

    print("[INFO] Fixed joint configuration q:", q)
    print("[INFO] Joint velocities qd:", qd)

    # Identity base pose for SDF queries
    pose_np = np.eye(4, dtype=np.float32)

    # ------------- Create a spherical obstacle -------------
    sphere_radius = 0.05  # meters

    sphere_collision = p.createCollisionShape(
        shapeType=p.GEOM_SPHERE,
        radius=sphere_radius
    )

    sphere_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=sphere_radius,
        rgbaColor=[1, 0, 0, 0.7],
        specularColor=[0.4, 0.4, 0]
    )

    # Create the obstacle at some dummy position; we will move it per-sample
    obstacle_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=sphere_collision,
        baseVisualShapeIndex=sphere_visual,
        basePosition=[0.8, 0.0, 0.8],
        baseOrientation=[0, 0, 0, 1]
    )

    # ------------- Sampling region for query points -------------
    # Choose a box that reasonably covers Panda workspace
    X_MIN, X_MAX = -0.6, 0.6
    Y_MIN, Y_MAX = -0.6, 0.6
    Z_MIN, Z_MAX =  0.0, 1.2

    # How many samples
    N_SAMPLES = 500

    # ------------- Logging containers -------------
    xs, ys, zs = [], [], []
    sdf_vals = []
    real_dists = []
    errors = []
    abs_errors = []
    runtime_sdf = []
    runtime_real = []

    # ------------- Main evaluation loop -------------
    print("[INFO] Starting SDF vs PyBullet distance evaluation...")
    start_wall = time.time()

    for i in range(N_SAMPLES):
        # Sample a random query point in world frame
        x = np.random.uniform(X_MIN, X_MAX)
        y = np.random.uniform(Y_MIN, Y_MAX)
        z = np.random.uniform(Z_MIN, Z_MAX)
        x_query = np.array([x, y, z], dtype=np.float32)

        # Move the spherical obstacle center to this point
        p.resetBasePositionAndOrientation(
            obstacle_id,
            [float(x), float(y), float(z)],
            [0, 0, 0, 1]
        )

        # Let Bullet update contact info (one step is enough)
        p.stepSimulation()

        # ------------ 1) Distance from PyBullet mesh-based collision ------------
        t0 = time.perf_counter()
        real_dist = functions.compute_min_center_distance(
            robot_id=robot.robot,
            obstacle_id=obstacle_id,
            sphere_radius=sphere_radius,
            distance_threshold=2.0   # large enough compared to workspace
        )
        t1 = time.perf_counter()

        # ------------ 2) Distance from Bernstein SDF (polynomial model) ------------
        # NOTE: query_sdf expects:
        #   x_np: (3,)
        #   pose_np: (4,4)
        #   theta_np: (7,)
        # and returns (dst, grad_q)
        dst_sdf, grad_q = query_sdf(
            x_query,
            pose_np,
            q
        )
        t2 = time.perf_counter()

        # ------------ Log data ------------
        xs.append(x)
        ys.append(y)
        zs.append(z)
        sdf_vals.append(dst_sdf)
        real_dists.append(real_dist)

        err = dst_sdf - real_dist
        errors.append(err)
        abs_errors.append(abs(err))

        runtime_real.append(t1 - t0)
        runtime_sdf.append(t2 - t1)

        if (i + 1) % 50 == 0:
            print(
                f"[{i+1}/{N_SAMPLES}] "
                f"x=({x:.3f},{y:.3f},{z:.3f})  "
                f"SDF={dst_sdf:.4f}  "
                f"real={real_dist:.4f}  "
                f"err={err:.4f}"
            )
        
        PAUSE_BETWEEN_SAMPLES = 0.01
        time.sleep(PAUSE_BETWEEN_SAMPLES)

    total_wall = time.time() - start_wall

    # ------------- Compute stats -------------
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    zs = np.array(zs, dtype=np.float32)
    sdf_vals = np.array(sdf_vals, dtype=np.float32)
    real_dists = np.array(real_dists, dtype=np.float32)
    errors = np.array(errors, dtype=np.float32)
    abs_errors = np.array(abs_errors, dtype=np.float32)
    runtime_sdf = np.array(runtime_sdf, dtype=np.float32)
    runtime_real = np.array(runtime_real, dtype=np.float32)

    print("\n================ Evaluation Summary ================")
    print(f"Total samples: {N_SAMPLES}")
    print(f"Total wall time: {total_wall:.2f} s")
    print(f"Mean |SDF - real|: {abs_errors.mean():.6f} m")
    print(f"Median |error|:     {np.median(abs_errors):.6f} m")
    print(f"Max |error|:        {abs_errors.max():.6f} m")

    # If you want to focus on “near robot” region:
    near_mask = real_dists < 0.1  # closer than 10 cm to robot
    if np.any(near_mask):
        print(f"\nNear samples (real_dist < 0.10 m): {near_mask.sum()}")
        print(f"  Mean |error| near: {abs_errors[near_mask].mean():.6f} m")
        print(f"  Max |error| near:  {abs_errors[near_mask].max():.6f} m")
    else:
        print("\nNo samples with real_dist < 0.10 m (consider changing sampling box).")

    print("\nRuntime:")
    print(f"  mean PyBullet distance time: {runtime_real.mean():.6e} s")
    print(f"  mean SDF query time:         {runtime_sdf.mean():.6e} s")

    # ------------- Save CSV -------------
    df = pd.DataFrame({
        "x": xs,
        "y": ys,
        "z": zs,
        "sdf_dist": sdf_vals,
        "real_dist": real_dists,
        "error": errors,
        "abs_error": abs_errors,
        "runtime_real": runtime_real,
        "runtime_sdf": runtime_sdf,
    })

    out_path = os.path.join("./output", f"sdf_vs_pybullet_{int(time.time())}.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[INFO] Results saved to: {out_path}")

    # keep GUI open
    print("[INFO] Press Ctrl+C to exit.")
    while True:
        p.stepSimulation()
        time.sleep(stepsize)
