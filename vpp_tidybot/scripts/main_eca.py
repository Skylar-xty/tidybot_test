# main.py

import time
import pandas as pd
import numpy as np
import math
import os

import functions

import sys
sys.path.append('./src')

import cvxpy as cp
import pybullet as p
from Panda import Panda
from rdf import query_sdf, query_sdf_batch

SEED = 28
np.random.seed(SEED)

if __name__ == "__main__":
    times, dists, gammas, real_dists, tar_dists, pred_dists, pred_distsv = [], [], [], [], [], [], []
    runtime = []
    os.makedirs("../output", exist_ok=True)

    acc_max = np.array([15, 7.5, 10, 12.5, 15, 20, 20], dtype=np.float32)
    qd_lim  = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], dtype=np.float32)
    q_min_hardware = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    q_max_hardware = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

    duration, stepsize = 6.0, 2e-3
    robot = Panda(stepsize)
    robot.setControlMode("torque")

    lambda1, lambda2, lambda3 = 3, 100, 100
    alpha = 1e-2  # damping term for torque regularization
    sphere_radius = 0.05

    # ----- Obstacles (red balls) -----
    sphere_collision = p.createCollisionShape(
        shapeType=p.GEOM_SPHERE,
        radius=sphere_radius
    )
    sphere_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=sphere_radius,
        rgbaColor=[1, 0, 0, 1],
        specularColor=[0.4, 0.4, 0]
    )

    # Base positions with a decent gap between them
    base1 = np.array([0.35, -0.30, 0.50], dtype=np.float32)
    base2 = base1 + np.array([0.45, 0.15, 0.0], dtype=np.float32)  # ~0.29 m apart at rest
    x = base1  # kept for legacy use in some computations

    obstacle_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=sphere_collision,
        baseVisualShapeIndex=sphere_visual,
        basePosition=base1.tolist(),
        baseOrientation=[0, 0, 0, 1]
    )
    obstacle_id2 = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=sphere_collision,
        baseVisualShapeIndex=sphere_visual,
        basePosition=base2.tolist(),
        baseOrientation=[0, 0, 0, 1]
    )

    # ----- Target (green ball) between the reds, slightly offset -----
    debug = False
    # this stays fixed; obstacles wiggle around it
    target_pos = (base1 + base2) / 2.0 + np.array([0.0, -0.05, 0.05], dtype=np.float32)

    target_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.02,
        rgbaColor=[0, 1, 0, 1],
        specularColor=[0.4, 0.4, 0]
    )
    target_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=target_visual,
        basePosition=target_pos,
        baseOrientation=[0, 0, 0, 1]
    )

    sol_debug = True
    time.sleep(1)
    start_time = time.time()

    for i in range(int(duration / stepsize)):
        iter_start = time.perf_counter()
        if i % int(1.0 / stepsize) == 0:
            print(f"Simulation time: {robot.t:.3f} s")

        # ---------------- EE task force ----------------
        end_pos = robot.solveForwardKinematics()[0]
        fx = -50.0 * (end_pos - target_pos)

        # anisotropic damping matrix D in task space
        e1 = fx / (np.linalg.norm(fx) + 1e-8)
        e2 = np.array([1.0, 0.0, 0.0]) - np.dot([1.0, 0.0, 0.0], e1) * e1
        e2 /= (np.linalg.norm(e2) + 1e-8)
        e3 = np.cross(e1, e2)
        Q = np.column_stack((e1, e2, e3))
        Lambda = np.diag([lambda1, lambda2, lambda3])
        D = Q @ Lambda @ Q.T

        xdot = robot.getEndVelocity()
        fc = -D @ (xdot - fx)

        dist_to_target = np.linalg.norm(end_pos - target_pos)

        # ---------------- Move obstacles ----------------
        t_sim = robot.t
        # smaller amplitudes so the gap never closes:
        # relative motion between centers is at most ~0.1m in x, 0.05m in z,
        # baseline gap ~0.29m -> always > 0.18m, spheres are radius 0.05 (0.1m diameter)
        offset1 = np.array([
            0.05 * np.sin(0.5 * t_sim),
            0.00,
            0.025 * np.cos(0.5 * t_sim)
        ], dtype=np.float32)

        offset2 = np.array([
            -0.05 * np.sin(0.5 * t_sim),
            0.00,
            -0.025 * np.cos(0.5 * t_sim)
        ], dtype=np.float32)

        pos1 = base1 + offset1
        pos2 = base2 + offset2

        p.resetBasePositionAndOrientation(obstacle_id,  pos1.tolist(), [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(obstacle_id2, pos2.tolist(), [0, 0, 0, 1])

        # ---------------- Joint state, SDF, gamma ----------------
        q, qd = robot.getJointStates()
        qe = functions.compute_qe(q, qd)  # stopping configuration under max decel

        x0 = np.array(x, dtype=np.float32).reshape(1, 3)
        x_query = np.array(x0, dtype=np.float32).reshape(1, 3)
        pose = np.eye(4, dtype=np.float32)

        theta_np = np.stack([q, qe], axis=0).astype(np.float32)   # (B=2, 7)
        points_np = np.stack(
            [pos1.astype(np.float32), pos2.astype(np.float32)],
            axis=0
        ).astype(np.float32)                                      # (N=2, 3)
        pose_np = np.broadcast_to(pose, (2, 4, 4)).astype(np.float32)

        dsts, grad_qs = query_sdf_batch(points_np, pose_np, theta_np)

        dst,  grad  = dsts[0, 0], grad_qs[0, 0]
        dst2, grad2 = dsts[1, 0], grad_qs[1, 0]
        dst3, grad3 = dsts[0, 1], grad_qs[0, 1]
        dst4, grad4 = dsts[1, 1], grad_qs[1, 1]

        real_dist = functions.compute_min_center_distance(
            robot_id=robot.robot,
            obstacle_id=obstacle_id,
            sphere_radius=sphere_radius,
            distance_threshold=2.0
        )
        real_dist2 = functions.compute_min_center_distance(
            robot_id=robot.robot,
            obstacle_id=obstacle_id2,
            sphere_radius=sphere_radius,
            distance_threshold=2.0
        )

        print(f"[SDF distance]: {dst:.5f}m; [real dst] = {real_dist:.5f}m; delta = {real_dist-dst:.5f}m")
        if min(real_dist, real_dist2) <= 0.05:
            print("Collision with obstacle (sphere)!")
            ecollision = True
        else:
            ecollision = False

        Gamma, grad_gamma = functions.compute_gamma_and_grad(q, qd, threshold=2.5)

        qdd_lb, qdd_ub = functions.compute_joint_acceleration_bounds_vec(
            q, qd, q_min_hardware, q_max_hardware, qd_lim, acc_max,
            dt=0.02, viability=True
        )

        # pick the closest SDF sample & gradient
        dist_s = [dst, dst2, dst3, dst4]
        min_idx = int(np.argmin(dist_s))
        if min_idx == 0:
            sel_grad = grad3  # like your original tweak
        elif min_idx == 1:
            sel_grad = grad4
        elif min_idx == 2:
            sel_grad = grad3
        else:
            sel_grad = grad4

        # ---------------------------------------------------------------------
        # QP: task + joint accel bounds + SOFT external collision constraint
        # ---------------------------------------------------------------------
        # fix inverted accel limits if any
        for idx in range(7):
            if qdd_lb[idx] > qdd_ub[idx]:
                qdd_lb[idx] = qdd_ub[idx] - 1e-4

        M = np.array(robot.getMassMatrix(q))
        tau_id = robot.solveInverseDynamics(q, qd, [0] * 7)
        M_inv = np.linalg.inv(M)

        u = cp.Variable(7)  # joint torques
        J = np.array(robot.getJacobian())
        JT_pinv = np.linalg.pinv(J.T)

        objective = cp.sum_squares(JT_pinv @ u - fc) + alpha * cp.sum_squares(u)

        constraints = [
            M_inv @ u >= qdd_lb + M_inv @ tau_id,
            M_inv @ u <= qdd_ub + M_inv @ tau_id,
        ]

        # External avoidance – only when near, with a slack
        d_min = min(dist_s)
        d_activate = 0.25  # start caring when closer than this
        eps_ext = 0.08     # desired minimum distance
        beta_slack = 1e3   # weight on slack (bigger = harder constraint)

        if d_min < d_activate:
            dt_ext = 0.02
            g_ext = sel_grad                       # (7,)
            c_ext = float(g_ext.dot(qd) * dt_ext)
            g_eff_ext = 0.5 * g_ext * dt_ext**2
            rhs_ext = eps_ext - d_min - c_ext + float(g_eff_ext @ M_inv @ tau_id)

            s = cp.Variable()  # slack for external constraint
            constraints += [
                s >= 0,
                g_eff_ext @ M_inv @ u + s >= rhs_ext
            ]
            objective += beta_slack * cp.square(s)

        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=cp.OSQP)

        if sol_debug:
            print(f"QP Status: {prob.status}, obj={prob.value}")

        status_ok = prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and (u.value is not None)

        if not status_ok:
            # Fallback: brake to reduce velocity within accel bounds
            print("QP failed or infeasible, using fallback torques.")
            brake_horizon = 0.1  # seconds
            qdd_cmd = np.clip(-np.array(qd) / brake_horizon, qdd_lb, qdd_ub)
            tau_cmd = np.array(robot.solveInverseDynamics(q, qd, qdd_cmd.tolist()))
        else:
            tau_cmd = np.array(u.value).reshape(7)

        tau_ext_max = np.array([87, 87, 87, 87, 12, 12, 12]) * 0.5
        if np.random.rand() < 0.1:
            tau_noise = np.random.uniform(-tau_ext_max, tau_ext_max)
        else:
            tau_noise = np.zeros(7)

        tau_with_disturb = tau_cmd + tau_noise
        robot.setTargetTorques(tau_with_disturb.tolist())
        robot.step()

        iter_end = time.perf_counter()
        runtime.append(iter_end - iter_start)

        # ---------------- logging ----------------
        dist_sc, collision = math.inf, False
        for l1 in range(7):
            for l2 in range(7):
                if abs(l1 - l2) > 1 and not {l1, l2} == {4, 6}:
                    pts = robot.getClosestPoints(l1, l2)
                    if len(pts) == 0:
                        continue
                    dmin = min(pt[8] for pt in pts)
                    dist_sc = min(dist_sc, dmin)
                    if dmin < 0:
                        collision = True

        times.append(robot.t)
        dists.append(dist_sc)
        gammas.append(Gamma)
        tar_dists.append(dist_to_target)

        time.sleep(robot.stepsize)

    # --- Save results ---
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.2f}s")
    print(f"Total runtime: {sum(runtime[1:]):.2f}s, "
          f"avg={np.mean(runtime[1:]):.4f}s, "
          f"max={np.max(runtime[1:]):.4f}s, "
          f"min={np.min(runtime[1:]):.4f}s, "
          f"std={np.std(runtime[1:]):.4f}s")

    df = pd.DataFrame({
        "time": times,
        "dist": dists,
        "gamma": gammas,
        "tar_dist": tar_dists,
    })
    df.to_csv(f"../output/{time.time()}.csv", index=False)
    print("Results saved.")
