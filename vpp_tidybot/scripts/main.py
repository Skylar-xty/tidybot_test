# main.py

import time
import pandas as pd
import numpy as np
import math
import os

import functions

import sys
sys.path.append('./src')

import time
import numpy as np
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
    alpha = 1e-2  # Damping term
    # # Radius of ball
    sphere_radius = 0.05

    # 1) Create a collision shape (do not use -1 if you want it to participate in physical collisions)
    sphere_collision = p.createCollisionShape(
        shapeType=p.GEOM_SPHERE,
        radius=sphere_radius
    )

    # 2) Create a visual shape (red)
    sphere_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=sphere_radius,
        rgbaColor=[1, 0, 0, 1],    # Red，alpha=1
        specularColor=[0.4,0.4,0]  
    )

    # 3) Assemble it into a multi-body object (mass set to 0 to represent a static obstacle).
    base1 = np.array([0.35, -0.30, 0.50], dtype=np.float32)
    x = base1
    base2 = base1 + np.array([0.45, 0.35, 0.0], dtype=np.float32)

    # 3) Assemble them into multi-body objects (mass=0 => static obstacles that we can teleport)
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

    ### target_pos = np.array([0.0, -0.6, 0.3])
    debug = False
    # target_pos = np.array([0.0, -0.0, 0.0])
    target_pos = np.array([0.5, -0.0, 0.7])
    target_pos = (base1 + base2) / 2.0 + np.array([0.0, -0.05, 0.05], dtype=np.float32)

    target_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.02,
        rgbaColor=[0, 1, 0, 1],    # Red，alpha=1
        specularColor=[0.4,0.4,0]  
    )
    target_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=target_visual,
        basePosition=target_pos,
        baseOrientation=[0,0,0,1]
    )

    sol_debug = True
    time.sleep(1)
    start_time = time.time()

    for i in range(int(duration / stepsize)):
        iter_start = time.perf_counter()
        if i % int(1.0 / stepsize) == 0:
            print(f"Simulation time: {robot.t:.3f} s")

        # --- Calculate the end-effector force fc of the main task ---
        end_pos = robot.solveForwardKinematics()[0]
        # fx = -50 * (end_pos - np.array([0, 0, 0.3]))
        fx = -50 * (end_pos - target_pos)
        e1 = fx / np.linalg.norm(fx)
        e2 = np.array([1, 0, 0]) - np.dot([1, 0, 0], e1) * e1
        e2 /= np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        Q = np.column_stack((e1, e2, e3))
        Lambda = np.diag([lambda1, lambda2, lambda3])
        D = Q @ Lambda @ Q.T
        xdot = robot.getEndVelocity()
        fc = -D @ (xdot - fx)

        # target_pos = np.array([0.0, -0.6, 0.3])
        dist_to_target = np.linalg.norm(end_pos - target_pos)

        # new_z = math.sin(i/900*math.pi)*0.1
        # new_pos = [x[0], x[1], x[2] ]#+ new_z]
        # p.resetBasePositionAndOrientation(obstacle_id, new_pos, [0,0,0,1])

        t = robot.t

        offset1 = np.array([
            0.10 * np.sin(0.5 * t),   # wiggle in x
            0.00,                     # keep y fixed for now
            0.05 * np.cos(0.5 * t)    # small up-down in z
        ], dtype=np.float32)

        offset2 = np.array([
            -0.10 * np.sin(0.5 * t),  # opposite wiggle in x
            0.00,
            -0.05 * np.cos(0.5 * t)   # opposite in z
        ], dtype=np.float32)

        pos1 = base1 + offset1
        pos2 = base2 + offset2

        # Update obstacle poses in PyBullet
        p.resetBasePositionAndOrientation(obstacle_id,  pos1.tolist(), [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(obstacle_id2, pos2.tolist(), [0, 0, 0, 1])

        # --- Read joint states & self-collision gamma & gradient ---
        q, qd = robot.getJointStates()

        # Compute the stopping point if decelerating at max deceleration
        qe = functions.compute_qe(q, qd)

        # X0 is the obstacle 1 position
        x0 = np.array(x, dtype=np.float32).reshape(1,3)
        x_query = np.array(x0, dtype=np.float32).reshape(1,3)

        # Identity pose I don't know why this is calculated
        pose = np.eye(4)
        # # dst, grad = query_sdf(
        # #     x_query,
        # #     pose,
        # #     np.array(q, dtype=np.float32)
        # # )
        # # dst2, grad2 = query_sdf(
        # #     x_query,
        # #     pose,
        # #     np.array(qe, dtype=np.float32)
        # # )
        # # dst3, grad3 = query_sdf(
        # #     x0 + np.array([0.5, 0.1, 0.0]),
        # #     pose,
        # #     np.array(q, dtype=np.float32)
        # # )
        # # dst4, grad4 = query_sdf(
        # #     x0 + np.array([0.5, 0.1, 0.0]),
        # #     pose,
        # #     np.array(qe, dtype=np.float32)
        # # )

        # # qe is end point if decelerate to zero at max deceleration, q is the current state. Why 
        # # this is being made, I don't know.
        theta_np = np.stack([q, qe], axis=0).astype(np.float32)   # (B=2, 7)
        points_np = np.stack([
            pos1.astype(np.float32),
            pos2.astype(np.float32),
        ], axis=0).astype(np.float32)   
        # points_np = np.stack([
        #     x_query,
        #     x0 + np.array([0.4, 0.1, 0.0], dtype=np.float32)
        # ], axis=0).astype(np.float32)                             # (N=2, 3)

        pose_np = np.broadcast_to(pose, (2, 4, 4)).astype(np.float32)  # (B=2, 4, 4)

        # # ------ A single batch query returns (B,N) and (B,N,7) ------
        dsts, grad_qs = query_sdf_batch(points_np, pose_np, theta_np)  # need_index=False

        # ------ Corresponding back to the original four variables ------
        dst,  grad  = dsts[0, 0], grad_qs[0, 0]   # q,  x_query
        dst2, grad2 = dsts[1, 0], grad_qs[1, 0]   # qe, x_query
        dst3, grad3 = dsts[0, 1], grad_qs[0, 1]   # q,  x0+[0.5,0.1,0.0]
        dst4, grad4 = dsts[1, 1], grad_qs[1, 1]   # qe, x0+[0.5,0.1,0.0]

        
        real_dist = functions.compute_min_center_distance(
            robot_id=robot.robot,
            obstacle_id=obstacle_id,
            sphere_radius=sphere_radius,
            distance_threshold=2.0  # Set a value slightly larger than the workspace based on the maximum possible distance in the scene
        )
        real_dist2 = functions.compute_min_center_distance(
            robot_id=robot.robot,
            obstacle_id=obstacle_id2,
            sphere_radius=sphere_radius,
            distance_threshold=2.0  # Set a value slightly larger than the workspace based on the maximum possible distance in the scene
        )
        ecollision = False
        # print(f"[SDF distance]: {min(dst,dst3):.5f}m; [SDF end-distance]: {min(dst2,dst4):.5f}m; [real dst] = {min(real_dist,real_dist2):.5f}m; delta = {real_dist-dst:.5f}m; dist to target = {dist_to_target:.5f}m")
        # print(f"[SDF distance]: {min(dst,dst2,dst3,dst4):.5f}m; [real dst] = {min(real_dist,real_dist2):.5f}m; delta = {real_dist-dst:.5f}m; dist to target = {dist_to_target:.5f}m")
        print(f"[SDF distance]: {dst:.5f}m; [real dst] = {real_dist:.5f}m; delta = {real_dist-dst:.5f}m")
        if min(real_dist, real_dist2) <= 0.05:
            print("Collision!")
            ecollision = True

        Gamma, grad_gamma = functions.compute_gamma_and_grad(q, qd, threshold=2.5)

        qdd_lb, qdd_ub = functions.compute_joint_acceleration_bounds_vec(
            q, qd, q_min_hardware, q_max_hardware, qd_lim, acc_max, dt=0.02, viability=True)

        
        dist_s = [dst, dst2, dst3, dst4]
        # dist_s = [dst3, dst4]
        # Find the index of the minimum value（0→dst, 1→dst2, 2→dst3, 3→dst4）
        min_idx = int(np.argmin(dist_s))

        #### # If the minimum value is dst or dst2 (index 0 or 1), use grad; otherwise, use grad2
        if min_idx == 0:
            sel_grad = grad
        elif min_idx == 1:
            sel_grad = grad2
        elif min_idx == 2:
            sel_grad = grad3
        else:
            sel_grad = grad4
        ####


        if min_idx == 0:
            sel_grad = grad3
        elif min_idx == 1:
            sel_grad = grad4
        
            
        if False:
        # if min(dst, dst2, dst3, dst4) < 0.106:
        # if min(dst3, dst4) < 0.1:

            dt = 0.02
            # 1) Calculate intermediate quantities
            c     = np.dot(sel_grad, qd) * dt                    # grad·qd * dt
            g_eff = 0.5 * sel_grad * dt**2                       # 0.5 * grad * dt^2
            qdd_cmd = np.where(g_eff > 0, qdd_ub, qdd_lb)
            # qdd_cmd = np.where(g_eff > 0, acc_max, -acc_max)
            # if (c+g_eff @ qdd_cmd)<0:
            #     print(c,g_eff @ qdd_cmd,c+g_eff @ qdd_cmd)
            # qdd_cmd = np.where(g_eff > 0, acc_max, -acc_max)
            tau_cmd = robot.solveInverseDynamics(q, qd, qdd_cmd.tolist())
            robot.setTargetTorques(tau_cmd)
            robot.step()

        else:
            soft = False
            for idx in range(7):
                if qdd_lb[idx] > qdd_ub[idx]:
                    qdd_lb[idx] = qdd_ub[idx] - 1e-4

            # --- Constructing the QP: Minimize ∥J⁺ x - fc∥^2 + α ∥y∥^2 ---
            M = np.array(robot.getMassMatrix(q))
            tau_id = robot.solveInverseDynamics(q, qd, [0]*7)
            M_inv = np.linalg.inv(M)

            if debug:
                print(f"\n=== DEBUG QP Solver (t={robot.t:.3f}s) ===")
                print(f"Joint positions q: {q}")
                print(f"Joint velocities qd: {qd}")
                print(f"Mass matrix condition number: {np.linalg.cond(M):.2e}")
                print(f"tau_id (gravity comp): {tau_id}")

            u = cp.Variable(7)  # torque
            y = cp.Variable(7)  # acceleration
            J = np.array(robot.getJacobian())
            JT_pinv = np.linalg.pinv(J.T)

            if debug:
                print(f"Jacobian shape: {J.shape}, condition: {np.linalg.cond(J):.2e}")
                print(f"Target force fc: {fc}")

            objective = cp.sum_squares(JT_pinv @ u - fc) + alpha * cp.sum_squares(u)

            constraints = [
                M_inv @ u >= qdd_lb + M_inv @ tau_id,
                M_inv @ u <= qdd_ub + M_inv @ tau_id,
            ]

            constraints = []
            if debug:
                print(f"Acceleration bounds: lb={qdd_lb}, ub={qdd_ub}")

            if grad_gamma is not None:
                dt      = 0.02
                grad_q  = grad_gamma[:7]
                grad_qd = grad_gamma[7:]
                scale = 0.5
                g_eff   = scale * (0.5 * grad_q * dt**2 + grad_qd * dt)
                c_const = grad_q.dot(qd) * dt
                ε       = 5e-2
                rhs = ε - c_const + g_eff @ M_inv @ tau_id
                rhs *= 0.5

                if debug:
                    print(f"Self-collision constraint active:")
                    print(f"  Gamma: {Gamma:.5f}")
                    print(f"  grad_gamma shape: {grad_gamma.shape}")
                    print(f"  g_eff: {g_eff}")
                    print(f"  c_const: {c_const:.5f}")
                    print(f"  epsilon: {ε}")
                    print(f"  RHS: {rhs:.5f}")

                #constraints.append(g_eff @ M_inv @ u >= ε-c_const+g_eff @ M_inv @ tau_id)
                # constraints.append(g_eff @ M_inv @ u >= rhs)
                # # -----------------------------------------------------
                # # EXTERNAL SDF COLLISION AVOIDANCE CONSTRAINT
                # # -----------------------------------------------------
                dt_ext = 0.02
                g_ext = sel_grad                     # (7,) from SDF
                c_ext = g_ext.dot(qd) * dt_ext
                g_eff_ext = 0.5 * g_ext * dt_ext**2
                eps_ext = 0.08                       # safe threshold
                d_min = min(dst, dst2, dst3, dst4)   # nearest obstacle distance

                constraints.append(
                    g_eff_ext @ M_inv @ u >= eps_ext - d_min - c_ext + g_eff_ext @ M_inv @ tau_id
                    )
                print("With self-collision constraint")
                # # ----------------------------------------------------- 
                prob = cp.Problem(cp.Minimize(objective), constraints)
                try:
                    prob.solve(solver=cp.OSQP)
                    if sol_debug:
                        print(f"QP Status: {prob.status}")
                        print(f"QP Objective value: {prob.value:.5f}")
                except cp.SolverError as e:
                    if sol_debug:
                        print("QP infeasible, relax constraint, use soft constraint")
                        print(f"QP Solver Error: {e}")
                    soft = True
                    qdd_cmd = np.where(g_eff > 0, qdd_ub, qdd_lb)
                    tau_cmd = robot.solveInverseDynamics(q, qd, qdd_cmd.tolist())
                    robot.setTargetTorques(tau_cmd)
                    robot.step()
                while prob.status != cp.OPTIMAL and ε > 1e-3:
                    if sol_debug:
                        print(f"QP infeasible, reducing epsilon from {ε:.4f} to {ε-1e-1:.4f}")
                    ε = ε - 1e-1
                    constraints[-1] = (g_eff @ M_inv @ u >= ε-c_const+g_eff @ M_inv @ tau_id)
                    prob = cp.Problem(cp.Minimize(objective), constraints)
                    prob.solve(solver=cp.OSQP)
                if ε < 1e-3:
                    if sol_debug:
                        print("QP still infeasible, relax constraint, use soft constraint")
                    soft = True
                    qdd_cmd = np.where(g_eff > 0, qdd_ub, qdd_lb)
                    tau_cmd = robot.solveInverseDynamics(q, qd, qdd_cmd.tolist())
                    robot.setTargetTorques(tau_cmd)
                    robot.step()


            else:
                # # -----------------------------------------------------
                # # EXTERNAL SDF COLLISION AVOIDANCE CONSTRAINT
                # # -----------------------------------------------------
                dt_ext = 0.02
                g_ext = sel_grad                     # (7,) from SDF
                c_ext = g_ext.dot(qd) * dt_ext
                g_eff_ext = 0.5 * g_ext * dt_ext**2
                eps_ext = 0.08                       # safe threshold
                d_min = min(dst, dst2, dst3, dst4)   # nearest obstacle distance

                constraints.append(
                    g_eff_ext @ M_inv @ u >= eps_ext - d_min - c_ext + g_eff_ext @ M_inv @ tau_id
                    )
                
                print("No self-collision constraint (grad_gamma is None)")
                # # -----------------------------------------------------
                if debug:
                    print("No self-collision constraint (grad_gamma is None)") 
                prob = cp.Problem(cp.Minimize(objective), constraints)
                prob.solve(solver=cp.OSQP)
                if sol_debug:
                    print(f"QP Status: {prob.status}")
                    print(f"QP Objective value: {prob.value:.5f}")

            if not soft:
                tau_cmd = np.array(u.value) 
                if debug:
                    print(f"Computed torques: {tau_cmd}")
                    print(f"Torque norm: {np.linalg.norm(tau_cmd):.3f}")
                # tau_ext_max = np.array([5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
                tau_ext_max = np.array([87, 87, 87, 87, 12, 12, 12])*0.5
                if np.random.rand() < 0.1:   
                    tau_noise = np.random.uniform(-tau_ext_max, tau_ext_max)
                    # print(f"tau_noise={tau_noise}")
                    tau_noise = np.zeros(7)
                else:
                    tau_noise = np.zeros(7)
                tau_with_disturb = tau_cmd + tau_noise
                robot.setTargetTorques(tau_with_disturb.tolist())
                # --- Execution Control & Visualization ---
                # robot.setTargetTorques(x.value.tolist())
                robot.step()
        iter_end = time.perf_counter()
        runtime.append(iter_end - iter_start)
                

        # Update camera perspective (optional)
        # new_yaw = (robot.cam_base_yaw - 60.0 * robot.t) % 360
        # p.resetDebugVisualizerCamera(
        #     cameraDistance=robot.cam_dist,
        #     cameraYaw=new_yaw,
        #     cameraPitch=robot.cam_pitch,
        #     cameraTargetPosition=robot.cam_target
        # )

        # --- Self-collision detection & logging ---
        dist, collision = math.inf, False
        for l1 in range(7):
            for l2 in range(7):
                if abs(l1 - l2) > 1 and not {l1, l2} == {4,6}:
                    pts = robot.getClosestPoints(l1, l2)
                    dmin = min(cp[8] for cp in pts)
                    dist = min(dist, dmin)
                    if dmin < 0:
                        collision = True
        times.append(robot.t)
        dists.append(dist)
        gammas.append(Gamma)
        #real_dists.append(min(real_dist, real_dist2))
        #tar_dists.append(dist_to_target)
        #pred_dists.append(min(dst, dst3))
        #pred_distsv.append(min(dst, dst3, dst2, dst4))

        # if collision or ecollision:
        #     print(f"t={robot.t:.3f}s: Collision! dist={dist}")
        #     break
        # else:
        #     # print(f"t={robot.t:.3f}s: Safe. dist={dist}, gamma={Gamma}")
        #     pass

        time.sleep(robot.stepsize)

    # --- Save results ---
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.2f}s")
    print(f"Total runtime: {sum(runtime[1:]):.2f}s, avg={np.mean(runtime[1:]):.4f}s, max={np.max(runtime[1:]):.4f}s, min={np.min(runtime[1:]):.4f}s, std={np.std(runtime[1:]):.4f}s")
    # df = pd.DataFrame({"time": times, "dist": dists, "gamma": gammas, "real_dist": real_dists, "tar_dist": tar_dists, "pred_dist": pred_dists, "pred_distv": pred_distsv})
    df = pd.DataFrame({"time": times, "dist": dists, "gamma": gammas, "tar_dist": tar_dists})
    # df = pd.DataFrame({"runtime": runtime})
    df.to_csv(f"../output/{time.time()}.csv", index=False)
    print("Results saved.")
