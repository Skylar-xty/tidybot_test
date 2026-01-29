# main.py

import time
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt

import functions

import sys
sys.path.append('./src')

import cvxpy as cp
import pybullet as p
from Panda import Panda
from rdf import query_sdf_batch

SEED = 28
np.random.seed(SEED)
esc_debug = False
sca_debug = True
# all_debug = esc_debug and sca_debug
debug = False
sol_debug = True  # set True if you want QP status spam

gamma_thr = 10# self-collision threshold for Gamma
duration, stepsize = 26.0, 2e-3
# duration, stepsize = 20.0, 2e-2
if __name__ == "__main__":
    # logging buffers
    times, dists, gammas, tar_dists = [], [], [], []
    runtime = []
    os.makedirs("../output", exist_ok=True)

    # joint limits etc. (hardware limits)
    # acc_max = np.array([15, 7.5, 10, 12.5, 15, 20, 20], dtype=np.float32)
    # qd_lim  = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], dtype=np.float32)
    # q_min_hardware = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    # q_max_hardware = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])


    q_min_hardware = np.array(
        [-2.96706, -2.0944, -2.96706, -2.0944, -2.96706, -2.0944, -3.05433],
        dtype=np.float32,
    )

    q_max_hardware = np.array(
        [ 2.96706,  2.0944,  2.96706,  2.0944,  2.96706,  2.0944,  3.05433],
        dtype=np.float32,
    )

    qd_lim = np.array(
        [1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562],
        dtype=np.float32,
    )

    acc_max = np.array([15, 7.5, 10, 12.5, 15, 20, 20], dtype=np.float32)

    robot = Panda(stepsize)
    robot.setControlMode("torque")
    p.setCollisionFilterPair(robot.robot, robot.robot, 8, 6, enableCollision=0)
    # main task stiffness matrix in task space
    lambda1, lambda2, lambda3 = 3, 100, 100
    alpha = 1e-2  # torque regularization

    # obstacle radius (for real distance check)
    sphere_radius = 0.05

    # ---------------------------------------------------------
    # Create three spherical obstacles (two scripted, one mouse)
    # ---------------------------------------------------------
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

    # Base positions for scripted obstacles
    base1 = np.array([0.10, -0.30, 0.50], dtype=np.float32)
    base2 = base1 + np.array([0.55, 0.15, 0.0], dtype=np.float32)

    if esc_debug:
        obstacle_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=sphere_collision,
            baseVisualShapeIndex=sphere_visual,
            basePosition=base1.tolist(),
            baseOrientation=[0, 0, 0, 1]
        )
        obstacle_id2 = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=sphere_collision,
            baseVisualShapeIndex=sphere_visual,
            basePosition=base2.tolist(),
            baseOrientation=[0, 0, 0, 1]
        )

        # Third obstacle: mouse-draggable (non-zero mass)
        mouse_init_pos = (base1 + base2) / 2.0 + np.array([0.15, 0.15, 0.10], dtype=np.float32)
        obstacle_id3 = p.createMultiBody(
            baseMass=0.2,  # non-zero so we can drag it in the GUI
            baseCollisionShapeIndex=sphere_collision,
            baseVisualShapeIndex=sphere_visual,
            basePosition=mouse_init_pos.tolist(),
            baseOrientation=[0, 0, 0, 1]
        )

    # ---------------------------------------------------------
    # Target: near the gap between obstacle 1 and 2 so reachable
    # ---------------------------------------------------------
    # target_pos = (base1 + base2) / 2.0 + np.array([0.0, -0.05, 0.05], dtype=np.float32)
    # target_pos = base1
    
    # Target on the base (assuming base is at origin)
    # target_pos = np.array([0.0, 0.0, 0], dtype=np.float32)
    target_pos = np.array([0.3, 0.0, 0.1], dtype=np.float32)

    target_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.02,
        rgbaColor=[0, 1, 0, 1],
        specularColor=[0.4, 0.4, 0]
    )
    target_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=target_visual,
        basePosition=target_pos.tolist(),
        baseOrientation=[0, 0, 0, 1]
    )

    time.sleep(1)
    start_time_wall = time.time()

    # identity pose for SDF query (world frame)
    pose = np.eye(4, dtype=np.float32)

    # -----------------
    # main control loop
    # -----------------
    try:
        for i in range(int(duration / stepsize)):
            iter_start = time.perf_counter()

            if i % int(1.0 / stepsize) == 0:
                print(f"Simulation time: {robot.t:.3f} s")

            # ------------------------------
            # 1) main task: task-space force
            # ------------------------------
            end_pos = robot.solveForwardKinematics()[0]  # 3-vector
            fx = -50.0 * (end_pos - target_pos)

            # build anisotropic damping matrix D = Q Λ Qᵀ
            e1 = fx / (np.linalg.norm(fx) + 1e-9)
            e2 = np.array([1.0, 0.0, 0.0]) - np.dot([1.0, 0.0, 0.0], e1) * e1
            e2 /= (np.linalg.norm(e2) + 1e-9)
            e3 = np.cross(e1, e2)
            Q = np.column_stack((e1, e2, e3))
            Lambda = np.diag([lambda1, lambda2, lambda3])
            D = Q @ Lambda @ Q.T

            xdot = robot.getEndVelocity()
            fc = -D @ (xdot - fx)

            dist_to_target = np.linalg.norm(end_pos - target_pos)

            # -----------------------------------
            # 2) move scripted obstacles (1 & 2)
            # -----------------------------------
            t = robot.t

            offset1 = np.array([
                0.10 * np.sin(0.5 * t),
                0.00,
                0.05 * np.cos(0.5 * t)
            ], dtype=np.float32)

            offset2 = np.array([
                -0.10 * np.sin(0.5 * t),
                0.00,
                -0.05 * np.cos(0.5 * t)
            ], dtype=np.float32)

            pos1 = base1 + offset1
            pos2 = base2 + offset2

            # p.resetBasePositionAndOrientation(obstacle_id,  pos1.tolist(), [0, 0, 0, 1])
            # p.resetBasePositionAndOrientation(obstacle_id2, pos2.tolist(), [0, 0, 0, 1])

            # ------------------------------------
            # 3) get mouse-draggable obstacle pose
            # ------------------------------------
            if esc_debug:
                pos3, orn3 = p.getBasePositionAndOrientation(obstacle_id3)
                pos3 = np.array(pos3, dtype=np.float32)
                # keep it from falling through the table
                pos3[2] = max(pos3[2], 0.45)
                p.resetBasePositionAndOrientation(obstacle_id3, pos3.tolist(), orn3)
                p.resetBaseVelocity(obstacle_id3, [0, 0, 0], [0, 0, 0])

            # ----------------------------------------
            # 4) joint states, stopping configuration
            # ----------------------------------------
            q, qd = robot.getJointStates()          # 7, 7
            # q = np.array(q)
            # qd = np.array(qd)
            qe = functions.compute_qe(q, qd)        # "emergency" stop q

            # ----------------------------------------
            # 5) SDF query for all 3 obstacles, q & qe
            # ----------------------------------------
            if esc_debug:
                theta_np = np.stack([q, qe], axis=0).astype(np.float32)  # (B=2,7)
                points_np = np.stack([
                    pos1.astype(np.float32),
                    pos2.astype(np.float32),
                    pos3.astype(np.float32),
                ], axis=0).astype(np.float32)  # (N=3,3)

                pose_np = np.broadcast_to(pose, (2, 4, 4)).astype(np.float32)  # (B=2,4,4)

                dsts, grad_qs = query_sdf_batch(points_np, pose_np, theta_np)  # (2,3), (2,3,7)

                # unpack for readability
                dst,  grad  = dsts[0, 0], grad_qs[0, 0]   # q,  obs1
                dst2, grad2 = dsts[1, 0], grad_qs[1, 0]   # qe, obs1
                dst3, grad3 = dsts[0, 1], grad_qs[0, 1]   # q,  obs2
                dst4, grad4 = dsts[1, 1], grad_qs[1, 1]   # qe, obs2
                dst5, grad5 = dsts[0, 2], grad_qs[0, 2]   # q,  obs3
                dst6, grad6 = dsts[1, 2], grad_qs[1, 2]   # qe, obs3

                # ----------------------------------------
                # 6) "real" distances from Bullet for sanity
                # ----------------------------------------
                real_dist  = functions.compute_min_center_distance(
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
                real_dist3 = functions.compute_min_center_distance(
                    robot_id=robot.robot,
                    obstacle_id=obstacle_id3,
                    sphere_radius=sphere_radius,
                    distance_threshold=2.0
                )
                min_real = min(real_dist, real_dist2, real_dist3)

                print(f"[SDF q distance min]: {min(dst, dst3, dst5):.5f} m; "
                    f"[real min]: {min_real:.5f} m; "
                    f"delta ≈ {min_real - min(dst, dst3, dst5):.5f} m")

                ecollision = min_real <= 0.05
                if ecollision:
                    print("External collision detected (Bullet sphere contact).")

            # Check for actual collisions using PyBullet's contact points
            # Reset colors first
            for j in range(p.getNumJoints(robot.robot)):
                p.changeVisualShape(robot.robot, j, rgbaColor=[1, 1, 1, 1])

            contacts = p.getContactPoints(bodyA=robot.robot)
            if len(contacts) > 0:
                # print(f"Collision detected at t={robot.t:.3f}! {len(contacts)} contact points.")
                for c in contacts:
                    # c[2] is bodyB, c[3] is linkIndexA, c[4] is linkIndexB
                    bodyB = c[2]
                    linkA = c[3]
                    linkB = c[4]
                    
                    # Visualize contact point
                    p.addUserDebugLine(c[5], [c[5][0], c[5][1], c[5][2]+0.2], [1, 0, 0], lineWidth=5, lifeTime=0.1)

                    if bodyB == robot.robot:
                        print(f"  - Self-collision between link {linkA} and link {linkB}")
                        p.changeVisualShape(robot.robot, linkA, rgbaColor=[1, 0, 0, 1])
                        p.changeVisualShape(robot.robot, linkB, rgbaColor=[1, 0, 0, 1])
                    elif bodyB == robot.plane and linkA == 0:
                        # Ignore collision between mobile base (link 0) and plane (body 0)
                        pass
                    else:
                        print(f"  - Collision with body {bodyB} (link {linkA} on robot)")
                        p.changeVisualShape(robot.robot, linkA, rgbaColor=[1, 0, 0, 1])

            # ----------------------------------------
            # 7) self-collision gamma + joint acc bounds
            # ----------------------------------------
            Gamma, grad_gamma = functions.compute_gamma_and_grad(q, qd, threshold=gamma_thr)
            # if grad_gamma is not None:
            #     print(f"Self-collision Gamma: {Gamma:.5f}, grad_gamma norm: {np.linalg.norm(grad_gamma):.5f}")

            qdd_lb, qdd_ub = functions.compute_joint_acceleration_bounds_vec(
                q, qd, q_min_hardware, q_max_hardware, qd_lim, acc_max, dt=0.02, viability=True
            )

            # make sure lower <= upper
            for idx in range(7):
                if qdd_lb[idx] > qdd_ub[idx]:
                    qdd_lb[idx] = qdd_ub[idx] - 1e-4

            # ----------------------------------------
            # 8) pick the closest SDF distance & gradient
            # ----------------------------------------
            if esc_debug:
                dist_s = [dst, dst2, dst3, dst4, dst5, dst6]
                min_idx = int(np.argmin(dist_s))

                if   min_idx == 0: sel_grad = grad
                elif min_idx == 1: sel_grad = grad2
                elif min_idx == 2: sel_grad = grad3
                elif min_idx == 3: sel_grad = grad4
                elif min_idx == 4: sel_grad = grad5
                else:              sel_grad = grad6

            # ----------------------------------------
            # 9) QP: min ||J^+ u - f_c||^2 + α ||u||^2
            #     s.t. joint acc bounds + external SDF
            #     (+ optional self-collision)
            # ----------------------------------------
            soft = False
            M = np.array(robot.getMassMatrix(q))
            tau_id = robot.solveInverseDynamics(q, qd, [0.0] * 7)
            M_inv = np.linalg.inv(M)

            if debug:
                print(f"\n=== DEBUG QP Solver (t={robot.t:.3f}s) ===")
                print(f"q = {q}")
                print(f"qd = {qd}")
                print(f"cond(M) = {np.linalg.cond(M):.2e}")
                print(f"tau_id  = {tau_id}")

            u = cp.Variable(7)  # torque decision variable
            J = np.array(robot.getJacobian())  # 3x7
            JT_pinv = np.linalg.pinv(J.T)      # 7x3 -> 7x3, so JT_pinv @ u is 3-vector

            objective = cp.sum_squares(JT_pinv @ u - fc) + alpha * cp.sum_squares(u)

            constraints = [
                M_inv @ u >= qdd_lb + M_inv @ tau_id,
                M_inv @ u <= qdd_ub + M_inv @ tau_id,
            ]
            # constraints =[]

            # # ------ External SDF collision avoidance constraint ------
            if esc_debug:
                dt_ext = stepsize
                g_ext = sel_grad                 # (7,)
                c_ext = g_ext.dot(qd) * dt_ext
                g_eff_ext = 0.5 * g_ext * dt_ext**2
                eps_ext = 0.08                   # safe threshold in meters-ish
                d_min = min(dist_s)              # nearest SDF distance among all

                constraints.append(
                    g_eff_ext @ M_inv @ u >= eps_ext - d_min - c_ext + g_eff_ext @ M_inv @ tau_id
                )

            # ------ Optional self-collision constraint ------
            if grad_gamma is not None and sca_debug:
                dt_sc = stepsize
                grad_q  = grad_gamma[:7]
                grad_qd = grad_gamma[7:]
                scale = 1
                g_eff_sc = scale * (0.5 * grad_q * dt_sc**2 + grad_qd * dt_sc)
                c_const_sc = grad_q.dot(qd) * dt_sc
                eps_sc = 1e-3
                rhs_sc = eps_sc - c_const_sc + g_eff_sc @ M_inv @ tau_id
                print(f"Self-collision constraint rhs: {rhs_sc:.5f}")
                # rhs_sc *= 0.5  # make it less aggressive

                constraints.append(g_eff_sc @ M_inv @ u >= rhs_sc)

                # ------ Solve QP ------
                prob = cp.Problem(cp.Minimize(objective), constraints)
                try:
                    prob.solve(solver=cp.OSQP)
                    if sol_debug:
                        print(f"QP Status: {prob.status}, obj={prob.value}")
                except cp.SolverError as e:
                    if sol_debug:
                        print("QP solver error:", e)
                    prob = None
                while prob.status != cp.OPTIMAL and eps_sc > 1e-4:
                    eps_sc = eps_sc - 0.01
                    rhs_sc = eps_sc - c_const_sc + g_eff_sc @ M_inv @ tau_id
                    constraints[-1] = (g_eff_sc @ M_inv @ u >= rhs_sc)
                    prob = cp.Problem(cp.Minimize(objective), constraints)
                    prob.solve(solver=cp.OSQP)
                    print(f"QP infeasible, relax constraint, reduce epsilon to {eps_sc:.2f}, now status: {prob.status}")
                if prob.status != cp.OPTIMAL:
                    print("QP still infeasible, relax constraint, use soft constraint (BRAKING)")
                    soft = True
                    # Fallback: Braking strategy to prevent drifting
                    # Use strong damping and clip to hardware limits (ignoring viability bounds)
                    # qdd_cmd = -15.0 * np.array(qd)
                    # # qdd_cmd = np.clip(qdd_cmd, -acc_max, acc_max)
                    # qdd_cmd = np.clip(qdd_cmd, qdd_lb, qdd_ub)
                    qdd_cmd = np.where(g_eff_sc > 0, qdd_ub, qdd_lb)
                    tau_cmd = robot.solveInverseDynamics(q, qd, qdd_cmd.tolist())
                    tau_cmd = np.array(robot.solveInverseDynamics(list(q), list(qd), qdd_cmd.tolist()))
                else:
                    tau_cmd = np.array(u.value).reshape(7)

            else:
                # ------ Solve QP ------
                prob = cp.Problem(cp.Minimize(objective), constraints)
                try:
                    prob.solve(solver=cp.OSQP)
                    if sol_debug:
                        print(f"QP Status: {prob.status}, obj={prob.value}")
                except cp.SolverError as e:
                    if sol_debug:
                        print("QP solver error:", e)
                    prob = None
                # tau_cmd = None
                if prob is None or prob.status != cp.OPTIMAL or u.value is None:
                    # Fallback: saturate accelerations based on external gradient sign
                    soft = True
                    qdd_cmd = np.where(g_eff_ext> 0.0, qdd_ub, qdd_lb)
                    tau_cmd = robot.solveInverseDynamics(q, qd, qdd_cmd.tolist())
                else:
                    tau_cmd = np.array(u.value).reshape(7)
                # -----------------------------------------------------------------
            # 10) send torques (with small random perturbation if you want it)
            # -----------------------------------------------------------------
            tau_ext_max = np.array([87, 87, 87, 87, 12, 12, 12]) * 0.5
            if np.random.rand() < 0.01:
                tau_noise = np.random.uniform(-tau_ext_max, tau_ext_max)
                # if you don't want noise, comment next line
                tau_noise = np.zeros(7)
            else:
                tau_noise = np.zeros(7)

            tau_with_disturb = tau_cmd + tau_noise
            robot.setTargetTorques(tau_with_disturb.tolist())
            robot.step()

            iter_end = time.perf_counter()
            runtime.append(iter_end - iter_start)

            # ----------------------------------------
            # 11) self-collision distance logging
            # ----------------------------------------
            dist, collision = math.inf, False
            for l1 in range(7):
                for l2 in range(7):
                    if abs(l1 - l2) > 1 and not {l1, l2} == {4, 6}:
                        pts = robot.getClosestPoints(l1, l2)
                        if len(pts) == 0:
                            continue
                        dmin = min(contact[8] for contact in pts)
                        dist = min(dist, dmin)
                        if dmin < 0:
                            collision = True

            times.append(robot.t)
            dists.append(dist)
            gammas.append(Gamma)
            tar_dists.append(dist_to_target)

            # Add gamma and gamma_grad logging
            if grad_gamma is not None:
                gamma_grad_norm = np.linalg.norm(grad_gamma)
            else:
                gamma_grad_norm = 0

            # Append gamma_grad_norm to a new list
            if 'gamma_grads' not in locals():
                gamma_grads = []
            gamma_grads.append(gamma_grad_norm)

            # let GUI breathe a bit
            time.sleep(robot.stepsize)


        # # ----------------------------------------
        # # 11) self-collision distance logging
        # # ----------------------------------------
        # # --- Visualization (Self-Collision) ---
        # # Reset colors
        # for link_idx in range(-1, p.getNumJoints(robot.robot)):
        #     p.changeVisualShape(robot.robot, link_idx, rgbaColor=[1, 1, 1, 1])

        # contacts = p.getContactPoints(bodyA=robot.robot, bodyB=robot.robot)
        # for c in contacts:
        #     # c[5] is positionOnA
        #     p.addUserDebugLine(c[5], [c[5][0], c[5][1], c[5][2]+0.2], [1, 0, 0], lineWidth=5, lifeTime=robot.stepsize*2)
        #     p.changeVisualShape(robot.robot, c[3], rgbaColor=[1, 0, 0, 1])
        #     p.changeVisualShape(robot.robot, c[4], rgbaColor=[1, 0, 0, 1])
        #     # --------------------------------------

        #     dist, collision = math.inf, False
        #     for l1 in range(7):
        #         for l2 in range(7):
        #             if abs(l1 - l2) > 1 and not {l1, l2} == {4, 6}:
        #                 pts = robot.getClosestPoints(l1, l2)
        #                 if len(pts) == 0:
        #                     continue
        #                 dmin = min(contact[8] for contact in pts)
        #                 dist = min(dist, dmin)
        #                 if dmin < 0:
        #                     collision = True

        #     times.append(robot.t)
        #     dists.append(dist)
        #     gammas.append(Gamma)
        #     tar_dists.append(dist_to_target)

        #     # Add gamma and gamma_grad logging
        #     if grad_gamma is not None:
        #         gamma_grad_norm = np.linalg.norm(grad_gamma)
        #     else:
        #         gamma_grad_norm = 0

        #     # Append gamma_grad_norm to a new list
        #     if 'gamma_grads' not in locals():
        #         gamma_grads = []
        #     gamma_grads.append(gamma_grad_norm)

        #     # let GUI breathe a bit
        #     time.sleep(robot.stepsize)
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
        sys.stdout.flush()  
    finally:
        # -------------
        # Save results
        # -------------
        print("Executing the finally block.")  # Debugging statement
        elapsed = time.time() - start_time_wall
        print(f"Total wall time: {elapsed:.2f}s")
        print(
            f"Total runtime (sum per-iter): {sum(runtime[1:]):.2f}s, "
            f"avg={np.mean(runtime[1:]):.4f}s, "
            f"max={np.max(runtime[1:]):.4f}s, "
            f"min={np.min(runtime[1:]):.4f}s, "
            f"std={np.std(runtime[1:]):.4f}s"
        )

        df = pd.DataFrame(
            {
                "time": times,
                "dist": dists,
                "gamma": gammas,
                "tar_dist": tar_dists,
            }
        )
        df.to_csv(f"../output/{time.time()}.csv", index=False)
        print("Results saved.")

        # Plot gamma and gamma_grad after simulation
        plt.figure(figsize=(10, 6))
        plt.plot(times, gammas, label='Gamma', color='blue')
        plt.plot(times, gamma_grads, label='Gamma Gradient Norm', color='orange')
        plt.xlabel('Time (s)')
        plt.ylabel('Values')
        plt.title('Gamma and Gamma Gradient Norm Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig('../output/gamma_plot_g10.png')
        plt.show()
