# main.py

import time
import pandas as pd
import numpy as np
import math
import os
import sys
import numpy.linalg as npl

sys.path.append('./src')

import cvxpy as cp
import pybullet as p
from Panda_move import Panda 
from rdf import query_sdf_batch_nine
import functions

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SEED = 28
np.random.seed(SEED)
esc_debug = True
sca_debug = False
sol_debug = True # Set True if you want QP solver details

# --- Robot Limits (9 DOF: X, Y, J1..J7) ---
base_q_min = np.array([-10.0, -10.0], dtype=np.float32)
base_q_max = np.array([ 10.0,  10.0], dtype=np.float32)
base_qd_lim = np.array([1.0, 1.0], dtype=np.float32)   
base_acc_max = np.array([2.0, 2.0], dtype=np.float32)

q_min_hardware = np.array([-2.96706, -2.0944, -2.96706, -2.0944, -2.96706, -2.0944, -3.05433], dtype=np.float32)
q_max_hardware = np.array([ 2.96706,  2.0944,  2.96706,  2.0944,  2.96706,  2.0944,  3.05433], dtype=np.float32)
qd_lim_arm = np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562], dtype=np.float32)
acc_max_arm = np.array([15, 7.5, 10, 12.5, 15, 20, 20], dtype=np.float32)

q_min_all = np.concatenate([base_q_min, q_min_hardware])
q_max_all = np.concatenate([base_q_max, q_max_hardware])
qd_lim_all = np.concatenate([base_qd_lim, qd_lim_arm])
acc_max_all = np.concatenate([base_acc_max, acc_max_arm])

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _unflatten_mat(m):
    return np.array(m, dtype=np.float32).reshape(4, 4)

def _compute_ray_from_mouse(mouse_x, mouse_y, width, height, view_mat, proj_mat):
    ndc_x = 2.0 * mouse_x / width - 1.0
    ndc_y = 1.0 - 2.0 * mouse_y / height
    ndc_near = np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float32)
    ndc_far  = np.array([ndc_x, ndc_y,  1.0, 1.0], dtype=np.float32)
    view = _unflatten_mat(view_mat).T
    proj = _unflatten_mat(proj_mat).T
    inv_vp = npl.inv(proj @ view)
    near_world = inv_vp @ ndc_near; near_world /= near_world[3]
    far_world  = inv_vp @ ndc_far;  far_world  /= far_world[3]
    return near_world[:3], far_world[:3]

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
duration, stepsize = 26.0, 4e-3

if __name__ == "__main__":
    times, dists, gammas, tar_dists = [], [], [], []
    runtime = []
    os.makedirs("../output", exist_ok=True)

    # -------------------------------------------------------------------------
    # DEBUG: Ensure correct URDF is loaded
    # -------------------------------------------------------------------------
    urdf_path = "/Users/stav.42/f_lab/tidybot_model/model/stanford_tidybot/base_move.urdf"
    print(f"DEBUG: Loading URDF from {urdf_path}")
    
    robot = Panda(stepsize, urdf_path=urdf_path)
    robot.setControlMode("torque")
    p.setCollisionFilterPair(robot.robot, robot.robot, 8, 6, enableCollision=0)
    
    # Task params
    lambda1, lambda2, lambda3 = 3, 100, 100
    alpha = 1e-2
    sphere_radius = 0.05

    # Obstacles
    sphere_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius)
    sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=sphere_radius, rgbaColor=[1, 0, 0, 1], specularColor=[0.4, 0.4, 0])
    
    base1 = np.array([0.10, -0.30, 0.50], dtype=np.float32)
    base2 = base1 + np.array([0.55, 0.15, 0.0], dtype=np.float32)

    obstacle_ids = []
    if esc_debug:
        obstacle_ids.append(p.createMultiBody(0.0, sphere_collision, sphere_visual, basePosition=base1.tolist()))
        obstacle_ids.append(p.createMultiBody(0.0, sphere_collision, sphere_visual, basePosition=base2.tolist()))
        mouse_init_pos = (base1 + base2) / 2.0 + np.array([0.15, 0.15, 0.10], dtype=np.float32)
        obstacle_ids.append(p.createMultiBody(0.2, sphere_collision, sphere_visual, basePosition=mouse_init_pos.tolist()))

    target_pos = np.array([1.5, 0.5, 0.5], dtype=np.float32) 
    target_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 1], specularColor=[0.4, 0.4, 0])
    target_id = p.createMultiBody(0, baseVisualShapeIndex=target_visual, basePosition=target_pos.tolist())

    time.sleep(1)
    start_time_wall = time.time()
    depth_param = p.addUserDebugParameter("target depth (m)", 0.0, 2.0, 0.6)

    # DEBUG: Text IDs
    debug_text_id = -1
    debug_line_id = -1

    for i in range(int(duration / stepsize)):
        iter_start = time.perf_counter()

        # GUI
        keys = p.getKeyboardEvents()
        shift_down = p.B3G_SHIFT in keys and (keys[p.B3G_SHIFT] & p.KEY_IS_DOWN)
        width, height, view_mat, proj_mat, *_ = p.getDebugVisualizerCamera()
        for ev in p.getMouseEvents():
            if len(ev) >= 5 and ev[3] == 0 and (ev[4] & p.KEY_WAS_TRIGGERED) and shift_down:
                depth = p.readUserDebugParameter(depth_param)
                ray_from, ray_to = _compute_ray_from_mouse(ev[1], ev[2], width, height, view_mat, proj_mat)
                ray_dir = ray_to - ray_from
                ray_dir /= (np.linalg.norm(ray_dir) + 1e-9)
                target_pos = ray_from + depth * ray_dir
                p.resetBasePositionAndOrientation(target_id, target_pos.tolist(), [0, 0, 0, 1])
                p.resetBaseVelocity(target_id, [0, 0, 0], [0, 0, 0])

        if i % int(1.0 / stepsize) == 0:
            print(f"Simulation time: {robot.t:.3f} s")

        # --------------------
        # 1. State (9 DOF)
        # --------------------
        q, qd = robot.getJointStates() 
        base_pos_xy = q[0:2]
        q_arm  = q[2:]   
        qd_arm = qd[2:]

        # --------------------
        # 2. Main Task
        # --------------------
        end_pos = robot.solveForwardKinematics()[0]
        
        # Deadzone fix to prevent jitter at target
        dist_to_target = np.linalg.norm(end_pos - target_pos)
        if dist_to_target < 0.02:
            fx = np.zeros(3)
            D = np.eye(3) * 10.0
        else:
            fx = -50.0 * (end_pos - target_pos)
            e1 = fx / (np.linalg.norm(fx) + 1e-9)
            e2 = np.array([1.0, 0.0, 0.0]) - np.dot([1.0, 0.0, 0.0], e1) * e1
            e2 /= (np.linalg.norm(e2) + 1e-9)
            e3 = np.cross(e1, e2)
            Q = np.column_stack((e1, e2, e3))
            Lambda = np.diag([lambda1, lambda2, lambda3])
            D = Q @ Lambda @ Q.T

        xdot = robot.getEndVelocity()
        fc = -D @ (xdot - fx)

        # --------------------
        # 3. Obstacles Update
        # --------------------
        t = robot.t
        offset1 = np.array([0.10 * np.sin(0.5 * t), 0.00, 0.05 * np.cos(0.5 * t)], dtype=np.float32)
        offset2 = np.array([-0.10 * np.sin(0.5 * t), 0.00, -0.05 * np.cos(0.5 * t)], dtype=np.float32)
        pos1 = base1 + offset1
        pos2 = base2 + offset2

        if esc_debug:
            pos3, orn3 = p.getBasePositionAndOrientation(obstacle_ids[2])
            pos3 = np.array(pos3, dtype=np.float32)
            pos3[2] = max(pos3[2], 0.45)
            p.resetBasePositionAndOrientation(obstacle_ids[2], pos3.tolist(), orn3)
            p.resetBaseVelocity(obstacle_ids[2], [0,0,0], [0,0,0])

        # --------------------
        # 4. SDF (Arm only)
        # --------------------
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = base_pos_xy[0]
        pose[1, 3] = base_pos_xy[1]
        
        qe_arm = functions.compute_qe(q_arm, qd_arm)
        d_min = 100.0
        sel_grad_arm = np.zeros(7)
        closest_obs_idx = -1

        if esc_debug:
            theta_np = np.stack([q_arm, qe_arm], axis=0).astype(np.float32)
            points_np = np.stack([pos1, pos2, pos3], axis=0).astype(np.float32)
            pose_np = np.broadcast_to(pose, (2, 4, 4)).astype(np.float32)

            dsts, grad_qs = query_sdf_batch_nine(points_np, pose_np, theta_np)
            all_dists = dsts.flatten()
            all_grads = grad_qs.reshape(-1, 9)
            
            if len(all_dists) > 0:
                min_idx = np.argmin(all_dists)
                d_min = all_dists[min_idx]
                sel_grad_arm = all_grads[min_idx]
                
                # DEBUG: Visualize lowest distance
                debug_text_id = p.addUserDebugText(f"Min Dist: {d_min:.3f}", 
                                                   [base_pos_xy[0], base_pos_xy[1], 1.5], 
                                                   textColorRGB=[1,0,0], 
                                                   replaceItemUniqueId=debug_text_id)
                
            print("Gradient Shape:", sel_grad_arm.shape)
            
            real_dist  = functions.compute_min_center_distance(
                robot_id=robot.robot,
                obstacle_id=obstacle_ids[0],
                sphere_radius=sphere_radius,
                distance_threshold=2.0
            )
            real_dist2 = functions.compute_min_center_distance(
                robot_id=robot.robot,
                obstacle_id=obstacle_ids[1],
                sphere_radius=sphere_radius,
                distance_threshold=2.0
            )
            real_dist3 = functions.compute_min_center_distance(
                robot_id=robot.robot,
                obstacle_id=obstacle_ids[2],
                sphere_radius=sphere_radius,
                distance_threshold=2.0
            )
            min_real = min(real_dist, real_dist2, real_dist3)
            # print(f"[SDF q distance min]: {d_min:.5f} m; "
            #     f"[real min]: {min_real:.5f} m; "
            #     f"delta ≈ {min_real - d_min:.5f} m")

            ecollision = min_real <= 0.05
            if ecollision:
                print("External collision detected (Bullet sphere contact).")

        # for j in range(p.getNumJoints(robot.robot)):
        #     p.changeVisualShape(robot.robot, j, rgbaColor=[1, 1, 1, 1])
    
        # contacts = p.getContactPoints(bodyA=robot.robot)
        # if len(contacts) > 0:
        #     # print(f"Collision detected at t={robot.t:.3f}! {len(contacts)} contact points.")
        #     for c in contacts:
        #         # c[2] is bodyB, c[3] is linkIndexA, c[4] is linkIndexB
        #         bodyB = c[2]
        #         linkA = c[3]
        #         linkB = c[4]
                
        #         # Visualize contact point
        #         p.addUserDebugLine(c[5], [c[5][0], c[5][1], c[5][2]+0.2], [1, 0, 0], lineWidth=5, lifeTime=0.1)

        #         if bodyB == robot.robot:
        #             print(f"  - Self-collision between link {linkA} and link {linkB}")
        #             p.changeVisualShape(robot.robot, linkA, rgbaColor=[1, 0, 0, 1])
        #             p.changeVisualShape(robot.robot, linkB, rgbaColor=[1, 0, 0, 1])
        #         elif bodyB == robot.plane and linkA == 0:
        #             # Ignore collision between mobile base (link 0) and plane (body 0)
        #             pass
        #         else:
        #             print(f"  - Collision with body {bodyB} (link {linkA} on robot)")
        #             p.changeVisualShape(robot.robot, linkA, rgbaColor=[1, 0, 0, 1])

            
        # --------------------
        # 5. QP (9 Vars)
        # --------------------
        u = cp.Variable(9)
        
        M = robot.getMassMatrix(q.tolist()) 
        M_inv = np.linalg.inv(M)
        tau_id = robot.solveInverseDynamics(q.tolist(), qd.tolist(), [0.0]*9)
        
        J = robot.getJacobian() # 3x9
        JT_pinv = np.linalg.pinv(J.T)

        qdd_lb, qdd_ub = functions.compute_joint_acceleration_bounds_vec(
            q, qd, q_min_all, q_max_all, qd_lim_all, acc_max_all, dt=0.02, viability=True
        )

        objective = cp.sum_squares(JT_pinv @ u - fc) + alpha * cp.sum_squares(u)
        constraints = [
            M_inv @ u >= qdd_lb + M_inv @ tau_id,
            M_inv @ u <= qdd_ub + M_inv @ tau_id,
        ]

        if False:
            print(f"\n=== DEBUG QP Solver (t={robot.t:.3f}s) ===")
            print(f"q = {q}")
            print(f"qd = {qd}")
            print(f"cond(M) = {np.linalg.cond(M):.2e}")
            print(f"tau_id  = {tau_id}")

        # External SDF
        if esc_debug:
            dt_ext = 0.02
            eps_ext = 0.08
            
            # --- DEBUG: CHECK FOR ZERO GRADIENT ISSUE ---
            # You are setting Base X/Y gradient to 0.0. 
            # This means the QP thinks moving the base has NO EFFECT on collision.
            # This is likely why collision avoidance "feels disabled".
            base_gradient_hack = np.zeros(2) 
            
            # To fix it properly, we need the surface normal. 
            # But for now, let's just log if d_min is small but we aren't stopping.
            if d_min < 0.15:
                print(f"!!! COLLISION WARNING !!! Dist: {d_min:.4f} m")
                print(f"    Arm Gradient Mag: {np.linalg.norm(sel_grad_arm):.4f}")
                print(f"    Base Velocity: {qd[0]:.3f}, {qd[1]:.3f}")
                
            g_ext_full = sel_grad_arm            
            c_ext = g_ext_full.dot(qd) * dt_ext
            g_eff_ext = 0.5 * g_ext_full * dt_ext**2
            
            # RHS Calculation for Debugging
            rhs_val = eps_ext - d_min - c_ext + g_eff_ext @ M_inv @ tau_id
            
            # We can't print 'rhs_val' directly because it contains cvxpy vars, 
            # but we can print the scalar parts:
            if d_min < 0.15:
                 print(f"    Constraint Margin (eps - d): {eps_ext - d_min:.4f}")

            constraints.append(
                g_eff_ext @ M_inv @ u >= eps_ext - d_min - c_ext + g_eff_ext @ M_inv @ tau_id
            )

        # Self Collision
        # Gamma, grad_gamma_arm = functions.compute_gamma_and_grad(q_arm, qd_arm, threshold=2.5)
        # if grad_gamma_arm is not None and sca_debug:
        #     dt_sc = 0.02
        #     grad_q_arm  = grad_gamma_arm[:7]
        #     grad_qd_arm = grad_gamma_arm[7:]
            
        #     grad_q_full = np.concatenate([np.zeros(2), grad_q_arm])
        #     grad_qd_full = np.concatenate([np.zeros(2), grad_qd_arm])
            
        #     g_eff_sc = (0.5 * grad_q_full * dt_sc**2 + grad_qd_full * dt_sc)
        #     c_const_sc = grad_q_full.dot(qd) * dt_sc
        #     eps_sc = 1e-3
        #     rhs_sc = eps_sc - c_const_sc + g_eff_sc @ M_inv @ tau_id
            
        #     constraints.append(g_eff_sc @ M_inv @ u >= rhs_sc)

        # --------------------
        # 6. Solve
        # --------------------
        prob = cp.Problem(cp.Minimize(objective), constraints)
        try:
            prob.solve(solver=cp.OSQP)
        except cp.SolverError:
            prob = None

        if prob is None or prob.status != cp.OPTIMAL or u.value is None:
            print("QP Fail: Braking")
            qdd_cmd = -12.0 * np.array(qd)
            qdd_cmd = np.clip(qdd_cmd, -acc_max_all, acc_max_all)
            tau_cmd = np.array(robot.solveInverseDynamics(q.tolist(), qd.tolist(), qdd_cmd.tolist()))
        else:
            tau_cmd = np.array(u.value).reshape(9)
            if esc_debug and d_min < 0.10:
                print(f"    QP Optimal. Base Force: {tau_cmd[0]:.2f}, {tau_cmd[1]:.2f}")

        # Noise
        tau_noise = np.zeros(9)
        if np.random.rand() < 0.01:
            noise_vec = np.array([50,50] + [40]*7)
            tau_noise = np.random.uniform(-noise_vec, noise_vec)

        tau_final = tau_cmd + tau_noise
        robot.setTargetTorques(tau_final.tolist())
        robot.step()

        iter_end = time.perf_counter()
        runtime.append(iter_end - iter_start)

        times.append(robot.t)
        dists.append(0.0) 
        # gammas.append(Gamma if grad_gamma_arm is not None else 0.0)
        tar_dists.append(dist_to_target)

        time.sleep(robot.stepsize)

    elapsed = time.time() - start_time_wall
    print(f"Total wall time: {elapsed:.2f}s")
    df = pd.DataFrame({"time": times, "dist": dists, "gamma": gammas, "tar_dist": tar_dists})
    df.to_csv(f"../output/{time.time()}.csv", index=False)
    print("Results saved.")