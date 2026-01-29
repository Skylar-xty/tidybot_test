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

    lambda1, lambda2, lambda3 = 5, 100, 100
    alpha = 1e-2  # 阻尼项权重
    # 球半径
    sphere_radius = 0.05

    # 1) 创建碰撞形状（如果你希望它能参与物理碰撞就不要用 -1）
    sphere_collision = p.createCollisionShape(
        shapeType=p.GEOM_SPHERE,
        radius=sphere_radius
    )

    # 2) 创建可视化形状（红色）
    sphere_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=sphere_radius,
        rgbaColor=[1, 0, 0, 1],    # 红色，alpha=1
        specularColor=[0.4,0.4,0]  # 高光颜色，可选
    )

    # 3) 把它组装成一个多体对象（质量设为0表示静态障碍物）
    x = [0.0, -0.4, 0.5]  # 你要放置的位置
    obstacle_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=sphere_collision,
        baseVisualShapeIndex=sphere_visual,
        basePosition=x,
        baseOrientation=[0,0,0,1]
    )
    obstacle_id2 = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=sphere_collision,
        baseVisualShapeIndex=sphere_visual,
        basePosition=x + np.array([0.4, 0.1, 0.0]),
        baseOrientation=[0,0,0,1]
    )

    target_pos = np.array([0.0, -0.6, 0.3])
    # target_pos = np.array([0.0, -0.0, 0.3])

    target_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.02,
        rgbaColor=[0, 1, 0, 1],    # 红色，alpha=1
        specularColor=[0.4,0.4,0]  # 高光颜色，可选
    )
    target_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=target_visual,
        basePosition=target_pos,
        baseOrientation=[0,0,0,1]
    )

    time.sleep(1)
    start_time = time.time()

    for i in range(int(duration / stepsize)):
        iter_start = time.perf_counter()
        if i % int(1.0 / stepsize) == 0:
            print(f"Simulation time: {robot.t:.3f} s")

        # --- 计算主任务的末端力 fc ---
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

        new_z = math.sin(i/900*math.pi)*0.1
        new_pos = [x[0], x[1], x[2] ]#+ new_z]
        p.resetBasePositionAndOrientation(obstacle_id, new_pos, [0,0,0,1])

        # --- 读取关节状态 & 自碰撞 gamma & 梯度 ---
        q, qd = robot.getJointStates()
        qe = functions.compute_qe(q, qd)
        x0 = np.array(x, dtype=np.float32).reshape(1,3)
        x_query = np.array(new_pos, dtype=np.float32).reshape(1,3)
        pose = np.eye(4)

        theta_np = np.stack([q, qe], axis=0).astype(np.float32)   # (B=2, 7)
        points_np = np.stack([
            x_query,
            x0 + np.array([0.4, 0.1, 0.0], dtype=np.float32)
        ], axis=0).astype(np.float32)                             # (N=2, 3)

        pose_np = np.broadcast_to(pose, (2, 4, 4)).astype(np.float32)  # (B=2, 4, 4)

        # ------ 一次批量查询：返回 (B,N) 与 (B,N,7) ------
        dsts, grad_qs = query_sdf_batch(points_np, pose_np, theta_np)  # need_index=False

        # ------ 对应回原来的四个变量 ------
        dst,  grad  = dsts[0, 0], grad_qs[0, 0]   # q,  x_query
        dst2, grad2 = dsts[1, 0], grad_qs[1, 0]   # qe, x_query
        dst3, grad3 = dsts[0, 1], grad_qs[0, 1]   # q,  x0+[0.5,0.1,0.0]
        dst4, grad4 = dsts[1, 1], grad_qs[1, 1]   # qe, x0+[0.5,0.1,0.0]

        real_dist = functions.compute_min_center_distance(
            robot_id=robot.robot,
            obstacle_id=obstacle_id,
            sphere_radius=sphere_radius,
            distance_threshold=2.0  # 根据场景最大可能距离设个比 workspace 大一点的值
        )
        real_dist2 = functions.compute_min_center_distance(
            robot_id=robot.robot,
            obstacle_id=obstacle_id2,
            sphere_radius=sphere_radius,
            distance_threshold=2.0  # 根据场景最大可能距离设个比 workspace 大一点的值
        )

        ecollision = False
        # print(f"[SDF distance]: {min(dst,dst3):.5f}m; [SDF end-distance]: {min(dst2,dst4):.5f}m; [real dst] = {min(real_dist,real_dist2):.5f}m; delta = {real_dist-dst:.5f}m; dist to target = {dist_to_target:.5f}m")
        print(f"[SDF distance]: {min(dst,dst3):.5f}m; [real dst] = {min(real_dist,real_dist2):.5f}m; delta = {real_dist-dst:.5f}m; dist to target = {dist_to_target:.5f}m")
        if min(real_dist, real_dist2) <= 0.05:
            print("Collision!")
            ecollision = True

        Gamma, grad_gamma = functions.compute_gamma_and_grad(q, qd, threshold=2.5)

        qdd_lb, qdd_ub = functions.compute_joint_acceleration_bounds_vec(
            q, qd, q_min_hardware, q_max_hardware, qd_lim, acc_max, dt=0.02, viability=True)

        
        # dist_s = [dst, dst2, dst3, dst4]
        dist_s = [dst, dst3]
        # 找到最小值的下标（0→dst, 1→dst2, 2→dst3, 3→dst4）
        min_idx = int(np.argmin(dist_s))

        # 如果最小值是 dst 或 dst2（下标 0 或 1），就用 grad，否则用 grad2
        # if min_idx == 0:
        #     sel_grad = grad
        # elif min_idx == 1:
        #     sel_grad = grad2
        # elif min_idx == 2:
        #     sel_grad = grad3
        # else:
        #     sel_grad = grad4

        if min_idx == 0:
            sel_grad = grad
        elif min_idx == 1:
            sel_grad = grad3


        # if min_idx == 0:
        #     sel_grad = grad3
        # elif min_idx == 1:
        #     sel_grad = grad4
        
            
        # if False:
        # if min(dst, dst2, dst3, dst4) < 0.106:
        if min(dst, dst3) < 0.106:
        # if min(dst3, dst4) < 0.1:

            dt = 0.02
            # 1) 计算中间量
            c     = np.dot(sel_grad, qd) * dt                    # grad·qd * dt
            g_eff = 0.5 * sel_grad * dt**2                       # 0.5 * grad * dt^2
            qdd_cmd = np.where(g_eff > 0, qdd_ub, qdd_lb)
            # qdd_cmd = np.where(g_eff > 0, acc_max, -acc_max)
            # if (c+g_eff @ qdd_cmd)<0:
            #     print(c,g_eff @ qdd_cmd,c+g_eff @ qdd_cmd)
            # qdd_cmd = np.where(g_eff > 0, acc_max, -acc_max)
            tau_cmd = robot.solveInverseDynamics(q, qd, qdd_cmd.tolist())
            robot.setTargetTorques(tau_cmd)
            print("Using fallback controller.")
            robot.step()

        else:
            soft = False
            for idx in range(7):
                if qdd_lb[idx] > qdd_ub[idx]:
                    qdd_lb[idx] = qdd_ub[idx] - 1e-4

            # --- 构建 QP：最小化 ∥J⁺ x - fc∥² + α ∥y∥² ---
            M = np.array(robot.getMassMatrix(q))
            tau_id = robot.solveInverseDynamics(q, qd, [0]*7)
            M_inv = np.linalg.inv(M)

            u = cp.Variable(7)  # torque
            y = cp.Variable(7)  # acceleration
            J = np.array(robot.getJacobian())
            JT_pinv = np.linalg.pinv(J.T)

            objective = cp.sum_squares(JT_pinv @ u - fc) + alpha * cp.sum_squares(u)

            constraints = [
                M_inv @ u >= qdd_lb + M_inv @ tau_id,
                M_inv @ u <= qdd_ub + M_inv @ tau_id,
            ]

            if grad_gamma is not None:
                dt      = 0.02
                grad_q  = grad_gamma[:7]
                grad_qd = grad_gamma[7:]
                g_eff   = 0.5 * grad_q * dt**2 + grad_qd * dt
                c_const = grad_q.dot(qd) * dt
                ε       = 4e-1
                constraints.append(g_eff @ M_inv @ u >= ε-c_const+g_eff @ M_inv @ tau_id) 
                prob = cp.Problem(cp.Minimize(objective), constraints)
                try:
                    prob.solve(solver=cp.OSQP)
                except cp.SolverError:
                    print("QP infeasible, relax constraint, use soft constraint")
                    soft = True
                    qdd_cmd = np.where(g_eff > 0, qdd_ub, qdd_lb)
                    tau_cmd = robot.solveInverseDynamics(q, qd, qdd_cmd.tolist())
                    robot.setTargetTorques(tau_cmd)
                    robot.step()
                while prob.status != cp.OPTIMAL and ε > 1e-3:
                    print("QP infeasible, relax constraint, reduce epsilon")
                    ε = ε - 1e-1
                    constraints[-1] = (g_eff @ M_inv @ u >= ε-c_const+g_eff @ M_inv @ tau_id)
                    prob = cp.Problem(cp.Minimize(objective), constraints)
                    prob.solve(solver=cp.OSQP)
                if ε < 1e-3:
                    print("QP still infeasible, relax constraint, use soft constraint")
                    soft = True
                    qdd_cmd = np.where(g_eff > 0, qdd_ub, qdd_lb)
                    tau_cmd = robot.solveInverseDynamics(q, qd, qdd_cmd.tolist())
                    robot.setTargetTorques(tau_cmd)
                    robot.step()


            else:
                prob = cp.Problem(cp.Minimize(objective), constraints)
                prob.solve(solver=cp.OSQP)

            if not soft:
                tau_cmd = np.array(u.value) 
                # tau_ext_max = np.array([5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
                tau_ext_max = np.array([87, 87, 87, 87, 12, 12, 12])*0.5
                if np.random.rand() < 0.1:   # 10% 概率打扰
                    tau_noise = np.random.uniform(-tau_ext_max, tau_ext_max)
                    # print(f"tau_noise={tau_noise}")
                    tau_noise = np.zeros(7)
                else:
                    tau_noise = np.zeros(7)
                tau_with_disturb = tau_cmd + tau_noise
                robot.setTargetTorques(tau_with_disturb.tolist())
                # --- 执行控制 & 可视化 ---
                # robot.setTargetTorques(x.value.tolist())
                robot.step()
        iter_end = time.perf_counter()
        runtime.append(iter_end - iter_start)
                

        # 更新相机视角（可选）
        # new_yaw = (robot.cam_base_yaw - 60.0 * robot.t) % 360
        # p.resetDebugVisualizerCamera(
        #     cameraDistance=robot.cam_dist,
        #     cameraYaw=new_yaw,
        #     cameraPitch=robot.cam_pitch,
        #     cameraTargetPosition=robot.cam_target
        # )

        # --- 自碰撞检测 & 日志 ---
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
        real_dists.append(min(real_dist, real_dist2))
        tar_dists.append(dist_to_target)
        pred_dists.append(min(dst, dst3))
        pred_distsv.append(min(dst, dst3, dst2, dst4))

        if collision or ecollision:
            print(f"t={robot.t:.3f}s: Collision! dist={dist}")
            break
        else:
            # print(f"t={robot.t:.3f}s: Safe. dist={dist}, gamma={Gamma}")
            pass

        time.sleep(robot.stepsize)

    # --- 保存结果 ---
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.2f}s")
    print(f"Total runtime: {sum(runtime[1:]):.2f}s, avg={np.mean(runtime[1:]):.4f}s, max={np.max(runtime[1:]):.4f}s, min={np.min(runtime[1:]):.4f}s, std={np.std(runtime[1:]):.4f}s")
    df = pd.DataFrame({"time": times, "dist": dists, "gamma": gammas, "real_dist": real_dists, "tar_dist": tar_dists, "pred_dist": pred_dists, "pred_distv": pred_distsv})
    # df = pd.DataFrame({"runtime": runtime})
    df.to_csv(f"../output/{time.time()}.csv", index=False)
    print("Results saved.")
    