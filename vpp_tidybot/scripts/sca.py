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

    target_pos = np.array([0.0, 0.0, 0.3])
    # target_pos = np.array([0.0, -0.0, 0.3])


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


        # --- 读取关节状态 & 自碰撞 gamma & 梯度 ---
        q, qd = robot.getJointStates()
        qe = functions.compute_qe(q, qd)



        Gamma, grad_gamma = functions.compute_gamma_and_grad(q, qd, threshold=2.52)

        qdd_lb, qdd_ub = functions.compute_joint_acceleration_bounds_vec(
            q, qd, q_min_hardware, q_max_hardware, qd_lim, acc_max, dt=0.02, viability=True)
        
        if True:
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
            # JT_pinv = np.linalg.pinv(J.T)
            JT_pinv = np.linalg.solve(J @ J.T + 1e-4 * np.eye(J.shape[0]), J)

            objective = cp.sum_squares(JT_pinv @ u - fc) + alpha * cp.sum_squares(u)

            constraints = [
                M_inv @ u >= qdd_lb + M_inv @ tau_id,
                M_inv @ u <= qdd_ub + M_inv @ tau_id,
            ]
            if False:
                pass
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

        # pred_dists.append(min(dst, dst3))
        # pred_distsv.append(min(dst2, dst4))

        if collision:
            print(f"t={robot.t:.3f}s: Collision! dist={dist}")
            break
        else:
            # print(f"t={robot.t:.3f}s: Safe. dist={dist}, gamma={Gamma}")
            pass
        iter_end = time.perf_counter()
        runtime.append(iter_end - iter_start)
        time.sleep(robot.stepsize)

    # --- 保存结果 ---
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.2f}s")
    print(f"Total runtime: {sum(runtime[1:]):.2f}s, avg={np.mean(runtime[1:]):.4f}s, max={np.max(runtime[1:]):.4f}s, min={np.min(runtime[1:]):.4f}s, std={np.std(runtime[1:]):.4f}s")
    df = pd.DataFrame({"time": times, "dist": dists, "gamma": gammas})
    # df = pd.DataFrame({"runtime": runtime})
    df.to_csv(f"../output/{time.time()}.csv", index=False)
    print("Results saved.")
    