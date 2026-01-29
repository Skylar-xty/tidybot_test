import sys
import torch
import torch.nn as nn

sys.path.append('./src')
import sys
import os
import pybullet as p

# 动态添加 scripts 文件夹到模块搜索路径
# scripts_path = r"/Users/stav.42/f_lab/vpp_tidybot/scripts"
scripts_path = r"G:\\My Drive\\PENN\\vpp_tidybot_test\\vpp_tidybot\\scripts"
sys.path.append(scripts_path)

from Panda import Panda  # 从 Panda.py 文件中导入 Panda 类
import time
import numpy as np
import cvxpy as cp
# from panda_noball import Panda
from torch.autograd.functional import hessian
import matplotlib.pyplot as plt
from scipy.io import savemat
duration = 12
stepsize = 2e-3

q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

robot = Panda(stepsize)
robot.setControlMode("torque")

lambda1 = 5
lambda2 = 100
lambda3 = 100

# plt.ion()
# plt.figure(1)
# t = [0]
# t_now = 0

# class BinaryClassifier(nn.Module):
#     def __init__(self):
#         super(BinaryClassifier, self).__init__()
#         self.linear1 = nn.Linear(7, 80)
#         self.linear2 = nn.Linear(80, 50)
#         self.linear3 = nn.Linear(50, 30)
#         self.linear4 = nn.Linear(30, 10)
#         self.linear5 = nn.Linear(10, 2)
#         self.leakyrelu = nn.LeakyReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()
#
#     def forward(self, x):
#         out = self.linear1(x)
#         out = self.tanh(out)
#         out = self.linear2(out)
#         out = self.tanh(out)
#         out = self.linear3(out)
#         out = self.tanh(out)
#         out = self.linear4(out)
#         out = self.tanh(out)
#         out = self.linear5(out)
#         return out

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.linear1 = nn.Linear(7, 40)
        self.linear2 = nn.Linear(40, 40)
        self.linear3 = nn.Linear(40, 40)
        self.linear4 = nn.Linear(40, 40)
        self.linear5 = nn.Linear(40, 2)
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.linear1(x)
        out = self.tanh(out)
        out = self.linear2(out)
        out = self.tanh(out)
        out = self.linear3(out)
        out = self.tanh(out)
        out = self.linear4(out)
        out = self.tanh(out)
        out = self.linear5(out)
        return out

device = torch.device('cuda')
# model = torch.load(f"./SCA_boundary_tanh.pt",map_location=torch.device('cpu'))
# model = torch.load(f"./SCA_boundary_tanh.pt")
# model = torch.load(f"./SCA_boundary_5000.pt")
model = torch.load(f"./Classification/NN_result/SCA_boundary_2000.pt", weights_only=False, map_location=device)

print(torch.cuda.is_available())

def f(inputs):
    return model(inputs)[1] - model(inputs)[0]

Tau_SCA = []
mindvector = []
times = []
ee_pos = []
# For comparison plotting
dists_to_target = []  # Euclidean EE->target distance
height_diffs = []     # Optional: Z-only difference
runtime = []   
# Define a fixed target position (world frame) and create a PyBullet visual marker once
target_position = np.array([0.3, 0.0, 0.1])
target_body = None
try:
    # create a small green sphere as marker (visual target)
    target_vis = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 1])
    target_body = p.createMultiBody(baseVisualShapeIndex=target_vis, basePosition=target_position)
except Exception:
    # If pybullet isn't available or Panda already manages bodies, ignore
    target_body = None

start_time0 = time.time()
for i in range(int(duration / stepsize)):
    iter_start = time.perf_counter()
    if i % 1000 == 0:
        print("Simulation time: {:.3f}".format(robot.t))

    end_pos = robot.solveForwardKinematics()[0]
    # fx = [-50 * (end_pos[0] - 0.3), -50 * (end_pos[1] - 0.3), -50 * (end_pos[2] - 0.5)]
    # fx = [-50 * (end_pos[0] - 0), -50 * (end_pos[1] - 0), -50 * (end_pos[2] - 0.3)]
    k_x = -25
    fx = [k_x * (end_pos[0] - target_position[0]), k_x* (end_pos[1] - target_position[1]), k_x * (end_pos[2] - target_position[2])]

    # log EE->target distances for NPZ
    try:
        dist_to_target = float(np.linalg.norm(np.asarray(end_pos) - np.asarray(target_position)))
        dists_to_target.append(dist_to_target)
        height_diffs.append(float(abs(end_pos[2] - target_position[2])))
    except Exception:
        pass

    #### Compute damping matrix ####
    e1 = np.array(fx)/np.linalg.norm(fx)
    e2_0 = np.array([1, 0, 0])
    e3_0 = np.array([0, 1, 0])
    e2 = e2_0 - np.dot(e2_0, (e1/np.linalg.norm(e1)))*(e1/np.linalg.norm(e1))
    e2 = e2/np.linalg.norm(e2)
    e3 = e3_0 - np.dot(e3_0, (e1/np.linalg.norm(e1)))*(e1/np.linalg.norm(e1)) - np.dot(e3_0, (e2 / np.linalg.norm(e2))) * (e2 / np.linalg.norm(e2))
    e3 = e3/np.linalg.norm(e3)
    Q = np.zeros([3, 3])
    Q[:, 0] = np.transpose(e1)
    Q[:, 1] = np.transpose(e2)
    Q[:, 2] = np.transpose(e3)

    Lambda = np.diag([lambda1, lambda2, lambda3])

    D = Q @ Lambda @ np.transpose(Q)

    xdot = robot.getEndVelocity()

    #### Compute desired target force ####
    fc = -D @ (np.transpose(np.array(xdot)) - np.transpose(fx))

    #### Compute Jacobian ####
    Jacobian = np.zeros([7, 3])
    Jacobian[..., 0] = np.transpose(np.array(robot.getJacobian()[0]))
    Jacobian[..., 1] = np.transpose(np.array(robot.getJacobian()[1]))
    Jacobian[..., 2] = np.transpose(np.array(robot.getJacobian()[2]))
    # target_torque = np.transpose(np.linalg.pinv(Jacobian)) @ fc

    #### Compute parameters of QP ####

    ## Dynamics constraints ##
    q = robot.getJointStates()[0]
    q_dot = robot.getJointStates()[1]
    q_ddot_test = [0, 0, 0, 0, 0, 0, 0]
    tau = robot.solveInverseDynamics(q, q_dot, q_ddot_test)
    b = np.array(robot.getMassMatrix(q)) @ np.transpose(np.array(q_ddot_test)) - np.transpose(np.array(tau))

    alpha1 = 10
    alpha2 = 100

    alpha1_JL = 100
    alpha2_JL = 100

    ## Self-collision avoidance constraints ##
    start_time = time.time()
    epsilon_SCA = 10
    inputs = torch.tensor(q, dtype=torch.float32, requires_grad=True).to(device)
    output = model(inputs)[1] - model(inputs)[0]
    # dTau = torch.autograd.grad(output, inputs, retain_graph=True, create_graph=True)[0].detach().numpy()
    dTau = torch.autograd.grad(output, inputs, retain_graph=True, create_graph=True)[0].cpu().detach().numpy()
    hTau = hessian(f,inputs).cpu().detach().numpy()
    righthandside = -alpha1 * (output.cpu().detach().numpy() - epsilon_SCA) - alpha2 * dTau @ np.array(q_dot) - np.array(
        q_dot) @ hTau @ np.transpose(np.array(q_dot))
    end_time = time.time()
    Tau_SCA.append(output.cpu().detach().numpy())
    # print('SCA par cal time: ' + str(end_time - start_time))
    # print(hTau)

    epsilon_joint_limit = 0.00

    righthandside1 = -alpha1_JL*(np.array(q) - q_min - epsilon_joint_limit) - alpha2_JL*np.array(q_dot)
    righthandside2 = -alpha1_JL*(np.array(q) - q_max + epsilon_joint_limit) - alpha2_JL*np.array(q_dot)

    ## External Collision Avoidance
    

    #### Launch QP to get target torque ####
    x = cp.Variable(7) # Torque
    y = cp.Variable(7) # q ddot

    prob = cp.Problem(cp.Minimize(cp.square(cp.norm2(np.transpose(np.transpose(np.linalg.pinv(Jacobian)))@x - fc))),
                       [np.array(robot.getMassMatrix(q))@y-x == b,
                                  dTau@y >= righthandside,
                                  y >= righthandside1,
                                  y <= righthandside2])

    # prob = cp.Problem(cp.Minimize(cp.square(cp.norm2(np.transpose(Jacobian)@x - fc))),
    #                    [np.array(robot.getMassMatrix(q))@y-x == b,
    #                               dTau@y >= righthandside,
    #                               y >= righthandside1,
    #                               y <= righthandside2])

    # prob = cp.Problem(cp.Minimize(cp.square(cp.norm2(np.transpose(np.transpose(np.linalg.pinv(Jacobian)))@x - fc))),
    #                 [np.array(robot.getMassMatrix(q))@y-x == b,
    #                               y >= righthandside1,
    #                               y <= righthandside2])

    # prob = cp.Problem(cp.Minimize(cp.square(cp.norm2(np.transpose(Jacobian) @ x - fc))),
    #                                       [np.array(robot.getMassMatrix(q))@y-x == b,
    #                                        dTau@y >= righthandside])
    #prob = cp.Problem(cp.Minimize(cp.square(cp.norm2(np.linalg.pinv(Jacobian) @ x - fc))))
    # prob = cp.Problem(cp.Minimize(cp.square(cp.norm2( x - Jacobian @ fc))),
    #                    [np.array(robot.getMassMatrix(q))@y-x == b])
    prob.solve()
    print("QP status:", prob.status)
    target_torque = [x.value[0], x.value[1], x.value[2], x.value[3], x.value[4], x.value[5], x.value[6]]
    robot.setTargetTorques(target_torque)
    times.append(robot.t)
    ee_pos.append(np.asarray(end_pos).copy())
    robot.step()
    mindvector.append(robot.getClosestPoints(1, 6)[0][8])

    # if i % 100 == 0:
        # print(robot.solveForwardKinematics())
    iter_end = time.perf_counter()
    runtime.append(iter_end - iter_start)
    time.sleep(robot.stepsize)
    # print(time.time() - start_time)

elapsed = time.time() - start_time0
print(f"Total time: {elapsed:.2f}s")
print(f"Total runtime: {sum(runtime[1:]):.2f}s, avg={np.mean(runtime[1:]):.4f}s, max={np.max(runtime[1:]):.4f}s, min={np.min(runtime[1:]):.4f}s, std={np.std(runtime[1:]):.4f}s")

ee_pos  = np.vstack(ee_pos)      # (N, 3)
times   = np.asarray(times)

# ---- (Optional) End-effector normalized jerk (dimensionless) ----
# Compute end-effector acceleration & jerk numerically
ee_vel  = np.diff(ee_pos, axis=0) / stepsize            # (N-1, 3)
ee_acc  = np.diff(ee_vel, axis=0) / stepsize            # (N-2, 3)
ee_jerk = np.diff(ee_acc, axis=0) / stepsize            # (N-3, 3)
print("ee_jerk top 5:", ee_jerk[:5])
T = duration
path_len = np.sum(np.linalg.norm(np.diff(ee_pos, axis=0), axis=1))
print("Path length (m):", np.round(path_len, 4))
NJ = (1.0 / (path_len**2*T**5)) * np.sum(np.sum(ee_jerk**2, axis=1)) * stepsize
print("End-effector Normalized Jerk (dimensionless):", np.round(NJ, 6))
from scipy.signal import savgol_filter
dt = stepsize
# choose an odd window; ~0.1–0.2 s is a good start -> e.g., 51 samples at 500 Hz
win = 51; poly = 5

# Direct jerk from positions (3rd derivative)
ee_jerk_sg = savgol_filter(ee_pos, window_length=win, polyorder=poly,
                           deriv=3, delta=dt, axis=0, mode="interp")
# trim edges (half window) to avoid boundary artifacts
k = win//2
ee_jerk_valid = ee_jerk_sg[k:-k]

# recompute path length on the same valid segment
ee_pos_valid = ee_pos[k:-k]
L = np.sum(np.linalg.norm(np.diff(ee_pos_valid, axis=0), axis=1)) + 1e-12
T_valid = dt * (ee_pos_valid.shape[0]-1)

NJ_dimless = (1.0 / (L**2*T_valid**5)) * np.sum(np.sum(ee_jerk_valid**2, axis=1)) * dt
print("Normalized Jerk (smoothed):", NJ_dimless)
mdic = {"a": Tau_SCA, "label": "experiment"}
savemat("Tau_SCA_active.mat", mdic)

mdic = {"a": mindvector, "label": "experiment"}
savemat("mindist_active.mat", mdic)

plt.plot(range(int(duration / stepsize)), Tau_SCA)
plt.show()

# ---- Save NPZ for comparison plotter ----
try:
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    os.makedirs(out_dir, exist_ok=True)
    ts = str(int(time.time()))
    # Reconstruct time axis from stepsize and step count for robustness
    time_axis = np.arange(len(times), dtype=np.float32) * np.float32(stepsize)
    payload = {
        'time': time_axis,
        'tar_dist': np.asarray(dists_to_target, dtype=np.float32),
        'height_diff': np.asarray(height_diffs, dtype=np.float32),
        'stepsize': np.float32(stepsize),
    }
    np.savez(os.path.join(out_dir, f'sca_run_{ts}.npz'), **payload)
    np.savez(os.path.join(out_dir, 'sca_latest.npz'), **payload)
    print(f"Saved SCA NPZ to: {out_dir}")
except Exception as e:
    print("Warning: failed to save SCA NPZ:", e)
# plt.plot(range(int(duration / stepsize)), mindvector)
# plt.show()
