import math
import numpy as np
from typing import Tuple, List, Union, Optional
import torch
from Transformer import gamma_model
import pybullet as p           # 新增：用于自碰撞检测

def acc_bounds_from_pos(q, qd, qmin, qmax, dt):
    qddmax1 = -qd/dt
    qddmax2 = -qd**2/(2*(qmax-q))
    qddmax3 = 2*(qmax-q-dt*qd)/(dt**2)
    qddmin2 = qd**2/(2*(q-qmin))
    qddmin3 = 2 * (qmin - q - dt * qd) / (dt**2)

    if qd >= 0:
        qdd_lb = qddmin3
        if qddmax3 > qddmax1:
            qdd_ub = qddmax3
        else:
            qdd_ub = min(qddmax1, qddmax2)
    else:
        qdd_ub = qddmax3
        if qddmin3 < qddmax1:
            qdd_lb = qddmin3
        else:
            qdd_lb = max(qddmax1, qddmin2)

    return qdd_lb, qdd_ub

def acc_bounds_from_viability(q, qd, qmin, qmax, qdd_max, dt):
    """
    Compute acceleration bounds (qdd_lb, qdd_ub) for state viability 
    w.r.t. position [qmin, qmax] using Algorithm 2 from the paper.
    """
    # Precompute common terms
    a = dt**2
    # Upper‐bound quadratic coeffs
    b = dt * (2*qd + qdd_max * dt)
    c = qd**2 - 2 * qdd_max * (qmax - q - dt*qd)
    qdd1 = -qd / dt

    # Discriminant and UB root
    delta = b**2 - 4 * a * c
    if delta >= 0:
        root_ub = (-b + math.sqrt(delta)) / (2 * a)
        qdd_ub = max(qdd1, root_ub)
    else:
        qdd_ub = qdd1

    # Lower‐bound quadratic coeffs
    b = 2 * dt * qd - qdd_max * dt**2
    c = qd**2 - 2 * qdd_max * (q + dt*qd - qmin)
    delta = b**2 - 4 * a * c
    if delta >= 0:
        root_lb = (-b - math.sqrt(delta)) / (2 * a)
        qdd_lb = min(qdd1, root_lb)
    else:
        qdd_lb = qdd1

    return qdd_lb, qdd_ub

def compute_joint_acceleration_bounds(q, qd, qmin, qmax, qdot_max, qdd_max, dt, viability=True):
    """
    Algorithm 3: Compute joint acceleration bounds by combining
    1) position limits via acc_bounds_from_pos,
    2) velocity limits,
    3) viability limits via acc_bounds_from_viability,
    4) trivial |qdd| ≤ qdd_max constraint.
    Returns (qdd_lb, qdd_ub).
    """
    # 1) Position-based bounds
    lb_pos, ub_pos = acc_bounds_from_pos(q, qd, qmin, qmax, dt)

    # 2) Velocity-based bounds (assume symmetric ±qdot_max)
    lb_vel = (-qdot_max - qd) / dt
    ub_vel = ( qdot_max - qd) / dt

    # 3) Viability-based bounds
    if viability:
        lb_viab, ub_viab = acc_bounds_from_viability(q, qd, qmin, qmax, qdd_max, dt)
    else:
        lb_viab = -qdd_max
        ub_viab =  qdd_max
    # If viability is not used, we can set the bounds to trivial limits

    # 4) Trivial acceleration limits
    lb_triv = -qdd_max
    ub_triv =  qdd_max

    # Combine all lower‐bounds (take max) and upper‐bounds (take min)
    qdd_lb = max(lb_pos, lb_vel, lb_viab, lb_triv)
    qdd_ub = min(ub_pos, ub_vel, ub_viab, ub_triv)

    return qdd_lb, qdd_ub

def compute_joint_acceleration_bounds_vec(
    q: np.ndarray,
    qd: np.ndarray,
    qmin: np.ndarray,
    qmax: np.ndarray,
    qdot_max: np.ndarray,
    qdd_max: np.ndarray,
    dt: float,
    viability: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized over n joints. Returns (qdd_lb, qdd_ub), each an array of length n.
    Each element i is compute_joint_acceleration_bounds(q[i], qd[i], ..., dt).
    """
    # ensure numpy arrays
    q         = np.asarray(q)
    qd        = np.asarray(qd)
    qmin      = np.asarray(qmin)
    qmax      = np.asarray(qmax)
    qdot_max  = np.asarray(qdot_max)
    qdd_max   = np.asarray(qdd_max)
    viability = bool(viability)

    n = q.size
    qdd_lb = np.empty(n)
    qdd_ub = np.empty(n)

    for i in range(n):
        lb, ub = compute_joint_acceleration_bounds(
            q[i], qd[i],
            qmin[i], qmax[i],
            qdot_max[i] if qdot_max.size>1 else float(qdot_max),
            qdd_max[i]  if qdd_max.size>1  else float(qdd_max),
            dt,
            viability
        )
        qdd_lb[i] = lb
        qdd_ub[i] = ub

    return qdd_lb, qdd_ub



def compute_gamma_and_grad(
    q: np.ndarray,
    qd: np.ndarray,
    threshold: float
) -> Tuple[float, Optional[np.ndarray]]:
    """
    返回:
      - gamma_val: float
      - full_grad: np.ndarray of shape (14,) when gamma < threshold, else None
    """
    # 1) 构造 tensor（都不需要在这里对 q 开启 grad）
    q_t  = torch.tensor(q,  dtype=torch.float32, requires_grad=False)
    qd_t = torch.tensor(qd, dtype=torch.float32, requires_grad=False)

    # 2) 前向，解包 gamma 和 x
    gamma_t, x = gamma_model(q_t.unsqueeze(0), qd_t.unsqueeze(0))
    # gamma_t: 标量 tensor； x: shape (1,14), requires_grad=True

    gamma_val = gamma_t.item()

    # 3) 如果需要梯度，就反向并从 x.grad 里拿
    if gamma_val < threshold:
        gamma_t.backward()
        grad14 = x.grad.squeeze(0).cpu().numpy()  # (14,)
        return gamma_val, grad14
    else:
        return gamma_val, None
    
def compute_min_center_distance(robot_id: int,
                                obstacle_id: int,
                                sphere_radius: float,
                                distance_threshold: float = 1.0) -> float:
    """
    计算机器人任意 link 到 障碍物中心 的最小距离。

    PyBullet getClosestPoints 返回的是“表面到表面”的距离，
    对于一个半径为 r 的球体，它等于：  |link_surface - sphere_surface|
    因此，如果要得到“link_surface 到 球心”的距离，需要加上 sphere_radius。

    Args:
        robot_id:           Panda.loadURDF 返回的 id
        obstacle_id:        createMultiBody 返回的 id
        sphere_radius:      球的半径
        distance_threshold: 搜索距离阈值，> 0 即可

    Returns:
        float: 最小的 link→球心 距离
    """
    # 在两者之间查找所有距离 < distance_threshold 的点对
    pts = p.getClosestPoints(bodyA=robot_id,
                             bodyB=obstacle_id,
                             distance=distance_threshold)
    if not pts:
        # 超过阈值，则认为都在更远处
        return float('inf')
    # pts[i][8] 是 “表面到表面” 的距离
    min_surface_dist = min(pt[8] for pt in pts)
    # 转成 “表面到球心” 距离
    return min_surface_dist + sphere_radius

def feasible_qdd_region(
    grad_gamma: np.ndarray,  # shape (14,)
    qd: np.ndarray,           # shape (7,)
    dt: float,
    qdd_lb: np.ndarray,       # shape (7,)
    qdd_ub: np.ndarray        # shape (7,)
):
    """
    返回 Delta gamma 在盒约束 [qdd_lb, qdd_ub] 上的最小值和最大值，
    以及判断是否存在能使 Delta gamma > 0 的 qdd。
    
    Δγ ≈ q̈ᵀ * g_eff, 其中
      g_eff = 0.5 * ∇_qγ * dt**2 + ∇_{q̇}γ * dt.
    """
    # 拆分梯度
    grad_q  = grad_gamma[:7]
    grad_qd = grad_gamma[7:]

    # 1) 计算有效梯度方向
    g_eff = 0.5 * grad_q * dt**2 + grad_qd * dt   # shape (7,)
    c = grad_q.dot(qd) * dt

    # 2) 在盒约束上，Δγ 的最小/最大值出现在每个维度要么取下界，要么取上界
    #    如果 g_eff[i] > 0，则 Δγ 对 q̈[i] 单调递增 → 最小在下界，最大在上界
    #    如果 g_eff[i] < 0，则 Δγ 对 q̈[i] 单调递减 → 最小在上界，最大在下界
    corner_min = np.where(g_eff>0, qdd_lb, qdd_ub)
    corner_max = np.where(g_eff>0, qdd_ub, qdd_lb)

    delta_gamma_min = np.dot(g_eff, corner_min)+ c
    delta_gamma_max = np.dot(g_eff, corner_max)+ c

    # 3) 可行性判断
    exists_positive = (delta_gamma_max > 0) and (delta_gamma_min < delta_gamma_max)
    if not exists_positive:
        print("Warning: no feasible qdd in original box can increase Gamma!")
        return corner_max

    # new_lb = qdd_lb.copy()
    # new_ub = qdd_ub.copy()
    # # 计算用于“最坏情况”迭代的 delta_min
    # # （此时 corner_min 对所有 j 都是 worst-case for Δγ）
    # for i in range(7):
    #     # 其他分量固定在 corner_min 上
    #     # worst_other = c + Σ_{j≠i} g_eff[j]*corner_min[j]
    #     worst_other = delta_gamma_min - g_eff[i] * corner_min[i]

    #     if g_eff[i] > 0:
    #         # 需要 g_eff[i]*q̈[i] > - worst_other
    #         bound_i = - worst_other / g_eff[i]
    #         new_lb[i] = max(new_lb[i], bound_i)
    #     else:
    #         # g_eff[i] < 0 时只收缩上界
    #         bound_i = - worst_other / g_eff[i]
    #         new_ub[i] = min(new_ub[i], bound_i)
    return corner_max

joint_acceleration_limits = [(-15, 15), (-7.5, 7.5), (-10, 10), (-12.5, 12.5), (-15, 15), (-20, 20), (-20, 20)]
def compute_qe(q, qd):
    """
    计算每个关节在匀减速（|a|=amax）停下来时的位置 qe。

    Args:
        q: list 或 1D numpy array，当前关节位置，长度 = n_joints
        qd: list 或 1D numpy array，当前关节速度，长度 = n_joints
        joint_acceleration_limits: list of (amin, amax)，每个关节允许的加速度范围

    Returns:
        qe: list，停下后每个关节的位置
    """
    qe = []
    for j, vel in enumerate(qd):
        # 取正向加速度上限来做最保守估计
        a_max = joint_acceleration_limits[j][1]
        if vel == 0:
            qe_j = q[j]
        else:
            # 停止所需时间 t = |v| / a_max
            t_stop = abs(vel) / a_max
            # 匀减速位移 Δ = 0.5 * v * t
            delta = 0.5 * vel * t_stop
            qe_j = q[j] + delta
        qe.append(qe_j)
    return qe

joint_acceleration_limits_base = [(-5, 5), (-5, 5), (-15, 15), (-7.5, 7.5), (-10, 10), (-12.5, 12.5), (-15, 15), (-20, 20), (-20, 20)]
def compute_qe_base(q, qd):
    """
    计算每个关节在匀减速（|a|=amax）停下来时的位置 qe。

    Args:
        q: list 或 1D numpy array，当前关节位置，长度 = n_joints
        qd: list 或 1D numpy array，当前关节速度，长度 = n_joints
        joint_acceleration_limits: list of (amin, amax)，每个关节允许的加速度范围

    Returns:
        qe: list，停下后每个关节的位置
    """
    qe = []
    for j, vel in enumerate(qd):
        # 取正向加速度上限来做最保守估计
        a_max = joint_acceleration_limits_base[j][1]
        if vel == 0:
            qe_j = q[j]
        else:
            # 停止所需时间 t = |v| / a_max
            t_stop = abs(vel) / a_max
            # 匀减速位移 Δ = 0.5 * v * t
            delta = 0.5 * vel * t_stop
            qe_j = q[j] + delta
        qe.append(qe_j)
    return qe
