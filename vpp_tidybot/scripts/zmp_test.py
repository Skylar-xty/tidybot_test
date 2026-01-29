# test_zmp_constraint.py
#
# Test script to implement and visualize Zero Moment Point (ZMP) stability constraints.
# This script runs a simulation where the robot moves, computes the ZMP based on multi-body dynamics,
# and visualizes it relative to the support polygon.
# It also generates the Linear Constraint Matrices (A_zmp, b_zmp) for a QP controller (A*qdd <= b).

import argparse
import math
import time
import sys
import os

import numpy as np
import torch
import cvxpy as cp

from isaaclab.app import AppLauncher

# Argument Parser
def parse_args():
    p = argparse.ArgumentParser("Tidybot ZMP Constraint Test")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--use_gravity", action="store_true", default=True)
    p.add_argument("--base_L", type=float, default=0.50, help="Base length (x)")
    p.add_argument("--base_W", type=float, default=0.50, help="Base width (y)")
    
    # Excite Options
    p.add_argument("--excite_mode", type=str, choices=["none", "arm_swing_traj"], default="arm_swing_traj")
    p.add_argument("--arm_swing_amp", type=float, default=0.9, help="Amplitude fraction")
    p.add_argument("--arm_swing_freq", type=float, default=1.5, help="Hz")
    p.add_argument("--arm_swing_phase_rand", action="store_true")
    
    # Rollout params
    p.add_argument("--settle_steps", type=int, default=60, help="Steps to settle at neutral pose")
    p.add_argument("--rollout_steps", type=int, default=300, help="Steps to execute swing trajectory")
    p.add_argument("--qd_abs", type=float, default=1.5, help="Initial randomization velocity scale")

    # Topple Detection
    p.add_argument("--margin_thresh", type=float, default=0.0)
    p.add_argument("--roll_thresh", type=float, default=0.35) 
    p.add_argument("--pitch_thresh", type=float, default=0.35)
    p.add_argument("--z_min", type=float, default=0.00)
    
    # QP Controller
    p.add_argument("--use_qp", action="store_true", default=True, help="Enable ZMP QP (monitoring or solving)")
    p.add_argument("--use_qp_solver", action="store_true", default=True,
                   help="If set, actually solve the QP and apply the filtered command (requires --use_qp).")
    
    p.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to run (0 for infinite)")

    return p.parse_args()

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Plot Helper
def plot_results(history_ref, history_safe, history_topple, episode_idx, base_filename="episode_qdd"):
    """
    Plot joint accelerations (ref vs safe) and topple status.
    history_ref: list of numpy arrays (N_steps, D)
    history_safe: list of numpy arrays (N_steps, D)
    """
    if plt is None:
        print("[WARN] matplotlib not installed, skipping plot.")
        return

    hist_ref = np.array(history_ref) # [T, D]
    hist_safe = np.array(history_safe) # [T, D]
    
    steps = np.arange(len(hist_ref))
    D = hist_ref.shape[1]
    
    fig, axes = plt.subplots(D, 1, figsize=(10, 2*D), sharex=True)
    if D == 1: axes = [axes]
    
    toppled = history_topple[-1] if len(history_topple) > 0 else False
    
    # Determine Status
    status_color = 'red' if toppled else 'green'
    status_text = "TOPPLED" if toppled else "STABLE"
    
    max_acc = 0.0

    for j in range(D):
        ax = axes[j]
        # Reference
        ax.plot(steps, hist_ref[:, j], 'r--', label='qdd_ref', alpha=0.6)
        # Safe
        
        # Check if safe contains valid data (not just zeros if solver was off)
        # Just plot it anyway
        ax.plot(steps, hist_safe[:, j], 'g-', label='qdd_safe', alpha=0.8)
        
        max_val = np.max(np.abs(hist_ref[:, j]))
        if max_val > max_acc: max_acc = max_val
        
        ax.set_ylabel(f'J{j}')
        ax.grid(True)
        if j == 0:
            ax.legend(loc='upper right')
            ax.set_title(f"Episode {episode_idx} | Result: {status_text}", color=status_color, fontweight='bold')
    
    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    
    fname = f"{base_filename}_{episode_idx}.png"
    plt.savefig(fname)
    print(f"[INFO] Saved plot to {fname}")
    plt.close(fig)


args = parse_args()

# Launch App
app_launcher = AppLauncher(headless=args.headless, cli_args=["--/telemetry/enable=false"])
simulation_app = app_launcher.app

# Imports after app launch
from isaaclab.scene import InteractiveScene
from isaaclab_tasks.direct.tidybot_v1.tidybot_env_v1_cfg import TidybotEnvCfg
from isaaclab_tasks.direct.tidybot_v1.tidybot_v1_env import TidybotEnv
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.utils.math import matrix_from_quat

# --------------------------
# Excite / Swing Helpers
# --------------------------
def smoothstep_01(x: torch.Tensor) -> torch.Tensor:
    """C1 smooth ramp on [0,1]."""
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def swing_envelope(t_sec: torch.Tensor, T_total: float, T_ramp: float) -> torch.Tensor:
    """
    Smooth on/off envelope: 0 -> 1 -> 0 with smoothstep ramps.
    t_sec: [N] seconds
    """
    if T_ramp <= 0.0:
        return torch.ones_like(t_sec)

    up = smoothstep_01(t_sec / T_ramp)
    down = smoothstep_01((T_total - t_sec) / T_ramp)
    return torch.minimum(up, down)

def build_swing_q_target(
    q_ref: torch.Tensor,                 # [N,7] reference posture
    t_step: int,
    dt: float,
    amp: float,
    freq_hz: float,
    phase: torch.Tensor,                 # [N]
    qmin: torch.Tensor, qmax: torch.Tensor,  # [N,7]
    joints_main=(0,2),
    couple=True,
    ramp_sec: float = 0.25,
    total_steps: int = 10000, # Large number if continuous 
):
    """
    Return q_des [N,7] for a swing motion around q_ref.
    amp is in "fraction of joint range" (safer than raw radians).
    """
    N = q_ref.shape[0]
    device = q_ref.device
    t_sec = torch.full((N,), float(t_step) * float(dt), device=device)

    # Simplified envelope for continuous running test (just ramp up)
    # T_total = float(total_steps) * float(dt)
    # env = swing_envelope(t_sec, T_total=T_total, T_ramp=float(ramp_sec))  # [N]
    
    # Just ramp up
    env = smoothstep_01(t_sec / ramp_sec)

    w = 2.0 * np.pi * float(freq_hz)
    s = torch.sin(torch.tensor(w * float(t_step) * float(dt), device=device) + phase)  # scalar->broadcast ok

    # amplitude in radians as fraction of range
    half_range = 0.5 * (qmax - qmin)
    A = amp * half_range  # [N,7]

    dq = torch.zeros_like(q_ref)
    
    # Iterate over all joints in joints_main with alternating signs to create swing/whip effect
    sign = 1.0
    for j_idx in joints_main:
        dq[:, j_idx] = sign * env * A[:, j_idx] * s
        sign *= -1.0

    if couple:
        # small phase-lead on another joint to mimic whip
        s2 = torch.sin(torch.tensor(w * float(t_step) * float(dt), device=device) + phase + 1.2)
        dq[:, 3] = 0.35 * env * A[:, 3] * s2

    q_des = q_ref + dq
    q_des = torch.minimum(torch.maximum(q_des, qmin), qmax)
    return q_des

# --------------------------
# Topple / Math helpers
# --------------------------
def quat_to_rpy_wxyz(q: torch.Tensor):
    """q: [N,4] in (w,x,y,z) -> roll,pitch,yaw [N] each."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def get_articulation_com_w(robot) -> torch.Tensor:
    """Return COM position in world: [N,3]."""
    com = getattr(robot.data, "com_pos_w", None)
    if isinstance(com, torch.Tensor) and com.numel() > 0:
        return com

    # Fallback to manual calc if not in data (should be enabled by default for articulation if configured)
    body_pos = getattr(robot.data, "body_pos_w", None)   # [N,B,3]
    body_mass = getattr(robot.data, "body_mass", None)   # [B] or [N,B]
    if isinstance(body_pos, torch.Tensor) and isinstance(body_mass, torch.Tensor) and body_pos.numel() > 0:
        if body_mass.ndim == 1:
            m = body_mass.unsqueeze(0).unsqueeze(-1)     # [1,B,1]
        else:
            m = body_mass.unsqueeze(-1)                  # [N,B,1]
        com = (body_pos * m).sum(dim=1) / (m.sum(dim=1) + 1e-9)
        return com

    return robot.data.root_pos_w


def stability_margin_rect(robot, base_L: float, base_W: float) -> torch.Tensor:
    """Rectangular support margin using COM projection in base frame."""
    com_w = get_articulation_com_w(robot)               # [N,3]
    root_xy_w = robot.data.root_pos_w[:, :2]            # [N,2]
    _, _, yaw = quat_to_rpy_wxyz(robot.data.root_quat_w)

    cy, sy = torch.cos(yaw), torch.sin(yaw)
    Rinv = torch.stack(
        [torch.stack([ cy, sy], dim=-1),
         torch.stack([-sy, cy], dim=-1)],
        dim=-2
    )  # [N,2,2]

    dxy = (com_w[:, :2] - root_xy_w).unsqueeze(-1)      # [N,2,1]
    com_b = torch.einsum("nij,njk->nik", Rinv, dxy).squeeze(-1)  # [N,2]

    hx = base_L / 2.0
    hy = base_W / 2.0
    margin_x = hx - com_b[:, 0].abs()
    margin_y = hy - com_b[:, 1].abs()
    return torch.minimum(margin_x, margin_y)


def detect_topple(robot, base_L, base_W,
                  margin_thresh=0.0, roll_thresh=0.8, pitch_thresh=0.8, z_min=0.05):
    """Return risk_topple, physical_topple, margin, roll, pitch, yaw."""
    # margin = stability_margin_rect(robot, base_L, base_W)
    
    # [Fix] The variable 'margin' was actually computed but implementation of detect_topple in 
    #       reference script calls stability_margin_rect right away.
    margin = stability_margin_rect(robot, base_L, base_W)
    
    roll, pitch, yaw = quat_to_rpy_wxyz(robot.data.root_quat_w)
    
    risk_topple = margin < margin_thresh
    
    physical_topple = (roll.abs() > roll_thresh) | (pitch.abs() > pitch_thresh) 
    
    return risk_topple, physical_topple, margin, roll, pitch, yaw

# --------------------------
# Sampling utilities (from sample_toppling.py)
# --------------------------
def sample_q_qd(robot, arm_ids, qd_abs=1.5):
    limits = robot.data.soft_joint_pos_limits[:, arm_ids, :]  # [E,J,2]
    qmin = limits[..., 0]
    qmax = limits[..., 1]
    u = torch.rand_like(qmin)
    q0 = qmin + (qmax - qmin) * u
    qd0 = (torch.rand_like(q0) * 2.0 - 1.0) * qd_abs
    return q0, qd0

def clamp_q(robot, arm_ids, q):
    limits = robot.data.soft_joint_pos_limits[:, arm_ids, :]
    qmin = limits[..., 0]
    qmax = limits[..., 1]
    return torch.minimum(torch.maximum(q, qmin), qmax)

class ZMPConstraintBuilder:
    def __init__(self, robot, num_envs, base_L=0.5, base_W=0.5, device="cuda:0"):
        """
        Helper to build ZMP constraints: x_min <= x_zmp <= x_max, y_min <= y_zmp <= y_max
        formulated as linear constraints on joint accelerations (qdd):
        A_zmp * qdd <= b_zmp
        """
        self.robot = robot
        self.num_envs = num_envs
        self.device = device
        
        # Support Polygon (Rectangle centered at Root)
        # In Local Base Frame.
        self.x_min = -base_L / 2.0
        self.x_max = base_L / 2.0
        self.y_min = -base_W / 2.0
        self.y_max = base_W / 2.0
        
        # Cache previous Jacobian for finite-diff J_dot
        self.J_prev = None
        self.last_update_time = -1.0
        
    def update(self, dt):
        """Update internal state like J_dot approximation."""
        # Get Current Jacobians for all links
        # shape: [num_envs, num_links, 6, num_dof]
        J_current = self.robot.root_physx_view.get_jacobians().clone()
        
        if self.J_prev is None:
            self.J_dot = torch.zeros_like(J_current)
        else:
            self.J_dot = (J_current - self.J_prev) / dt
            
        self.J_prev = J_current
        
    def compute_zmp_from_acc(self):
        """
        Compute Current ZMP based on realized body accelerations (Ground Truth Monitor).
        Uses formula:
          x_zmp = (sum m_i (z_i + g) x_i - sum m_i z_i x_dd_i) / sum m_i (z_i_dd + g)
        Assuming flat ground at z=0.
        """
        # Data
        # mass: [num_envs, num_links] (or broadcastable)
        masses_flat = self.robot.root_physx_view.get_masses().to(self.robot.device)
        masses = masses_flat.view(self.num_envs, -1) # [num_envs, num_links]
        
        # position: [num_envs, num_links, 3]
        pos_w = self.robot.data.body_pos_w 
        
        # acceleration: [num_envs, num_links, 6] (lin_acc, ang_acc)
        # Note: ISAAC LAB sometimes requires explicit accel computation or subscription.
        # robot.data.body_acc_w is available if enabled.
        acc_w = self.robot.data.body_acc_w 
        
        g = 9.81
        
        # Terms
        # m_i * (z_i_dd + g)
        # acc_w[..., 2] is z_dd
        force_z = masses * (acc_w[..., 2] + g) # [num_envs, num_links]
        
        # Angular Momentum Terms
        # tau_inertial = I * alpha + w x (I * w)
        # alpha is ang_acc, w is ang_vel
        
        # 1. Get Angular Velocity & Acceleration
        ang_vel = self.robot.data.body_ang_vel_w # [E, L, 3]
        # Assuming acc_w is 6D (lin, ang)
        if acc_w.shape[-1] == 6:
            ang_acc = acc_w[..., 3:6]
        else:
            # Fallback if only linear accel provided (unlikely in ArticulationData default but possible)
            ang_acc = torch.zeros_like(ang_vel)
            
        # 2. Get Inertia Tensor in World Frame
        # get_inertias() returns flattened 3x3 inertia tensor per link? [E*L, 9]
        inertias_flat = self.robot.root_physx_view.get_inertias().to(self.robot.device)
        
        # Check shape to be sure. If it's [E*L*9], we view as [E, L, 3, 3]
        # Or if it's [E*L, 3] (diagonal), we diag_embed.
        
        # Based on error (39 vs 13 implies factor of 3 mismatch in dimension 1 when viewed as -1,3), 
        # it seems input has 9 elements per link.
        I_local_mat = inertias_flat.view(self.num_envs, -1, 3, 3) # [E, L, 3, 3]
        
        # Rot matrices [E, L, 3, 3]
        quat = self.robot.data.body_quat_w
        R = matrix_from_quat(quat)
        
        # I_world = R * I_local * R^T
        # I_local_mat is already 3x3
        I_world = torch.matmul(R, torch.matmul(I_local_mat, R.transpose(-1, -2)))
        
        # 3. Compute Rate of Angular Momentum
        # term1 = I * alpha
        term1 = torch.matmul(I_world, ang_acc.unsqueeze(-1)).squeeze(-1) # [E, L, 3]
        # term2 = w x (I * w)
        Iw = torch.matmul(I_world, ang_vel.unsqueeze(-1)).squeeze(-1)
        term2 = torch.cross(ang_vel, Iw, dim=-1)
        
        tau_inertial = term1 + term2 # [E, L, 3]
        tau_x = tau_inertial[..., 0]
        tau_y = tau_inertial[..., 1]
        
        # Moment calculation components 
        # numerator X: sum ( x_i * F_z - z_i * F_x - tau_y )
        # numerator Y: sum ( y_i * F_z - z_i * F_y + tau_x )
        
        x_i = pos_w[..., 0]
        y_i = pos_w[..., 1]
        z_i = pos_w[..., 2]
        
        x_dd_i = acc_w[..., 0]
        y_dd_i = acc_w[..., 1]
        
        # Numerators [num_envs, num_links] -> sum over links
        num_x = (force_z * x_i) - (masses * x_dd_i * z_i) - tau_y
        num_y = (force_z * y_i) - (masses * y_dd_i * z_i) + tau_x
        
        denom = force_z # sum over links gives F_z_total
        
        sum_num_x = torch.sum(num_x, dim=1)
        sum_num_y = torch.sum(num_num_y := num_y, dim=1)
        sum_denom = torch.sum(denom, dim=1)
        
        # Avoid div by zero
        sum_denom = torch.where(sum_denom.abs() < 1e-5, torch.ones_like(sum_denom), sum_denom)
        
        p_zmp_x = sum_num_x / sum_denom
        p_zmp_y = sum_num_y / sum_denom
        
        return torch.stack([p_zmp_x, p_zmp_y], dim=1)


    def get_constraint_matrices(self, qd, g=9.81):
        """
        Build inequalites A * qdd <= b
        derived from:
           x_min <= ZMP_x_local(qdd) <= x_max
           y_min <= ZMP_y_local(qdd) <= y_max
        """
        # Data
        # Use cached default masses from data if possible, else physx view
        if hasattr(self.robot.data, "default_mass"):
            masses_data = self.robot.data.default_mass # [E, L]
            if masses_data.ndim == 1:
                masses_data = masses_data.unsqueeze(0).expand(self.num_envs, -1)
            masses = masses_data.unsqueeze(-1).to(self.device) # [E, L, 1]
        else:
            masses_flat = self.robot.root_physx_view.get_masses().to(self.robot.device)
            masses = masses_flat.view(self.num_envs, -1).unsqueeze(-1).to(self.device)

        pos = self.robot.data.body_pos_w.to(self.device)  # [E, L, 3]
        x_i = pos[..., 0:1]
        y_i = pos[..., 1:2]
        z_i = pos[..., 2:3]
        
        # Root State for Projection
        root_pos = self.robot.data.root_pos_w.to(self.device)  # [E, 3]
        r_x = root_pos[:, 0].view(-1, 1) # [E, 1]
        r_y = root_pos[:, 1].view(-1, 1) # [E, 1]
        
        root_quat = self.robot.data.root_quat_w.to(self.device)
        _, _, yaw = quat_to_rpy_wxyz(root_quat)
        c_psi = torch.cos(yaw).view(-1, 1) # [E, 1]
        s_psi = torch.sin(yaw).view(-1, 1)
        
        # Translation Jacobians: [E, L, 3, dof] 
        J_full = self.robot.root_physx_view.get_jacobians().to(self.device)  # [E, L, 6, D]
        J_trans = J_full[..., 0:3, :] # [E, L, 3, D]
        J_rot = J_full[..., 3:6, :]  # [E, L, 3, D] (angular part)
        J_x = J_trans[..., 0, :]
        J_y = J_trans[..., 1, :]
        J_z = J_trans[..., 2, :]
        
        # Bias accel eta (approx. J_dot * qd)
        # eta_full = J_dot(q, qd) @ qd, split into translational/rotational parts.
        if self.J_dot is None:
            eta_full = torch.zeros((self.num_envs, J_full.shape[1], 6), device=J_full.device, dtype=J_full.dtype)
        else:
            eta_full = torch.matmul(self.J_dot.to(self.device), qd.to(self.device).unsqueeze(1).unsqueeze(-1)).squeeze(-1)  # [E, L, 6]

        eta_trans = eta_full[..., 0:3]  # [E, L, 3]
        eta_rot   = eta_full[..., 3:6]  # [E, L, 3]

        eta_x = eta_trans[..., 0:1]
        eta_y = eta_trans[..., 1:2]
        eta_z = eta_trans[..., 2:3]

        # --- [Scheme A] Add swing angular-momentum-rate effects into the ZMP numerator ---
        # Monitor uses: N_x -= tau_y, N_y += tau_x, where
        #   tau = I_world * alpha + omega x (I_world * omega)
        # and alpha is approximated by: alpha = J_rot @ qdd + (J_dot_rot @ qd) = J_rot @ qdd + eta_rot.
        #
        # Here we construct tau = C_tau @ qdd + d_tau, so we can keep linear constraints in qdd.
        omega = self.robot.data.body_ang_vel_w.to(self.device)          # [E, L, 3]
        quat  = self.robot.data.body_quat_w.to(self.device)             # [E, L, 4]
        R     = matrix_from_quat(quat)                                  # [E, L, 3, 3]

        inertias_flat = self.robot.root_physx_view.get_inertias().to(self.device)
        # PhysX returns 9 inertia entries per link; reshape to [E, L, 3, 3]
        I_local = inertias_flat.view(self.num_envs, -1, 3, 3)           # [E, L, 3, 3]
        I_world = torch.matmul(R, torch.matmul(I_local, R.transpose(-1, -2)))  # [E, L, 3, 3]

        # tau = I_world * alpha + omega x (I_world * omega), with alpha = J_rot*qdd + eta_rot
        C_tau = torch.matmul(I_world, J_rot)                            # [E, L, 3, D]
        I_eta = torch.matmul(I_world, eta_rot.unsqueeze(-1)).squeeze(-1)  # [E, L, 3]
        I_omg = torch.matmul(I_world, omega.unsqueeze(-1)).squeeze(-1)    # [E, L, 3]
        d_tau = I_eta + torch.cross(omega, I_omg, dim=-1)               # [E, L, 3]
        
        # --- Build World Frame Coefficients ---
        # Numerator X (N_x): C_nx @ qdd + d_nx
        term_nx = masses * (x_i * J_z - z_i * J_x) # [E, L, D] (Mass broadcast)
        C_nx = torch.sum(term_nx, dim=1) # [E, D]
        
        term_dnx = masses * (x_i * (eta_z + g) - z_i * eta_x) 
        d_nx = torch.sum(term_dnx, dim=1).view(self.num_envs)
        
        # Numerator Y (N_y): C_ny @ qdd + d_ny
        term_ny = masses * (y_i * J_z - z_i * J_y)
        C_ny = torch.sum(term_ny, dim=1)
        
        term_dny = masses * (y_i * (eta_z + g) - z_i * eta_y)
        d_ny = torch.sum(term_dny, dim=1).view(self.num_envs)
        
        # [ADD] Include angular-momentum-rate terms (match compute_zmp_from_acc sign convention):
        #   N_x <- N_x - sum_i tau_{y,i}
        #   N_y <- N_y + sum_i tau_{x,i}
        C_nx = C_nx - torch.sum(C_tau[..., 1, :], dim=1)                # [E, D]
        d_nx = d_nx - torch.sum(d_tau[..., 1], dim=1).view(self.num_envs)  # [E]
        C_ny = C_ny + torch.sum(C_tau[..., 0, :], dim=1)                # [E, D]
        d_ny = d_ny + torch.sum(d_tau[..., 0], dim=1).view(self.num_envs)  # [E]

        # Denominator (D): C_d @ qdd + d_d
        term_d = masses * J_z
        C_d = torch.sum(term_d, dim=1)
        
        term_dd = masses * (eta_z + g)
        d_d = torch.sum(term_dd, dim=1).view(self.num_envs)
        
        # --- Project to Base Frame ---
        # ZMP_local_x = c(ZMP_world_x - r_x) + s(ZMP_world_y - r_y)
        #             = (c N_x + s N_y - (c r_x + s r_y) D) / D
        # Let Px_N = c N_x + s N_y
        # Let Off_x = c r_x + s r_y
        # ZMP_local_x = (Px_N - Off_x * D) / D
        
        # Coeffs for Px_N = C_pxn @ qdd + d_pxn
        C_pxn = c_psi * C_nx + s_psi * C_ny
        d_pxn = c_psi.view(-1) * d_nx + s_psi.view(-1) * d_ny
        
        Off_x = c_psi.view(-1) * r_x.view(-1) + s_psi.view(-1) * r_y.view(-1)
        
        # ZMP_local_y = -s(ZMP_world_x - r_x) + c(ZMP_world_y - r_y)
        #             = (-s N_x + c N_y - (-s r_x + c r_y) D) / D
        C_pyn = -s_psi * C_nx + c_psi * C_ny
        d_pyn = -s_psi.view(-1) * d_nx + c_psi.view(-1) * d_ny
        
        Off_y = -s_psi.view(-1) * r_x.view(-1) + c_psi.view(-1) * r_y.view(-1)
        
        constraints_A = []
        constraints_b = []
        
        # Helper for "ZMP_component >= Limit" => (Limit + Offset) * D <= Projected_N
        # => [ (Limit + Offset) C_d - C_pn ] qdd <= d_pn - (Limit + Offset) d_d
        
        def add_ge_constraint(limit, C_pn, d_pn, Offset):
            # Val >= Limit
            L = limit + Offset
            # (L * C_d - C_pn) qdd <= d_pn - L * d_d
            row_A = L.view(-1, 1) * C_d - C_pn
            row_b = d_pn - L * d_d
            constraints_A.append(row_A)
            constraints_b.append(row_b)

        def add_le_constraint(limit, C_pn, d_pn, Offset):
            # Val <= Limit -> Projected_N <= (Limit + Offset) * D
            # => [ C_pn - (Limit + Offset) C_d ] qdd <= (Limit + Offset) d_d - d_pn
            L = limit + Offset
            row_A = C_pn - L.view(-1, 1) * C_d
            row_b = L * d_d - d_pn
            constraints_A.append(row_A)
            constraints_b.append(row_b)

        # 1. x_min <= ZMP_local_x
        add_ge_constraint(self.x_min, C_pxn, d_pxn, Off_x)
        # 2. ZMP_local_x <= x_max
        add_le_constraint(self.x_max, C_pxn, d_pxn, Off_x)
        
        # 3. y_min <= ZMP_local_y
        add_ge_constraint(self.y_min, C_pyn, d_pyn, Off_y)
        # 4. ZMP_local_y <= y_max
        add_le_constraint(self.y_max, C_pyn, d_pyn, Off_y)
        
        return torch.stack(constraints_A, dim=1), torch.stack(constraints_b, dim=1)

def solve_qp_per_env(A_in, b_in, qdd_ref_in, slack_weight: float = 1000.0):
    """
    Solve a small QP per environment (CPU loop; cvxpy).

        minimize    ||qdd - qdd_ref||^2
        subject to  A * qdd <= b

    Inputs can be either:
      - batched tensors: A_in [E, M, D], b_in [E, M], qdd_ref_in [E, D]
      - lists of per-env tensors with the same per-env shapes.

    Returns:
      qdd_safe [E, D] on the same device/dtype as qdd_ref_in.
    """
    if isinstance(A_in, torch.Tensor):
        A_batch = A_in
        b_batch = b_in
        ref_batch = qdd_ref_in
    else:
        # assume list/tuple
        A_batch = torch.stack(list(A_in), dim=0)
        b_batch = torch.stack(list(b_in), dim=0)
        ref_batch = torch.stack(list(qdd_ref_in), dim=0)

    device = ref_batch.device
    dtype = ref_batch.dtype
    E = A_batch.shape[0]
    D = ref_batch.shape[1]
    M = A_batch.shape[1]

    qdd_safe_list = []

    for i in range(E):
        A = A_batch[i].detach().cpu().numpy().astype(np.float64)         # [M, D]
        b = b_batch[i].detach().cpu().numpy().reshape(-1).astype(np.float64)  # [M]
        ref = ref_batch[i].detach().cpu().numpy().astype(np.float64)     # [D]

        qdd = cp.Variable(D)

        prob = cp.Problem(cp.Minimize(cp.sum_squares(qdd - ref)),
                          [A @ qdd <= b])

        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception:
            try:
                prob.solve(warm_start=True, verbose=False)
            except Exception:
                pass

        if prob.status not in ["optimal", "optimal_inaccurate"] or qdd.value is None:
            # Slack fallback (soften constraints)
            z = cp.Variable(M)
            prob_slack = cp.Problem(
                cp.Minimize(cp.sum_squares(qdd - ref) + float(slack_weight) * cp.sum_squares(z)),
                [A @ qdd <= b + z]
            )
            try:
                prob_slack.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            except Exception:
                try:
                    prob_slack.solve(warm_start=True, verbose=False)
                except Exception:
                    pass

        if qdd.value is None:
            qdd_safe_list.append(torch.as_tensor(ref, dtype=dtype))
        else:
            qdd_safe_list.append(torch.as_tensor(qdd.value, dtype=dtype))

    return torch.stack(qdd_safe_list, dim=0).to(device=device, dtype=dtype)

def main():
    # 1. Config
    cfg = TidybotEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.dt = args.dt
    # [Fix] Decimation 1 ensures freq of control == freq of physics (essential for swing dynamics)
    cfg.sim.render_interval = 1 
    
    cfg.scene.robot.spawn.articulation_props.fix_root_link = False # Let it tip!
    cfg.scene.robot.spawn.rigid_props.disable_gravity = not args.use_gravity
    success_zmp = 0
    failure_zmp = 0
    # [Mod] Increase gains for stronger swing if requested
    if args.excite_mode == "arm_swing_traj" and "iiwa" in cfg.scene.robot.actuators:
        print("[INFO] Increasing actuator limits and stiffness for swing trajectory.")
        # Note: cfg is a config object, we can modify it before env creation
        cfg.scene.robot.actuators["iiwa"].stiffness = 1500.0
        cfg.scene.robot.actuators["iiwa"].damping = 60.0
        cfg.scene.robot.actuators["iiwa"].effort_limit_sim = 2000.0
        # If effort_limit_sim exists in the actuator config class (it might be simply effort_limit or part of a struct)
        # Tidybot uses ImplicitActuatorCfg usually.
    
    # 2. Environment
    env = TidybotEnv(cfg)
    env.reset()
    robot = env.scene["robot"]
    sim_dt = cfg.sim.dt
    
    # 3. ZMP Builder
    zmp_builder = ZMPConstraintBuilder(robot, args.num_envs, args.base_L, args.base_W)
    
    # 3b. Setup Swing
    arm_ids = env._arm_dof_idx
    
    # [Mod] Match initialization logic from sample_toppling.py
    if args.excite_mode == "arm_swing_traj":
        print("[INFO] Setting up arm swing trajectory execution...")
        
        # Prepare params
        if args.arm_swing_phase_rand:
            phase = (torch.rand(args.num_envs, device=env.device) * 2.0 * math.pi)
        else:
            phase = torch.zeros(args.num_envs, device=env.device)
            
        limits = robot.data.soft_joint_pos_limits[:, arm_ids, :]  # [E,J,2]
        qmin, qmax = limits[..., 0], limits[..., 1]
        
        # [Mod] Randomize initial state using standard sampling logic
        # Sample q0, qd0
        print("[INFO] Sampling random initial arm configuration...")
        q0, qd0 = sample_q_qd(robot, arm_ids, qd_abs=args.qd_abs)
        q0 = clamp_q(robot, arm_ids, q0)
        
        # Write to sim state directly
        q_all = robot.data.joint_pos.clone()
        dq_all = robot.data.joint_vel.clone()
        q_all[:, arm_ids] = q0
        
        # Ideally we want qd=0 for settle
        dq_all[:] = 0.0
        
        robot.write_joint_state_to_sim(q_all, dq_all)
        robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
        robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
        
        # Update references
        q_ref = q0.clone()
    else:
        # Default behavior for other modes
        q_ref = robot.data.joint_pos.clone()
        q_ref[:, arm_ids] = 0.0

    
    # 4. Visualization
    marker_cfg = SPHERE_MARKER_CFG.copy()
    marker_cfg.prim_path = "/Visuals/ZMP"
    marker_cfg.markers["sphere"].radius = 0.05
    marker_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.0, 1.0) # Magenta ZMP
    zmp_marker = VisualizationMarkers(marker_cfg)
    
    print("[INFO] Starting Simulation...")
    global_step = 0
    episode_step = 0
    episode_count = 0
    
    # Previous state for integration integration
    # Ideally should use robot state, but for command integration we need consistent history
    q_cmd_prev = robot.data.joint_pos.clone()
    qd_cmd_prev = robot.data.joint_vel.clone()
    episode_violation_occurred = False
    detected_topple = False

    # Plotting History
    ep_qdd_ref = []
    ep_qdd_safe = []
    ep_topple_status = []

    while simulation_app.is_running():
        # Reset if episode done
        total_ep_steps = args.settle_steps + args.rollout_steps
        if episode_step >= total_ep_steps:
            # End of Episode Plotting
            plot_results(ep_qdd_ref, ep_qdd_safe, ep_topple_status, episode_count)
            episode_count += 1
            
            if args.num_episodes > 0 and episode_count >= args.num_episodes:
                print(f"[INFO] Reached total episodes {args.num_episodes}. Exiting.")
                break

            # Reset Logic
            env.reset()
            episode_step = 0
            episode_violation_occurred = False
            detected_topple = False
            ep_qdd_ref = []
            ep_qdd_safe = []
            ep_topple_status = []

            # Reset prev cmd
            q_cmd_prev = robot.data.joint_pos.clone()
            qd_cmd_prev = robot.data.joint_vel.clone()

            if args.arm_swing_phase_rand:
                phase = (torch.rand(args.num_envs, device=env.device) * 2.0 * math.pi)
            print(f"[INFO] Reset environment at global step {global_step}. Starting Episode {episode_count}")
            
            # Reset history in zmp builder?
            zmp_builder.J_prev = None

        # --- ZMP & Control Loop ---
        
        # 0. Update Physics State (Jacobians etc) from PREVIOUS step result
        #    Note: On step 0, this uses initial state.
        zmp_builder.update(sim_dt)
        
        # 1. Measured State
        q_meas = robot.data.joint_pos #[E, D]
        qd_meas = robot.data.joint_vel
        # Full velocity for ZMP matrices (Base + Joints)
        root_vel = robot.data.root_vel_w # [E, 6]
        qd_full = torch.cat([root_vel, qd_meas], dim=1) # [E, 13]

        # 2. Control Logic
        q_cmd = q_meas.clone()
        
        # Init Default Recording Values
        current_qdd_ref = torch.zeros_like(q_meas) # Default 0
        current_qdd_safe = torch.zeros_like(q_meas) # Default 0

        if args.excite_mode == "arm_swing_traj":
            violation_flag = 0

            # 1. Settle Phase: Hold Ref
            if episode_step < args.settle_steps:
                # Hold q_ref
                robot.set_joint_position_target(q_ref)
                q_cmd = q_ref.clone()
                # keep command history consistent for QP integration
                q_cmd_prev = q_cmd.clone()
                qd_cmd_prev = torch.zeros_like(qd_cmd_prev)
                
            # 2. Rollout Phase: Swing
            else:
                swing_t = episode_step - args.settle_steps
                q_des = build_swing_q_target(
                    q_ref=q_ref,
                    t_step=swing_t,
                    dt=sim_dt,
                    amp=args.arm_swing_amp,
                    freq_hz=args.arm_swing_freq,
                    phase=phase,
                    qmin=qmin, qmax=qmax,
                    joints_main=[0,2,4,5],
                    couple=True,
                    ramp_sec=0.25 # Sharp ramp
                )
                
                if args.use_qp:
                    # ZMP-QP safety filter around the nominal swing trajectory.
                    # We construct a nominal desired joint acceleration qdd_ref via PD tracking of q_des,
                    # then enforce ZMP constraints A*qdd <= b (joint columns only).
                    kp = 100.0
                    kd = 20.0

                    # finite-difference desired velocity for a consistent qdd_ref
                    if swing_t > 0:
                        q_des_prev = build_swing_q_target(
                            q_ref=q_ref,
                            t_step=swing_t - 1,
                            dt=sim_dt,
                            amp=args.arm_swing_amp,
                            freq_hz=args.arm_swing_freq,
                            phase=phase,
                            qmin=qmin, qmax=qmax,
                            joints_main=[0, 2, 4, 5],
                            couple=True,
                            ramp_sec=0.25
                        )
                        qd_des = (q_des - q_des_prev) / sim_dt
                    else:
                        qd_des = torch.zeros_like(q_des)

                    qdd_ref = kp * (q_des - q_meas) + kd * (qd_des - qd_meas)
                    current_qdd_ref = qdd_ref.detach().clone() # Store

                    # 2) Build constraints A_zmp * qdd_full <= b_zmp, then keep only joint columns
                    A_zmp, b_zmp = zmp_builder.get_constraint_matrices(qd_full)
                    # Separate Base (first 6 cols) and Joints (rest)
                    num_bases = 6
                    A_joints = A_zmp[:, :, num_bases:]   # [E, 4, 7]
                    b_safe = b_zmp                        # [E, 4]

                    # 3) Monitoring: check whether the nominal qdd_ref violates the constraints
                    # Constraint: A * qdd <= b  -->  A * qdd - b <= 0
                    # Violation if val > 0
                    val = torch.matmul(A_joints, qdd_ref.unsqueeze(-1)).squeeze(-1) - b_safe  # [E, 4]
                    max_violation, _ = torch.max(val, dim=1)                                  # [E]

                    for env_i in range(args.num_envs):
                        if max_violation[env_i] > 1e-3:
                            violation_flag = 1
                            episode_violation_occurred = True

                    # 4) Either solve the QP and apply a filtered command, or just execute nominal.
                    if args.use_qp_solver:
                        # Solve per-env QP: min ||qdd - qdd_ref||^2  s.t. A_joints*qdd <= b_safe
                        qdd_safe = solve_qp_per_env(A_joints, b_safe, qdd_ref)  # [E, 7]
                        current_qdd_safe = qdd_safe.detach().clone() # Store

                        print("[INFO] QP Solver applied to enforce ZMP constraints.")
                        print("qdd_safe:", qdd_safe.cpu().numpy())
                        print("qdd_ref:", qdd_ref.cpu().numpy())
                        # Convert qdd_safe -> position target by simple integration on the *command* state.
                        # This keeps compatibility with ImplicitActuator position targets.
                        qd_cmd = qd_cmd_prev + qdd_safe * sim_dt
                        q_cmd = q_cmd_prev + qd_cmd * sim_dt
                        q_cmd = clamp_q(robot, arm_ids, q_cmd)

                        # update command history
                        q_cmd_prev = q_cmd.clone()
                        qd_cmd_prev = qd_cmd.clone()
                    else:
                        # Monitoring-only mode: execute the nominal desired position command.
                        q_cmd = q_des
                        q_cmd_prev = q_cmd.clone()
                        qd_cmd_prev = qd_meas.clone()
                else:
                    # Direct PD (Unconstrained)
                    q_cmd = q_des
                
                robot.set_joint_position_target(q_cmd)
        
        # Apply & Step
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(sim_dt)
        
        # --- ZMP Visualization only ---
        # (Re-calculate ZMP from actual resulting acceleration to see if we stayed safe)
        # zmp_builder.update(sim_dt) -> Done at top of loop next time, but for viz we can do approx
        
        # 1. Measured ZMP (from accelerations)
        zmp_pos_measured = zmp_builder.compute_zmp_from_acc() # [E, 2]
        
        # Visualize
        # ZMP is calculated in World frame x/y if we used world pos/acc.
        # Wait, compute_zmp_from_acc used robot.data.body_pos_w. So result is World Frame
        # BUT the logic x_min/x_max is usually relative to Base Frame.
        # Let's verify result coordinate space.
        
        # transform world ZMP to base frame for checking constraints
        root_pos = robot.data.root_pos_w
        root_quat = robot.data.root_quat_w
        # (Inverse transform not implemented here for brevity, assume visual check in world)
        
        # Show marker at ZMP (z=0)
        zmp_world_3d = torch.zeros((args.num_envs, 3), device=robot.device)
        zmp_world_3d[:, 0] = zmp_pos_measured[:, 0]
        zmp_world_3d[:, 1] = zmp_pos_measured[:, 1]
        zmp_world_3d[:, 2] = 0.02 # Slightly above ground
        
        zmp_marker.visualize(zmp_world_3d)
        
        # 2. Compute QP Matrices (A, b)
        # qd_current = robot.data.joint_vel # [E, D]
        
        # Concatenate base velocity (twist) with joint velocity because robot is floating base
        # (fix_root_link=False). PhysX Jacobian includes 6 spatial DOFs for root.
        # root_vel = robot.data.root_vel_w # [E, 6]
        # qd_full = torch.cat([root_vel, qd_current], dim=1) # [E, 13]
        
        # A_zmp, b_zmp = zmp_builder.get_constraint_matrices(qd_full)
        
        # Topple Check
        risk, phys, margin, roll, pitch, yaw = detect_topple(
            robot,
            args.base_L, args.base_W,
            margin_thresh=args.margin_thresh,
            roll_thresh=args.roll_thresh,
            pitch_thresh=args.pitch_thresh,
            z_min=args.z_min
        )
        if phys.any():
            success_zmp += 1
            detected_topple = True
            if violation_flag:
                print("[success] successfully detected violation before topple.")
            else:
                print("[WARN] Physical Topple detected without prior violation warning!")
            # print(f"[WARN] Step {global_step} Physical Topple detected! Roll={roll[0]:.2f}, Pitch={pitch[0]:.2f}")
        elif risk.any():
            success_zmp += 1
            detected_topple = True
            if violation_flag:
                print("[success] successfully detected violation before risk topple.")
            else: 
                print("[WARN] Risk Topple detected without prior violation warning!")
            # print(f"[INFO] Step {global_step} Stability Risk: margin={margin[0]:.3f}")
        elif episode_violation_occurred and not detected_topple and episode_step >= total_ep_steps - 5:
            failure_zmp += 1
            print("[INFO] Constraint violation detected, but no topple occurred yet.")
            
        # Store Plot Data (use first env only for simplicity)
        ep_qdd_ref.append(current_qdd_ref[0].detach().cpu().numpy())
        ep_qdd_safe.append(current_qdd_safe[0].detach().cpu().numpy())
        ep_topple_status.append(detected_topple)
        
        # violation_flag = 0
        # Debug Print
        if global_step % 20 == 0:
            print(f"Step {global_step} (Ep Step {episode_step}):")
            print(f"  Measured ZMP (World): {zmp_pos_measured[0].cpu().numpy()}")
            # print(f"  Constraint Sample (Row 0): {A_zmp[0,0, :5].cpu().numpy()}... <= {b_zmp[0,0].item()}")
            
            # Check violation
            # Ideally: A * qdd_measured <= b
            # but measuring qdd from noisy sim is hard.
        
        global_step += 1
        episode_step += 1
    # print(f"ZMP Detection Summary: {success_zmp} successful detections, {failure_zmp} failures.")
if __name__ == "__main__":
    main()
