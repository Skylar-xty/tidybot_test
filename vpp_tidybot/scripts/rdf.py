import torch
from RDF_TB.bf_sdf_copy import BPSDF
from RDF_TB.panda_layer.panda_layer_textured import PandaLayer
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'RDF_TB/models/BP_8.pt'
sdf_model = torch.load(model_path, map_location=device, weights_only=False)

panda_layer = PandaLayer(device)
bpsdf = BPSDF(
    n_func=8,
    domain_min=-1.0,
    domain_max=1.0,
    robot=panda_layer,
    device=device
)

import torch
from RDF_TB.bf_sdf_copy import BPSDF
from RDF_TB.panda_layer.panda_layer_textured import PandaLayer
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'RDF_TB/models/BP_8.pt'
sdf_model = torch.load(model_path, map_location=device, weights_only=False)

panda_layer = PandaLayer(device)
bpsdf = BPSDF(
    n_func=8,
    domain_min=-1.0,
    domain_max=1.0,
    robot=panda_layer,
    device=device
)

def query_sdf_batch_nine(x_np: np.ndarray,
                    pose_np: np.ndarray,
                    theta_np: np.ndarray,
                    need_index: bool = False):
    """
    Compute SDF and gradients w.r.t 9 DOF (Base X, Base Y, + 7 Arm Joints).
    
    Args:
        x_np: (N, 3) Query points (Obstacles)
        pose_np: (B, 4, 4) Base Poses
        theta_np: (B, 7) Arm Joint Configurations
        
    Returns:
        dsts: (B, N) - Distances
        grad_qs: (B, N, 9) - Gradients [dx, dy, dq1, dq2, ..., dq7]
    """
    # -------- 1) Normalize input shapes --------    
    x_np = np.asarray(x_np, dtype=np.float32).reshape(-1, 3) # (N, 3)
    
    theta_np = np.asarray(theta_np, dtype=np.float32)
    if theta_np.ndim == 1:
        theta_np = theta_np.reshape(1, 7)
    
    B = theta_np.shape[0]
    N = x_np.shape[0]

    pose_np = np.asarray(pose_np, dtype=np.float32)
    if pose_np.shape == (4, 4):
        pose_np = np.broadcast_to(pose_np, (B, 4, 4))

    # -------- 2) Convert tensor --------
    x_t     = torch.from_numpy(x_np).to(device)       # (N,3)
    pose_t  = torch.from_numpy(pose_np).to(device)    # (B,4,4)
    theta_t = torch.from_numpy(theta_np).to(device)   # (B,7)

    # -------- 3) Finite Difference Setup (9 DOF) --------
    # We need to evaluate: 
    # 1 (Original) + 1 (Base X) + 1 (Base Y) + 7 (Arm Perturbs) = 10 total evaluations
    delta = 0.001
    total_perturbs = 10

    # --- A. Expand Theta (Arm Joints) ---
    # Shape: (B, 10, 7)
    # Index 0: Original
    # Index 1: Base X Perturb (Arm static)
    # Index 2: Base Y Perturb (Arm static)
    # Index 3-9: Arm Perturbs
    
    # Prepare Theta Batch
    theta_expanded = theta_t.unsqueeze(1).repeat(1, total_perturbs, 1) # (B, 10, 7)
    
    # Create identity matrix for arm perturbations (7x7)
    eye_arm = torch.eye(7, device=device) * delta
    
    # Apply arm perturbations to indices 3 to 9
    theta_expanded[:, 3:, :] = theta_expanded[:, 3:, :] + eye_arm

    # --- B. Expand Pose (Base Joints) ---
    # Shape: (B, 10, 4, 4)
    pose_expanded = pose_t.unsqueeze(1).repeat(1, total_perturbs, 1, 1) # (B, 10, 4, 4)
    
    # Perturb X (Index 1)
    pose_expanded[:, 1, 0, 3] += delta
    
    # Perturb Y (Index 2)
    pose_expanded[:, 2, 1, 3] += delta
    
    # Note: We DO NOT perturb Rotation (Theta) as requested.

    # -------- 4) Forward Pass (Batch Eval) --------
    # Flatten batch dimensions: (B * 10, ...)
    theta_flat = theta_expanded.reshape(-1, 7)
    pose_flat = pose_expanded.reshape(-1, 4, 4)
    
    with torch.no_grad():
        # Use non-derivative version; we are calculating gradients manually via FD
        sdf_flat, _ = bpsdf.get_whole_body_sdf_batch(
            x_t, pose_flat, theta_flat, sdf_model,
            use_derivative=False 
        )
        
    # -------- 5) Calculate Gradients --------
    # Reshape back to (B, 10, N)
    sdf_vals = sdf_flat.view(B, total_perturbs, N)
    
    # SDF Value (Baseline at index 0)
    dsts = sdf_vals[:, 0, :] # (B, N)
    
    # Differences: (Perturbed - Original) / delta
    # Indices 1: end contain the perturbed values
    diffs = (sdf_vals[:, 1:, :] - dsts.unsqueeze(1)) / delta # (B, 9, N)
    
    # Transpose to (B, N, 9)
    # Output order: [dx, dy, dq1, dq2, dq3, dq4, dq5, dq6, dq7]
    grad_qs = diffs.transpose(1, 2) 

    # Convert to numpy
    dsts_np = dsts.cpu().numpy().astype(np.float32)
    grad_qs_np = grad_qs.cpu().numpy().astype(np.float32)

    # -------- 6) Optional Index Return --------
    if not need_index:
        return dsts_np, grad_qs_np

    with torch.no_grad():
        _, _, idx = bpsdf.get_whole_body_sdf_batch(
            x_t, pose_t, theta_t, sdf_model,
            use_derivative=False, return_index=True
        )
        link_ids = idx.cpu().numpy().astype(np.int32)

    return dsts_np, grad_qs_np, link_ids

def query_sdf(x_np: np.ndarray,
                         pose_np: np.ndarray,
                         theta_np: np.ndarray):
    """
    查询 SDF 距离、最近连杆索引，并计算 dst 对 q 的梯度。

    Args:
        x_np:     (3,) numpy array
        pose_np:  (4,4) numpy array
        theta_np: (7,) numpy array

    Returns:
        dst       (float): SDF 距离
        link_id   (int):   最近连杆索引
        grad_q    (np.ndarray shape (7,)): ∂dst/∂q
    """
    # 1) 转 numpy→tensor，theta 要 requires_grad
    x_t     = torch.from_numpy(x_np.reshape(1,3).astype(np.float32)).to(device)
    pose_t  = torch.from_numpy(pose_np.reshape(1,4,4).astype(np.float32)).to(device)
    theta_t = torch.from_numpy(theta_np.reshape(1,7).astype(np.float32)) \
                   .to(device).requires_grad_(True)

    # # 2) 前向计算
    # sdf_vals, _, idx = bpsdf.get_whole_body_sdf_batch(
    #     x_t, pose_t, theta_t, sdf_model,
    #     use_derivative=False,   # 这里关闭 BPSDF 自带的导数输出
    #     return_index=True
    # )
    # # squeeze 成标量
    # sdf_val = sdf_vals.squeeze()   # shape=()

    # # 3) 反向传播
    # # 清空可能存在的旧梯度
    # if theta_t.grad is not None:
    #     theta_t.grad.zero_()
    # sdf_val.backward()             # 计算 ∂sdf_val/∂theta_t

    # # 4) 拷贝到 numpy
    # dst     = float(sdf_val.item())
    # link_id = int(idx.item())
    # grad_q  = theta_t.grad.detach().cpu().numpy().reshape(-1)  # (7,)

    # return dst, grad_q#, link_id
    with torch.no_grad():
        sdf, d_sdf = bpsdf.get_whole_body_sdf_with_joints_grad_batch(
            x_t, pose_t, theta_t, sdf_model
        )
        grad_q = d_sdf.squeeze(0).squeeze(0).cpu().numpy()  # (7,)
        sdf_val = sdf.squeeze()
        dst     = float(sdf_val.item())

    return dst, grad_q

def query_sdf_batch(x_np: np.ndarray,
                    pose_np: np.ndarray,
                    theta_np: np.ndarray,
                    need_index: bool = False):
# -------- 1) Normalize input shapes --------    
    x_np = np.asarray(x_np, dtype=np.float32)
    # Key: Regardless of whether it comes in as (3,), (1,3), (2,1,3), or (N,3), all are folded into (N,3)
    if x_np.size % 3 != 0:
        raise ValueError("The number of elements in x_np must be a multiple of 3")
    
    # I suppose this is to solve the case with two query points
    x_np = x_np.reshape(-1, 3)   # <<< This line solves the (2,1,3) problem

    theta_np = np.asarray(theta_np, dtype=np.float32)
    if theta_np.ndim == 1:
        theta_np = theta_np.reshape(1, 7)
    elif theta_np.shape[-1] != 7:
        raise ValueError("theta_np should have shape (..., 7)")

    B = theta_np.shape[0] # batch size - Number of query poses
    N = x_np.shape[0]     # number of query points

    pose_np = np.asarray(pose_np, dtype=np.float32)
    if pose_np.shape == (4, 4):
        pose_np = np.broadcast_to(pose_np, (B, 4, 4))
    elif pose_np.shape != (B, 4, 4):
        raise ValueError("pose_np should have shape (4,4) or (B,4,4)")

    # -------- 2) Convert tensor --------
    x_t     = torch.from_numpy(x_np).to(device)       # (N,3)
    pose_t  = torch.from_numpy(pose_np).to(device)    # (B,4,4)
    theta_t = torch.from_numpy(theta_np).to(device)   # (B,7)

    # -------- 3) Forward --------
    with torch.no_grad():
        sdf0, d_sdf = bpsdf.get_whole_body_sdf_with_joints_grad_batch(
            x_t, pose_t, theta_t, sdf_model
        )
        dsts    = sdf0.detach().cpu().numpy().astype(np.float32)   # (B,N)
        grad_qs = d_sdf.detach().cpu().numpy().astype(np.float32)  # (B,N,7)

    if not need_index:
        return dsts, grad_qs

    with torch.no_grad():
        sdf_vals, _, idx = bpsdf.get_whole_body_sdf_batch(
            x_t, pose_t, theta_t, sdf_model,
            use_derivative=False, return_index=True
        )
        link_ids = idx.detach().cpu().numpy().astype(np.int32)     # (B,N)

    return dsts, grad_qs, link_ids

