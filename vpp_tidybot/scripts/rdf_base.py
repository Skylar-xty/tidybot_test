# import torch
# from RDF_TB.bf_sdf_copy import BPSDF
# from RDF_TB.panda_layer.panda_layer_textured import PandaLayer
# import numpy as np

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model_path = 'RDF_TB/models/BP_8.pt'
# sdf_model = torch.load(model_path, map_location=device, weights_only=False)

# panda_layer = PandaLayer(device)
# bpsdf = BPSDF(
#     n_func=8,
#     domain_min=-1.0,
#     domain_max=1.0,
#     robot=panda_layer,
#     device=device
# )

import torch
from RDF_TB.bf_sdf_copy import BPSDF
from RDF_TB.panda_layer.panda_layer_textured import PandaLayerMovableBase
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'RDF_TB/models/BP_8.pt'
sdf_model = torch.load(model_path, map_location=device, weights_only=False)

panda_layer = PandaLayerMovableBase(device)
bpsdf = BPSDF(
    n_func=8,
    domain_min=-1.0,
    domain_max=1.0,
    robot=panda_layer,
    device=device
)



def query_sdf_batch_base(x_np: np.ndarray,
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
        theta_np = theta_np.reshape(1, 9)
    elif theta_np.shape[-1] != 9:
        raise ValueError("theta_np should have shape (..., 9)")

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
    theta_t = torch.from_numpy(theta_np).to(device)   # (B,9)

    # -------- 3) Forward --------
    with torch.no_grad():
        sdf0, d_sdf = bpsdf.get_whole_body_sdf_with_joints_grad_batch_base(
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

