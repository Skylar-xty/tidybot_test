import torch
from panda_layer.panda_layer import PandaLayer
from bf_sdf import BPSDF  # 假设 BPSDF 在 your_module 中

# 1. 加载训练好的模型
model_path = 'models/BP_24.pt'        # 与训练时 n_func 保持一致
model = torch.load(model_path)

# 2. 实例化 PandaLayer 和 BPSDF
device = 'cuda'                      # 或 'cpu'
panda = PandaLayer(device)
bpsdf = BPSDF(
    n_func=24,
    domain_min=-1.0,
    domain_max=1.0,
    robot=panda,
    device=device
)

# 3. 准备输入：单个查询点 + 机械臂当前位姿和关节角
#    如果只有一个点和一组关节，batch size 都设为 1
x = torch.tensor([[0.2, 0.1, 0.5]], device=device)       # 形状 (1, 3)
theta = torch.tensor([[0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.5]], 
                     device=device)                     # 形状 (1, 7)
pose = torch.eye(4, device=device).unsqueeze(0)         # 形状 (1, 4, 4)

# 4. 调用 SDF 计算，得到到最近连杆的距离和对应连杆索引
#    return_index=True 会额外返回索引张量 idx，表示哪个连杆的距离最小
sdf_vals, _, idx = bpsdf.get_whole_body_sdf_batch(
    x, pose, theta, model,
    use_derivative=False,
    return_index=True
)

# 5. 结果处理
distance = sdf_vals.item()   # 标量距离
link_id  = idx.item()        # 最小距离对应的连杆编号

print(f"点 {x.cpu().numpy()} 到连杆 {link_id} 的距离为 {distance:.4f} m")
