import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from camera_head import CameraHead  # CameraHeadのパスに応じて調整してね
import torch.optim as optim

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ハイパーパラメータ
B = 2
C = 1024
num_iterations = 4
lr = 1e-3
epochs = 1000

# モデル初期化
model = CameraHead(dim_in=C, input_gt_cam_params=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# ReduceLROnPlateau スケジューラ設定
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
	optimizer,
	mode='min',          # 損失が減るのを監視
	factor=0.5,          # lrを半分にする
	patience=50,         # 50エポック改善がなければlr減少
	verbose=True,
	min_lr=1e-6          # 最小学習率
)


def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1"):
	"""
	Computes translation, rotation, and focal loss for a batch of pose encodings.
	
	Args:
		pred_pose_enc: (N, D) predicted pose encoding
		gt_pose_enc: (N, D) ground truth pose encoding
		loss_type: "l1" (abs error) or "l2" (euclidean error)
	Returns:
		loss_T: translation loss (mean)
		loss_R: rotation loss (mean)
		loss_FL: focal length/intrinsics loss (mean)
	
	NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
		So here we use l1 loss.
	"""
	if loss_type == "l1":
		# Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
		loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
		loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
		loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
	elif loss_type == "l2":
		# L2 norm for each component
		loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
		loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
		loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
	else:
		raise ValueError(f"Unknown loss type: {loss_type}")

	# Check/fix numerical issues (nan/inf) for each loss component
#	loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
#	loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
#	loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

	# Clamp outlier translation loss to prevent instability, then average
	loss_T = loss_T.clamp(max=100).mean()
	loss_R = loss_R.mean()
	loss_FL = loss_FL.mean()

	return loss_T, loss_R, loss_FL


def criterion(pred, target,
	weight_trans=1.0,       # weight for translation loss
	weight_rot=1.0,         # weight for rotation loss
	weight_focal=0.5,       # weight for focal length loss
	):
	loss_T, loss_R, loss_FL = camera_loss_single(pred, target)
	return loss_T * weight_trans + loss_R * weight_rot + loss_FL * weight_focal


loss_history = []

for epoch in range(epochs):
	S = np.random.randint(1, 25)
	input_tensor = torch.randn(B, S, C, device=device)

	# 正解データ生成
	trans = torch.randn(B, S, 3, device=device)
	quat = torch.randn(B, S, 4, device=device)
	quat = quat / quat.norm(dim=-1, keepdim=True)
	fl = torch.abs(torch.randn(B, S, 2, device=device))
	target = torch.cat([trans, quat, fl], dim=-1)

	input_tensor[:, :, :9] = target

	optimizer.zero_grad()
	pred_list = model([input_tensor.view(B, S, 1, -1)], num_iterations=num_iterations)
	pred = pred_list[-1]

	loss = criterion(pred, target)
	loss.backward()
	optimizer.step()

	loss_history.append(loss.item())
	scheduler.step(loss)

	if (epoch + 1) % 10 == 0:
		current_lr = optimizer.param_groups[0]['lr']
		print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, LR: {current_lr:.6e}")

# loss曲線をプロットして保存
plt.figure()
plt.plot(loss_history, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png', dpi=200)
plt.close()

print("Loss curve saved as loss_curve.png")
