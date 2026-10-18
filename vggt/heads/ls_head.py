import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
try:
	from train_utils.general import check_and_fix_inf_nan
except (ImportError, ModuleNotFoundError):
	def check_and_fix_inf_nan(x, name="tensor"):
		if torch.is_tensor(x):
			return torch.nan_to_num(x)
		return x

from vggt.layers import Mlp
from vggt.layers.block import Block
from vggt.heads.head_act import inverse_log_transform
from vggt.utils.geometry import convert_coord_map_to_depth_map, unproject_depth_map_to_point_map_torch
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri


class AddCoords(nn.Module):
	def __init__(self, rank=2):
		super().__init__()
		self.rank = rank


	def forward(self, input_tensor):
		"""
		input_tensor: [batch_size, channels, height, width]
		"""
		batch_size, _, height, width = input_tensor.size()
		
		coords_h = torch.linspace(0, 1, height, device=input_tensor.device, dtype=input_tensor.dtype)
		coords_w = torch.linspace(0, 1, width, device=input_tensor.device, dtype=input_tensor.dtype)
		
		# [height, width] -> [1, 1, height, width]
		yy, xx = torch.meshgrid(coords_h, coords_w, indexing='ij')
		xx = xx.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
		yy = yy.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
		
		out = torch.cat([input_tensor, xx, yy], dim=1)
		return out



class CoordConv(nn.Module):
	def __init__(self, in_channels, out_channels, **kwargs):
		super().__init__()
		self.add_coords = AddCoords()
		self.conv = nn.Conv2d(in_channels + 2, out_channels, **kwargs)

	def forward(self, x):
		x = self.add_coords(x)
		x = self.conv(x)
		return x



def normalize_intrinsics(intrinsics, width, height):
	K_norm = intrinsics.clone()
	
	K_norm[..., 0, 0] /= width
	K_norm[..., 0, 2] /= width
	
	K_norm[..., 1, 1] /= height
	K_norm[..., 1, 2] /= height
	
	return K_norm



class LSHead(nn.Module):
	def __init__(
		self,
		dim_in: int = 2048,
		latent_dim: int = 32,
		mode: str = "analytical",
	):
		super().__init__()
		self.mode = mode
		self.add_coords = AddCoords()

		coord_extra = 2 
		
		dim_in_depth = (3*3 + 3*4 + 3*3 + 2) + coord_extra # 23 + 2 = 25 channels
		dim_out_depth = 1
		dim_in_point = (dim_in_depth + 3*4) # 35 + 2 = 37 channels
		dim_out_point = 3

		self.token_norm = nn.LayerNorm(dim_in)

		kernel_size = 3
		self.layer_depth = nn.Sequential(
			nn.Conv2d(dim_in_depth, latent_dim, kernel_size=kernel_size, padding='same'),
			nn.GroupNorm(1, latent_dim),
			nn.SiLU(),
			nn.Conv2d(latent_dim, latent_dim, kernel_size=kernel_size, padding='same'),
			nn.GroupNorm(1, latent_dim),
			nn.SiLU(),
			nn.Conv2d(latent_dim, dim_out_depth, kernel_size=kernel_size, padding='same'),
		)

		self.layer_point = nn.Sequential(
			nn.Conv2d(dim_in_point, latent_dim, kernel_size=kernel_size, padding='same'),
			nn.GroupNorm(1, latent_dim),
			nn.SiLU(),
			nn.Conv2d(latent_dim, latent_dim, kernel_size=kernel_size, padding='same'),
			nn.GroupNorm(1, latent_dim),
			nn.SiLU(),
			nn.Conv2d(latent_dim, dim_out_point, kernel_size=kernel_size, padding='same'),
		)


	def forward_depth(self, x):
		x = self.add_coords(x)
		depth_features = checkpoint(self.layer_depth, x, use_reentrant=False)
		depth_map = torch.exp(depth_features) 
		return depth_map


	def forward_point(self, x):
		x = self.add_coords(x)
		point_features = checkpoint(self.layer_point, x, use_reentrant=False)
		point_map = inverse_log_transform(point_features) 
		return point_map


	def forward(self, 
		aggregated_tokens_list: list, 
		cam_extrinsic,
		cam_intrinsic,
		proj_extrinsic,
		proj_intrinsic,
		coord_map,
	) -> torch.Tensor:
		tokens = aggregated_tokens_list[-1]

		if not isinstance(coord_map, torch.Tensor):
			coord_map = torch.tensor(np.array([graycode_to_coords(coord) for coord in coord_map]))

		B, N, H, W, _ = coord_map.shape
		BN = B * N
		
		cam_intrinsic = cam_intrinsic.reshape(BN, 3, 3)
		cam_extrinsic = cam_extrinsic.reshape(BN, 3, 4)
		proj_intrinsic = proj_intrinsic.reshape(BN, 3, 3)
		proj_extrinsic = proj_extrinsic.reshape(BN, 3, 4)

		if self.mode == "analytical":
			depth_map = convert_coord_map_to_depth_map(coord_map.reshape(BN, H, W, 2), cam_intrinsic, proj_intrinsic, proj_extrinsic)
			depth_map = check_and_fix_inf_nan(depth_map, "depth_map")
			world_points = unproject_depth_map_to_point_map_torch(depth_map, cam_extrinsic, cam_intrinsic)
			world_points = check_and_fix_inf_nan(world_points, "world_points")

		elif self.mode in ["proj_depth", "world_points", "scale"]:
			coord_map_input = coord_map.reshape(BN, H, W, 2).permute(0, 3, 1, 2)
			
			def broadcast_to_map(x, H, W):
				# (BN, D) -> (BN, D, 1, 1) -> (BN, D, H, W)
				return x.reshape(BN, -1, 1, 1).expand(-1, -1, H, W)

			features = [coord_map_input]
			to_cat = [
				proj_intrinsic.reshape(BN, 9), 
				proj_extrinsic.reshape(BN, 12),
				normalize_intrinsics(cam_intrinsic, W, H).reshape(BN, 9)
			]
			
			if self.mode == "world_points":
				to_cat.append(cam_extrinsic.reshape(BN, 12))

			for tensor in to_cat:
				features.append(broadcast_to_map(tensor, H, W))
			
			input_tensor = torch.cat(features, dim=1)

			if self.mode == "proj_depth":
				depth_map = self.forward_depth(input_tensor) # (BN, 1, H, W)
				depth_map = check_and_fix_inf_nan(depth_map, "depth_map")
				world_points = unproject_depth_map_to_point_map_torch(
					depth_map.squeeze(1), cam_extrinsic, cam_intrinsic
				)
				world_points = check_and_fix_inf_nan(world_points, "world_points")

			elif self.mode == "world_points":
				# (BN, 3, H, W) -> permute -> (BN, H, W, 3)
				world_points = self.forward_point(input_tensor).permute(0, 2, 3, 1)
				world_points = check_and_fix_inf_nan(world_points, "world_points")
				depth_map = torch.zeros_like(world_points[...,0])

			else: # scale mode
				depth_map = convert_coord_map_to_depth_map(coord_map.reshape(BN, H, W, 2), cam_intrinsic, proj_intrinsic, proj_extrinsic)
				depth_map = check_and_fix_inf_nan(depth_map, "depth_map")
				
				scale_map = self.forward_depth(input_tensor) # (BN, 1, H, W)
				scale = scale_map.mean(dim=(2, 3), keepdim=True) # (BN, 1, 1, 1)
				
				depth_map = depth_map.reshape(BN, 1, H, W) * scale
				world_points = unproject_depth_map_to_point_map_torch(
					depth_map.squeeze(1), cam_extrinsic, cam_intrinsic
				)
				world_points = check_and_fix_inf_nan(world_points, "world_points")

		else:
			raise NotImplementedError

		return depth_map.reshape(B, N, H, W), world_points.reshape(B, N, H, W, 3)


if __name__ == '__main__':
	import torch
	from torchview import draw_graph

	# 1. 動作モードとパラメータの設定
	# layer_pointを可視化するため "world_points" モードを選択
	mode = "world_points"
	batch_size = 1
	num_cams = 1
	height, width = 64, 64
	
	# 2. モデルのインスタンス化
	# 引数はクラス定義のデフォルト値に合わせています
	model = LSHead(dim_in=2048, latent_dim=32, mode=mode)

	# 3. ダミーデータの作成 (forward引数に合わせる)
	dummy_tokens = [torch.randn(batch_size * num_cams, 1, 2048)] # aggregated_tokens_list[-1]用
	dummy_cam_ext = torch.randn(batch_size, num_cams, 3, 4)
	dummy_cam_int = torch.randn(batch_size, num_cams, 3, 3)
	dummy_proj_ext = torch.randn(batch_size, num_cams, 3, 4)
	dummy_proj_int = torch.randn(batch_size, num_cams, 3, 3)
	dummy_coords = torch.randn(batch_size, num_cams, height, width, 2)

	# 4. 可視化の実行
	# LSHeadのforwardは複数の引数を取るため、x ではなく input_data としてタプルで渡します
	model_graph = draw_graph(
		model,
		input_data=(
			dummy_tokens, 
			dummy_cam_ext, 
			dummy_cam_int, 
			dummy_proj_ext, 
			dummy_proj_int, 
			dummy_coords
		),
		graph_name=f"LSHead_{mode}",
		expand_nested=True,
		depth=3,				# グラフの深さ（レイヤーの階層）
		save_graph=True,
		filename="lshead_architecture",
		directory="."
	)

	print(f"Architecutre diagram has been saved as 'lshead_architecture.pdf' (or png) in the current directory.")
