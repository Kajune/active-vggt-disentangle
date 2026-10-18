# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.raft_head import RAFTHead
from vggt.heads.track_head import TrackHead
from vggt.heads.ls_head import LSHead
from vggt.layers.vision_transformer import vit_large
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri



class VGGT(nn.Module, PyTorchModelHubMixin):
	def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
				 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True, enable_pattern_track=False, enable_ls=False,
				 input_cam_params_mode=None, enable_projection=False, num_max_projectors=4, use_raft=False,
				 input_pattern_mode=None, pattern_residual_weight=0, learnable_residual_weight=False, input_proj_params=False, coded_coord=False, 
				 ls_mode="analytical", use_pred_cam_for_ls=True, use_pred_proj_for_ls=False):
		super().__init__()

		self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, input_cam_params_mode=input_cam_params_mode,
									input_pattern_mode=input_pattern_mode, 
									pattern_residual_weight=pattern_residual_weight, learnable_residual_weight=learnable_residual_weight,
									input_proj_params=input_proj_params)

		self.camera_head = CameraHead(dim_in=2 * embed_dim, input_gt_cam_params=input_cam_params_mode=="head") if enable_camera else None
		self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
		self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
		self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

		if enable_projection:
			self.input_pattern_mode = input_pattern_mode

			if self.input_pattern_mode == "branch":
				pattern_embed_dim = 384
				self.pattern_aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=pattern_embed_dim, 
													depth=1, patch_embed="dinov2_vits14_reg")
			else:
				pattern_embed_dim = 0

			self.projector_head = CameraHead(dim_in=2 * embed_dim + 2 * pattern_embed_dim)
			self.use_raft = use_raft
			if self.use_raft:
				self.coord_head = RAFTHead()
			else:
				self.coord_head = DPTHead(dim_in=2 * embed_dim + 2 * pattern_embed_dim, output_dim=8*2+1 if coded_coord else 3, activation="linear", conf_activation="expp1")
			self.pattern_track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_pattern_track else None

			if enable_ls:
				self.ls_head = LSHead(dim_in=2 * embed_dim + 2 * pattern_embed_dim, mode=ls_mode)
				self.use_pred_cam_for_ls = use_pred_cam_for_ls
				self.use_pred_proj_for_ls = use_pred_proj_for_ls
			else:
				self.ls_head = None

		else:
			self.input_pattern_mode = None
			self.projector_head = None
			self.coord_head = None
			self.pattern_track_head = None
			self.ls_head = None


	def forward(self, images: torch.Tensor, query_points: torch.Tensor = None, 
				patterns: torch.Tensor = None, cam_params = None, proj_params = None):
		"""
		Forward pass of the VGGT model.

		Args:
			images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
				B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
			query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
				Shape: [N, 2] or [B, N, 2], where N is the number of query points.
				Default: None

		Returns:
			dict: A dictionary containing the following predictions:
				- pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
				- depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
				- depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
				- world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
				- world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
				- images (torch.Tensor): Original input images, preserved for visualization

				If query_points is provided, also includes:
				- track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
				- vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
				- conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
		"""        
		# If without batch dimension, add it
		if len(images.shape) == 4:
			images = images.unsqueeze(0)

		if query_points is not None and len(query_points.shape) == 2:
			query_points = query_points.unsqueeze(0)

		if patterns is not None and len(patterns.shape) == 4:
			patterns = patterns.unsqueeze(0)
#			images = torch.cat([images, patterns], dim=1)

		if self.input_pattern_mode in ["residual", "cross_attn"]:
			aggregated_tokens_list, patch_start_idx = self.aggregator(images, 
				patterns=patterns, proj_params=proj_params, cam_params=cam_params)
		else:
			aggregated_tokens_list, patch_start_idx = self.aggregator(images, cam_params=cam_params)

		predictions = {}

		with torch.cuda.amp.autocast(enabled=False):
			if self.camera_head is not None:
				pose_enc_list = self.camera_head(aggregated_tokens_list)
				predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
				predictions["pose_enc_list"] = pose_enc_list
				
			if self.depth_head is not None:
				depth, depth_conf = self.depth_head(
					aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
				)
				predictions["depth"] = depth
				predictions["depth_conf"] = depth_conf

			if self.point_head is not None:
				pts3d, pts3d_conf = self.point_head(
					aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
				)
				predictions["world_points"] = pts3d
				predictions["world_points_conf"] = pts3d_conf

			if self.input_pattern_mode == "branch" and (self.projector_head is not None or self.coord_head is not None):
				pattern_tokens_list, _ = self.pattern_aggregator(patterns)
				aggregated_tokens_list_for_projection = [torch.cat([aggregated_tokens, pattern_tokens_list[0]], dim=-1) 
																for aggregated_tokens in aggregated_tokens_list]
			else:
				aggregated_tokens_list_for_projection = aggregated_tokens_list

			if self.projector_head is not None:
				proj_pose_enc_list = self.projector_head(aggregated_tokens_list_for_projection)
				predictions["proj_pose_enc"] = proj_pose_enc_list[-1]  # pose encoding of the last iteration
				predictions["proj_pose_enc_list"] = proj_pose_enc_list

			if self.coord_head is not None:
				if self.use_raft:
					coord = self.coord_head(images=images, patterns=patterns)
					predictions["coord"] = coord
					predictions["coord_conf"] = torch.ones_like(coord[...,:1])

				else:
					coord, coord_conf = self.coord_head(
						aggregated_tokens_list_for_projection, images=images, patch_start_idx=patch_start_idx
					)
					predictions["coord"] = coord
					predictions["coord_conf"] = coord_conf

			if self.ls_head is not None and "coord" in predictions:
				if self.use_pred_cam_for_ls or cam_params is None:
					image_hw = images.shape[-2:]
					cam_extrinsic, cam_intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], image_hw)
				else:
					cam_extrinsic, cam_intrinsic = cam_params

				if self.use_pred_proj_for_ls or proj_params is None:
					proj_extrinsic, proj_intrinsic = pose_encoding_to_extri_intri(predictions["proj_pose_enc"], (1, 1))
				else:
					proj_extrinsic, proj_intrinsic = proj_params

				predictions["depth_from_coord"], predictions["world_points_from_coord"] = self.ls_head(
					aggregated_tokens_list_for_projection, 
					cam_extrinsic, cam_intrinsic,
					proj_extrinsic, proj_intrinsic,
					predictions["coord"],
				)

		if self.track_head is not None and query_points is not None:
			track_list, vis, conf = self.track_head(
				aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
			)
			predictions["track"] = track_list
			predictions["vis"] = vis
			predictions["conf"] = conf

		if self.pattern_track_head is not None and query_points is not None:
			track_list, vis, conf = self.pattern_track_head(
				aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
			)
			predictions["pattern_track"] = track_list
			predictions["pattern_vis"] = vis
			predictions["pattern_conf"] = conf

		if not self.training:
			predictions["images"] = images  # store the images for visualization during inference

		return predictions

