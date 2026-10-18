# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from vggt.utils.pose_enc import extri_intri_to_pose_encoding

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
	"""
	The Aggregator applies alternating-attention over input frames,
	as described in VGGT: Visual Geometry Grounded Transformer.

	Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

	Args:
		img_size (int): Image size in pixels.
		patch_size (int): Size of each patch for PatchEmbed.
		embed_dim (int): Dimension of the token embeddings.
		depth (int): Number of blocks.
		num_heads (int): Number of attention heads.
		mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
		num_register_tokens (int): Number of register tokens.
		block_fn (nn.Module): The block type used for attention (Block by default).
		qkv_bias (bool): Whether to include bias in QKV projections.
		proj_bias (bool): Whether to include bias in the output projection.
		ffn_bias (bool): Whether to include bias in MLP layers.
		patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
		aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
		aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
		qk_norm (bool): Whether to apply QK normalization.
		rope_freq (int): Base frequency for rotary embedding. -1 to disable.
		init_values (float): Init scale for layer scale.
	"""

	def __init__(
		self,
		img_size=518,
		patch_size=14,
		embed_dim=1024,
		depth=24,
		num_heads=16,
		mlp_ratio=4.0,
		num_register_tokens=4,
		block_fn=Block,
		qkv_bias=True,
		proj_bias=True,
		ffn_bias=True,
		patch_embed="dinov2_vitl14_reg",
		aa_order=["frame", "global"],
		aa_block_size=1,
		qk_norm=True,
		rope_freq=100,
		init_values=0.01,
		input_pattern_mode=None,
		pattern_residual_weight=0.01,
		learnable_residual_weight=False,
		input_cam_params_mode=None,
		input_proj_params=False,
	):
		super().__init__()

		self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

		# Initialize rotary position embedding if frequency > 0
		self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
		self.position_getter = PositionGetter() if self.rope is not None else None

		self.frame_blocks = nn.ModuleList(
			[
				block_fn(
					dim=embed_dim,
					num_heads=num_heads,
					mlp_ratio=mlp_ratio,
					qkv_bias=qkv_bias,
					proj_bias=proj_bias,
					ffn_bias=ffn_bias,
					init_values=init_values,
					qk_norm=qk_norm,
					rope=self.rope,
				)
				for _ in range(depth)
			]
		)

		self.global_blocks = nn.ModuleList(
			[
				block_fn(
					dim=embed_dim,
					num_heads=num_heads,
					mlp_ratio=mlp_ratio,
					qkv_bias=qkv_bias,
					proj_bias=proj_bias,
					ffn_bias=ffn_bias,
					init_values=init_values,
					qk_norm=qk_norm,
					rope=self.rope,
				)
				for _ in range(depth)
			]
		)

		self.depth = depth
		self.aa_order = aa_order
		self.patch_size = patch_size
		self.aa_block_size = aa_block_size

		# Validate that depth is divisible by aa_block_size
		if self.depth % self.aa_block_size != 0:
			raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

		self.aa_block_num = self.depth // self.aa_block_size

		# Note: We have two camera tokens, one for the first frame and one for the rest
		# The same applies for register tokens
		self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
		self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
		self.projector_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
		self.pattern_register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
		if learnable_residual_weight:
			self.pattern_residual_weight = nn.Parameter(torch.tensor(pattern_residual_weight))
		else:
			self.pattern_residual_weight = pattern_residual_weight
		self.input_pattern_mode = input_pattern_mode
		if self.input_pattern_mode == "cross_attn":
			self.cross_attn = nn.MultiheadAttention(
				embed_dim=embed_dim,
				num_heads=num_heads,
				batch_first=True  # [B, L, C]
			)
			nn.init.constant_(self.cross_attn.out_proj.weight, 1e-3)
			nn.init.constant_(self.cross_attn.out_proj.bias, 0.)
		self.input_cam_params_mode = input_cam_params_mode
		self.input_proj_params = input_proj_params

		# The patch tokens start after the camera and register tokens
		self.patch_start_idx = 1 + num_register_tokens

		# Initialize parameters with small values
		nn.init.normal_(self.camera_token, std=1e-6)
		nn.init.normal_(self.register_token, std=1e-6)
		nn.init.normal_(self.projector_token, std=1e-6)
		nn.init.normal_(self.pattern_register_token, std=1e-6)

		# Register normalization constants as buffers
		for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
			self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

		self.use_reentrant = False # hardcoded to False

	def __build_patch_embed__(
		self,
		patch_embed,
		img_size,
		patch_size,
		num_register_tokens,
		interpolate_antialias=True,
		interpolate_offset=0.0,
		block_chunks=0,
		init_values=1.0,
		embed_dim=1024,
	):
		"""
		Build the patch embed layer. If 'conv', we use a
		simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
		"""

		if "conv" in patch_embed:
			self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
		else:
			vit_models = {
				"dinov2_vitl14_reg": vit_large,
				"dinov2_vitb14_reg": vit_base,
				"dinov2_vits14_reg": vit_small,
				"dinov2_vitg2_reg": vit_giant2,
			}

			self.patch_embed = vit_models[patch_embed](
				img_size=img_size,
				patch_size=patch_size,
				num_register_tokens=num_register_tokens,
				interpolate_antialias=interpolate_antialias,
				interpolate_offset=interpolate_offset,
				block_chunks=block_chunks,
				init_values=init_values,
			)

			# Disable gradient updates for mask token
			if hasattr(self.patch_embed, "mask_token"):
				self.patch_embed.mask_token.requires_grad_(False)

	def forward(self, images: torch.Tensor, 
		patterns: Optional[torch.Tensor] = None,
		cam_params: Optional[Tuple[torch.Tensor]] = None,
		proj_params: Optional[Tuple[torch.Tensor]] = None) -> Tuple[List[torch.Tensor], int]:
		"""
		Args:
			images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
				B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

		Returns:
			(list[torch.Tensor], int):
				The list of outputs from the attention blocks,
				and the patch_start_idx indicating where patch tokens begin.
		"""
		B, S, C_in, H, W = images.shape

		if C_in != 3:
			raise ValueError(f"Expected 3 input channels, got {C_in}")

		# Normalize images and reshape for patch embed
		images = (images - self._resnet_mean) / self._resnet_std

		# Reshape to [B*S, C, H, W] for patch embedding
		images = images.view(B * S, C_in, H, W)
		patch_tokens = self.patch_embed(images)

		if isinstance(patch_tokens, dict):
			patch_tokens = patch_tokens["x_norm_patchtokens"]

		_, P, C = patch_tokens.shape

		# Expand camera and register tokens to match batch size and sequence length
		camera_token = slice_expand_and_flatten(self.camera_token, B, S)		# [B*S, 1, embed_dim]
		register_token = slice_expand_and_flatten(self.register_token, B, S)	# [B*S, num_register_tokens, embed_dim]

		if self.input_cam_params_mode == "camera_token" or self.input_pattern_mode == "head":
			cam_param_features = extri_intri_to_pose_encoding(*cam_params, image_size_hw=(H, W))
			cam_param_features = cam_param_features.view(B * S, 1, 9)
			camera_token = camera_token.clone()
			camera_token[:, :, :9] = cam_param_features

		# Concatenate special tokens with patch tokens
		tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

		if patterns is not None:
			Bp, Sp, Cp, Hp, Wp = patterns.shape
			assert Bp == B and Sp == S

			patterns_flat = patterns.view(B * S, Cp, Hp, Wp)
			pattern_patch_tokens = self.patch_embed(patterns_flat)

			if isinstance(pattern_patch_tokens, dict):
				pattern_patch_tokens = pattern_patch_tokens["x_norm_patchtokens"]

			projector_token = slice_expand_and_flatten(self.projector_token, B, S)					# [B*S, 1, embed_dim]
			pattern_register_token = slice_expand_and_flatten(self.pattern_register_token, B, S)	# [B*S, num_register_tokens, embed_dim]

			if self.input_cam_params_mode == "projector_token" and cam_params is not None:
				cam_param_features = extri_intri_to_pose_encoding(*cam_params, image_size_hw=(H, W))
				cam_param_features = cam_param_features.view(B * S, 1, 9)
				projector_token = projector_token.clone()
				projector_token[:, :, 9:18] = cam_param_features

			if self.input_proj_params and proj_params is not None:
				proj_param_features = extri_intri_to_pose_encoding(*proj_params, image_size_hw=(Hp, Wp))
				proj_param_features = proj_param_features.view(B * S, 1, 9)
				projector_token = projector_token.clone()
				projector_token[:, :, :9] = proj_param_features

			pattern_tokens = torch.cat([projector_token, pattern_register_token, pattern_patch_tokens], dim=1)

		else:
			pattern_tokens = None

		pos = None
		if self.rope is not None:
			pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

		if self.patch_start_idx > 0:
			# do not use position embedding for special tokens (camera and register tokens)
			# so set pos to 0 for the special tokens
			pos = pos + 1
			pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
			pos = torch.cat([pos_special, pos], dim=1)

		# update P because we added special tokens
		_, P, C = tokens.shape

		frame_idx = 0
		global_idx = 0
		output_list = []

		for bi in range(self.aa_block_num):
			for attn_type in self.aa_order:
				if attn_type == "frame":
					tokens, frame_idx, frame_intermediates = self._process_frame_attention(
						tokens, B, S, P, C, frame_idx, pos=pos,
					)
				elif attn_type == "global":
					tokens, global_idx, global_intermediates = self._process_global_attention(
						tokens, B, S, P, C, global_idx, pos=pos, pattern_tokens=pattern_tokens,
					)
				else:
					raise ValueError(f"Unknown attention type: {attn_type}")

			for i in range(len(frame_intermediates)):
				# concat frame and global intermediates, [B x S x P x 2C]
				concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
				output_list.append(concat_inter)

		if self.input_cam_params_mode == "head":
			cam_param_features = extri_intri_to_pose_encoding(*cam_params, image_size_hw=(H, W))
			cam_param_features = cam_param_features.view(B, S, 9)
			output_list[-1][:,:,0,:9] = cam_param_features

		del concat_inter
		del frame_intermediates
		del global_intermediates
		return output_list, self.patch_start_idx

	def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
		"""
		Process frame attention blocks. We keep tokens in shape (B*S, P, C).
		"""
		# If needed, reshape tokens or positions:
		if tokens.shape != (B * S, P, C):
			tokens = tokens.view(B, S, P, C).view(B * S, P, C)

		if pos is not None and pos.shape != (B * S, P, 2):
			pos = pos.view(B, S, P, 2).view(B * S, P, 2)

		intermediates = []

		# by default, self.aa_block_size=1, which processes one block at a time
		for _ in range(self.aa_block_size):
			if self.training:
				tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
			else:
				tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
			frame_idx += 1
			intermediates.append(tokens.view(B, S, P, C))

		return tokens, frame_idx, intermediates

	def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, pattern_tokens=None):
		"""
		Process global attention blocks. We keep tokens in shape (B, S*P, C).
		"""
		if tokens.shape != (B, S * P, C):
			tokens = tokens.view(B, S, P, C).view(B, S * P, C)

		if pattern_tokens is not None:
			pattern_tokens = pattern_tokens.view(B, S, P, C).view(B, S * P, C)
			if self.input_pattern_mode == "cross_attn":
				def cross_attn_forward(q, k, v):
					return self.cross_attn(q, k, v, need_weights=False)[0]
				attn_out = checkpoint(cross_attn_forward, tokens, pattern_tokens, pattern_tokens, use_reentrant=self.use_reentrant)
				tokens = tokens + attn_out * self.pattern_residual_weight

			elif self.input_pattern_mode == "residual":
				tokens = tokens + pattern_tokens * self.pattern_residual_weight

			else:
				raise NotImplementedError

		if pos is not None and pos.shape != (B, S * P, 2):
			pos = pos.view(B, S, P, 2).view(B, S * P, 2)

		intermediates = []

		# by default, self.aa_block_size=1, which processes one block at a time
		for _ in range(self.aa_block_size):
			if self.training:
				tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
			else:
				tokens = self.global_blocks[global_idx](tokens, pos=pos)

			global_idx += 1
			intermediates.append(tokens.view(B, S, P, C))

		return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
	"""
	Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
	1) Uses the first position (index=0) for the first frame only
	2) Uses the second position (index=1) for all remaining frames (S-1 frames)
	3) Expands both to match batch size B
	4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
	   followed by (S-1) second-position tokens
	5) Flattens to (B*S, X, C) for processing

	Returns:
		torch.Tensor: Processed tokens with shape (B*S, X, C)
	"""

	# Slice out the "query" tokens => shape (1, 1, ...)
	query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
	# Slice out the "other" tokens => shape (1, S-1, ...)
	others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
	# Concatenate => shape (B, S, ...)
	combined = torch.cat([query, others], dim=1)

	# Finally flatten => shape (B*S, ...)
	combined = combined.view(B * S, *combined.shape[2:])
	return combined
