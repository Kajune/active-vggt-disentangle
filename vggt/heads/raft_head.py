import os
from typing import List, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.utils import flow_to_image, make_grid


class RAFTHead(nn.Module):
	def __init__(
		self,
	) -> None:
		super().__init__()
		weights = Raft_Large_Weights.DEFAULT
		self.preprocess = weights.transforms()
		self.model = raft_large(weights=weights, progress=True)


	def forward(
		self,
		images: torch.Tensor,
		patterns: torch.Tensor,
	) -> Union[torch.Tensor]:
		B, S, C, H, W = images.shape

		assert images.shape == patterns.shape

		pad_h = (8 - H % 8) % 8
		pad_w = (8 - W % 8) % 8
		img_in = F.pad(images, (0, pad_w, 0, pad_h), mode='constant', value=0)
		pat_in = F.pad(patterns, (0, pad_w, 0, pad_h), mode='constant', value=0)

		img_in, pat_in = self.preprocess(img_in, pat_in)
		list_of_flows = self.model(img_in.view(-1, C, H+pad_h, W+pad_w), 
								   pat_in.view(-1, C, H+pad_h, W+pad_w))
		full_flow = list_of_flows[-1]
		pred_flow = full_flow[:, :, :H, :W]

		grid_y, grid_x = torch.meshgrid(torch.arange(H, device=pred_flow.device), torch.arange(W, device=pred_flow.device), indexing='ij')
		grid = torch.stack((grid_x, grid_y), dim=-1).float()
		
		coords_pixel = grid.unsqueeze(0) + pred_flow.permute(0, 2, 3, 1)
		coords = coords_pixel.clone()
		coords[..., 0] /= (W - 1)
		coords[..., 1] /= (H - 1)

		return coords.view(B, S, H, W, 2)
