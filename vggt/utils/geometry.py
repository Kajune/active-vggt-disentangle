# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
import cv2


from vggt.dependency.distortion import apply_distortion, iterative_undistortion, single_undistortion


def unproject_depth_map_to_point_map(
	depth_map: np.ndarray, extrinsics_cam: np.ndarray, intrinsics_cam: np.ndarray
) -> np.ndarray:
	"""
	Unproject a batch of depth maps to 3D world coordinates.

	Args:
		depth_map (np.ndarray): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
		extrinsics_cam (np.ndarray): Batch of camera extrinsic matrices of shape (S, 3, 4)
		intrinsics_cam (np.ndarray): Batch of camera intrinsic matrices of shape (S, 3, 3)

	Returns:
		np.ndarray: Batch of 3D world coordinates of shape (S, H, W, 3)
	"""
	if isinstance(depth_map, torch.Tensor):
		depth_map = depth_map.cpu().numpy()
	if isinstance(extrinsics_cam, torch.Tensor):
		extrinsics_cam = extrinsics_cam.cpu().numpy()
	if isinstance(intrinsics_cam, torch.Tensor):
		intrinsics_cam = intrinsics_cam.cpu().numpy()

	world_points_list = []
	for frame_idx in range(depth_map.shape[0]):
		cur_world_points, _, _ = depth_to_world_coords_points(
			depth_map[frame_idx].squeeze(-1), extrinsics_cam[frame_idx], intrinsics_cam[frame_idx]
		)
		world_points_list.append(cur_world_points)
	world_points_array = np.stack(world_points_list, axis=0)

	return world_points_array


def depth_to_world_coords_points(
	depth_map: np.ndarray,
	extrinsic: np.ndarray,
	intrinsic: np.ndarray,
	eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Convert a depth map to world coordinates.

	Args:
		depth_map (np.ndarray): Depth map of shape (H, W).
		intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
		extrinsic (np.ndarray): Camera extrinsic matrix of shape (3, 4). OpenCV camera coordinate convention, cam from world.

	Returns:
		tuple[np.ndarray, np.ndarray]: World coordinates (H, W, 3) and valid depth mask (H, W).
	"""
	if depth_map is None:
		return None, None, None

	# Valid depth mask
	point_mask = depth_map > eps

	# Convert depth map to camera coordinates
	cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

	# Multiply with the inverse of extrinsic matrix to transform to world coordinates
	# extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
	cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

	R_cam_to_world = cam_to_world_extrinsic[:3, :3]
	t_cam_to_world = cam_to_world_extrinsic[:3, 3]

	# Apply the rotation and translation to the camera coordinates
	world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3
	# world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

	return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""
	Convert a depth map to camera coordinates.

	Args:
		depth_map (np.ndarray): Depth map of shape (H, W).
		intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

	Returns:
		tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
	"""
	H, W = depth_map.shape
	assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
	assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

	# Intrinsic parameters
	fu, fv = intrinsic[0, 0], intrinsic[1, 1]
	cu, cv = intrinsic[0, 2], intrinsic[1, 2]

	# Generate grid of pixel coordinates
	u, v = np.meshgrid(np.arange(W), np.arange(H))

	# Unproject to camera coordinates
	x_cam = (u - cu) * depth_map / fu
	y_cam = (v - cv) * depth_map / fv
	z_cam = depth_map

	# Stack to form camera coordinates
	cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

	return cam_coords


def unproject_depth_map_to_point_map_torch(
	depth_map: torch.Tensor, extrinsics_cam: torch.Tensor, intrinsics_cam: torch.Tensor, eps=1e-8
) -> torch.Tensor:
	"""
	Args:
		depth_map: (S, H, W) or (S, H, W, 1)
		extrinsics_cam: (S, 3, 4)
		intrinsics_cam: (S, 3, 3)
	Returns:
		(S, H, W, 3)
	"""
	if depth_map.dim() == 4 and depth_map.shape[-1] == 1:
		depth_map = depth_map.squeeze(-1)

	world_points_list = []
	for i in range(depth_map.shape[0]):
		world, _, _ = depth_to_world_coords_points_torch(
			depth_map[i], extrinsics_cam[i], intrinsics_cam[i], eps
		)
		world_points_list.append(world)

	return torch.stack(world_points_list, dim=0)


def depth_to_world_coords_points_torch(
	depth_map: torch.Tensor,
	extrinsic: torch.Tensor,
	intrinsic: torch.Tensor,
	eps: float = 1e-8,
):
	if depth_map is None:
		return None, None, None

	point_mask = depth_map > eps
	cam_coords_points = depth_to_cam_coords_points_torch(depth_map, intrinsic)

	# (3,4) -> (4,4) for inversion
	ex4 = torch.eye(4, device=extrinsic.device, dtype=extrinsic.dtype)
	ex4[:3, :4] = extrinsic
	cam_to_world = closed_form_inverse_se3(ex4[None])[0]  # (4,4)

	R = cam_to_world[:3, :3]
	t = cam_to_world[:3, 3]

	world_coords = cam_coords_points @ R.T + t

	return world_coords, cam_coords_points, point_mask


def depth_to_cam_coords_points_torch(depth_map: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
	"""
	Args:
		depth_map: (H, W)
		intrinsic: (3, 3)
	Returns:
		(H, W, 3)
	"""
	H, W = depth_map.shape
	fu, fv = intrinsic[0, 0], intrinsic[1, 1]
	cu, cv = intrinsic[0, 2], intrinsic[1, 2]

	u = torch.arange(W, device=depth_map.device).unsqueeze(0).expand(H, W)
	v = torch.arange(H, device=depth_map.device).unsqueeze(1).expand(H, W)

	x = (u - cu) * depth_map / fu
	y = (v - cv) * depth_map / fv
	z = depth_map

	return torch.stack((x, y, z), dim=-1).float()


def closed_form_inverse_se3(se3, R=None, T=None):
	"""
	Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

	If `R` and `T` are provided, they must correspond to the rotation and translation
	components of `se3`. Otherwise, they will be extracted from `se3`.

	Args:
		se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
		R (optional): Nx3x3 array or tensor of rotation matrices.
		T (optional): Nx3x1 array or tensor of translation vectors.

	Returns:
		Inverted SE3 matrices with the same type and device as `se3`.

	Shapes:
		se3: (N, 4, 4)
		R: (N, 3, 3)
		T: (N, 3, 1)
	"""
	# Check if se3 is a numpy array or a torch tensor
	is_numpy = isinstance(se3, np.ndarray)

	# Validate shapes
	if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
		raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

	# Extract R and T if not provided
	if R is None:
		R = se3[:, :3, :3]  # (N,3,3)
	if T is None:
		T = se3[:, :3, 3:]  # (N,3,1)

	# Transpose R
	if is_numpy:
		# Compute the transpose of the rotation for NumPy
		R_transposed = np.transpose(R, (0, 2, 1))
		# -R^T t for NumPy
		top_right = -np.matmul(R_transposed, T)
		inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
	else:
		R_transposed = R.transpose(1, 2)  # (N,3,3)
		top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
		inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
		inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

	inverted_matrix[:, :3, :3] = R_transposed
	inverted_matrix[:, :3, 3:] = top_right

	return inverted_matrix


# TODO: this code can be further cleaned up


def project_world_points_to_camera_points_batch(world_points, cam_extrinsics):
	"""
	Transforms 3D points to 2D using extrinsic and intrinsic parameters.
	Args:
		world_points (torch.Tensor): 3D points of shape BxSxHxWx3.
		cam_extrinsics (torch.Tensor): Extrinsic parameters of shape BxSx3x4.
	Returns:
	"""
	# TODO: merge this into project_world_points_to_cam
	
	# device = world_points.device
	# with torch.autocast(device_type=device.type, enabled=False):
	ones = torch.ones_like(world_points[..., :1])  # shape: (B, S, H, W, 1)
	world_points_h = torch.cat([world_points, ones], dim=-1)  # shape: (B, S, H, W, 4)

	# extrinsics: (B, S, 3, 4) -> (B, S, 1, 1, 3, 4)
	extrinsics_exp = cam_extrinsics.unsqueeze(2).unsqueeze(3)

	# world_points_h: (B, S, H, W, 4) -> (B, S, H, W, 4, 1)
	world_points_h_exp = world_points_h.unsqueeze(-1)

	# Now perform the matrix multiplication
	# (B, S, 1, 1, 3, 4) @ (B, S, H, W, 4, 1) broadcasts to (B, S, H, W, 3, 1)
	camera_points = torch.matmul(extrinsics_exp, world_points_h_exp).squeeze(-1)

	return camera_points



def project_world_points_to_cam(
	world_points,
	cam_extrinsics,
	cam_intrinsics=None,
	distortion_params=None,
	default=0,
	only_points_cam=False,
):
	"""
	Transforms 3D points to 2D using extrinsic and intrinsic parameters.
	Args:
		world_points (torch.Tensor): 3D points of shape Px3.
		cam_extrinsics (torch.Tensor): Extrinsic parameters of shape Bx3x4.
		cam_intrinsics (torch.Tensor): Intrinsic parameters of shape Bx3x3.
		distortion_params (torch.Tensor): Extra parameters of shape BxN, which is used for radial distortion.
	Returns:
		torch.Tensor: Transformed 2D points of shape BxNx2.
	"""
	device = world_points.device
	# with torch.autocast(device_type=device.type, dtype=torch.double):
	with torch.autocast(device_type=device.type, enabled=False):
		N = world_points.shape[0]  # Number of points
		B = cam_extrinsics.shape[0]  # Batch size, i.e., number of cameras
		world_points_homogeneous = torch.cat(
			[world_points, torch.ones_like(world_points[..., 0:1])], dim=1
		)  # Nx4
		# Reshape for batch processing
		world_points_homogeneous = world_points_homogeneous.unsqueeze(0).expand(
			B, -1, -1
		)  # BxNx4

		# Step 1: Apply extrinsic parameters
		# Transform 3D points to camera coordinate system for all cameras
		cam_points = torch.bmm(
			cam_extrinsics, world_points_homogeneous.transpose(-1, -2)
		)

		if only_points_cam:
			return None, cam_points

		# Step 2: Apply intrinsic parameters and (optional) distortion
		image_points = img_from_cam(cam_intrinsics, cam_points, distortion_params, default=default)

		return image_points, cam_points



def img_from_cam(cam_intrinsics, cam_points, distortion_params=None, default=0.0):
	"""
	Applies intrinsic parameters and optional distortion to the given 3D points.

	Args:
		cam_intrinsics (torch.Tensor): Intrinsic camera parameters of shape Bx3x3.
		cam_points (torch.Tensor): 3D points in camera coordinates of shape Bx3xN.
		distortion_params (torch.Tensor, optional): Distortion parameters of shape BxN, where N can be 1, 2, or 4.
		default (float, optional): Default value to replace NaNs in the output.

	Returns:
		pixel_coords (torch.Tensor): 2D points in pixel coordinates of shape BxNx2.
	"""

	# Normalized device coordinates (NDC)
	cam_points = cam_points / cam_points[:, 2:3, :]
	ndc_xy = cam_points[:, :2, :]

	# Apply distortion if distortion_params are provided
	if distortion_params is not None:
		x_distorted, y_distorted = apply_distortion(distortion_params, ndc_xy[:, 0], ndc_xy[:, 1])
		distorted_xy = torch.stack([x_distorted, y_distorted], dim=1)
	else:
		distorted_xy = ndc_xy

	# Prepare cam_points for batch matrix multiplication
	cam_coords_homo = torch.cat(
		(distorted_xy, torch.ones_like(distorted_xy[:, :1, :])), dim=1
	)  # Bx3xN
	# Apply intrinsic parameters using batch matrix multiplication
	pixel_coords = torch.bmm(cam_intrinsics, cam_coords_homo)  # Bx3xN

	# Extract x and y coordinates
	pixel_coords = pixel_coords[:, :2, :]  # Bx2xN

	# Replace NaNs with default value
	pixel_coords = torch.nan_to_num(pixel_coords, nan=default)

	return pixel_coords.transpose(1, 2)  # BxNx2




def cam_from_img(pred_tracks, intrinsics, extra_params=None):
	"""
	Normalize predicted tracks based on camera intrinsics.
	Args:
	intrinsics (torch.Tensor): The camera intrinsics tensor of shape [batch_size, 3, 3].
	pred_tracks (torch.Tensor): The predicted tracks tensor of shape [batch_size, num_tracks, 2].
	extra_params (torch.Tensor, optional): Distortion parameters of shape BxN, where N can be 1, 2, or 4.
	Returns:
	torch.Tensor: Normalized tracks tensor.
	"""

	# We don't want to do intrinsics_inv = torch.inverse(intrinsics) here
	# otherwise we can use something like
	#     tracks_normalized_homo = torch.bmm(pred_tracks_homo, intrinsics_inv.transpose(1, 2))

	principal_point = intrinsics[:, [0, 1], [2, 2]].unsqueeze(-2)
	focal_length = intrinsics[:, [0, 1], [0, 1]].unsqueeze(-2)
	tracks_normalized = (pred_tracks - principal_point) / focal_length

	if extra_params is not None:
		# Apply iterative undistortion
		try:
			tracks_normalized = iterative_undistortion(
				extra_params, tracks_normalized
			)
		except:
			tracks_normalized = single_undistortion(
				extra_params, tracks_normalized
			)

	return tracks_normalized


def convert_coord_map_to_depth_map(
	coord_map, 				# [N,H,W,2] projector pixel (u_p, v_p)
	intrinsics_cam, 		# [N,3,3]  K_cam
	intrinsics_proj, 		# [N,3,3]  K_proj
	extrinsics_proj,		# [N,3,4]  [R|t]  (proj -> cam)
	use_ray_ray=False,
	max_depth_scale=20,
):
	if torch.is_tensor(coord_map):
		backend = torch
		transpose = lambda x, order: x.permute(order)
		reshape = lambda x, *a: x.reshape(*a)
		broadcast_to = lambda x, shape: x.expand(shape)
		inv = torch.inverse
		nan = torch.tensor(float("nan"), device=coord_map.device, dtype=coord_map.dtype)
	else:
		backend = np
		transpose = lambda x, order: np.transpose(x, order)
		reshape = np.reshape
		broadcast_to = np.broadcast_to
		inv = np.linalg.inv
		nan = np.nan

	N, H, W, _ = coord_map.shape

	# -------------------------
	# Camera rays
	# -------------------------
	u = backend.arange(W, dtype=coord_map.dtype if backend is torch else np.float64)
	v = backend.arange(H, dtype=coord_map.dtype if backend is torch else np.float64)
	if backend is torch:
		u = u.to(coord_map.device)
		v = v.to(coord_map.device)
	uu, vv = backend.meshgrid(u, v, indexing="xy")

	pix_cam = backend.stack([uu, vv, backend.ones_like(uu)], axis=-1)[None]
	Kc_inv = inv(intrinsics_cam)

	d_c = reshape(pix_cam, (1, -1, 3))
	d_c = transpose(Kc_inv @ transpose(d_c, (0,2,1)), (0,2,1))
	d_c = reshape(d_c, (N, H, W, 3))
	d_c = d_c / (backend.linalg.norm(d_c, axis=-1, keepdims=True) + 1e-12)

	# -------------------------
	# Extrinsics
	# -------------------------
	R = extrinsics_proj[..., :3]
	t = extrinsics_proj[..., 3:]	# [N,3,1]
	baseline = backend.linalg.norm(t)

	# =========================================================
	# Ray–Ray triangulation (existing behavior)
	# =========================================================
	if use_ray_ray:
		pix_proj = backend.stack(
			[coord_map[...,0], coord_map[...,1], backend.ones_like(coord_map[...,0])],
			axis=-1
		)
		Kp_inv = inv(intrinsics_proj)

		d_p = reshape(pix_proj, (N, -1, 3))
		d_p = transpose(Kp_inv @ transpose(d_p, (0,2,1)), (0,2,1))
		d_p = reshape(d_p, (N, H, W, 3))
		d_p = d_p / (backend.linalg.norm(d_p, axis=-1, keepdims=True) + 1e-12)

		d_p_cam = reshape(d_p, (N, -1, 3))
		d_p_cam = transpose(R @ transpose(d_p_cam, (0,2,1)), (0,2,1))
		d_p_cam = reshape(d_p_cam, (N, H, W, 3))

		o_p_cam = transpose(t, (0,2,1))
		o_p_cam = broadcast_to(o_p_cam[:,None], (N, H, W, 3))
		o_c = backend.zeros_like(o_p_cam)

		def dot(a, b):
			return backend.sum(a * b, axis=-1, keepdims=True)

		u_vec = d_c
		v_vec = d_p_cam
		w_vec = o_p_cam - o_c

		a = dot(u_vec, u_vec)
		b = dot(u_vec, v_vec)
		c = dot(v_vec, v_vec)
		d = dot(u_vec, w_vec)
		e = dot(v_vec, w_vec)

		den = a * c - b * b
		eps = 1e-12
		mask_valid = (backend.abs(den) > eps) #& (coord_map[...,:1] > 1e-2)
		safe_den = backend.where(backend.abs(den) > eps, den, backend.ones_like(den) * eps)

		s = backend.where(
			mask_valid,
			(d * c - b * e) / safe_den,
			nan
		)

		X_c = s * u_vec
		depth = X_c[..., 2:3]

	# =========================================================
	# Ray–Plane (light-sectioning)
	# =========================================================
	else:
		u_p = coord_map[...,0]	# [N,H,W]

		Kp_inv = inv(intrinsics_proj)

		# --- build two rays defining the light plane ---
		p1 = backend.stack(
			[u_p, backend.zeros_like(u_p), backend.ones_like(u_p)],
			axis=-1
		)
		p2 = backend.stack(
			[u_p, backend.ones_like(u_p), backend.ones_like(u_p)],
			axis=-1
		)

		p1 = reshape(p1, (N, -1, 3))
		p2 = reshape(p2, (N, -1, 3))

		r1 = transpose(Kp_inv @ transpose(p1, (0,2,1)), (0,2,1))
		r2 = transpose(Kp_inv @ transpose(p2, (0,2,1)), (0,2,1))

		r1 = reshape(r1, (N, H, W, 3))
		r2 = reshape(r2, (N, H, W, 3))

		# --- plane normal in projector coord ---
		if backend is torch:
			n_p = torch.linalg.cross(r1, r2, dim=-1)
		else:
			n_p = backend.cross(r1, r2, axisa=-1, axisb=-1, axisc=-1)

		# --- transform plane to camera coord ---
		n_c = reshape(n_p, (N, -1, 3))
		n_c = transpose(R @ transpose(n_c, (0,2,1)), (0,2,1))
		n_c = reshape(n_c, (N, H, W, 3))

		t_cam = reshape(t, (-1,1,1,3))
		d_plane = -backend.sum(n_c * t_cam, axis=-1, keepdims=True)

		den = backend.sum(n_c * d_c, axis=-1, keepdims=True)
		eps = 1e-6
		mask_valid = (backend.abs(den) > eps) #& (u_p[...,None] > 1e-2)
		safe_den = backend.where(backend.abs(den) > eps, den, backend.ones_like(den) * eps)

#		s = backend.where(
#			mask_valid,
#			-d_plane / safe_den,
#			nan
#		)
		s = -d_plane / safe_den

		X_c = s * d_c
		depth = X_c[..., 2:3]

	# -------------------------
	# Final cleanup
	# -------------------------
	if max_depth_scale is not None:
		if backend is torch:
			depth = torch.clamp(depth, 0, baseline * max_depth_scale)
			depth = torch.where(torch.isfinite(depth), depth, nan)
	#		depth = torch.where(mask_valid, depth, nan)
		else:
			depth = backend.clip(depth, 0, baseline * max_depth_scale)
			depth = backend.where(backend.isfinite(depth), depth, nan)
	#		depth = backend.where(mask_valid, depth, nan)

	return depth


def convert_coord_map_to_depth_map_cv(
	coord_map: np.ndarray, 			# [N, H, W, 2]  projector pixel (u_p, v_p)
	intrinsics_cam: np.ndarray, 	# [N, 3, 3]     K_cam
	intrinsics_proj: np.ndarray, 	# [N, 3, 3]     K_proj
	extrinsics_proj: np.ndarray,	# [N, 3, 4]     [R|t] : X_c = R X_p + t
) -> np.ndarray:	# [N, H, W, 1]
	N, H, W, _ = coord_map.shape
	depth_map = np.zeros((N, H, W, 1), dtype=np.float32)

	for i in range(N):
		Kc = intrinsics_cam[i]
		Kp = intrinsics_proj[i]
		Rt_p = extrinsics_proj[i]

		P_c = Kc @ np.hstack([np.eye(3), np.zeros((3,1))])
		P_p = Kp @ Rt_p

		u, v = np.meshgrid(np.arange(W), np.arange(H))
		pts_c = np.stack([u.flatten(), v.flatten()], axis=0).astype(np.float32)
		pts_p = coord_map[i].reshape(-1, 2).T.astype(np.float32)

		points_4d = cv2.triangulatePoints(P_c, P_p, pts_c, pts_p)
		points_3d = points_4d[:3] / points_4d[3]

		depth_map[i, ..., 0] = points_3d[2].reshape(H, W)

	return depth_map
