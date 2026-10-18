import os, sys, json, argparse, csv, copy, yaml
from glob import glob
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch3d.loss import chamfer_distance
from hydra import initialize, compose
from omegaconf import OmegaConf
from data.datasets.blender import BlenderDataset
import cv2
from vggt.utils.visual_util import create_frustum

def to_homogeneous(E):
	if E.shape == (3, 4):
		Eh = np.eye(4)
		Eh[:3, :4] = E
		return Eh
	elif E.shape == (4, 4):
		return E
	else:
		raise ValueError(f"Invalid extrinsic shape: {E.shape}")


def compute_param_accuracy(intrinsic, intrinsic_gt, extrinsic, extrinsic_gt, resolution, align=False):
	"""
	FOV誤差(度), 回転誤差(度), 並進誤差(絶対値), align_matrix を返す
	"""
	W, H = resolution

	def focal_to_fov(fx, fy):
		fov_x = 2 * np.arctan(W / (2 * fx))
		fov_y = 2 * np.arctan(H / (2 * fy))
		return fov_x, fov_y

	def estimate_rigid(A, B, iters=1000, lr=1e-2, device="cpu"):
		pts_A = torch.tensor(A[:, :3, 3], dtype=torch.float32, device=device)
		pts_B = torch.tensor(B[:, :3, 3], dtype=torch.float32, device=device)

		# 初期値
		log_scale = torch.zeros(1, requires_grad=True, device=device)
		rot_vec = torch.zeros(3, requires_grad=True, device=device)  # axis-angle
		t = torch.zeros(3, requires_grad=True, device=device)

		optimizer = torch.optim.Adam([log_scale, rot_vec, t], lr=lr)

		def so3_exp(w):
			theta = torch.norm(w) + 1e-8
			k = w / theta

			K = torch.stack([
				torch.stack([torch.zeros_like(k[0]), -k[2], k[1]]),
				torch.stack([k[2], torch.zeros_like(k[0]), -k[0]]),
				torch.stack([-k[1], k[0], torch.zeros_like(k[0])])
			])

			I = torch.eye(3, device=w.device)
			return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)

		for _ in range(iters):
			optimizer.zero_grad()
			s = torch.exp(log_scale)
			R = so3_exp(rot_vec)
			pts_A_hat = s * (pts_A @ R.T) + t
			loss = torch.linalg.norm(pts_A_hat - pts_B, dim=-1).mean()
			loss.backward()
			optimizer.step()

		with torch.no_grad():
			s = torch.exp(log_scale).item()
			R = so3_exp(rot_vec).cpu().numpy()
			t = t.cpu().numpy()

		T = np.eye(4)
		T[:3, :3] = s * R
		T[:3, 3] = t
		return T

	align_matrix = np.eye(4)

	if align:
		align_matrix = estimate_rigid(extrinsic, extrinsic_gt)
		extrinsic = np.array([align_matrix @ E for E in extrinsic])

	fov_errors = []
	rot_errors = []
	trans_errors = []

	for K, K_gt, E, E_gt in zip(intrinsic, intrinsic_gt, extrinsic, extrinsic_gt):
		fx, fy = K[0, 0], K[1, 1]
		fx_gt, fy_gt = K_gt[0, 0], K_gt[1, 1]

		fov_x, fov_y = focal_to_fov(fx, fy)
		fov_x_gt, fov_y_gt = focal_to_fov(fx_gt, fy_gt)
		fov_errors.append(np.mean(np.abs([[fov_x - fov_x_gt], [fov_y - fov_y_gt]])))

		R = E[:3, :3]
		R_gt = E_gt[:3, :3]
		R_err = R @ R_gt.T
		angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
		rot_errors.append(np.degrees(angle))

		t = E[:3, 3]
		t_gt = E_gt[:3, 3]
		trans_errors.append(np.linalg.norm(t - t_gt))

	return float(np.mean(fov_errors)), float(np.mean(rot_errors)), float(np.mean(trans_errors)), align_matrix


def visualize_poses(pcd, cam_intrinsic, cam_extrinsic, cam_intrinsic_gt, cam_extrinsic_gt, image_resolution,
					proj_intrinsic, proj_extrinsic, proj_intrinsic_gt, proj_extrinsic_gt, proj_resolution):
	pose_geometries = []
	lines = []
	line_colors = []
	points = []

	def pose_center(p):
		return p[:3, 3]

	idx = 0

	for cam_mat, cam_pose, cam_pose_gt in zip(cam_intrinsic, cam_extrinsic, cam_extrinsic_gt):
		cam = create_frustum(cam_mat, *image_resolution, scale=0.05)
		cam.transform(cam_pose)
		cam.paint_uniform_color([0, 0, 1])
		pose_geometries.append(cam)

		cam_gt = create_frustum(cam_mat, *image_resolution, scale=0.05)
		cam_gt.transform(cam_pose_gt)
		cam_gt.paint_uniform_color([0, 1, 0])
		pose_geometries.append(cam_gt)

		p1 = pose_center(cam_pose)
		p2 = pose_center(cam_pose_gt)
		points.extend([p1, p2])
		lines.append([idx, idx + 1])
		line_colors.append([0, 0, 0])
		idx += 2

	# ---- プロジェクタは None チェック ----
	if proj_intrinsic is not None and proj_extrinsic is not None and \
	   proj_intrinsic_gt is not None and proj_extrinsic_gt is not None:

		for proj_mat, proj_pose, proj_pose_gt in zip(proj_intrinsic, proj_extrinsic, proj_extrinsic_gt):
			proj = create_frustum(proj_mat, *proj_resolution, scale=0.05)
			proj.transform(np.linalg.inv(proj_pose))
			proj.paint_uniform_color([1, 0.5, 0])
			pose_geometries.append(proj)

			proj_gt = create_frustum(proj_mat, *proj_resolution, scale=0.05)
			proj_gt.transform(np.linalg.inv(proj_pose_gt))
			proj_gt.paint_uniform_color([1, 0, 0])
			pose_geometries.append(proj_gt)

			p1 = pose_center(np.linalg.inv(proj_pose))
			p2 = pose_center(np.linalg.inv(proj_pose_gt))
			points.extend([p1, p2])
			lines.append([idx, idx + 1])
			line_colors.append([0, 0, 0])
			idx += 2

	line_set = o3d.geometry.LineSet()
	line_set.points = o3d.utility.Vector3dVector(points)
	line_set.lines = o3d.utility.Vector2iVector(lines)
	line_set.colors = o3d.utility.Vector3dVector(line_colors)

	o3d.visualization.draw_geometries([pcd, line_set] + pose_geometries)


def compute_accuracy(dataset_config, output_dir, phase="after", remove_bg=True, verbose=False):
	with initialize(version_base=None, config_path="config"):
		config = compose(config_name=dataset_config)

	common_config = config["data"]["val"]["common_config"]
	dataset_cfg = config["data"]["val"]["dataset"]["dataset_configs"]

	results = []

	dataset_name = os.path.basename(output_dir)

	if dataset_name == "klab_sl2":
		dir_pattern = os.path.join(output_dir, "*/*/*/*/*/*")
	elif dataset_name == "rov" or dataset_name == "rov_tmp":
		dir_pattern = os.path.join(output_dir, "*/*/*")
	elif dataset_name == "endoscope2":
		dir_pattern = os.path.join(output_dir, "*/*")
	else:
		raise NotImplementedError

	for path in sorted(glob(dir_pattern)):
		print(path)
		cfg_copy = OmegaConf.to_container(dataset_cfg[0], resolve=True)

		if dataset_name == "klab_sl2":
			parts = path.split(os.sep)[-6:]
			config_name, scene, illum, intensity, pattern, mode = parts
			cfg_copy["scenes"] = [
				os.path.join(dataset_name, config_name, scene, illum, intensity, pattern)
			]

		elif dataset_name == "rov" or dataset_name == "rov_tmp":
			parts = path.split(os.sep)[-3:]
			scene, illum, mode = parts
			config_name = "default"
			intensity = "default"
			pattern = "laser"
			cfg_copy["scenes"] = [
				os.path.join("rov", scene, illum)
			]


		elif dataset_name == "endoscope2":
			parts = path.split(os.sep)[-2:]
			scene, mode = parts
			config_name = "default"
			intensity = "default"
			illum = "default"
			pattern = "doe"
			cfg_copy["scenes"] = [
				scene
			]

		else:
			raise NotImplementedError

		del cfg_copy["_target_"]
		dataset = BlenderDataset(common_config, **cfg_copy)
		if dataset_name == "klab_sl2":
			batch = dataset.get_data(0, ids="all")
		elif dataset_name == "rov" or dataset_name == "rov_tmp":
			index = int(np.loadtxt(os.path.join(path, phase, "reference_frame.txt")))
			ids = np.arange(index, dataset.get_frame_num(0), 10)
			batch = dataset.get_data(0, ids=ids)
		elif dataset_name == "endoscope2":
			batch = dataset.get_data(0, ids="all")
		else:
			raise NotImplementedError

		image = np.array(batch["images"])[0]
		image_resolution = (image.shape[1], image.shape[0])
		proj_resolution = (1, 1)

		camera_params = np.load(os.path.join(path, phase, "camera_params.npz"))
		cam_intrinsic = camera_params["intrinsic"]
		cam_extrinsic = camera_params["extrinsic"]
		cam_intrinsic_gt = np.array(batch["intrinsics"])
		cam_extrinsic_gt = np.array(batch["extrinsics"])
		proj_intrinsic = camera_params["proj_intrinsic"][:1] if "proj_intrinsic" in camera_params else None
		proj_extrinsic = camera_params["proj_extrinsic"][:1] if "proj_extrinsic" in camera_params else None
		proj_intrinsic_gt = np.array(batch["proj_intrinsics"])[:1] if "proj_intrinsics" in batch else None
		proj_extrinsic_gt = np.array(batch["proj_extrinsics"])[:1] if "proj_extrinsics" in batch else None

#		print(batch["coords"][0].shape, batch["coords"][0].min(), batch["coords"][0].max())

#		coords = batch["coords"]
#		for i in range(len(coords)):
#			coord_vis = np.concatenate([coords[i], np.zeros_like(coords[i][...,:1])], axis=-1)
#			cv2.imshow("", coord_vis)
#			cv2.waitKey()

		if "patterns" in batch:
			pH, pW, _ = batch["patterns"][0].shape
			proj_intrinsic_gt[:,0] /= pW
			proj_intrinsic_gt[:,1] /= pH

		cam_extrinsic = np.array([to_homogeneous(E) for E in cam_extrinsic])
		cam_extrinsic_gt = np.array([to_homogeneous(E) for E in cam_extrinsic_gt])
		if proj_extrinsic is not None and proj_extrinsic_gt is not None:
			proj_extrinsic = np.array([to_homogeneous(E) for E in proj_extrinsic])
			proj_extrinsic_gt = np.array([to_homogeneous(E) for E in proj_extrinsic_gt])

		pcd_result = o3d.io.read_point_cloud(os.path.join(path, phase, "result_pointmap.ply"))
		if remove_bg and pcd_result.has_colors():
			colors = np.asarray(pcd_result.colors)
			points = np.asarray(pcd_result.points)
			mask = ~np.all(colors >= [0.95, 0.95, 0.95], axis=1)
			pcd_result.points = o3d.utility.Vector3dVector(points[mask])
			pcd_result.colors = o3d.utility.Vector3dVector(colors[mask])

		cam_focal_error, cam_rot_error, cam_trans_error, align_matrix = \
			compute_param_accuracy(cam_intrinsic, cam_intrinsic_gt, cam_extrinsic, cam_extrinsic_gt, image_resolution, align=len(cam_extrinsic) >= 3)

		if proj_intrinsic is None or proj_extrinsic is None:
			proj_focal_error, proj_rot_error, proj_trans_error = None, None, None

		else:
			proj_extrinsic = np.array([align_matrix @ E for E in proj_extrinsic])
			proj_focal_error, proj_rot_error, proj_trans_error, _ = \
				compute_param_accuracy(proj_intrinsic, proj_intrinsic_gt, proj_extrinsic, proj_extrinsic_gt, proj_resolution)

		print(cam_focal_error, cam_rot_error, cam_trans_error, proj_focal_error, proj_rot_error, proj_trans_error)

		results.append((
			config_name, scene, illum, intensity, pattern, mode,
			cam_focal_error, cam_rot_error, cam_trans_error,
			proj_focal_error, proj_rot_error, proj_trans_error,
		))

		if verbose:
			pcd_result.transform(align_matrix)
			cam_extrinsic = np.array([align_matrix @ E for E in cam_extrinsic])
			visualize_poses(pcd_result, cam_intrinsic, cam_extrinsic, cam_intrinsic_gt, cam_extrinsic_gt, image_resolution,
							proj_intrinsic, proj_extrinsic, proj_intrinsic_gt, proj_extrinsic_gt, proj_resolution)

	return results


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset_config', type=str, required=True)
	parser.add_argument('--output_dir', type=str, required=True)
	parser.add_argument('--result', type=str, default="result.csv")
	parser.add_argument('--phase', type=str, default='after')
	parser.add_argument('--remove_bg', action='store_true')
	parser.add_argument('--verbose', action="store_true")
	args = parser.parse_args()

	header = ["config", "scene", "illum", "intensity", "pattern", "mode", 
			"cam_focal_error", "cam_rot_error", "cam_trans_error", 
			"proj_focal_error", "proj_rot_error", "proj_trans_error"]

	result_path = os.path.join(args.output_dir, args.result)
	with open(result_path, "w") as f:
		writer = csv.writer(f)
		writer.writerow(header)

		results = compute_accuracy(
			args.dataset_config,
			args.output_dir, 
			phase=args.phase,
			remove_bg=args.remove_bg,
			verbose=args.verbose)

		for result in results:
			writer.writerow(result)
