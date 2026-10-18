import os, json, argparse, csv, copy, yaml
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


def align_pointclouds(pcd_src: o3d.geometry.PointCloud, pcd_tgt: o3d.geometry.PointCloud, intrinsic : np.ndarray, image : np.ndarray,
	iterations: int = 500, lr: float = 1e-2, sample_points: int = 1024, device: str = "cuda") -> o3d.geometry.PointCloud:
	"""
	Aligns pcd_src to pcd_tgt by optimizing scale and z-offset to minimize chamfer distance.
	"""
	# Convert Open3D point clouds to torch tensors
	src_pts_np = np.asarray(pcd_src.points)
	tgt_pts_np = np.asarray(pcd_tgt.points)
	src_pts = torch.tensor(src_pts_np, dtype=torch.float32, device=device)
	tgt_pts = torch.tensor(tgt_pts_np, dtype=torch.float32, device=device)

	src_pts_proj = (intrinsic @ src_pts_np.T).T
	src_pts_proj /= src_pts_proj[:,2:]
	src_valid_mask = (0 <= src_pts_proj[:,0]) & (src_pts_proj[:,0] < image.shape[1]) & (0 <= src_pts_proj[:,1]) & (src_pts_proj[:,1] < image.shape[0])
	tgt_pts_proj = (intrinsic @ tgt_pts_np.T).T
	tgt_pts_proj /= tgt_pts_proj[:,2:]
	tgt_valid_mask = (0 <= tgt_pts_proj[:,0]) & (tgt_pts_proj[:,0] < image.shape[1]) & (0 <= tgt_pts_proj[:,1]) & (tgt_pts_proj[:,1] < image.shape[0])

	src_indices = src_pts_proj[src_valid_mask].astype(np.int32)
	tgt_indices = tgt_pts_proj[tgt_valid_mask].astype(np.int32)

	depth_image_src = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
	depth_image_tgt = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

	depth_image_src[src_indices[:,1], src_indices[:,0]] = src_pts_np[src_valid_mask][:,2]
	depth_image_tgt[tgt_indices[:,1], tgt_indices[:,0]] = tgt_pts_np[tgt_valid_mask][:,2]

#	cv2.imshow("", depth_image_src / np.max(depth_image_src))
#	cv2.waitKey()
#	cv2.imshow("", depth_image_tgt / np.max(depth_image_tgt))
#	cv2.waitKey()

	if np.any(src_valid_mask) and np.any(tgt_valid_mask):
		init_scale = tgt_pts_np[tgt_valid_mask][:,2].mean() / (src_pts_np[src_valid_mask][:,2].mean() + 1e-3)
		init_offset = 0
	else:
		src_z_min, src_z_mean, src_z_max = src_pts_np[:, 2].min(), src_pts_np[:, 2].mean(), src_pts_np[:, 2].max()
		tgt_z_min, tgt_z_mean, tgt_z_max = tgt_pts_np[:, 2].min(), tgt_pts_np[:, 2].mean(), tgt_pts_np[:, 2].max()
		src_mean_radius = np.mean(np.linalg.norm(src_pts_np - np.mean(src_pts_np, axis=0), axis=-1))
		tgt_mean_radius = np.mean(np.linalg.norm(tgt_pts_np - np.mean(tgt_pts_np, axis=0), axis=-1))

		init_scale = tgt_mean_radius / src_mean_radius
		init_offset = tgt_z_mean - src_z_mean * init_scale

	scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32, device=device))
	z_offset = nn.Parameter(torch.tensor(init_offset, dtype=torch.float32, device=device))

	optimizer = optim.Adam([scale, z_offset], lr=lr)
	src_num = src_pts.shape[0]
	tgt_num = tgt_pts.shape[0]

	for it in range(iterations):
		optimizer.zero_grad()

		src_idx = np.random.choice(src_num, min(sample_points, src_num), replace=False)
		tgt_idx = np.random.choice(tgt_num, min(sample_points, tgt_num), replace=False)

		src_sample = src_pts[src_idx]
		tgt_sample = tgt_pts[tgt_idx]

		transformed_src = src_sample * scale
		transformed_src[:, 2] += z_offset

		# Chamfer Distance
		distance, _ = chamfer_distance(transformed_src.unsqueeze(0), tgt_sample.unsqueeze(0), 
			batch_reduction=None, point_reduction=None, single_directional=True)
		clamp_max = torch.quantile(distance, 0.5).item()
		loss = torch.clamp(distance, min=0.0, max=clamp_max).mean()
		loss.backward()
		optimizer.step()

	# Apply optimized transformation
	final_src = src_pts * scale.detach()
	final_src[:, 2] += z_offset.detach()

	# Convert back to Open3D point cloud
	aligned_pcd = copy.deepcopy(pcd_src)
	aligned_pcd.points = o3d.utility.Vector3dVector(final_src.cpu().numpy())

	return aligned_pcd


def compute_chamfer_distance(pcd_result, pcd_gt, gt_to_result_thresh, result_to_gt_thresh, scale):
	shape_size = np.max(pcd_gt.get_axis_aligned_bounding_box().get_extent())

#	pcd_gt.paint_uniform_color([0, 1, 0])
#	pcd_result.paint_uniform_color([0, 0, 1])
#	frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
#	o3d.visualization.draw_geometries([pcd_result, pcd_gt, frame])

#	if with_icp:
#		reg_p2p = o3d.pipelines.registration.registration_icp(
#			pcd_result, pcd_gt, 0.1, np.eye(4),
#			o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True))
#		pcd_result.transform(reg_p2p.transformation)

	dist = np.array(pcd_gt.compute_point_cloud_distance(pcd_result))
	mask = dist < shape_size * gt_to_result_thresh
	gt_to_result_accuracy = np.mean(dist[mask]) * scale
	gt_to_result_coverage = np.mean(mask)
	print("GT to result: %.4f (Coverage: %.3f)" % (gt_to_result_accuracy, gt_to_result_coverage))

#	o3d.visualization.draw_geometries([pcd_gt.select_by_index(np.where(mask)[0]), pcd_result])

	pcd_result = pcd_result.uniform_down_sample(every_k_points=10)
	dist = np.array(pcd_result.compute_point_cloud_distance(pcd_gt))
	mask = dist < shape_size * result_to_gt_thresh
	result_to_gt_accuracy = np.mean(dist[mask]) * scale
	result_to_gt_coverage = np.mean(mask)
	print("Result to GT: %.4f (Coverage: %.3f)" % (result_to_gt_accuracy, result_to_gt_coverage))

	return gt_to_result_accuracy, gt_to_result_coverage, result_to_gt_accuracy, result_to_gt_coverage


def compute_accuracy(dataset_config, output_dir, gt_to_result_thresh, result_to_gt_thresh, remove_bg, scale, phase="after"):
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

		pcd_gt = o3d.geometry.PointCloud()
		all_points = np.concatenate([pts[mask].reshape(-1, 3) for pts, mask in zip(batch['world_points'], batch['point_masks'])], axis=0)
		all_colors = np.concatenate([img[mask].reshape(-1, 3) for img, mask in zip(batch['images'], batch['point_masks'])], axis=0) / 255.0
		pcd_gt.points = o3d.utility.Vector3dVector(all_points)
		pcd_gt.colors = o3d.utility.Vector3dVector(all_colors)

		pcd_result = o3d.io.read_point_cloud(os.path.join(path, phase, "result_pointmap.ply"))
		if remove_bg and pcd_result.has_colors():
			colors = np.asarray(pcd_result.colors)
			points = np.asarray(pcd_result.points)
			mask = ~np.all(colors >= [0.95, 0.95, 0.95], axis=1)
			pcd_result.points = o3d.utility.Vector3dVector(points[mask])
			pcd_result.colors = o3d.utility.Vector3dVector(colors[mask])

		camera_params = np.load(os.path.join(path, phase, "camera_params.npz"))
		extrinsic = camera_params["extrinsic"]
		extrinsic_gt = np.array(batch["extrinsics"])
		pose_pred = np.eye(4)
		pose_pred[:3] = extrinsic[0]
		pose_gt = np.eye(4)
		pose_gt[:3] = extrinsic_gt[0]
		pcd_result.transform(np.linalg.inv(pose_pred))
#		pcd_result.transform(np.linalg.inv(pose_gt))
		pcd_gt.transform(pose_gt)

		intrinsic_gt = np.array(batch["intrinsics"])[0]
		image = np.array(batch["images"])[0]

		pcd_result = align_pointclouds(pcd_result, pcd_gt, intrinsic_gt, image)

		pcd_result.transform(np.linalg.inv(pose_gt))
		pcd_gt.transform(np.linalg.inv(pose_gt))

		save_dir = os.path.join(path, phase)
		os.makedirs(save_dir, exist_ok=True)
		o3d.io.write_point_cloud(os.path.join(save_dir, "aligned_result.ply"), pcd_result)
		o3d.io.write_point_cloud(os.path.join(save_dir, "gt_pointcloud.ply"), pcd_gt)

		gt_to_result_accuracy, gt_to_result_coverage, result_to_gt_accuracy, result_to_gt_coverage = \
			compute_chamfer_distance(pcd_result, pcd_gt, gt_to_result_thresh, result_to_gt_thresh, scale)

		results.append((
			config_name, scene, illum, intensity, pattern, mode,
			gt_to_result_accuracy, gt_to_result_coverage,
			result_to_gt_accuracy, result_to_gt_coverage
		))

	return results


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset_config', type=str, required=True)
	parser.add_argument('--output_dir', type=str, required=True)
	parser.add_argument('--result', type=str, default="result.csv")
	parser.add_argument('--gt_to_result_thresh', type=float, default=0.05)
	parser.add_argument('--result_to_gt_thresh', type=float, default=0.05)
	parser.add_argument('--remove_bg', action='store_true')
	parser.add_argument('--scale', type=float, default=1e3)
	parser.add_argument('--phase', type=str, default='after')
	args = parser.parse_args()

	header = ["config", "scene", "illum", "intensity", "pattern", "mode", "gt_to_result_accuracy", "gt_to_result_coverage", "result_to_gt_accuracy", "result_to_gt_coverage"]

	result_path = os.path.join(args.output_dir, args.result)
	with open(result_path, "w") as f:
		writer = csv.writer(f)
		writer.writerow(header)

		results = compute_accuracy(
			args.dataset_config,
			args.output_dir, 
			args.gt_to_result_thresh, 
			args.result_to_gt_thresh, 
			args.remove_bg,
			scale=args.scale,
			phase=args.phase)

		for result in results:
			writer.writerow(result)
