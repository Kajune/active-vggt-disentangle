import torch
import torch.distributed as dist

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map, convert_coord_map_to_depth_map
from vggt.utils.visual_util import predictions_to_ply, create_frustum, integrate_line_sets
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
import os, sys, glob, re, argparse, cv2, json
from datetime import datetime
import open3d as o3d
import numpy as np
from natsort import natsorted
from hydra import initialize, compose
from hydra.utils import instantiate
from evaluate import compute_chamfer_distance
from evaluate_pose import to_homogeneous, compute_param_accuracy



class Tester:
	def __init__(self, weight_path=None, conf_thres=0.5):
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
		self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

		# Initialize the model and load the pretrained weights.
		# This will automatically download the model weights the first time it's run, which may take a while.
		if weight_path is not None:
			print("Loading", weight_path)
			self.model = VGGT.from_pretrained(weight_path).to(self.device)
#			with open(weight_path, "rb") as f:
#				checkpoint = torch.load(f, map_location="cpu")
#			self.model.load_state_dict(checkpoint["model"], strict=False)
#			self.model.to(self.device)

		else:
			self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)

		self.model.eval()

		self.conf_thres = conf_thres


	def __call__(self, batch):
		with torch.no_grad():
			for key in batch.keys():
				if isinstance(batch[key], torch.Tensor):
					batch[key] = batch[key].cuda()

			images = batch["images"]
			patterns = batch["patterns"] if "patterns" in batch else None

			start = datetime.now()
			predictions = self.model(images=images.unsqueeze(0), patterns=patterns.unsqueeze(0),
				proj_params=(batch["proj_extrinsics"].unsqueeze(0), batch["proj_intrinsics"].unsqueeze(0)) 
					if "proj_intrinsics" in batch and "proj_extrinsics" in batch else None,
				cam_params=(batch["extrinsics"].unsqueeze(0), batch["intrinsics"].unsqueeze(0)),
			)

			end = datetime.now()
			elapsed_ms = (end - start).total_seconds() * 1000

			# Convert pose encoding to extrinsic and intrinsic matrices
			print("Converting pose encoding to extrinsic and intrinsic matrices...")
			extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], (images.shape[3], images.shape[2]))
			predictions["extrinsic"] = extrinsic
			predictions["intrinsic"] = intrinsic

			if "proj_pose_enc" in predictions:
				proj_extrinsic_, proj_intrinsic_ = pose_encoding_to_extri_intri(predictions["proj_pose_enc"], (1, 1))
				predictions["proj_extrinsic"] = proj_extrinsic_
				predictions["proj_intrinsic"] = proj_intrinsic_

			# Convert tensors to numpy
			for key in predictions.keys():
				if isinstance(predictions[key], torch.Tensor):
					predictions[key] = predictions[key].cpu().numpy().squeeze(0)
			for key in batch.keys():
				if isinstance(batch[key], torch.Tensor):
					batch[key] = batch[key].cpu().numpy()

		# Generate world points from depth map
		print("Computing world points from depth map...")
		depth_map = predictions["depth"]  # (S, H, W, 1)
		world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
		predictions["world_points_from_depth"] = world_points

		if "coord" in predictions and "world_points_from_coord" not in predictions:
			print("Computing world points from coord map...")
			coord_map = predictions["coord"]  # (S, H, W, 2)
			if coord_map.shape[-1] != 2:
				coord_map = np.array([graycode_to_coords(coord) for coord in coord_map])
			depth_map = convert_coord_map_to_depth_map(coord_map, predictions["intrinsic"], 
				predictions["proj_intrinsic"] if "proj_intrinsics" not in batch else batch["proj_intrinsics"], 
				predictions["proj_extrinsic"] if "proj_extrinsics" not in batch else batch["proj_extrinsics"])
			world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
			predictions["world_points_from_coord"] = world_points

		# Convert predictions to ply
		plyscene_pointmap = predictions_to_ply(
			predictions,
			conf_thres=self.conf_thres * 100,
			sample_by_conf=False,
		)
		plyscene_depth = predictions_to_ply(
			predictions,
			conf_thres=self.conf_thres * 100,
			sample_by_conf=False,
			prediction_mode="Depthmap and Camera Branch",
		)
		plyscene_coord = predictions_to_ply(
			predictions,
			conf_thres=self.conf_thres * 100,
			sample_by_conf=False,
			prediction_mode="Coordmap and Camera Branch",
		)
		plyscene_pointmap_gt = predictions_to_ply(
			batch,
			conf_thres=self.conf_thres * 100,
			sample_by_conf=False,
		)


		pose_geometries = []
		if "intrinsic" in predictions and "extrinsic" in predictions:
			for image, cam_mat, cam_pose in zip(images, predictions["intrinsic"], predictions["extrinsic"]):
				cam = create_frustum(cam_mat, image.shape[2], image.shape[1])
				pose = np.eye(4)
				pose[:3] = cam_pose
				cam.transform(pose)
				pose_geometries.append(cam)

		if "proj_intrinsic" in predictions and "proj_extrinsic" in predictions:
			for proj_mat, proj_pose in zip(predictions["proj_intrinsic"], predictions["proj_extrinsic"]):
				proj = create_frustum(proj_mat, 1, 1)
				pose = np.eye(4)
				pose[:3] = proj_pose
				proj.transform(np.linalg.inv(pose))
				pose_geometries.append(proj)
				break

		plyscene_pose = integrate_line_sets(pose_geometries)

		pose_geometries_gt = []
		for image, cam_mat, cam_pose in zip(images, batch["intrinsics"], batch["extrinsics"]):
			cam = create_frustum(cam_mat, image.shape[2], image.shape[1])
			pose = np.eye(4)
			pose[:3] = cam_pose
			cam.transform(pose)
			pose_geometries_gt.append(cam)

		for proj_mat, proj_pose in zip(batch["proj_intrinsics"], batch["proj_extrinsics"]):
			proj = create_frustum(proj_mat, 1, 1)
			pose = np.eye(4)
			pose[:3] = proj_pose
			proj.transform(np.linalg.inv(pose))
			pose_geometries_gt.append(proj)
			break

		plyscene_pose_gt = integrate_line_sets(pose_geometries_gt)

		return batch, predictions, plyscene_pointmap, plyscene_depth, plyscene_coord, plyscene_pose, plyscene_pointmap_gt, plyscene_pose_gt, elapsed_ms


def process_batch(batch, data_conf):
	# Normalize camera extrinsics and points. The function returns new tensors.
	normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths, normalized_proj_extrinsics = \
		normalize_camera_extrinsics_and_points_batch(
			extrinsics=batch["extrinsics"].unsqueeze(0),
			cam_points=batch["cam_points"].unsqueeze(0),
			world_points=batch["world_points"].unsqueeze(0),
			depths=batch["depths"].unsqueeze(0),
			point_masks=batch["point_masks"].unsqueeze(0),
			proj_extrinsics=batch["proj_extrinsics"].unsqueeze(0) if "proj_extrinsics" in batch else None,
			scale_by_points=data_conf.common_config.scale_by_points,
			scale_by_proj_baseline=data_conf.common_config.scale_by_proj_baseline,
		)

	# Replace the original values in the batch with the normalized ones.
	batch["extrinsics"] = normalized_extrinsics.squeeze(0)
	batch["cam_points"] = normalized_cam_points.squeeze(0)
	batch["world_points"] = normalized_world_points.squeeze(0)
	batch["depths"] = normalized_depths.squeeze(0)

	if "proj_extrinsics" in batch:
		batch["proj_extrinsics"] = normalized_proj_extrinsics.squeeze(0)

	return batch


def remove_bg(pcd):
	if not pcd.has_colors():
		return pcd
	colors = np.asarray(pcd.colors)
	points = np.asarray(pcd.points)
	mask = np.all(colors > [0, 0, 0], axis=1)
	pcd.points = o3d.utility.Vector3dVector(points[mask])
	pcd.colors = o3d.utility.Vector3dVector(colors[mask])
	return pcd


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="default")
	parser.add_argument('--weight_path', type=str)
	parser.add_argument('--conf_thresh', type=float, default=0.5)
	parser.add_argument('--num_images', type=int)
	parser.add_argument('--window', type=int, nargs=5)
	parser.add_argument('--gt_to_result_thresh', type=float, default=0.05)
	parser.add_argument('--result_to_gt_thresh', type=float, default=0.05)
	parser.add_argument('--scale', type=float, default=1e3)
	parser.add_argument('--remove_bg', type=bool, default=True)
	parser.add_argument('--output_dir', type=str, default="test")
	args = parser.parse_args()

	tester = Tester(args.weight_path, conf_thres=args.conf_thresh)

	with initialize(version_base=None, config_path="config"):
		cfg = compose(config_name=args.config)

	cfg.max_img_per_gpu = 1 	# Ensure batch_size=1
	cfg.data.test.common_config.inside_random = False

	if args.window is not None:
		cfg.data.test.common_config.window = args.window

	dist.init_process_group(backend="nccl")
	rank = dist.get_rank()
	device = torch.device(f"cuda:{rank}")

	test_dataset = instantiate(cfg.data.test, _recursive_=False).dataset
#	test_loader = test_dataset.get_loader(epoch=0)

	if args.num_images is not None:
		n_views = f"{args.num_images}views"
	elif args.window is not None:
		n_views = f"{args.window}slices"
	else:
		n_views = "allviews"
	output_dir_root = os.path.join("output", cfg.exp_name, args.output_dir, n_views)
	os.makedirs(output_dir_root, exist_ok=True)
	acc_file = open(os.path.join(output_dir_root, "accuracy.csv"), "w")

	for i in range(len(test_dataset)):
		batch = test_dataset[(i, args.num_images, 1.0)]
		batch = process_batch(batch, cfg.data.test)

		batch, predictions, plyscene_pointmap, plyscene_depth, plyscene_coord, plyscene_pose, \
			plyscene_pointmap_gt, plyscene_pose_gt, elapsed_ms = tester(batch)

		output_dir = os.path.join(output_dir_root, batch["seq_name"])
		os.makedirs(output_dir, exist_ok=True)
		path_list = batch["seq_name"].split("/")
		accuracy = [cfg.exp_name, n_views, *path_list]
		if i == 0:
			acc_file.write(", ".join(["exp_name", "n_views"] + [""] * len(path_list)) + ", ")

		np.savetxt(
			os.path.join(output_dir, "inference_time.txt"),
			[len(batch["images"]), elapsed_ms]
		)

		if args.remove_bg:
			plyscene_pointmap = remove_bg(plyscene_pointmap)
			plyscene_depth = remove_bg(plyscene_depth)

		o3d.io.write_point_cloud(os.path.join(output_dir, "result_pointmap.ply"), plyscene_pointmap)
		o3d.io.write_point_cloud(os.path.join(output_dir, "result_depth.ply"), plyscene_depth)
		o3d.io.write_line_set(os.path.join(output_dir, "result_pose.ply"), plyscene_pose)
		o3d.io.write_point_cloud(os.path.join(output_dir, "gt_pointmap.ply"), plyscene_pointmap_gt)
		o3d.io.write_line_set(os.path.join(output_dir, "gt_pose.ply"), plyscene_pose_gt)

		accuracy += [*compute_chamfer_distance(plyscene_pointmap, plyscene_pointmap_gt, args.gt_to_result_thresh, args.result_to_gt_thresh, args.scale)]
		if i == 0:
			acc_file.write(", ".join(["gt_to_result_accuracy_point", "gt_to_result_coverage_point", "result_to_gt_accuracy_point", "result_to_gt_coverage_point"]) + ", ")

		accuracy +=	[*compute_chamfer_distance(plyscene_depth, plyscene_pointmap_gt, args.gt_to_result_thresh, args.result_to_gt_thresh, args.scale)]
		if i == 0:
			acc_file.write(", ".join(["gt_to_result_accuracy_depth", "gt_to_result_coverage_depth", "result_to_gt_accuracy_depth", "result_to_gt_coverage_depth"]) + ", ")

		if "coord" in predictions:
			for pi, coord in enumerate(predictions['coord']):
				if coord.shape[-1] != 2:
					coord = graycode_to_coords(coord)
				coord_vis = np.concatenate([coord, np.zeros_like(coord[...,:1])], axis=-1)
				cv2.imwrite(os.path.join(output_dir, "coord%03d.png" % pi), coord_vis * 255)

			for pi, coord in enumerate(batch['coords']):
				if coord.shape[-1] != 2:
					coord = graycode_to_coords(coord)
				coord_vis = np.concatenate([coord, np.zeros_like(coord[...,:1])], axis=-1)
				cv2.imwrite(os.path.join(output_dir, "coord%03d_gt.png" % pi), coord_vis * 255)

			if args.remove_bg:
				plyscene_coord = remove_bg(plyscene_coord)

			o3d.io.write_point_cloud(os.path.join(output_dir, "result_coord.ply"), plyscene_coord)
			accuracy += [*compute_chamfer_distance(plyscene_coord, plyscene_pointmap_gt, args.gt_to_result_thresh, args.result_to_gt_thresh, args.scale)]

		else:
			accuracy += [None, None, None, None]
		if i == 0:
			acc_file.write(", ".join(["gt_to_result_accuracy_coord", "gt_to_result_coverage_coord", "result_to_gt_accuracy_coord", "result_to_gt_coverage_coord"]) + ", ")


		if "proj_extrinsics" in batch and "proj_intrinsics" in batch:
			np.savez_compressed(os.path.join(output_dir, "camera_params_gt.npz"),
				intrinsic=batch["intrinsics"], extrinsic=batch["extrinsics"],
				proj_intrinsic=batch["proj_intrinsics"], proj_extrinsic=batch["proj_extrinsics"])
		else:
			np.savez_compressed(os.path.join(output_dir, "camera_params_gt.npz"),
				intrinsic=batch["intrinsics"], extrinsic=batch["extrinsics"])

		if "proj_extrinsic" in predictions and "proj_intrinsic" in predictions:
			np.savez_compressed(os.path.join(output_dir, "camera_params.npz"),
				intrinsic=predictions["intrinsic"], extrinsic=predictions["extrinsic"],
				proj_intrinsic=predictions["proj_intrinsic"], proj_extrinsic=predictions["proj_extrinsic"])
		else:
			np.savez_compressed(os.path.join(output_dir, "camera_params.npz"),
				intrinsic=predictions["intrinsic"], extrinsic=predictions["extrinsic"])

		cam_intrinsic = predictions["intrinsic"]
		cam_extrinsic = predictions["extrinsic"]
		cam_intrinsic_gt = np.array(batch["intrinsics"])
		cam_extrinsic_gt = np.array(batch["extrinsics"])
		proj_intrinsic = predictions["proj_intrinsic"][:1] if "proj_intrinsic" in predictions else None
		proj_extrinsic = predictions["proj_extrinsic"][:1] if "proj_extrinsic" in predictions else None
		proj_intrinsic_gt = np.array(batch["proj_intrinsics"])[:1] if "proj_intrinsics" in batch else None
		proj_extrinsic_gt = np.array(batch["proj_extrinsics"])[:1] if "proj_extrinsics" in batch else None

		if "patterns" in batch:
			pH, pW, _ = batch["patterns"][0].shape
			proj_intrinsic_gt[:,0] /= pW
			proj_intrinsic_gt[:,1] /= pH

		cam_extrinsic = np.array([to_homogeneous(E) for E in cam_extrinsic])
		cam_extrinsic_gt = np.array([to_homogeneous(E) for E in cam_extrinsic_gt])
		if proj_extrinsic is not None and proj_extrinsic_gt is not None:
			proj_extrinsic = np.array([to_homogeneous(E) for E in proj_extrinsic])
			proj_extrinsic_gt = np.array([to_homogeneous(E) for E in proj_extrinsic_gt])

		image = np.array(batch["images"])[0]
		image_resolution = (image.shape[1], image.shape[0])
		cam_focal_error, cam_rot_error, cam_trans_error, align_matrix = \
			compute_param_accuracy(
				cam_intrinsic, cam_intrinsic_gt, 
				cam_extrinsic, cam_extrinsic_gt, 
				image_resolution, align=len(cam_extrinsic_gt) >= 3
			)
		accuracy += [cam_focal_error, cam_rot_error, cam_trans_error]
		if i == 0:
			acc_file.write(", ".join(["cam_focal_error", "cam_rot_error", "cam_trans_error"]) + ", ")

		if "proj_extrinsic" in predictions and "proj_intrinsic" in predictions:
			proj_resolution = (1, 1)
			proj_extrinsic = np.array([align_matrix @ E for E in proj_extrinsic])
			proj_focal_error, proj_rot_error, proj_trans_error, _ = \
				compute_param_accuracy(
					proj_intrinsic, proj_intrinsic_gt, 
					proj_extrinsic, proj_extrinsic_gt, 
					proj_resolution
				)
			accuracy += [proj_focal_error, proj_rot_error, proj_trans_error]
		else:
			accuracy += [None, None, None]
		if i == 0:
			acc_file.write(", ".join(["proj_focal_error", "proj_rot_error", "proj_trans_error"]) + "\n")

		acc_file.write(", ".join([str(acc) for acc in accuracy]) + "\n")

