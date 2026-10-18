import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map, convert_coord_map_to_depth_map
from vggt.utils.visual_util import predictions_to_ply, create_frustum, integrate_line_sets
import os, glob, re, argparse, cv2, json
from datetime import datetime
import open3d as o3d
import numpy as np
from natsort import natsorted



class Inferencer:
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


	def __call__(self, image_files, 
		cam_intrinsic=None, cam_extrinsic=None, 
		proj_intrinsic=None, proj_extrinsic=None, 
		pattern_files=None):
		images, cam_intrinsic_transforms = load_and_preprocess_images(image_files, mode="pad", return_transforms=True)
		images = images.to(self.device)

		if pattern_files is not None:
			patterns = load_and_preprocess_images(pattern_files, mode="pad").to(self.device)
			if len(patterns) != len(images) and len(patterns) == 1:
				patterns = patterns.expand(len(images), *patterns.shape[1:])
		else:
			patterns = None

		with torch.no_grad():
			with torch.cuda.amp.autocast(dtype=self.dtype):
				# Predict attributes including cameras, depth maps, and point maps.

				if cam_intrinsic is not None and cam_extrinsic is not None:
					cam_intrinsic_tensor = torch.from_numpy(cam_intrinsic).to(self.device).float()
					cam_extrinsic_tensor = torch.from_numpy(cam_extrinsic).to(self.device).float()
					cam_intrinsic_tensor = torch.stack([transform(cam_intrinsic_tensor) for transform in cam_intrinsic_transforms], dim=0).unsqueeze(0)
#					cam_intrinsic_tensor = cam_intrinsic_tensor.expand(1, len(images), *cam_intrinsic_tensor.shape[1:])
					cam_extrinsic_tensor = cam_extrinsic_tensor.unsqueeze(0)
					cam_params = (cam_extrinsic_tensor, cam_intrinsic_tensor)

				else:
					cam_params = None

				if proj_intrinsic is not None and proj_extrinsic is not None:
					proj_intrinsic_tensor = torch.from_numpy(proj_intrinsic).to(self.device).float()
					proj_extrinsic_tensor = torch.from_numpy(proj_extrinsic).to(self.device).float()
					proj_intrinsic_tensor = proj_intrinsic_tensor.expand(1, len(images), *proj_intrinsic_tensor.shape[1:])
					proj_extrinsic_tensor = proj_extrinsic_tensor.expand(1, len(images), *proj_extrinsic_tensor.shape[1:])
					proj_params = (proj_extrinsic_tensor, proj_intrinsic_tensor)

				else:
					proj_params = None

				start = datetime.now()
				predictions = self.model(images, patterns=patterns, cam_params=cam_params, proj_params=proj_params)
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
						predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension

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
				predictions["proj_intrinsic"] if proj_intrinsic is None else proj_intrinsic, 
				predictions["proj_extrinsic"] if proj_extrinsic is None else proj_extrinsic)
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

		return predictions, plyscene_pointmap, plyscene_depth, plyscene_coord, plyscene_pose, elapsed_ms


def get_intrinsics(angle_x, w, h):
	fx = fy = 0.5 * w / np.tan(0.5 * angle_x)
	cx, cy = w / 2, h / 2
	K = np.array([
		[fx, 0, cx],
		[0, fy, cy],
		[0, 0, 1]
	], dtype=np.float32)
	return K


def get_opencv_extrinsic_from_blender_cam(extri_blender):
	M_world_to_cam = np.linalg.inv(extri_blender)

	blender_to_opencv = np.diag([1, -1, -1])  # 3x3

	R = M_world_to_cam[:3, :3]
	t = M_world_to_cam[:3, 3:]

	R_cv = blender_to_opencv @ R
	t_cv = blender_to_opencv @ t

	# OpenCV Extrinsic: [R | t] (3x4)
	extri_opencv = np.hstack((R_cv, t_cv))
	return extri_opencv


def get_opencv_extrinsic_from_blender_proj(extri_blender):
	R_cv = extri_blender[:3, :3].T

	t_blender = extri_blender[:3, 3]
	t_cv = np.float32([1, -1, -1]) * t_blender

	extri_opencv = np.hstack((R_cv, t_cv.reshape(3, 1)))
	return extri_opencv


def graycode_to_coords(graycode, num_bits=8, coord_range=(0.0, 1.0)):
	"""
	Convert Gray code representation [H, W, num_bits*2] back to coordinates [H, W, 2].
	
	Args:
		graycode: ndarray of shape [H, W, num_bits*2]
		num_bits: number of bits per coordinate axis
		coord_range: (min_val, max_val) of coordinate normalization range
	"""
	H, W, C = graycode.shape
	assert C == num_bits * 2, f"Expected {num_bits*2} channels, got {C}"

	# Split x/y gray bits
	bits_x = graycode[..., :num_bits].astype(np.uint8)
	bits_y = graycode[..., num_bits:].astype(np.uint8)

	def bits_to_int(bits):
		"""Convert bit array (..., num_bits) to integer array (...)."""
		powers = (1 << np.arange(num_bits))
		return np.sum(bits * powers, axis=-1).astype(np.uint16)

	gray_x = bits_to_int(bits_x)
	gray_y = bits_to_int(bits_y)

	def gray_to_binary(gray):
		"""Convert gray code integer to binary integer."""
		binary = np.zeros_like(gray)
		for shift in range(num_bits):
			binary ^= (gray >> shift)
		return binary

	bin_x = gray_to_binary(gray_x)
	bin_y = gray_to_binary(gray_y)

	# Convert back to normalized coordinates
	max_val_int = 2**num_bits - 1
	min_val, max_val = coord_range

	coord_x = bin_x.astype(np.float32) / max_val_int * (max_val - min_val) + min_val
	coord_y = bin_y.astype(np.float32) / max_val_int * (max_val - min_val) + min_val

	coord = np.stack([coord_x, coord_y], axis=-1)  # [H, W, 2]
	return coord


def load_cam_param(param_path):
	_, ext = os.path.splitext(param_path)

	if ext == ".json":
		meta = json.load(open(param_path))

		cam_intri_opencv = get_intrinsics(meta["camera_angle_x"], meta["w"], meta["h"])
		cam_extri_opencv = [get_opencv_extrinsic_from_blender_cam(np.array(frame["transform_matrix"])) for frame in meta["frames"]]
		return cam_intri_opencv[np.newaxis], np.array(cam_extri_opencv)

	elif ext == ".npz":
		meta = np.load(param_path, allow_pickle=True)
		cam_intri_opencv = meta["cam_mat_0"]
		cam_extri_opencv = []
		i = 0
		while True:
			if "cam_pose_%d" % i in meta:
				cam_extri_opencv.append(meta["cam_pose_%d" % i])
				i += 1
			else:
				break
		return cam_intri_opencv[np.newaxis], np.array(cam_extri_opencv)

	else:
		raise NotImplementedError


def load_proj_param(param_path, resolution):
	_, ext = os.path.splitext(param_path)

	if ext == ".json":
		meta = json.load(open(param_path))

		proj = meta["projectors"][0]
		proj_intri_opencv = get_intrinsics(proj["proj_angle_x"], 1, 1)
		proj_extri_blender = np.array(proj["transform_matrix"])
		proj_extri_opencv = get_opencv_extrinsic_from_blender_proj(proj_extri_blender)
		return proj_intri_opencv[np.newaxis], proj_extri_opencv[np.newaxis]

	elif ext == ".npz":
		meta = np.load(param_path, allow_pickle=True)
		pfov = np.arctan(resolution[0] * 0.5 / meta["proj_mat_0"][0][0,0]) * 2
		proj_intri_opencv = get_intrinsics(pfov, 1, 1)
		return proj_intri_opencv[np.newaxis], meta["cam_proj_pose_0"][0][:3][np.newaxis]

	else:
		raise NotImplementedError



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_lists', type=str, nargs="+", default=[r"examples/tank/r_\d+\.png"])
	parser.add_argument('--output_dirs', type=str, nargs="+", default=["output/test/tank"])
	parser.add_argument('--weight_path', type=str)
	parser.add_argument('--conf_thresh', type=float, default=0.5)
	parser.add_argument('--param', type=str)
	parser.add_argument('--proj_resolution', type=int, nargs="+", default=[1024, 768])
	parser.add_argument('--pattern_files', type=str, nargs="+")
	args = parser.parse_args()

	inferencer = Inferencer(args.weight_path, conf_thres=args.conf_thresh)

	if args.param is not None:
		cam_intrinsic, cam_extrinsic = load_cam_param(args.param)
		proj_intrinsic, proj_extrinsic = load_proj_param(args.param, args.proj_resolution)
	else:
		cam_intrinsic = cam_extrinsic = None
		proj_intrinsic = proj_extrinsic = None

	for image_pattern, output_dir in zip(args.image_lists, args.output_dirs):
		dir_path = os.path.dirname(image_pattern)
		regex_str = os.path.basename(image_pattern)
		pattern = re.compile(regex_str)

		all_files = os.listdir(dir_path)
		image_files = natsorted(
			[os.path.join(dir_path, f) for f in all_files if pattern.fullmatch(f)]
		)

		num_pattern = re.compile(r"\d+")

		image_indices = []
		for fpath in image_files:
			filename = os.path.basename(fpath)
			m = num_pattern.search(filename)
			if m:
				image_indices.append(int(m.group()))
		image_indices = np.array(image_indices)

		if cam_extrinsic is not None:
			cam_extrinsic = cam_extrinsic[image_indices]

		predictions, plyscene_pointmap, plyscene_depth, plyscene_coord, plyscene_pose, elapsed_ms = \
			inferencer(image_files, cam_intrinsic, cam_extrinsic, proj_intrinsic, proj_extrinsic, args.pattern_files)

		os.makedirs(output_dir, exist_ok=True)
		np.savetxt(
			os.path.join(output_dir, "inference_time.txt"),
			[len(image_files), elapsed_ms]
		)

		o3d.io.write_point_cloud(os.path.join(output_dir, "result_pointmap.ply"), plyscene_pointmap)
		o3d.io.write_point_cloud(os.path.join(output_dir, "result_depth.ply"), plyscene_depth)
		o3d.io.write_line_set(os.path.join(output_dir, "result_pose.ply"), plyscene_pose)

		if "proj_extrinsic" in predictions and "proj_intrinsic" in predictions:
			np.savez_compressed(os.path.join(output_dir, "camera_params.npz"),
				intrinsic=predictions["intrinsic"], extrinsic=predictions["extrinsic"],
				proj_intrinsic=predictions["proj_intrinsic"], proj_extrinsic=predictions["proj_extrinsic"])
		else:
			np.savez_compressed(os.path.join(output_dir, "camera_params.npz"),
				intrinsic=predictions["intrinsic"], extrinsic=predictions["extrinsic"])

		if "coord" in predictions:
			for pi, coord in enumerate(predictions['coord']):
				if coord.shape[-1] != 2:
					coord = graycode_to_coords(coord)
				coord_vis = np.concatenate([coord, np.zeros_like(coord[...,:1])], axis=-1)
				cv2.imwrite(os.path.join(output_dir, "coord%03d.png" % pi), coord_vis * 255)

			o3d.io.write_point_cloud(os.path.join(output_dir, "result_coord.ply"), plyscene_coord)
