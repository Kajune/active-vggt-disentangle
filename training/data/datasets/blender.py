import os, sys, json
import numpy as np
import cv2
import logging
import random
import fnmatch
import open3d as o3d
from typing import *
import concurrent.futures

from data.dataset_util import *
from data.base_dataset import BaseDataset
from .compute_uw_params import compute_uw_params
from vggt.utils.geometry import convert_coord_map_to_depth_map, depth_to_world_coords_points, unproject_depth_map_to_point_map_torch


class BlenderDataset(BaseDataset):
	def __init__(
		self,
		common_conf,
		split: str = "train",
		scenes = None,
		dataset_dir: str = None,
		len_train: int = 100000,
		len_test: int = 10000,
		keep_original_shape : bool = False,
		include_projector : bool = False,
		proj_as_camera : bool = False,
		coded_coord : bool = False,
		identity_canonical_pose : bool = False,
		normalize_scale : bool = False,
		simulate_event : Optional[List[int]] = None,
	):
		super().__init__(common_conf=common_conf)

		self.debug = common_conf.debug
		self.training = common_conf.training
		self.get_nearby = common_conf.get_nearby
		self.load_depth = common_conf.load_depth
		self.inside_random = common_conf.inside_random
		self.allow_duplicate_img = common_conf.allow_duplicate_img
		self.keep_original_shape = keep_original_shape
		self.include_projector = include_projector
		self.proj_as_camera = proj_as_camera
		self.coded_coord = coded_coord
		self.identity_canonical_pose = identity_canonical_pose
		self.normalize_scale = normalize_scale
		self.simulate_event = simulate_event
		if hasattr(common_conf, "window"):
			self.window = common_conf.window
		else:
			self.window = None

		if dataset_dir is None:
			raise ValueError("dataset_dir must be specified.")

		self.dataset_dir = dataset_dir
		self.len_train = len_train if split == "train" else len_test
		sequence_list = []

		if scenes is None:
			for root, dirs, files in os.walk(self.dataset_dir):
				dirs.sort()
				files.sort()
				for file in files:
					if file == "transforms.json":
						sequence_list.append(root.replace(self.dataset_dir + "/", ""))
						break
		else:
			matched = []
			for root, dirs, files in os.walk(self.dataset_dir):
				dirs.sort()
				files.sort()
				rel_path = root.replace(self.dataset_dir + "/", "")
				for pattern in scenes:
					if fnmatch.fnmatch(rel_path, pattern):
						for file in files:
							if file == "transforms.json":
								matched.append(rel_path)
								break
						break
			sequence_list = matched

		for seq_name in sequence_list:
			metadata = json.load(open(os.path.join(self.dataset_dir, seq_name, "transforms.json")))
			if len(metadata["frames"]) == 0:
				sequence_list.remove(seq_name)

		print(f"[{split}] Sequence list:")
		for seq in sequence_list:
			print("\t", seq)

		self.sub_sequence_list = []
		if self.window is not None:
			for seq_name in sequence_list:
				metadata = json.load(open(os.path.join(self.dataset_dir, seq_name, "transforms.json")))

# index			0 	1 	2 	3 	4 	5 	6 	7 	8 	9 	10 	11	12	13	14 	15	16	17	18 	19 	20 	21 	22	23	24
# interval:	2	0 		1 		2 		3 		4 		5 		6 		7 		8 		9 		10 		11		12
# begin: 3 				 		 		3 		4 		5 		6 		7 		8 		9 		10 		11		12
# end: 10 				 		 		3 		4 		5 		6 		7 		8 		9
# step: 3 								<--------------->		<--------------->
# n_views: 2 							<------->				<------->

				interval, begin, end, step, n_views = self.window
				seq_len = len(metadata["frames"])
				end = min(end, seq_len)

				for i in range(begin, end, step):
					ids = np.arange(i * interval, (i + n_views) * interval, interval)
					ids = ids[ids < seq_len]
					if len(ids) == n_views:
						self.sub_sequence_list.append((seq_name, ids))

		else:
			for seq_name in sequence_list:
				self.sub_sequence_list.append((seq_name, None))

		self.sequence_list_len = len(self.sub_sequence_list)

		if self.sequence_list_len == 0:
			raise RuntimeError(f"No transforms.json found in {self.dataset_dir}")


	def __len__(self):
		if self.inside_random:
			return self.len_train
		else:
			return self.sequence_list_len


	def _get_intrinsics(self, angle_x, w, h):
		fx = fy = 0.5 * w / np.tan(0.5 * angle_x)
		cx, cy = w / 2, h / 2
		K = np.array([
			[fx, 0, cx],
			[0, fy, cy],
			[0, 0, 1]
		], dtype=np.float32)
		return K


	def _compute_3d_points(self, depth, mask, K, extri):
		"""
		Cam/world points を depth, mask, K, extri から生成する
		"""
		h, w = depth.shape
		i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
		pts_cam = np.stack([(i - K[0, 2]) * depth / K[0, 0],
							(j - K[1, 2]) * depth / K[1, 1],
							depth], axis=-1)

		pts_cam = pts_cam[mask > 0]
		pts_world = (extri[:3, :3] @ pts_cam.T + extri[:3, 3:4]).T
		return pts_cam, pts_world


	def get_opencv_extrinsic_from_blender_cam(self, extri_blender):
		M_world_to_cam = np.linalg.inv(extri_blender)

		blender_to_opencv = np.diag([1, -1, -1])  # 3x3

		R = M_world_to_cam[:3, :3]
		t = M_world_to_cam[:3, 3:]

		R_cv = blender_to_opencv @ R
		t_cv = blender_to_opencv @ t

		# OpenCV Extrinsic: [R | t] (3x4)
		extri_opencv = np.hstack((R_cv, t_cv))
		return extri_opencv


	def get_opencv_extrinsic_from_blender_proj(self, extri_blender):
#		axis_convert = np.array([
#			[1, 0,  0, 0],
#			[0, -1,  0, 0],
#			[0, 0, -1, 0],
#			[0, 0,  0, 1]
#		], dtype=np.float32)
		
#		return (axis_convert @ extri_blender @ np.linalg.inv(axis_convert))[:3]

		R_cv = extri_blender[:3, :3].T

		t_blender = extri_blender[:3, 3]
		t_cv = np.float32([1, -1, -1]) * t_blender

		extri_opencv = np.hstack((R_cv, t_cv.reshape(3, 1)))
		return extri_opencv


	def get_frame_num(self, seq_index: int = None):
		seq_name = self.sub_sequence_list[seq_index][0]
		metadata = json.load(open(os.path.join(self.dataset_dir, seq_name, "transforms.json")))
		return len(metadata["frames"])


	def _process_single_frame(self, anno, seq_name, metadata, target_image_shape, 
							  intri_fixed, proj_intrinsics, proj_extrinsics, patterns_imgs, cam_dists):
		filepath = anno["file_path"]
		if os.path.splitext(filepath)[1] == "":
			filepath += ".png"

		image_path = os.path.join(self.dataset_dir, seq_name, filepath)
		image = read_image_cv2(image_path)
		
		img_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
		mask = (img_raw[...,-1] / 255 > 0.5) if img_raw.shape[-1] == 4 else None

		depth_map = None
		if self.load_depth:
			depth_path = anno["depth_path"]
			if os.path.splitext(depth_path)[1] == "":
				depth_path += ".png"
			depth_path = os.path.join(self.dataset_dir, seq_name, depth_path)
			depth_map = 1 - read_depth(depth_path, 1)
			depth_map[depth_map == 1] = 0
			if mask is not None:
				depth_map[~mask] = 0
			else:
				mask = depth_map > 0
			depth_map *= anno["depth_far"]

		original_size = np.array(image.shape[:2])

		if intri_fixed is not None:
			intri_opencv = intri_fixed
		else:
			intri_opencv = self._get_intrinsics(metadata["camera_angle_x"], metadata["w"], metadata["h"])

		extri_blender = np.array(anno["transform_matrix"])
		extri_opencv = self.get_opencv_extrinsic_from_blender_cam(extri_blender)

		coord = np.zeros_like(image)
		proj_validity = np.zeros_like(image[...,:1])
		
		if depth_map is not None and "projectors" in metadata and self.include_projector:
			cam_coords_points = depth_to_cam_coords_points(depth_map, intri_opencv)
			cam_coords_points_homo = np.concatenate([cam_coords_points, np.ones_like(cam_coords_points[...,:1])], axis=-1)
			proj_pose_homo = np.linalg.inv(np.concatenate([proj_extrinsics[0], np.array([0,0,0,1])[None,:]], axis=0))
			cam_coords_points_homo = (proj_pose_homo @ cam_coords_points_homo.reshape(-1,4).T).T.reshape(*cam_coords_points_homo.shape)[...,:3]
			reprojected_points = (proj_intrinsics[0] @ cam_coords_points_homo.reshape(-1,3).T).T.reshape(*cam_coords_points_homo.shape)
			coord = reprojected_points[...,:2] / reprojected_points[...,2:]
			
			proj_validity = ((0 <= coord[...,0]) & (coord[...,0] <= 1) & (0 <= coord[...,1]) & (coord[...,1] <= 1) & (mask > 0))[...,np.newaxis]
			coord[~proj_validity[...,0]] *= 0
			if self.coded_coord:
				coord = coords_to_graycode(coord)

		if np.max(np.abs(cam_dists)) > 1e-4:
			image = cv2.undistort(image, intri_opencv, cam_dists, None, intri_opencv)
			map1, map2 = cv2.initUndistortRectifyMap(intri_opencv, cam_dists, None, intri_opencv, image.shape[:2][::-1], cv2.CV_32FC1)
			depth_map = cv2.remap(depth_map, map1, map2, interpolation=cv2.INTER_NEAREST)

		(
			image, depth_map, extri_opencv, intri_opencv,
			world_coords_points, cam_coords_points, point_mask,
			[coord, proj_validity], _,
		) = self.process_one_image(
			image, depth_map, extri_opencv, intri_opencv,
			original_size, target_image_shape, [coord, proj_validity], filepath=filepath,
		)

		if "pattern_file_path" in anno:
			pattern_filepath = os.path.join(self.dataset_dir, seq_name, anno["pattern_file_path"] + ".png")
			pattern = read_image_cv2(pattern_filepath)
			pattern = cv2.resize(pattern, (image.shape[1], image.shape[0]))
		elif len(patterns_imgs) > 0:
			pattern = cv2.resize(patterns_imgs[0], (image.shape[1], image.shape[0]))
		else:
			pattern = np.zeros_like(image)

		return {
			"image": image, "depth": depth_map, "pattern": pattern, "coord": coord,
			"proj_validity": proj_validity[...,0], "extrinsic": extri_opencv,
			"intrinsic": intri_opencv, "cam_point": cam_coords_points,
			"world_point": world_coords_points, "point_mask": point_mask,
			"image_path": image_path, "original_size": original_size
		}


	def get_data(
		self,
		seq_index: int = None,
		img_per_seq: int = None,
		seq_name: str = None,
		ids: list = None,
		aspect_ratio: float = 1.0,
		debug: bool = False,
	) -> dict:
		"""
		Retrieve data for a specific sequence.

		Args:
			seq_index (int): Index of the sequence to retrieve.
			img_per_seq (int): Number of images per sequence.
			seq_name (str): Name of the sequence.
			ids (list): Specific IDs to retrieve.
			aspect_ratio (float): Aspect ratio for image processing.

		Returns:
			dict: A batch of data including images, depths, and other metadata.
		"""
		self.aug_params = {}
		if self.inside_random:
			seq_index = random.randint(0, self.sequence_list_len - 1)

		start_ind = 0
		if seq_name is None:
			seq_name, ids = self.sub_sequence_list[seq_index]

		metadata = json.load(open(os.path.join(self.dataset_dir, seq_name, "transforms.json")))

		if ids is None:
			if img_per_seq is None:
				ids = np.arange(start_ind, len(metadata["frames"]))
			else:
				ids = np.random.choice(
					len(metadata["frames"]), img_per_seq, replace=self.allow_duplicate_img
				)
		if isinstance(ids, str) and ids == "all":
			ids = np.arange(len(metadata["frames"]))

		annos = [metadata["frames"][i] for i in ids]
		cam_dists = np.float32([metadata["k1"], metadata["k2"], metadata["p1"], metadata["p2"]])

		if self.keep_original_shape:
			target_image_shape = np.array([self.img_size, self.img_size])
		else:
			target_image_shape = self.get_target_shape(aspect_ratio)

		if "refractive_interface" in metadata:
			metadata = compute_uw_params(metadata, 10000, 0, 10)

		patterns_imgs = []
		pattern_shapes = []
		proj_intrinsics = []
		proj_extrinsics = []
		if "projectors" in metadata and self.include_projector:
			projectors = metadata["projectors"]

			for proj in projectors:
				pattern_path = os.path.join(self.dataset_dir, seq_name, proj["pattern"])
				pattern = read_image_cv2(pattern_path) / 255
#				pattern = cv2.resize(pattern, target_image_shape[::-1])
				patterns_imgs.append(pattern)
				pattern_shapes.append(pattern.shape[:2])

				if "fx" in proj and "fy" in proj and "cx" in proj and "cy" in proj:
					proj_intri_opencv = np.float32([
						[proj["fx"] / pattern.shape[1], 0, proj["cx"] / pattern.shape[1]],
						[0, proj["fy"] / pattern.shape[0], proj["cy"] / pattern.shape[0]],
						[0, 0, 1]
					])

				else:
					proj_intri_opencv = self._get_intrinsics(proj["proj_angle_x"], 1, 1)

				proj_extri_blender = np.array(proj["transform_matrix"])
				proj_extri_opencv = self.get_opencv_extrinsic_from_blender_proj(proj_extri_blender)
				proj_intrinsics.append(proj_intri_opencv)
				proj_extrinsics.append(proj_extri_opencv)

		images = []
		depths = []
		patterns = []
		coords = []
		proj_validities = []
		cam_points = []
		world_points = []
		point_masks = []
		extrinsics = []
		intrinsics = []
		image_paths = []
		proj_intrinsics_per_img = []
		proj_extrinsics_per_img = []
		original_sizes = []
		canonical_pose = None

		intri_fixed = None
		if "fx" in metadata and "fy" in metadata and "cx" in metadata and "cy" in metadata:
			intri_fixed = np.float32([
				[metadata["fx"], 0, metadata["cx"]],
				[0, metadata["fy"], metadata["cy"]],
				[0, 0, 1],
			])

		with concurrent.futures.ThreadPoolExecutor() as executor:
			futures = [
				executor.submit(
					self._process_single_frame, 
					anno, seq_name, metadata, target_image_shape,
					intri_fixed, proj_intrinsics, proj_extrinsics, 
					patterns_imgs, cam_dists
				) for anno in annos
			]

			results = [f.result() for f in futures]

		for i, res in enumerate(results):
			extri_opencv = res["extrinsic"]
			if self.identity_canonical_pose:
				extri_opencv_full = np.eye(4)
				extri_opencv_full[:3] = extri_opencv

				if canonical_pose is None:
					canonical_pose = extri_opencv_full.copy()
					extri_opencv_full = np.eye(4)
				else:
					extri_opencv_full = extri_opencv_full @ np.linalg.inv(canonical_pose)

				extri_opencv = extri_opencv_full[:3]

			images.append(res["image"])
			depths.append(res["depth"])
			patterns.append(res["pattern"])
			coords.append(res["coord"])
			proj_validities.append(res["proj_validity"])
			extrinsics.append(extri_opencv)
			intrinsics.append(res["intrinsic"])
			cam_points.append(res["cam_point"])
			world_points.append(res["world_point"])
			point_masks.append(res["point_mask"])
			image_paths.append(res["image_path"])
			original_sizes.append(res["original_size"])

			if self.include_projector:
				proj_intrinsics_per_img.append(proj_intrinsics[0])
				proj_extrinsics_per_img.append(proj_extrinsics[0])

		set_name = "blender"
		if self.sub_sequence_list[seq_index][1] is not None:
			seq_name += f"_{ids}"

		batch = {
			"seq_name": set_name + "_" + seq_name,
			"ids": ids,
			"frame_num": len(extrinsics),
			"images": images,
			"depths": depths,
			"extrinsics": extrinsics,
			"intrinsics": intrinsics,
			"cam_points": cam_points,
			"world_points": world_points,
			"point_masks": point_masks,
			"original_sizes": original_sizes,
			"tracks": None,
		}

		if "projectors" in metadata and self.include_projector:
			batch["patterns"] = patterns
			batch["coords"] = coords
#			batch["proj_depths"] = # TODO
			batch["proj_validities"] = proj_validities
			batch["proj_intrinsics"] = proj_intrinsics_per_img
			batch["proj_extrinsics"] = proj_extrinsics_per_img

			proj_extrinsics_world = []
			for cam_extri, proj_extri in zip(batch["extrinsics"], batch["proj_extrinsics"]):
				cam_extri_homo = np.eye(4)
				cam_extri_homo[:3] = cam_extri
				proj_extri_homo = np.eye(4)
				proj_extri_homo[:3] = proj_extri
				proj_extri_world = cam_extri_homo @ proj_extri_homo
				proj_extrinsics_world.append(proj_extri_world[:3])

			if self.proj_as_camera:
#				batch["frame_num"] = batch["frame_num"] * 2
				batch["images"] = batch["images"] + [batch["patterns"][0]]
#				batch["depths"] = batch["depths"] + batch["proj_deptsh"]
#				batch["extrinsics"] = batch["extrinsics"] + proj_extrinsics_world
#				batch["intrinsics"] = batch["intrinsics"] + batch["proj_intrinsics"]
#				batch["cam_points"] = batch["cam_points"] + # TODO
#				batch["world_points"] = batch["world_points"] + batch["world_points"]
#				batch["point_masks"] = batch["point_masks"] + batch["point_masks"]
#				batch["original_sizes"] = batch["original_sizes"] + pattern_shapes

		if self.normalize_scale:
			all_points = np.concatenate([pts.reshape(-1, 3) for pts in world_points], axis=0)
			mass_center = np.mean(all_points, axis=0)
			distances = np.linalg.norm(all_points - mass_center, axis=-1)
			mean_scale = np.mean(distances)

			if mean_scale < 1e-8:
				mean_scale = 1.0

			for i in range(len(world_points)):
				for key in ["world_points", "depths", "cam_points"]:
					batch[key][i] /= mean_scale

				batch["extrinsics"][i][:3,3] /= mean_scale

				if "proj_extrinsics" in batch:
					batch["proj_extrinsics"][i][:3,3] /= mean_scale

		if debug:
			debug_file_name = ((set_name + "_" + seq_name).replace("/", "-"))

			pcd = o3d.geometry.PointCloud()
			all_points = np.concatenate([pts.reshape(-1, 3) for pts in world_points], axis=0)
			all_colors = np.concatenate([img.reshape(-1, 3) for img in images], axis=0) / 255.0
			pcd.points = o3d.utility.Vector3dVector(all_points)
			pcd.colors = o3d.utility.Vector3dVector(all_colors)
			o3d.io.write_point_cloud("output/%s.ply" % debug_file_name, pcd)

			if self.coded_coord:
				coords = np.array([graycode_to_coords(coord) for coord in coords])

			coord_vis = np.concatenate([coords[0], np.zeros_like(coords[0][...,:1])], axis=-1)
			cv2.imwrite("output/%s_coord.png" % debug_file_name, coord_vis * 255)

			import torch
			depth_map_from_coord = convert_coord_map_to_depth_map(np.array(coords), np.array(intrinsics), 
																np.array(proj_intrinsics), np.array(proj_extrinsics))
			points_from_coord = unproject_depth_map_to_point_map_torch(torch.from_numpy(depth_map_from_coord).float(), 
				torch.from_numpy(np.float32(extrinsics)), torch.from_numpy(np.float32(intrinsics))).numpy()
#			points_from_coord, _, _ = depth_to_world_coords_points(depth_map_from_coord[0,...,0], extrinsics[0], intrinsics[0])
			pcd = o3d.geometry.PointCloud()
			all_points = np.concatenate([pts.reshape(-1, 3) for pts in points_from_coord], axis=0)
			all_colors = np.concatenate([img.reshape(-1, 3) for img in images], axis=0) / 255.0
			pcd.points = o3d.utility.Vector3dVector(all_points)
			pcd.colors = o3d.utility.Vector3dVector(all_colors)
			o3d.io.write_point_cloud("output/%s_coord.ply" % debug_file_name, pcd)

			h, w = pattern_shapes[0]
			coord_debug = (coords[0] * (w, h)).astype(np.int32)
			coord_debug[..., 0] = np.clip(coord_debug[..., 0], 0, w - 1)
			coord_debug[..., 1] = np.clip(coord_debug[..., 1], 0, h - 1)
			coord_debug = patterns_imgs[0][coord_debug[..., 1], coord_debug[..., 0]]
			cv2.imwrite("output/%s_coord_debug.png" % debug_file_name, coord_debug[...,::-1] * 255)
			cv2.imwrite("output/%s_img.png" % debug_file_name, images[0][...,::-1])
			cv2.imwrite("output/%s_depth.png" % debug_file_name, (depths[0] - np.min(depths[0])) / (np.max(depths[0]) - np.min(depths[0])) * 255)

		return batch
