import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import open3d as o3d


def create_camera_frustum(K, pose, image_shape, color=[1, 0, 0], scale=0.05):
	"""
	K: 3x3 intrinsic matrix
	pose: 4x4 extrinsic matrix
	image_shape: (H, W) tuple
	color: RGB color of the frustum lines
	scale: scalar to resize frustum size
	"""
	h, w = image_shape
	fx, fy = K[0, 0], K[1, 1]
	cx, cy = K[0, 2], K[1, 2]

	corners = np.array([
		[(0 - cx) / fx, (0 - cy) / fy, 1],
		[(w - cx) / fx, (0 - cy) / fy, 1],
		[(w - cx) / fx, (h - cy) / fy, 1],
		[(0 - cx) / fx, (h - cy) / fy, 1]
	]) * scale

	points = np.vstack([[0, 0, 0], corners])
	lines = [
		[0, 1], [0, 2], [0, 3], [0, 4],
		[1, 2], [2, 3], [3, 4], [4, 1]
	]

	points_world = (pose[:3, :3] @ points.T + pose[:3, 3:4]).T

	frustum = o3d.geometry.LineSet(
		points=o3d.utility.Vector3dVector(points_world),
		lines=o3d.utility.Vector2iVector(lines)
	)
	frustum.colors = o3d.utility.Vector3dVector([color] * len(lines))
	return frustum


def compute_rectify_rotation(pose1, pose2):
	R1 = pose1[..., :3, :3]
	T1 = pose1[..., :3, 3]
	R2 = pose2[..., :3, :3]
	T2 = pose2[..., :3, 3]

	diff = T2 - T1
	rel_t = np.stack([diff, -diff], axis=0)
	
	e1 = rel_t / np.linalg.norm(rel_t, axis=-1, keepdims=True)

	z_world1 = R1[..., :, 2]

	e2 = np.cross(z_world1, e1)
	e2 = e2 / np.linalg.norm(e2, axis=-1, keepdims=True)

	e3 = np.cross(e1, e2)
	
	R_new_candidates = np.stack([e1, e2, e3], axis=-1)

	R1_T = R1.swapaxes(-1, -2)
	R2_T = R2.swapaxes(-1, -2)

	R1_delta_candidates = R_new_candidates @ R1_T
	R2_delta_candidates = R_new_candidates @ R2_T

	traces = np.trace(R1_delta_candidates, axis1=-2, axis2=-1)
	best_idx = np.argmax(traces)

	selected_rel_t = rel_t[best_idx]
	selected_R_new = R_new_candidates[best_idx]
	R1_delta_init = R1_delta_candidates[best_idx]
	R2_delta_init = R2_delta_candidates[best_idx]

	return selected_rel_t, selected_R_new, R1_delta_init, R2_delta_init


class RectificationOptimizer(nn.Module):
	def __init__(self, K1, K2, T1, T2, size1, size2, target_size, share_intrinsic=False, normalize_disparity=False, num_points=20000):
		super().__init__()
		K1 = K1.copy()
		K2 = K2.copy()
		T1 = T1.copy()
		T2 = T2.copy()
		size1 = size1.copy()
		size2 = size2.copy()
		target_size = target_size.copy()
		self.target_size = torch.from_numpy(target_size) # (W, H)
		self.share_intrinsic = share_intrinsic
		self.normalize_disparity = normalize_disparity

		self.register_buffer('R1_orig', torch.from_numpy(T1[:3, :3]).double())
		self.register_buffer('t1_orig', torch.from_numpy(T1[:3, 3]).double())
		self.register_buffer('R2_orig', torch.from_numpy(T2[:3, :3]).double())
		self.register_buffer('t2_orig', torch.from_numpy(T2[:3, 3]).double())
		self.register_buffer('K1_orig', torch.from_numpy(K1).double())
		self.register_buffer('K2_orig', torch.from_numpy(K2).double())

		rel_t, new_R, R1_delta_init, R2_delta_init = compute_rectify_rotation(T1, T2)
		self.register_buffer('R', torch.from_numpy(new_R).double())
		self.register_buffer('t1', torch.from_numpy(T1[:3, 3]).double())
		self.register_buffer('t2', torch.from_numpy(T2[:3, 3]).double())
		

		baseline_length = torch.linalg.norm(torch.from_numpy(rel_t))

		def sample_points_around(center, sigma, n):
			points = torch.randn(n, 3, device=center.device) * sigma
			return center.view(1, 3) + points

		radius = baseline_length * 100.0
		pts1 = sample_points_around(self.t1_orig, radius, num_points)
		pts2 = sample_points_around(self.t2_orig, radius, num_points)
		p_all = torch.cat([pts1, pts2], dim=0) # (2 * num_points, 3)

		def is_visible(p_world, R_orig, t_orig, K_orig, size):
			p_cam = R_orig.T @ (p_world.T - t_orig.view(3, 1)) # (3, M)

			z_mask = p_cam[2, :] > 0
			
			p_img_homo = K_orig @ p_cam
			p_img = p_img_homo[:2, :] / (p_img_homo[2:3, :] + 1e-8)
			
			w_mask = (p_img[0, :] >= 0) & (p_img[0, :] < size[0])
			h_mask = (p_img[1, :] >= 0) & (p_img[1, :] < size[1])
			
			return z_mask & w_mask & h_mask

		mask1 = is_visible(p_all, self.R1_orig, self.t1_orig, self.K1_orig, size1)
		mask2 = is_visible(p_all, self.R2_orig, self.t2_orig, self.K2_orig, size2)		
		valid_points = p_all[mask1 & mask2]
		
		self.register_buffer('p_world', valid_points)

		p1_cam = (self.R.T @ (self.p_world.T - self.t1.view(3, 1))).T
		p2_cam = (self.R.T @ (self.p_world.T - self.t2.view(3, 1))).T
		p1_cam = p1_cam / p1_cam[:,2:]
		p2_cam = p2_cam / p2_cam[:,2:]
		mean_y1 = p1_cam[:,1].mean()
		mean_y2 = p2_cam[:,1].mean()
		mean_y = (mean_y1 + mean_y2) / 2
		angle = torch.arctan2(-mean_y, torch.ones_like(mean_y))
		self.pitch_delta = nn.Parameter(angle)

		p1_cam = (self.get_rectified_R().T @ (self.p_world.T - self.t1.view(3, 1))).T
		p2_cam = (self.get_rectified_R().T @ (self.p_world.T - self.t2.view(3, 1))).T
		p1_cam = p1_cam / p1_cam[:, 2:]
		p2_cam = p2_cam / p2_cam[:, 2:]

		x1_min, x1_max = p1_cam[:, 0].min(), p1_cam[:, 0].max()
		x2_min, x2_max = p2_cam[:, 0].min(), p2_cam[:, 0].max()
		y1_min, y1_max = p1_cam[:, 1].min(), p1_cam[:, 1].max()
		y2_min, y2_max = p2_cam[:, 1].min(), p2_cam[:, 1].max()

		range_x1, range_y1 = x1_max - x1_min, y1_max - y1_min
		range_x2, range_y2 = x2_max - x2_min, y2_max - y2_min
		target_w, target_h = self.target_size
		
		fx1 = (target_w * 0.9) / range_x1
		fy1 = (target_h * 0.9) / range_y1
		fx2 = (target_w * 0.9) / range_x2
		fy2 = (target_h * 0.9) / range_y2

		f_common_y = min(fy1, fy2)
		cy_common = (target_h / 2.0) - f_common_y * (min(y1_min, y2_min) + max(y1_max, y2_max)) / 2.0

		if self.normalize_disparity:
			disp_normalized = p1_cam[:, 0] - p2_cam[:, 0] 

			f_x_new = target_w / disp_normalized.std() * 0.1

			mid_x1 = (x1_min + x1_max) / 2.0
			mid_x2 = (x2_min + x2_max) / 2.0
			
			fx1 = f_x_new
			fx2 = f_x_new
			
			current_mean_disp = f_x_new * (mid_x1 - mid_x2)
			offset_cx = current_mean_disp / 2.0
			
			cx1 = (target_w / 2.0) - fx1 * mid_x1 - offset_cx
			cx2 = (target_w / 2.0) - fx2 * mid_x2 + offset_cx

		else:
			cx1 = (target_w / 2.0) - fx1 * (x1_min + x1_max) / 2.0
			cx2 = (target_w / 2.0) - fx2 * (x2_min + x2_max) / 2.0

		if self.share_intrinsic:
			fx_common = (fx1 + fx2) / 2.0
			cx_common = (cx1 + cx2) / 2.0
			self.k_new_params = nn.Parameter(torch.tensor([
				fx_common, f_common_y, cx_common, cy_common
			]))
		else:
			self.k_new_params = nn.Parameter(torch.tensor([
				fx1, fx2, f_common_y, cx1, cx2, cy_common
			]))

		"""
		cam1 = create_camera_frustum(K1.copy(), T1.copy(), [size1[1], size1[0]], color=[0, 0, 1], scale=1)
		cam2 = create_camera_frustum(K2.copy(), T2.copy(), [size2[1], size2[0]], color=[1, 0, 0], scale=1)
		T1[:3,:3] = new_R
		T2[:3,:3] = new_R
		new_cam1 = create_camera_frustum(self.get_rectified_K1(), T1, [target_size[1], target_size[0]], color=[0, 1, 1], scale=1)
		new_cam2 = create_camera_frustum(self.get_rectified_K2(), T2, [target_size[1], target_size[0]], color=[1, 1, 0], scale=1)

		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(self.p_world.numpy())

		o3d.visualization.draw_geometries([cam1, cam2, new_cam1, new_cam2, pcd])
		"""

	def get_rectified_K1(self):
		p = self.k_new_params
		if self.share_intrinsic:
			fx, fy, cx, cy = p[0], p[1], p[2], p[3]
		else:
			fx, fy, cx, cy = p[0], p[2], p[3], p[5]
			
		return torch.stack([
			torch.tensor([fx, 0.0, cx], device=p.device, dtype=p.dtype),
			torch.tensor([0.0, fy, cy], device=p.device, dtype=p.dtype),
			torch.tensor([0.0, 0.0, 1.0], device=p.device, dtype=p.dtype)
		])


	def get_rectified_K2(self):
		p = self.k_new_params
		if self.share_intrinsic:
			fx, fy, cx, cy = p[0], p[1], p[2], p[3]
		else:
			fx, fy, cx, cy = p[1], p[2], p[4], p[5]
			
		return torch.stack([
			torch.tensor([fx, 0.0, cx], device=p.device, dtype=p.dtype),
			torch.tensor([0.0, fy, cy], device=p.device, dtype=p.dtype),
			torch.tensor([0.0, 0.0, 1.0], device=p.device, dtype=p.dtype)
		])


	def get_rectified_R(self):
		c = torch.cos(self.pitch_delta)
		s = torch.sin(self.pitch_delta)
		
		R_tilt = torch.eye(3, device=self.pitch_delta.device, dtype=self.pitch_delta.dtype)
		R_tilt[1, 1] = c
		R_tilt[1, 2] = -s
		R_tilt[2, 1] = s
		R_tilt[2, 2] = c
		
		return R_tilt @ self.R


	def forward(self):
		K1_new = self.get_rectified_K1()
		K2_new = self.get_rectified_K2()
		R_rect = self.get_rectified_R()
		p1_cam = R_rect.T @ (self.p_world.T - self.t1.view(3, 1))
		p2_cam = R_rect.T @ (self.p_world.T - self.t2.view(3, 1))

		p1_img_homo = K1_new @ p1_cam
		p2_img_homo = K2_new @ p2_cam

		p1_img = (p1_img_homo[:2, :] / p1_img_homo[2:3, :].clamp(min=1e-8)).T
		p2_img = (p2_img_homo[:2, :] / p2_img_homo[2:3, :].clamp(min=1e-8)).T

		return p1_img, p2_img



def compute_joint_boundary_loss(p1_n, p2_n):
	dist1 = torch.relu(-p1_n) + torch.relu(p1_n - 1.0)
	dist2 = torch.relu(-p2_n) + torch.relu(p2_n - 1.0)
	joint_dist = torch.max(dist1, dist2)    
	return joint_dist.mean()


def rectify(img_size1, img_size2, K1, K2, T1, T2, share_intrinsic=False, normalize_disparity=False):
	device = "cuda" if torch.cuda.is_available() else "cpu"
	target_size = img_size1
	target_size_t = torch.from_numpy(np.array(target_size)).to(device)

	model = RectificationOptimizer(K1, K2, T1, T2, 
		np.array(img_size1), np.array(img_size2), 
		np.array(target_size), share_intrinsic=share_intrinsic, normalize_disparity=normalize_disparity).to(device)
	optimizer = optim.Adam(model.parameters(), lr=1e-3)

	with torch.set_grad_enabled(True):
		for i in range(0):
			optimizer.zero_grad()
			p1, p2 = model()

#			loss_epipolar = torch.mean((p1[:,1] - p2[:,1]).abs())
			p1_normalized = p1 / target_size_t
			p2_normalized = p2 / target_size_t
			loss_keep_in_image = compute_joint_boundary_loss(p1_normalized, p2_normalized)
			loss = loss_keep_in_image

#			loss_k = torch.mean((model.k_new_params - torch.tensor([K1[0,0], K1[1,1], K1[0,2], K1[1,2]], device=device))**2)
#			loss = loss_epipolar + 0.01 * loss_keep_in_image #+ 0.1 * loss_k
			loss.backward()
			optimizer.step()

			print(loss)
#			print(f"Iter {i}, Loss: {loss.item():.4f}, epi: {loss_epipolar.item():.4f}, center: {loss_keep_in_image.item():.4f}")

	model.eval()
	with torch.no_grad():
		K1_new = model.get_rectified_K1().cpu().numpy()
		K2_new = model.get_rectified_K2().cpu().numpy()
		R_new = model.get_rectified_R().cpu().numpy()
		
		R1_rect = R_new.T @ T1[:3, :3]
		R2_rect = R_new.T @ T2[:3, :3]

		H1 = K1_new @ R1_rect @ np.linalg.inv(K1)
		H2 = K2_new @ R2_rect @ np.linalg.inv(K2)

		T1_new = np.eye(4)
		T1_new[:3, :3] = R_new
		T1_new[:3, 3] = T1[:3, 3]
		
		T2_new = np.eye(4)
		T2_new[:3, :3] = R_new
		T2_new[:3, 3] = T2[:3, 3]

	return H1, H2, K1_new, K2_new, T1_new, T2_new


if __name__ == '__main__':
	def create_synthetic_images(img_size1, img_size2, K1, K2, T1, T2):
		img1 = np.zeros((img_size1[1], img_size1[0], 3), dtype=np.uint8)
		img2 = np.zeros((img_size2[1], img_size2[0], 3), dtype=np.uint8)
		
		num_pts = 20
		pts_3d = np.random.uniform(-1.5, 1.5, (num_pts, 3))
		pts_3d[:, 2] += 4.0
		
		def get_projections(pts_3d, K, T):
			pts_homo = np.hstack([pts_3d, np.ones((len(pts_3d), 1))])
			pts_cam = (T @ pts_homo.T)[:3, :]
			p_img = K @ pts_cam
			depth = p_img[2, :]
			p_img = (p_img[:2, :] / depth).T
			return p_img, depth

		p1, d1 = get_projections(pts_3d, K1, T1)
		p2, d2 = get_projections(pts_3d, K2, T2)

		for i in range(num_pts):
			color = [int(c) for c in np.random.randint(50, 255, 3)]
			cv2.circle(img1, (int(p1[i, 0]), int(p1[i, 1])), max(int(30 / d1[i]), 1), color, -1)
			cv2.circle(img2, (int(p2[i, 0]), int(p2[i, 1])), max(int(30 / d2[i]), 1), color, -1)
			
		return img1, img2

	img_size1 = (800, 600)
	img_size2 = (640, 480)
	
	K1 = np.array([[850, 0, 400], [0, 850, 300], [0, 0, 1]], dtype=np.float64)
	K2 = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
	
	T1 = np.eye(4) # Camera 1 at origin
	T2 = np.eye(4) # Camera 2 with offset and rotation
	T2[:3, :3] = cv2.Rodrigues(np.array([0.1, 0.25, 0.05]))[0]
	T2[:3, 3] = [0.6, 0.1, 0.1]

	print("Generating synthetic images...")
	img1, img2 = create_synthetic_images(img_size1, img_size2, K1, K2, np.linalg.inv(T1), np.linalg.inv(T2))
	img2_resized = cv2.resize(img2, img_size1)

	def draw_grid(img):
		canvas = img.copy()
		for y in range(0, canvas.shape[0], 40):
			cv2.line(canvas, (0, y), (canvas.shape[1], y), (0, 255, 0), 1)
		return canvas

	cv2.imshow("img1", img1)
	cv2.imshow("img2", img2)
	comparison = np.hstack([draw_grid(img1), draw_grid(img2_resized)])
	cv2.imshow("comparison", comparison)
	cv2.waitKey()

	print("Starting Optimization...")
	H1, H2, K1_new, K2_new, T1_n, T2_n = rectify(img_size1, img_size2, K1, K2, T1, T2)
	print("Finished!")

	rect1 = cv2.warpPerspective(img1, H1, img_size1)
	rect2 = cv2.warpPerspective(img2, H2, img_size1)

	cv2.imshow("img1_after", rect1)
	cv2.imshow("img2_after", rect2)
	comparison = np.hstack([draw_grid(rect1), draw_grid(rect2)])
	cv2.imshow("comparison_after", comparison)
	cv2.waitKey()
