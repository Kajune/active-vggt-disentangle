import numpy as np
import cv2
import os, sys, argparse, json, copy


def refract(ray_dir, interface_normal, mu):
	dot = np.sum(ray_dir * (-interface_normal), axis=-1)[...,np.newaxis]
	refracted_ray_dir = (ray_dir / mu - (dot + np.sqrt(mu ** 2 + dot ** 2 - 1.0)) / mu * (-interface_normal))
	return refracted_ray_dir


def compute_distance_to_plane(ray_origin, ray_dir, plane_normal, plane_origin):
	distance = -(np.sum(plane_normal * (ray_origin - plane_origin), axis=-1) / np.sum(plane_normal * ray_dir, axis=-1))
	return distance


def project(pos3d, cam_mat):
	pos3d = pos3d.copy()
	pos3d /= pos3d[...,2:]
	pos3d = (cam_mat @ pos3d.reshape(-1,3).T).T.reshape(*pos3d.shape)
	return pos3d


def normalize(vec):
	return vec / np.sqrt(np.expand_dims(np.sum(vec ** 2, axis=-1), axis=-1))


def solve2nd(coefs):
	a = coefs[...,0]
	b = coefs[...,1]
	c = coefs[...,2]

	D = b ** 2 - 4 * a * c
	D = D.astype(np.complex128)

	x0 = (-b + np.sqrt(D)) / (2 * a)
	x1 = (-b - np.sqrt(D)) / (2 * a)
	return np.stack([x0, x1], axis=-1)


def solve3rd(coefs):
	a = coefs[...,0]
	b = coefs[...,1]
	c = coefs[...,2]
	d = coefs[...,3]

	A0 = d / a
	A1 = c / a
	A2 = b / a
	p = A1 - A2 * A2 / 3
	q = A0 - A1 * A2 / 3 + A2 * A2 * A2 * 2 / 27
	D = (q / 2) ** 2 + (p / 3) ** 3

	r = np.sqrt(D.astype(np.complex128))
	o = complex(-0.5, np.sqrt(3) / 2)

	L = np.cbrt(-q / 2 + r.real).astype(np.complex128)
	R = np.cbrt(-q / 2 - r.real).astype(np.complex128)
	x0 = L + R - A2 / 3
	x1 = o * o * L + o * R - A2 / 3
	x2 = o * L + o * o * R - A2 / 3

	L_ = (-q / 2 + r) ** (1/3)
	R_ = (-q / 2 - r) ** (1/3)
	x0_ = L_ + R_ - A2 / 3
	x1_ = o * o * L_ + o * R_ - A2 / 3
	x2_ = o * L_ + o * o * R_ - A2 / 3

	# 還元不能の場合
	x0[D < 0] = x0_[D < 0]
	x1[D < 0] = x1_[D < 0]
	x2[D < 0] = x2_[D < 0]

	return np.stack([x0, x1, x2], axis=-1)


def solve4th(coefs):
	a = coefs[...,0]
	b = coefs[...,1]
	c = coefs[...,2]
	d = coefs[...,3]
	e = coefs[...,4]

	A0 = e / a
	A1 = d / a
	A2 = c / a
	A3 = b / a
	p = A2 - 6 * A3 * A3 / 16
	q = A1 - A2 * A3 / 2 + 8 * (A3 / 4) ** 3
	r = A0 - A1 * A3 / 4 + A2 * A3 * A3 / 16 - 3 * (A3 / 4) ** 4

	coefs2 = np.stack([
		np.ones_like(p),
		2 * p,
		p * p - 4 * r,
		-q * q
	]).T

	u = solve3rd(coefs2)
	t = u[:,0].real

	coefs3_1 = np.stack([
		np.ones_like(t),
		np.sqrt(t),
		(p + t) / 2 - np.sqrt(t) * q / (2 * t),
	]).T

	coefs3_2 = np.stack([
		np.ones_like(t),
		-np.sqrt(t),
		(p + t) / 2 + np.sqrt(t) * q / (2 * t),
	]).T

	y0 = solve2nd(coefs3_1)
	y1 = solve2nd(coefs3_2)

	x0 = y0[:,0] - A3 / 4
	x1 = y0[:,1] - A3 / 4
	x2 = y1[:,0] - A3 / 4
	x3 = y1[:,1] - A3 / 4

	return np.stack([x0, x1, x2, x3], axis=-1)


def solventh(coefs):
	comp = np.zeros((*coefs.shape, coefs.shape[-1]), dtype=np.complex128)
	for i in range(coefs.shape[-1] - 1):
		comp[...,0,i] = -coefs[...,i+1] / coefs[...,0]
		if i > 0:
			comp[...,i,i-1] = 1

	eigenvalues = np.linalg.eig(comp)[0]
	return eigenvalues


def refractive_forward_projection(pos3d, normal, depth, mu):
	n = -normalize(np.float32([normal[0], normal[1], 1.0]))
	d = depth

	por = normalize(np.cross(n, pos3d, axis=-1))
	z1 = -n
	z2 = np.cross(por, z1, axis=-1)

	v = np.sum(pos3d * z1, axis=-1)
	u = np.sum(pos3d * z2, axis=-1)

	coef1 = (mu - 1) * (mu + 1)
	coef2 = -2 * u * (mu - 1) * (mu + 1)
	coef3 = d * d * mu * mu - d * d + 2 * d * v + mu * mu * u * u - u * u - v * v
	coef4 = -2 * d * d * mu * mu * u
	coef5 = d * d * mu * mu * u * u
	coef1 = np.ones_like(coef2) * coef1
	coefs = np.stack([coef1, coef2, coef3, coef4, coef5], axis=-1)

	x0 = solve4th(coefs.astype(np.float64)).astype(np.complex64).T
	xx = x0.real

	a = 1 / mu
	n2 = np.float32([0, -1])
	vi = np.stack([xx, np.ones_like(xx) * d], axis=-1)
	bb = np.sum(vi * n2, axis=-1)
	bbb = np.sum(vi * vi, axis=-1)

	b = -bb - np.sqrt(bb * bb - (1 - mu * mu) * bbb)
	b /= mu

	vr = a * vi + b[...,np.newaxis] * n2
	vrd = np.stack([u, v], axis=-1)[np.newaxis]
	vrd = vrd - vi

	error = np.abs(vrd[...,0] * vr[...,1] - vrd[...,1] * vr[...,0])
	error[np.abs(x0.imag) > 1e-3] = np.inf
	best = xx[np.argmin(error, axis=0), np.arange(xx.shape[1])]
	return best[...,np.newaxis] * z2 + d * z1


def compute_uw_params(transforms, num_points, near, far, forward_projection=False):
	if "refractive_interface" not in transforms:
		return transforms

	transforms = copy.deepcopy(transforms)

	fx = transforms["w"] / np.tan(transforms["camera_angle_x"] / 2) / 2
#	fy = transforms["h"] / np.tan(transforms["camera_angle_y"] / 2) / 2
	fy = fx
	cx = transforms["cx"]
	cy = transforms["cy"]
	normal = np.float32([0, 0, 1])
	d = transforms["refractive_interface"]["d"]
	mu = transforms["refractive_interface"]["mu"]

	cam_mat = np.array([
		[fx, 0, cx],
		[0, fy, cy],
		[0, 0, 1],
	])

	pos2d = np.random.rand(num_points, 2)
	pos2d[...,0] *= transforms["w"]
	pos2d[...,1] *= transforms["h"]
#	import matplotlib.pyplot as plt
#	plt.scatter(pos2d[:,0], pos2d[:,1])

	pos2d_ = np.concatenate((pos2d, np.ones_like(pos2d[:,:1])), axis=-1)
	pos3d = (np.linalg.inv(cam_mat) @ pos2d_.T).T

	rays_o = np.zeros_like(pos3d)
	rays_v = pos3d - rays_o
	rays_v /= np.linalg.norm(rays_v, axis=-1)[...,np.newaxis]

	if forward_projection:
		depth = np.random.uniform(near, far, (num_points,))
		pos3d = depth[...,np.newaxis] * rays_v + rays_o
		pos3d = refractive_forward_projection(pos3d, normal[:2], d, mu)
		pos3d *= depth[...,np.newaxis]
		pos2d_dist = project(pos3d, cam_mat)[:,:2]
		flag = (0 <= pos2d_dist[:,0]) & (pos2d_dist[:,0] < transforms["w"]) & (0 <= pos2d_dist[:,1]) & (pos2d_dist[:,1] < transforms["h"])
		pos2d = pos2d[flag]
		pos3d = pos3d[flag]
		num_points = len(pos2d)

#		plt.scatter(pos2d[:,0], pos2d[:,1])
#		plt.show()

	else:
		distance_to_interface = compute_distance_to_plane(rays_o, rays_v, normal, normal * d)
		rays_o_refract = rays_o + rays_v * distance_to_interface[...,np.newaxis]
		rays_v = refract(rays_v, normal, mu)
		rays_v /= rays_v[...,2][...,np.newaxis]

		depth = np.random.uniform(near, far, (num_points,))
		pos3d = (depth - rays_o_refract[...,2])[...,np.newaxis] * rays_v + rays_o_refract

	ret, cam_mat_refract, dist, _, _ = cv2.calibrateCamera(
		pos3d.reshape(1,num_points,3).astype(np.float32), 
		pos2d.reshape(1,num_points,2).astype(np.float32), 
		(transforms["w"], transforms["h"]), cam_mat.copy(), None,
		flags=cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_K3)

	transforms["camera_angle_x"] = 2 * np.arctan(transforms["w"] / cam_mat_refract[0,0] / 2)
	transforms["camera_angle_y"] = 2 * np.arctan(transforms["h"] / cam_mat_refract[1,1] / 2)
	transforms["cx"] = cam_mat_refract[0,2]
	transforms["cy"] = cam_mat_refract[1,2]
	transforms["k1"] = dist[0,0]
	transforms["k2"] = dist[0,1]
	transforms["p1"] = dist[0,2]
	transforms["p2"] = dist[0,3]

	return transforms



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--transforms', type=str, required=True)
	parser.add_argument('--near', type=float, default=1.0)
	parser.add_argument('--far', type=float, default=10.0)
	parser.add_argument('--num_points', type=float, default=10000)
	args = parser.parse_args()

	transforms = json.load(open(args.transforms))
	print(json.dumps(compute_uw_params(transforms, args.num_points, args.near, args.far), indent=4))

