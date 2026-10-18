import numpy as np
import cv2
import torch

def get_bayer_matrix(n=16):
    """
    再帰的に n x n のベイヤマトリクスを生成し、0.0〜1.0 に正規化する
    """
    def _bayer(size):
        if size == 2:
            return np.array([[0, 2], [3, 1]], dtype=np.float32)
        half = size // 2
        m = _bayer(half)
        top = np.hstack([4 * m, 4 * m + 2])
        bottom = np.hstack([4 * m + 3, 4 * m + 1])
        return np.vstack([top, bottom])
    
    mat = _bayer(n)
    return mat / (n * n)

def measure_chaotropy_entropy(pattern_img, grid_size=(4, 4), white_ratio=0.5):
	"""
	生成された画像パターンと理想のパターンを比較し、
	シャノンエントロピー（ビット単位）を計測する。
	"""
	h, w = pattern_img.shape[:2]
	gw, gh = grid_size
	rows, cols = h // gh, w // gw
	
	# 1. 各グリッド（マス）の平均輝度から 0 または 1 の状態を復元
	blocks = pattern_img.reshape(rows, gh, cols, gw).mean(axis=(1, 3))
	actual_grid = (blocks > 127).astype(np.float32)
	
	# 2. 理想的な規則的パターンを作成（16x16 ベイヤマトリクスを使用）
	bayer_16x16 = get_bayer_matrix(16)
	yy, xx = np.indices((rows, cols))
	bayer_val = bayer_16x16[yy % 16, xx % 16]
	ideal_grid = (bayer_val < white_ratio).astype(np.float32)

	# 3. 理想パターンから外れている（反転している）マスの割合（エラー率 p）を計算
	mismatches = np.sum(actual_grid != ideal_grid)
	total_cells = rows * cols
	p = mismatches / total_cells
	
	# 4. 2値エントロピーの公式でエントロピーを算出
	p = np.clip(p, 1e-12, 1.0 - 1e-12)
	entropy_per_cell = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
	total_system_entropy = entropy_per_cell * total_cells
	
	return entropy_per_cell, total_system_entropy

def _solve_probability_from_entropy(target_entropy):
	"""
	目標エントロピー (0.0 〜 1.0) から、パターンを反転させる確率 p を二分探索で逆算する
	"""
	target_entropy = np.clip(target_entropy, 0.0, 1.0)
	if target_entropy <= 0.0:
		return 0.0
	if target_entropy >= 1.0:
		return 0.5  
	
	low, high = 0.0, 0.5
	for _ in range(30):
		mid = (low + high) / 2.0
		h = -mid * np.log2(np.clip(mid, 1e-12, 1.0)) - (1.0 - mid) * np.log2(np.clip(1.0 - mid, 1e-12, 1.0))
		if h < target_entropy:
			low = mid
		else:
			high = mid
	return (low + high) / 2.0

def generate_pattern_with_entropy(args):
	if args is None:
		return None
	
	width = args.get("width", 1024)
	height = args.get("height", 768)
	dtype = args.get("dtype", torch.float32)
	device = args.get("device", "cpu")

	target_entropy = args.get("target_entropy", 0.5)
	white_ratio = args.get("white_ratio", 0.5)  # 白のセルが登場する割合 (0.0 〜 1.0)
	grid_size = args.get("grid_size", [4, 4])
	blur_size = args.get("blur_size", 1)
	roi = args.get("roi", 1.0)
	contour = args.get("contour", False)

	grid_w, grid_h = grid_size[0], grid_size[1]
	rows, cols = height // grid_h, width // grid_w

	# target_entropy に一致する反転確率 p を逆算
	p = _solve_probability_from_entropy(target_entropy)

	# 1. 白の割合（white_ratio）を反映した【規則的なベースパターン】を作成（16x16 ベイヤマトリクスを使用）
	bayer_16x16 = get_bayer_matrix(16)
	yy, xx = np.indices((rows, cols))
	bayer_val = bayer_16x16[yy % 16, xx % 16]
	p_mat = (bayer_val < white_ratio).astype(np.float32)

	# 2. 確率 p に従って、一部のマス目を正確に反転させてエントロピーを調整する
	if p > 0.0:
		flip_mask = np.random.rand(rows, cols) < p
		p_mat = np.where(flip_mask, 1.0 - p_mat, p_mat)

	p_mat = (p_mat * 255).astype(np.uint8)
	pattern = cv2.resize(p_mat, (width, height), interpolation=cv2.INTER_NEAREST)

	if roi < 1.0:
		mask = np.zeros_like(pattern, dtype=np.uint8)
		cx, cy = width // 2, height // 2
		w_roi, h_roi = int(width * roi), int(height * roi)
		x1, y1 = cx - w_roi // 2, cy - h_roi // 2
		x2, y2 = cx + w_roi // 2, cy + h_roi // 2
		mask[y1:y2, x1:x2] = 1
		pattern = pattern * mask + int(white_ratio * 255) * (1 - mask)

	if contour:
		_, binary = cv2.threshold(pattern, 127, 255, cv2.THRESH_BINARY)
		contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
		color_pattern = ((np.ones((height, width, 3), dtype=np.float32) * white_ratio) * 255).astype(np.uint8)
		if hierarchy is not None:
			for i, h_idx in enumerate(hierarchy[0]):
				cv2.drawContours(color_pattern, contours, i, (255, 255, 255), 1)
	else:
		color_pattern = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
		if blur_size > 0:
			color_pattern = cv2.GaussianBlur(color_pattern, (blur_size, blur_size), 0)

	pattern = color_pattern.astype(np.float32).transpose(2, 0, 1)  # [C, H, W]
	return torch.tensor(pattern, dtype=dtype, device=device)

# --- デモ実行メイン処理 ---
def main():
	# 中間値の設定（他のパラメータを固定する際の基準値）
	DEFAULT_ENTROPY = 0.5
	DEFAULT_GRID_SIZE = [4, 4]
	DEFAULT_WHITE_RATIO = 0.5

	"""
	# 1. エントロピーを変化させるパターン
	print("=== 1. Varying Entropy ===")
	target_entropies = [0.0, 0.25, 0.5, 0.75, 1.0]
	for t_entropy in target_entropies:
		args = {
			"width": 1024,
			"height": 768,
			"target_entropy": t_entropy,
			"grid_size": DEFAULT_GRID_SIZE,
			"blur_size": 0,
			"white_ratio": DEFAULT_WHITE_RATIO,
			"device": "cpu"
		}
		tensor_pattern = generate_pattern_with_entropy(args)
		img = tensor_pattern.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		entropy_per_cell, _ = measure_chaotropy_entropy(gray, grid_size=tuple(DEFAULT_GRID_SIZE), white_ratio=DEFAULT_WHITE_RATIO)
		print(f"Entropy: {t_entropy:.2f} -> Measured: {entropy_per_cell:.4f} bits")
		cv2.imwrite(f"pattern_entropy_{t_entropy}.png", gray)

	# 2. グリッドサイズを変化させるパターン (1, 2, 4, 8, 16)
	print("\n=== 2. Varying Grid Size ===")
	grid_sizes = [[1, 1], [2, 2], [4, 4], [8, 8], [16, 16]]
	for g_size in grid_sizes:
		args = {
			"width": 1024,
			"height": 768,
			"target_entropy": DEFAULT_ENTROPY,
			"grid_size": g_size,
			"blur_size": 0,
			"white_ratio": DEFAULT_WHITE_RATIO,
			"device": "cpu"
		}
		tensor_pattern = generate_pattern_with_entropy(args)
		img = tensor_pattern.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		entropy_per_cell, _ = measure_chaotropy_entropy(gray, grid_size=tuple(g_size), white_ratio=DEFAULT_WHITE_RATIO)
		print(f"Grid Size: {g_size[0]} -> Measured Entropy: {entropy_per_cell:.4f} bits")
		cv2.imwrite(f"pattern_grid_{g_size[0]}x{g_size[1]}.png", gray)

	# 3. 白の割合（white_ratio）を変化させるパターン
	print("\n=== 3. Varying White Ratio ===")
	white_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
	for w_val in white_ratios:
		args = {
			"width": 1024,
			"height": 768,
			"target_entropy": DEFAULT_ENTROPY,
			"grid_size": DEFAULT_GRID_SIZE,
			"blur_size": 0,
			"white_ratio": w_val,
			"device": "cpu"
		}
		tensor_pattern = generate_pattern_with_entropy(args)
		img = tensor_pattern.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		entropy_per_cell, _ = measure_chaotropy_entropy(gray, grid_size=tuple(DEFAULT_GRID_SIZE), white_ratio=w_val)
		print(f"White Ratio: {w_val} -> Measured Entropy: {entropy_per_cell:.4f} bits")
		cv2.imwrite(f"pattern_white_ratio_{w_val}.png", gray)
	"""

	args = {
		"width": 1024,
		"height": 768,
		"target_entropy": 0.5,
		"grid_size": (4, 4),
		"blur_size": 0,
		"white_ratio": 0.1,
		"device": "cpu"
	}
	tensor_pattern = generate_pattern_with_entropy(args)
	img = tensor_pattern.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	cv2.imwrite(f"pattern_optimal.png", gray)

	print("\nすべてのパターン生成と保存が完了しました。")

if __name__ == "__main__":
	main()