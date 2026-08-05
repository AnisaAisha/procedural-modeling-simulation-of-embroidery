import numpy as np
import cv2
from PIL import Image

ROUGHNESS_PATH = "weave_roughness_map.png"
BUMP_OUT_PATH = "weave_bump_map.png"


def build_bump_map(height_np, blur_ksize=3, sobel_ksize=3):
    smoothed = cv2.GaussianBlur(height_np, (blur_ksize, blur_ksize), 0)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
    gz = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=sobel_ksize)
    m = max(np.abs(gx).max(), np.abs(gz).max(), 1e-6)
    return gx / m, gz / m


def encode_to_png(gx, gz, out_path):
    h, w = gx.shape
    img16 = np.zeros((h, w, 3), dtype=np.uint16)
    img16[..., 0] = np.clip((gx * 0.5 + 0.5) * 65535.0, 0, 65535).astype(np.uint16)
    img16[..., 1] = np.clip((gz * 0.5 + 0.5) * 65535.0, 0, 65535).astype(np.uint16)
    img16[..., 2] = 32768  # unused channel, kept mid-gray
    cv2.imwrite(out_path, img16[..., ::-1])


if __name__ == "__main__":
    rough_img = Image.open(ROUGHNESS_PATH).convert("L")
    rough_np = np.asarray(rough_img, dtype=np.float32) / 255.0

    gx_np, gz_np = build_bump_map(rough_np)
    encode_to_png(gx_np, gz_np, BUMP_OUT_PATH)
