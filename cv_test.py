import os
import glob
import csv
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from skimage.feature import graycomatrix, graycoprops

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff") 



def normalize_u8(img):
    img = img.astype(np.float32)
    p1 = np.percentile(img, 1)
    p99 = np.percentile(img, 99)
    img = (img - p1) / (p99 - p1 + 1e-6)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def norm01(m):
    a, b = np.percentile(m, 5), np.percentile(m, 95)
    return np.clip((m - a) / (b - a + 1e-6), 0, 1)

def safe_cv(std, mean):
    return float(std / (mean + 1e-6))


def tile_map(img_u8, tile=512, stride=256):
    H, W = img_u8.shape
    ys = list(range(0, max(1, H - tile + 1), stride))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if ys[-1] != H - tile: ys.append(H - tile)
    if xs[-1] != W - tile: xs.append(W - tile)

    mean_m = np.zeros((len(ys), len(xs)), np.float32)
    std_m  = np.zeros((len(ys), len(xs)), np.float32)
    edge_m = np.zeros((len(ys), len(xs)), np.float32)

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            t = img_u8[y:y+tile, x:x+tile]
            mean_m[i, j] = float(t.mean())
            std_m[i, j]  = float(t.std())

            gx = cv2.Sobel(t, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(t, cv2.CV_32F, 0, 1, ksize=3)
            edge_m[i, j] = float(np.mean(gx*gx + gy*gy))

    return {
        "mean": mean_m, "std": std_m, "edge": edge_m,
        "mean_n": norm01(mean_m), "std_n": norm01(std_m), "edge_n": norm01(edge_m),
        "ys": ys, "xs": xs, "tile": tile, "stride": stride,
    }


def blob_pick_density(img_u8, min_diam_px=60, max_diam_px=180, blob_color=0):
    blur = cv2.GaussianBlur(img_u8, (0, 0), 2.0)
    bg   = cv2.GaussianBlur(img_u8, (0, 0), 20.0)
    hp   = cv2.subtract(blur, bg)  

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = float(np.pi * (min_diam_px/2)**2 * 0.25)
    params.maxArea = float(np.pi * (max_diam_px/2)**2 * 2.0)

    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByConvexity = False

    params.filterByColor = True
    params.blobColor = int(blob_color)

    det = cv2.SimpleBlobDetector_create(params)
    kps = det.detect(hp)
    pts = np.array([kp.pt for kp in kps], dtype=np.float32) if kps else np.zeros((0, 2), np.float32)

    return pts, hp


def density_per_tile(points_xy, H, W, tile=512, stride=256):
    ys = list(range(0, max(1, H - tile + 1), stride))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if ys[-1] != H - tile: ys.append(H - tile)
    if xs[-1] != W - tile: xs.append(W - tile)

    dens = np.zeros((len(ys), len(xs)), np.float32)
    tile_area = float(tile * tile)

    if len(points_xy) == 0:
        return dens, norm01(dens), ys, xs

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            in_tile = (
                (points_xy[:, 0] >= x) & (points_xy[:, 0] < x + tile) &
                (points_xy[:, 1] >= y) & (points_xy[:, 1] < y + tile)
            )
            dens[i, j] = float(in_tile.sum()) / tile_area

    return dens, norm01(dens), ys, xs



def nn_distance_stats(points):
    if len(points) < 2:
        return np.nan, np.nan, np.nan
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)  
    nn = dists[:, 1]
    nn_mean = float(nn.mean())
    nn_std = float(nn.std())
    nn_cv = float(nn_std / (nn_mean + 1e-6))
    return nn_mean, nn_std, nn_cv



def hf_power_ratio(img_u8, r0=0.2):
    f = np.fft.fftshift(np.fft.fft2(img_u8.astype(np.float32)))
    mag = np.abs(f)

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    r_norm = r / (r.max() + 1e-6)

    hf = float(mag[r_norm > r0].sum())
    total = float(mag.sum())
    return hf / (total + 1e-6)


def glcm_features(img_u8, levels=32):
    g = (img_u8.astype(np.float32) * (levels - 1) / 255.0).astype(np.uint8)
    glcm = graycomatrix(g, distances=[1], angles=[0], levels=levels, symmetric=True, normed=True)

    contrast = float(graycoprops(glcm, "contrast")[0, 0])
    homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])
    asm = float(graycoprops(glcm, "ASM")[0, 0])

    P = glcm[:, :, 0, 0]
    entropy = float(-np.sum(P * np.log(P + 1e-12)))
    return contrast, homogeneity, asm, entropy


def good_tile_ratio(mean_n, edge_n, dens_n=None, mean_lo=0.30, mean_hi=0.70, edge_hi=0.40, dens_lo=None):
    good = (mean_n >= mean_lo) & (mean_n <= mean_hi) & (edge_n <= edge_hi)
    if dens_n is not None and dens_lo is not None:
        good = good & (dens_n >= dens_lo)
    return float(good.sum() / (good.size + 1e-9))


def save_heatmap(mat01, out_png, cmap="viridis", title=None):
    plt.figure(figsize=(5, 5))
    if title:
        plt.title(title)
    plt.imshow(mat01, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def process_one_image(path, out_dir, tile, stride, min_diam_px, max_diam_px, overlay_radius, blob_color=0):
    base = os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read: {path}")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_u8 = normalize_u8(img)

    maps = tile_map(img_u8, tile=tile, stride=stride)

    pts, hp = blob_pick_density(
        img_u8, min_diam_px=min_diam_px, max_diam_px=max_diam_px, blob_color=blob_color
    )

    dens, dens_n, _, _ = density_per_tile(pts, img_u8.shape[0], img_u8.shape[1], tile=tile, stride=stride)

    nn_mean, nn_std, nn_cv = nn_distance_stats(pts)

    tile_mean_cv = safe_cv(maps["mean"].std(), maps["mean"].mean())
    tile_edge_cv = safe_cv(maps["edge"].std(), maps["edge"].mean())
    tile_dens_cv = safe_cv(dens.std(), dens.mean())

    good_ratio = good_tile_ratio(
        mean_n=maps["mean_n"],
        edge_n=maps["edge_n"],
        dens_n=dens_n,
        mean_lo=0.30,
        mean_hi=0.70,
        edge_hi=0.40,
        dens_lo=None 
    )

    hf_ratio = float(hf_power_ratio(img_u8, r0=0.20))

    glcm_contrast, glcm_homogeneity, glcm_asm, glcm_entropy = glcm_features(img_u8, levels=32)

    sub = os.path.join(out_dir, base)
    os.makedirs(sub, exist_ok=True)

    cv2.imwrite(os.path.join(sub, "micrograph_u8.png"), img_u8)
    cv2.imwrite(os.path.join(sub, "highpass.png"), hp)

    save_heatmap(maps["mean_n"], os.path.join(sub, "heat_mean.png"), cmap="viridis", title="Tile mean (proxy ice)")
    save_heatmap(maps["edge_n"], os.path.join(sub, "heat_edge.png"), cmap="magma", title="Tile edge energy (proxy contam)")
    save_heatmap(dens_n, os.path.join(sub, "heat_density.png"), cmap="plasma", title="Particle density (candidates/px^2)")

    vis = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    for (x, y) in pts.astype(int):
        cv2.circle(vis, (int(x), int(y)), overlay_radius, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.imwrite(os.path.join(sub, "overlay_candidates.png"), vis)

    out = {
        "file": os.path.basename(path),
        "H": int(img_u8.shape[0]),
        "W": int(img_u8.shape[1]),

        "global_mean_u8": float(img_u8.mean()),
        "global_std_u8": float(img_u8.std()),

        "n_candidates": int(len(pts)),
        "cand_density_per_px2": float(len(pts) / (img_u8.size + 1e-9)),

        "tile": int(tile),
        "stride": int(stride),
        "min_diam_px": int(min_diam_px),
        "max_diam_px": int(max_diam_px),
        "blob_color": int(blob_color),

        "nn_dist_mean_px": float(nn_mean),
        "nn_dist_std_px": float(nn_std),
        "nn_dist_cv": float(nn_cv),

        "tile_mean_cv": float(tile_mean_cv),
        "tile_edge_cv": float(tile_edge_cv),
        "tile_density_cv": float(tile_dens_cv),

        "good_tile_ratio": float(good_ratio),

        "hf_power_ratio": float(hf_ratio),

        "glcm_contrast": float(glcm_contrast),
        "glcm_homogeneity": float(glcm_homogeneity),
        "glcm_asm": float(glcm_asm),
        "glcm_entropy": float(glcm_entropy),

        "out_dir": sub,
    }
    return out


def list_images(in_dir, recursive):
    pattern = "**/*" if recursive else "*"
    files = glob.glob(os.path.join(in_dir, pattern), recursive=recursive)
    out = []
    for f in files:
        if os.path.isfile(f) and f.lower().endswith(IMG_EXTS) and not f.lower().endswith(".mrc"):
            out.append(f)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--recursive", action="store_true")

    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)

    ap.add_argument("--min_diam_px", type=int, default=60)
    ap.add_argument("--max_diam_px", type=int, default=180)

    ap.add_argument("--overlay_radius", type=int, default=6)

    ap.add_argument("--blob_color", type=int, default=0, choices=[0, 255],
                    help="0=detect dark blobs, 255=detect bright blobs")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = list_images(args.in_dir, args.recursive)

    summary_path = os.path.join(args.out_dir, "summary.csv")
    rows = []

    for idx, f in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {os.path.basename(f)}")
        try:
            row = process_one_image(
                f, args.out_dir,
                tile=args.tile,
                stride=args.stride,
                min_diam_px=args.min_diam_px,
                max_diam_px=args.max_diam_px,
                overlay_radius=args.overlay_radius,
                blob_color=args.blob_color
            )
            rows.append(row)
        except Exception as e:
            print(f"  !! failed: {f}\n     {e}")

    if rows:
        keys = list(rows[0].keys())
        with open(summary_path, "w", newline="", encoding="utf-8") as fo:
            w = csv.DictWriter(fo, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    print(f"Done: {summary_path}")


if __name__ == "__main__":
    main()
