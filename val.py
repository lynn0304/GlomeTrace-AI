#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np

def load_mask(path: Path) -> np.ndarray:
    suf = path.suffix.lower()
    if suf == ".npy":
        arr = np.load(path)
    else:
        try:
            import imageio.v3 as iio
            arr = iio.imread(path)
        except Exception:
            from PIL import Image
            arr = np.array(Image.open(path))

    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0, ...]
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D after squeeze, got shape={arr.shape} for {path}")
    return arr

def binarize(arr: np.ndarray, th: float) -> np.ndarray:
    if arr.dtype == bool:
        return arr
    u = np.unique(arr)
    if u.size <= 2 and set(u.tolist()).issubset({0, 1}):
        return arr.astype(bool)
    if u.size <= 2 and set(u.tolist()).issubset({0, 255}):
        return (arr > 0)
    return (arr.astype(np.float32) >= th)

def confusion(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool).ravel()
    gt = gt.astype(bool).ravel()
    tp = np.logical_and(pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return tp, tn, fp, fn

def safe_div(n, d):
    return float(n) / float(d) if d != 0 else np.nan

def compute_metrics(pred_bool: np.ndarray, gt_bool: np.ndarray) -> dict:
    tp, tn, fp, fn = confusion(pred_bool, gt_bool)

    acc  = safe_div(tp + tn, tp + tn + fp + fn)
    prec = safe_div(tp, tp + fp)
    rec  = safe_div(tp, tp + fn)               
    spec = safe_div(tn, tn + fp)               
    iou  = safe_div(tp, tp + fp + fn)
    dice = safe_div(2 * tp, 2 * tp + fp + fn)

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "iou": iou,
        "dice": dice,
    }

def metrics_from_conf(tp, tn, fp, fn) -> dict:
    return {
        "accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "iou": safe_div(tp, tp + fp + fn),
        "dice": safe_div(2 * tp, 2 * tp + fp + fn),
    }

def summarize(df, metric_cols):
    out = {}
    for c in metric_cols:
        v = df[c].to_numpy(dtype=float)
        out[c + "_mean"] = float(np.nanmean(v))
        out[c + "_std"]  = float(np.nanstd(v, ddof=1)) if np.sum(~np.isnan(v)) > 1 else np.nan
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_mask_dir", type=str, help="root dir for predict mask")
    ap.add_argument("--gt_mask_dir", type=str, help="root dir for ground truth mask")
    ap.add_argument("--folds", type=str, default="fold_1,fold_2,fold_3,fold_4,fold_5",
                    help="Comma-separated fold folder names under root.")
    ap.add_argument("--pred_subdir", type=str, default="pred")
    ap.add_argument("--gt_subdir", type=str, default="gt")
    ap.add_argument("--th", type=float, default=0.5,
                    help="Threshold for binarizing pred/gt if not already binary.")
    ap.add_argument("--exts", type=str, default=".npy,.png,.tif,.tiff",
                    help="Allowed file extensions for masks (pred side).")
    ap.add_argument("--out_csv", type=str, default="metrics_all_images.csv")
    ap.add_argument("--out_fold_csv", type=str, default="metrics_by_fold.csv")
    args = ap.parse_args()

    folds = [f.strip() for f in args.folds.split(",") if f.strip()]
    exts = set(e.strip().lower() for e in args.exts.split(",") if e.strip())

    metric_cols = ["accuracy", "precision", "recall", "specificity", "iou", "dice"]

    rows = []
    fold_summ_rows = []

    TP_all = TN_all = FP_all = FN_all = 0

    for fold in folds:
        pred_dir = args.pred_mask_dir
        gt_dir   = args.gt_mask_dir

        if not pred_dir.exists() or not gt_dir.exists():
            raise FileNotFoundError(f"Missing pred/gt dir in fold={fold}: {pred_dir} / {gt_dir}")

        pred_files = [p for p in pred_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in exts]
        pred_map = {p.stem: p for p in pred_files}

        gt_files = [g for g in gt_dir.iterdir() if g.is_file()]
        gt_map = {g.stem: g for g in gt_files}

        common = sorted(set(pred_map.keys()) & set(gt_map.keys()))
        if not common:
            raise RuntimeError(
                f"No matched filenames in fold={fold}. "
                f"Check stems under {pred_dir} and {gt_dir}"
            )

        import pandas as pd
        fold_rows = []

        TPm = TNm = FPm = FNm = 0

        for k in common:
            p_path = pred_map[k]
            g_path = gt_map[k]

            pred = load_mask(p_path)
            gt   = load_mask(g_path)

            if pred.shape != gt.shape:
                raise ValueError(f"Shape mismatch in fold={fold}, name={k}: pred={pred.shape}, gt={gt.shape}")

            pred_b = binarize(pred, args.th)
            gt_b   = binarize(gt, args.th)

            m = compute_metrics(pred_b, gt_b)

            TPm += m["tp"]; TNm += m["tn"]; FPm += m["fp"]; FNm += m["fn"]

            row = {"fold": fold, "name": k, "pred_path": str(p_path), "gt_path": str(g_path)}
            row.update(m)
            rows.append(row)
            fold_rows.append(row)

        fold_df = pd.DataFrame(fold_rows)

        # macro: mean/std over images (your original behavior)
        macro = summarize(fold_df, metric_cols)

        # micro: metrics from aggregated confusion
        micro = metrics_from_conf(TPm, TNm, FPm, FNm)

        fold_sum = {
            "fold": fold,
            "n_images": int(len(fold_df)),

            # macro
            **{f"macro_{k}": v for k, v in macro.items()},

            # micro + confusion
            "micro_tp": int(TPm),
            "micro_tn": int(TNm),
            "micro_fp": int(FPm),
            "micro_fn": int(FNm),
            **{f"micro_{k}": v for k, v in micro.items()},
        }
        fold_summ_rows.append(fold_sum)

        # accumulate for 5-fold micro
        TP_all += TPm; TN_all += TNm; FP_all += FPm; FN_all += FNm

        print(f"[{fold}] n={len(fold_df)}")
        for c in metric_cols:
            print(f"  macro_{c:10s}: {fold_sum[f'macro_{c}_mean']:.6f} ± {fold_sum[f'macro_{c}_std']:.6f}")
        for c in metric_cols:
            print(f"  micro_{c:10s}: {fold_sum[f'micro_{c}']:.6f}")

    import pandas as pd
    all_df = pd.DataFrame(rows)
    fold_df = pd.DataFrame(fold_summ_rows)

    # 5-fold macro summary: mean/std across fold macro means
    five = {"fold": "5fold", "n_folds": len(folds)}

    # 5-fold macro summary: mean/std across fold macro means
    for c in metric_cols:
        fold_means = fold_df[f"macro_{c}_mean"].to_numpy(dtype=float)
        five[f"macro_{c}_mean"] = float(np.nanmean(fold_means))
        five[f"macro_{c}_std"]  = float(np.nanstd(fold_means, ddof=1)) if np.sum(~np.isnan(fold_means)) > 1 else np.nan

    # 5-fold micro (fold-level): mean/std across fold-level micro values  
    for c in metric_cols:
        micro_vals = fold_df[f"micro_{c}"].to_numpy(dtype=float)
        five[f"micro_{c}_mean"] = float(np.nanmean(micro_vals))
        five[f"micro_{c}_std"]  = float(np.nanstd(micro_vals, ddof=1)) if np.sum(~np.isnan(micro_vals)) > 1 else np.nan

    micro_global = metrics_from_conf(TP_all, TN_all, FP_all, FN_all)
    five.update({
        "micro_global_tp": int(TP_all),
        "micro_global_tn": int(TN_all),
        "micro_global_fp": int(FP_all),
        "micro_global_fn": int(FN_all),
        **{f"micro_global_{k}": v for k, v in micro_global.items()},
    })
    plot_confusion_matrix_counts(
        tn=TN_all, fp=FP_all, fn=FN_all, tp=TP_all,
        title="Micro Confusion Matrix (counts)",
        out_path="5fold_micro_confmat.png",
        use_log=False
    )

    fold_df2 = pd.concat([fold_df, pd.DataFrame([five])], ignore_index=True)

    all_df.to_csv(args.out_csv, index=False)
    fold_df2.to_csv(args.out_fold_csv, index=False)

    print("\n[5-fold summary]")
    for c in metric_cols:
        print(f"  macro_{c:10s}: {five[f'macro_{c}_mean']:.6f} ± {five[f'macro_{c}_std']:.6f}")
    for c in metric_cols:
        print(f"  micro_{c:10s}: {five[f'micro_{c}_mean']:.6f} ± {five[f'micro_{c}_std']:.6f}")
    for c in metric_cols:
        print(f"  micro_global_{c:3s}: {five[f'micro_global_{c}']:.6f}")

    print(f"\nSaved:\n- {args.out_csv}\n- {args.out_fold_csv}")

import numpy as np
import cv2
from pathlib import Path

def load_gray_image(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(img_path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).clip(0, 255).astype(np.uint8)
    return img

def load_mask_any(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        m = np.load(p)
    else:
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(path)
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    if m.ndim == 3:  
        m = m[..., 0]
    if m.ndim != 2:
        raise ValueError(f"mask must be 2D, got {m.shape} from {path}")
    return m

import matplotlib.pyplot as plt

def plot_confusion_matrix_counts(
    tn: int, fp: int, fn: int, tp: int,
    class_names=("Background (0)", "GBM (1)"),
    title="Micro Confusion Matrix (counts)",
    out_path=None,
    use_log=False,     
    figsize=(8, 6),
    dpi=150,
):
    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=np.float64)

    disp = np.log10(cm + 1.0) if use_log else cm

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(disp, cmap="Blues")

    ax.set_xticks([0, 1], labels=class_names)
    ax.set_yticks([0, 1], labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{int(cm[i, j]):,}",
                ha="center", va="center",
                fontsize=12,
                color="white" if disp[i, j] > disp.max() * 0.55 else "black"
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(count+1)" if use_log else "count")

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def binarize_mask(arr: np.ndarray, th: float = 0.5) -> np.ndarray:
    if arr.dtype == bool:
        return arr
    u = np.unique(arr)
    if u.size <= 2 and set(u.tolist()).issubset({0, 1}):
        return arr.astype(bool)
    if u.size <= 2 and set(u.tolist()).issubset({0, 255}):
        return arr > 0
    return arr.astype(np.float32) >= th

def overlay_tp_fp_fn(
    gray_img: np.ndarray,
    pred_bool: np.ndarray,
    gt_bool: np.ndarray,
    alpha: float = 0.55,
    draw_contour: bool = True,
    contour_px: int = 2,
) -> np.ndarray:
    if gray_img.ndim != 2:
        raise ValueError("gray_img must be 2D grayscale (H,W)")
    if pred_bool.shape != gt_bool.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred_bool.shape} vs {gt_bool.shape}")
    if gray_img.shape != pred_bool.shape:
        raise ValueError(f"img/mask shape mismatch: {gray_img.shape} vs {pred_bool.shape}")

    tp = pred_bool & gt_bool
    fp = pred_bool & (~gt_bool)
    fn = (~pred_bool) & gt_bool

    base = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR).astype(np.float32)

    overlay = np.zeros_like(base, dtype=np.float32)

    overlay[tp] = (0, 255, 0)     # TP green
    overlay[fp] = (0, 0, 255)     # FP red
    overlay[fn] = (255, 0, 0)     # FN blue

    out = base.copy()
    mask_any = tp | fp | fn
    out[mask_any] = (1 - alpha) * base[mask_any] + alpha * overlay[mask_any]

    out = np.clip(out, 0, 255).astype(np.uint8)

    if draw_contour:
        def _draw(region_bool, color_bgr):
            m = (region_bool.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(out, contours, -1, color_bgr, thickness=contour_px, lineType=cv2.LINE_AA)

        _draw(tp, (0, 255, 0))
        _draw(fp, (0, 0, 255))
        _draw(fn, (255, 0, 0))

    return out

import cv2
import numpy as np

def resize_to_shape(arr: np.ndarray, target_shape: tuple, is_mask: bool):
    """Resize (H,W) or (H,W,C) to (target_H, target_W)."""
    th, tw = target_shape
    h, w = arr.shape[:2]
    if (h, w) == (th, tw):
        return arr

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    out = cv2.resize(arr, (tw, th), interpolation=interp)
    return out

def fit_image_and_masks(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    ref: str = "gt",   
):
    if ref == "gt":
        H, W = gt_mask.shape[:2]
    elif ref == "image":
        H, W = image.shape[:2]
    else:
        raise ValueError("ref must be 'gt' or 'image'")

    image_r = resize_to_shape(image, (H, W), is_mask=False)
    pred_r  = resize_to_shape(pred_mask, (H, W), is_mask=True)
    gt_r    = resize_to_shape(gt_mask, (H, W), is_mask=True)

    return image_r, pred_r, gt_r


def make_tp_fp_fn_figure(
    img_path: str,
    pred_mask_path: str,
    gt_mask_path: str,
    out_path: str,
    th: float = 0.5,
    alpha: float = 0.55,
    draw_contour: bool = False,
    contour_px: int = 2,
):
    img = load_gray_image(img_path)
    pred_raw = load_mask_any(pred_mask_path)   
    gt_raw   = load_mask_any(gt_mask_path)

    img, pred_raw, gt_raw = fit_image_and_masks(img, pred_raw, gt_raw, ref="gt")

    pred = binarize_mask(pred_raw, th)
    gt   = binarize_mask(gt_raw, th)

    vis = overlay_tp_fp_fn(img, pred, gt, alpha=alpha, draw_contour=draw_contour, contour_px=contour_px)
    cv2.imwrite(out_path, vis)
    print(f"Saved: {out_path}")

def main_plot():
    import os
    for img_path in os.listdir("train/images"):
        make_tp_fp_fn_figure(img_path, img_path.replace(".jpg", ".png"), "masks/" + img_path.replace(".jpg", ".png"), img_path.replace(".jpg", ".png"))

if __name__ == "__main__":
    main()
