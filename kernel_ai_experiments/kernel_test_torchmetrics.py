import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import os

import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import cv2
from tqdm import tqdm
import tqdm

from monai.metrics import GeneralizedDiceScore, DiceMetric


# -------------------- IO --------------------


def parse_names(path: Path | None) -> Dict[int, str]:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        import json

        data = json.loads(path.read_text())
        return {int(k): str(v) for k, v in data.items()}
    lines = [
        ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    return {i: name for i, name in enumerate(lines)}


def read_gt_file(path: Path) -> Tuple[List[int], List[np.ndarray]]:
    """Return (labels, list_of_polygons) for GT. Polygons are (N,2) float in [0,1]."""
    labels, polys = [], []
    if not path.exists():
        return labels, polys
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        cls = int(float(parts[0]))
        coords = np.array(list(map(float, parts[1:])), dtype=np.float32)
        if coords.size < 6 or coords.size % 2 == 1:
            continue
        poly = coords.reshape(-1, 2)
        labels.append(cls)
        polys.append(poly)
    return labels, polys


def read_yolo_masks(
    path: Path, min_conf: float
) -> Tuple[List[int], List[float], List[np.ndarray]]:
    """Return (labels, scores, list_of_polygons) for predictions."""
    labels, scores, polys = [], [], []
    if not path.exists():
        return labels, scores, polys
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        cls = int(float(parts[0]))
        conf = float(parts[1])
        if conf < min_conf:
            continue
        coords = np.array(list(map(float, parts[2:])), dtype=np.float32)
        if coords.size < 6 or coords.size % 2 == 1:
            continue
        poly = coords.reshape(-1, 2)
        labels.append(cls)
        scores.append(conf)
        polys.append(poly)
    # sort by score desc
    order = np.argsort(-np.array(scores)) if scores else []
    labels = [labels[i] for i in order] if len(order) else labels
    scores = [scores[i] for i in order] if len(order) else scores
    polys = [polys[i] for i in order] if len(order) else polys
    return labels, scores, polys


def read_boxes_file(path: Path) -> Tuple[List[int], List[float], List[np.ndarray]]:
    """Read predicted boxes saved by apply_yolo_and_sam.py.
    Expected format per-line: class conf x1 y1 x2 y2 (pixel coords) OR class conf x_center y_center w h (normalized) —
    We try to detect which by value ranges (if coords in [0,1] assume normalized).
    Returns labels, scores, boxes (as numpy arrays [x1,y1,x2,y2] normalized to [0,1] if input was normalized, otherwise kept in pixel coords flagged by caller).
    """
    labels, scores, boxes = [], [], []
    if not path.exists():
        return labels, scores, boxes
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 6:
            continue
        cls = int(float(parts[0]))
        conf = float(parts[1])
        vals = list(map(float, parts[2:]))
        if len(vals) == 4:
            # Could be x1 y1 x2 y2 (pixels) or x_c y_c w h (normalized)
            a, b, c, d = vals
            # Heuristic: if values all <= 1.0 then normalized center format
            if max(vals) <= 1.0:
                x_c, y_c, w, h = a, b, c, d
                x1 = x_c - w / 2.0
                y1 = y_c - h / 2.0
                x2 = x_c + w / 2.0
                y2 = y_c + h / 2.0
                boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
            else:
                boxes.append(np.array([a, b, c, d], dtype=np.float32))
        else:
            continue
        labels.append(cls)
        scores.append(conf)
    return labels, scores, boxes


def read_sam_npy_masks(dirpath: Path) -> List[np.ndarray]:
    """Read per-image SAM masks saved as NPY (one file per image containing (N,H,W)).
    Returns list of masks as boolean arrays normalized to [0,1] coordinates when needed by rasterizers.
    """
    masks = []
    if not dirpath.exists():
        return masks
    for p in sorted(dirpath.glob("*.npy")):
        try:
            arr = np.load(p)
            if arr.ndim == 3:
                for i in range(arr.shape[0]):
                    masks.append(arr[i].astype(bool))
            elif arr.ndim == 2:
                masks.append(arr.astype(bool))
        except Exception:
            continue
    return masks


# -------------------- Geometry --------------------


def polys_to_mask(polys: List[np.ndarray], h: int, w: int) -> np.ndarray:
    """Rasterize list of normalized polygons to boolean mask stack (N, H, W)."""
    if not polys:
        return np.zeros((0, h, w), dtype=bool)
    masks = np.zeros((len(polys), h, w), dtype=np.uint8)
    for i, poly in enumerate(polys):
        # Scale to pixel coords
        xs = np.clip(poly[:, 0] * (w - 1), 0, w - 1)
        ys = np.clip(poly[:, 1] * (h - 1), 0, h - 1)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        pts = pts.reshape(-1, 1, 2)
        cv2.fillPoly(masks[i], [pts], 1)
    return masks.astype(bool)


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU for two boolean masks."""
    if mask1.shape != mask2.shape:
        # resize mask2 to mask1
        mask2 = cv2.resize(
            mask2.astype(np.uint8),
            (mask1.shape[1], mask1.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    inter = float(np.logical_and(mask1, mask2).sum())
    union = float(np.logical_or(mask1, mask2).sum())
    if union == 0.0:
        return 0.0
    return inter / union


def dice_score(
    masks_gt: np.ndarray, masks_pred: np.ndarray, eps: float = 1e-6
) -> float:
    """Compute  Dice Score using MONAI's DiceScore metric.

    The function treats foreground as the union of instance masks and computes
    the MONAI  Dice (include_background=False). Returns a float in [0,1].
    """

    # If both empty -> perfect
    if (masks_gt.size == 0 or masks_gt.sum() == 0) and (
        masks_pred.size == 0 or masks_pred.sum() == 0
    ):
        return 1.0

    # Determine H,W
    if masks_gt.size:
        H, W = masks_gt.shape[1], masks_gt.shape[2]
    elif masks_pred.size:
        H, W = masks_pred.shape[1], masks_pred.shape[2]
    else:
        return 1.0

    gt_fg = (
        (masks_gt.sum(axis=0) > 0).astype(np.float32)
        if masks_gt.size
        else np.zeros((H, W), dtype=np.float32)
    )
    pred_fg = (
        (masks_pred.sum(axis=0) > 0).astype(np.float32)
        if masks_pred.size
        else np.zeros((H, W), dtype=np.float32)
    )

    # Convert to tensors (B, C, H, W) with C=1 (foreground)
    y = torch.from_numpy(gt_fg[None, None, ...]).float()
    yp = torch.from_numpy(pred_fg[None, None, ...]).float()

    dmetric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    dmetric.reset()
    dmetric(y_pred=yp, y=y)
    score = dmetric.aggregate().item()

    return float(score)


def get_bbox_from_poly(poly: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    """Get bounding box [x1,y1,x2,y2] in absolute pixel coords from normalized polygon."""
    if poly.size == 0:
        return np.array([0, 0, 0, 0], dtype=np.int32)
    xs = np.clip(poly[:, 0] * (img_size[1] - 1), 0, img_size[1] - 1)
    ys = np.clip(poly[:, 1] * (img_size[0] - 1), 0, img_size[0] - 1)
    x1 = np.min(xs)
    y1 = np.min(ys)
    x2 = np.max(xs)
    y2 = np.max(ys)
    return np.array([x1, y1, x2, y2], dtype=np.int32)


# -------------------- Evaluation --------------------


def build_tm_batches(stems: List[str], gt_dir: Path, pred_dir: Path, min_conf: float):
    preds_ymasks_list, preds_smasks_list, preds_bbox_list, targets_list = [], [], [], []
    ymask_dices, smask_dices = [], []

    for s in tqdm.tqdm(stems, desc="Processing images"):
        gt_p = gt_dir / f"{s}.txt"
        # Prediction subfolders inside pred_dir
        yolo_base = pred_dir / "yolo"
        boxes_base = pred_dir / "yolo_bboxes"
        sam_base = pred_dir / "sam"

        # Predictions: prefer polygon .txt from pred_dir/yolo, boxes from pred_dir/yolo_bboxes,
        # and SAM outputs from pred_dir/sam (either {stem}.npy or folder {stem}/mask_*.png)
        pr_poly_p = yolo_base / f"{s}.txt"
        pr_boxes_p = boxes_base / f"{s}_boxes.txt"
        sam_npy_fp = sam_base / f"{s}.npy"
        sam_folder = sam_base / s

        # SAM masks
        sam_masks = []
        if sam_npy_fp.exists():
            try:
                arr = np.load(sam_npy_fp)
                for i in range(arr.shape[0]):
                    sam_masks.append(arr[i].astype(bool))
            except Exception:
                pass
        elif sam_folder.exists() and sam_folder.is_dir():
            mask_files = sorted(sam_folder.glob("mask_*.png"))
            for mf in mask_files:
                m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
                if m is None:
                    raise FileNotFoundError(f"Failed to read SAM mask PNG: {mf}")
                sam_masks.append((m > 0).astype(bool))

        img_size: Tuple[int, int] = sam_masks[0].shape
        assert len(img_size) == 2, "SAM mask must have shape (H,W)"

        gt_labels, gt_polys = read_gt_file(gt_p)
        gt_bboxes = np.array(
            [get_bbox_from_poly(p, img_size) for p in gt_polys], dtype=np.float32
        )
        pr_labels, pr_scores, pr_yolo_polys = read_yolo_masks(pr_poly_p, min_conf)

        # read boxes if present
        box_labels, box_scores, box_boxes = read_boxes_file(pr_boxes_p)

        # Rasterize GT and predicted polygons to masks at img_size
        gt_masks = polys_to_mask(gt_polys, img_size[0], img_size[1])
        pr_masks = polys_to_mask(pr_yolo_polys, img_size[0], img_size[1])

        # For boxes: convert normalized coords into absolute pixel (x1,y1,x2,y2) values
        # We no longer rasterize boxes into masks here; instead we keep absolute box coordinates
        boxes_xyxy_abs = []
        boxes_labels = box_labels
        boxes_scores = box_scores
        if box_boxes:
            for b in box_boxes:
                # if values in [0,1], treat as normalized coords (we already converted to x1,y1,x2,y2 normalized)
                if np.max(np.abs(b)) <= 1.0:
                    x1n, y1n, x2n, y2n = b
                    x1 = np.clip(x1n * (img_size[1] - 1), 0, img_size[1] - 1)
                    y1 = np.clip(y1n * (img_size[0] - 1), 0, img_size[0] - 1)
                    x2 = np.clip(x2n * (img_size[1] - 1), 0, img_size[1] - 1)
                    y2 = np.clip(y2n * (img_size[0] - 1), 0, img_size[0] - 1)
                else:
                    raise RuntimeError("Box coordinates must be normalized to [0,1]")
                boxes_xyxy_abs.append([x1, y1, x2, y2])

        # Build targets and preds for segmentation
        target = {
            "masks": torch.from_numpy(gt_masks),
            "boxes": torch.from_numpy(gt_bboxes),
            "labels": torch.tensor(gt_labels, dtype=torch.long),
        }

        # Merge predicted masks: include polygon masks and SAM masks and box-derived masks
        yolo_pred_masks = []
        sam_pred_masks = []

        yolo_pred_scores = []
        sam_pred_scores = []

        yolo_pred_labels = []
        sam_pred_labels = []

        # Polygons
        for lab, sc, pm in zip(pr_labels, pr_scores, pr_masks):
            yolo_pred_labels.append(lab)
            yolo_pred_scores.append(sc)
            yolo_pred_masks.append(pm)

        # SAM masks (no scores) — assign score 1.0 and label 0
        for sm in sam_masks:
            # ensure size matches
            if sm.shape != img_size:
                sm = cv2.resize(
                    sm.astype(np.uint8), img_size, interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            sam_pred_labels.append(0)
            sam_pred_scores.append(1.0)
            sam_pred_masks.append(sm.astype(bool))

        pr_yolo_masks_stack = (
            np.stack(yolo_pred_masks, axis=0)
            if yolo_pred_masks
            else np.zeros((0, img_size[0], img_size[1]), dtype=bool)
        )
        pr_sam_masks_stack = (
            np.stack(sam_pred_masks, axis=0)
            if sam_pred_masks
            else np.zeros((0, img_size[0], img_size[1]), dtype=bool)
        )

        yolo_mask_pred = {
            "masks": torch.from_numpy(pr_yolo_masks_stack),
            "scores": (
                torch.tensor(yolo_pred_scores, dtype=torch.float32)
                if yolo_pred_scores
                else torch.empty(0)
            ),
            "labels": (
                torch.tensor(yolo_pred_labels, dtype=torch.long)
                if yolo_pred_labels
                else torch.empty(0, dtype=torch.long)
            ),
        }
        sam_mask_pred = {
            "masks": torch.from_numpy(pr_sam_masks_stack),
            "scores": (
                torch.tensor(sam_pred_scores, dtype=torch.float32)
                if sam_pred_scores
                else torch.empty(0)
            ),
            "labels": (
                torch.tensor(sam_pred_labels, dtype=torch.long)
                if sam_pred_labels
                else torch.empty(0, dtype=torch.long)
            ),
        }
        bbox_preds = {
            # boxes from box predictions (absolute pixel xyxy) along with their scores/labels converted to torch tensors
            "boxes": (
                torch.from_numpy(np.array(boxes_xyxy_abs, dtype=np.float32))
                if "boxes_xyxy_abs" in locals() and boxes_xyxy_abs
                else torch.empty((0, 4), dtype=torch.float32)
            ),
            "scores": (
                torch.from_numpy(np.array(boxes_scores, dtype=np.float32))
                if boxes_scores
                else torch.empty((0,), dtype=torch.float32)
            ),
            "labels": (
                torch.from_numpy(np.array(boxes_labels, dtype=np.int64))
                if boxes_labels
                else torch.empty((0,), dtype=torch.int64)
            ),
        }

        preds_ymasks_list.append(yolo_mask_pred)
        preds_smasks_list.append(sam_mask_pred)
        preds_bbox_list.append(bbox_preds)
        targets_list.append(target)

        # Dice overall
        yolo_mask_dice = dice_score(
            gt_masks.astype(bool), pr_yolo_masks_stack.astype(bool)
        )
        ymask_dices.append(yolo_mask_dice)
        sam_mask_dice = dice_score(
            gt_masks.astype(bool), pr_sam_masks_stack.astype(bool)
        )
        smask_dices.append(sam_mask_dice)

    preds_lists = preds_ymasks_list, preds_smasks_list, preds_bbox_list
    mean_ymask_dice = float(np.mean(ymask_dices)) if ymask_dices else 0.0
    mean_smask_dice = float(np.mean(smask_dices)) if smask_dices else 0.0
    gdices = mean_ymask_dice, mean_smask_dice

    return preds_lists, targets_list, gdices


class InferenceTesting:
    def __init__(
        self,
        gt_dir: Path,
        pred_dir: Path,
        min_conf: float = 0.001,
        backbone: str = "",
        dataset: str = "",
    ):
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.min_conf = min_conf
        self.backbone = backbone
        self.dataset = dataset

        self.stems = self.collect_stems()

    def collect_stems(self):
        # expect prediction subfolders inside pred_dir: yolo, yolo_bboxes, sam
        yolo_dir = self.pred_dir / "yolo"
        stems_gt = {p.stem for p in self.gt_dir.glob("*.txt")}
        stems_yolo = (
            {p.stem for p in yolo_dir.glob("*.txt")} if yolo_dir.exists() else set()
        )
        stems = sorted(stems_gt | stems_yolo)
        if not stems:
            raise FileNotFoundError("No .txt files found in GT or pred_dir/yolo.")
        return stems

    def run_test(
        self,
        test_metric_p: str = "kernel_ai_experiments/InferenceTestResults/test_metrics.csv",
    ):
        preds_lists, targets_list, gdices = build_tm_batches(
            self.stems, self.gt_dir, self.pred_dir, self.min_conf
        )

        preds_ymasks_list, preds_smasks_list, preds_bbox_list = preds_lists
        mean_ymask_gdice, mean_smask_gdice = gdices
        bbox_map = MeanAveragePrecision(
            iou_type="bbox",
            max_detection_thresholds=[100, 200, 300],
            backend="faster_coco_eval",
        )
        ymask_map = MeanAveragePrecision(
            iou_type="segm",
            max_detection_thresholds=[100, 200, 300],
            backend="faster_coco_eval",
        )
        smask_map = MeanAveragePrecision(
            iou_type="segm",
            max_detection_thresholds=[100, 200, 300],
            backend="faster_coco_eval",
        )
        bbox_map.update(preds_bbox_list, targets_list)
        ymask_map.update(preds_ymasks_list, targets_list)
        smask_map.update(preds_smasks_list, targets_list)
        results_bbox = bbox_map.compute()
        results_ymask = ymask_map.compute()
        results_smask = smask_map.compute()

        result_dict = {
            "backbone": self.backbone,
            "dataset": self.dataset,
            "bbox_map_50:95": round(results_bbox["map"].item(), 4),
            "bbox_map_50": round(results_bbox["map_50"].item(), 4),
            "bbox_map_75": round(results_bbox["map_75"].item(), 4),
            "ymask_map_50:95": round(results_ymask["map"].item(), 4),
            "ymask_map_50": round(results_ymask["map_50"].item(), 4),
            "ymask_map_75": round(results_ymask["map_75"].item(), 4),
            "smask_map_50:95": round(results_smask["map"].item(), 4),
            "smask_map_50": round(results_smask["map_50"].item(), 4),
            "smask_map_75": round(results_smask["map_75"].item(), 4),
            "mean_ymask_dice": round(float(mean_ymask_gdice), 4),
            "mean_smask_dice": round(float(mean_smask_gdice), 4),
        }
        if os.path.exists(test_metric_p):
            # load existing CSV and append data
            df = pd.read_csv(test_metric_p)
            df = pd.concat([df, pd.DataFrame([result_dict])], ignore_index=True)
            df.to_csv(test_metric_p, index=False)
        else:
            # create new CSV
            df = pd.DataFrame([result_dict])
            df.to_csv(test_metric_p, index=False)


def main():
    p = argparse.ArgumentParser(description="YOLO polygon seg mAP with TorchMetrics")
    p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument(
        "--min-conf", type=float, default=0.001, help="Min confidence for preds"
    )

    args = p.parse_args()

    tester = InferenceTesting(args.gt_dir, args.pred_dir, args.min_conf)
    tester.run_test()


if __name__ == "__main__":
    main()
