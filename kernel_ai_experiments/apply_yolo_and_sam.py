from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import random
import hashlib

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Ultralytics YOLO
from ultralytics import YOLO  # pip install ultralytics

# Meta SAM (segment-anything)
# pip install git+https://github.com/facebookresearch/segment-anything.git
from segment_anything import sam_model_registry, SamPredictor


# ---------------- utility ----------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(images_dir: Path | list[str]) -> List[Path]:
    if isinstance(images_dir, list):
        return [
            Path(p)
            for p in images_dir
            if Path(p).suffix.lower() in IMG_EXTS and Path(p).is_file()
        ]
    paths = []
    for ext in IMG_EXTS:
        paths.extend(Path(images_dir).glob(f"*{ext}"))
    return sorted(paths)


def imread_rgb(path: Path) -> np.ndarray:
    # robust read for Unicode paths
    arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


def polygon_to_yolo_line(poly_xy: np.ndarray, conf: float, w: int, h: int) -> str:
    """
    poly_xy: (K,2) in pixel coords; returns YOLO seg line with class=0
    conf: confidence score (float)
    w,h: image size in pixels
    """
    x = np.clip(poly_xy[:, 0] / max(w, 1), 0.0, 1.0)
    y = np.clip(poly_xy[:, 1] / max(h, 1), 0.0, 1.0)
    coords = " ".join([f"{xi:.6f} {yi:.6f}" for xi, yi in zip(x, y)])
    return f"0 {conf} {coords}"


def save_boxes_txt(boxes_xyxy: np.ndarray, confs: List[float], out_path: Path):
    """Save boxes as lines: class conf x1 y1 x2 y2 in pixel coords (float)."""
    lines = []
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box.tolist()
        conf = float(confs[i]) if i < len(confs) else 1.0
        lines.append(f"0 {conf:.6f} {x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_boxes_yolo_norm_txt(
    boxes_xyxy: np.ndarray, confs: List[float], out_path: Path, W: int, H: int
):
    """Save boxes in YOLO detection format with normalization: class conf x_center y_center width height (all normalized).
    boxes_xyxy: (N,4) with x1,y1,x2,y2 in pixel coords
    """
    lines = []
    denom_w = float(max(W, 1))
    denom_h = float(max(H, 1))
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box.tolist()
        # clip
        x1c = max(0.0, min(x1, W - 1))
        y1c = max(0.0, min(y1, H - 1))
        x2c = max(0.0, min(x2, W - 1))
        y2c = max(0.0, min(y2, H - 1))
        w = max(0.0, x2c - x1c)
        h = max(0.0, y2c - y1c)
        x_center = (x1c + x2c) / 2.0
        y_center = (y1c + y2c) / 2.0
        x_c_norm = float(np.clip(x_center / denom_w, 0.0, 1.0))
        y_c_norm = float(np.clip(y_center / denom_h, 0.0, 1.0))
        w_norm = float(np.clip(w / denom_w, 0.0, 1.0))
        h_norm = float(np.clip(h / denom_h, 0.0, 1.0))
        conf = float(confs[i]) if i < len(confs) else 1.0
        lines.append(
            f"0 {conf:.6f} {x_c_norm:.6f} {y_c_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _seeded_color_for_instance(seed_str: str, idx: int) -> Tuple[int, int, int]:
    # deterministic color per instance using stem + idx
    h = hashlib.md5((seed_str + str(idx)).encode()).digest()
    return (int(h[0]), int(h[1]), int(h[2]))


def draw_boxes_on_image(
    img: np.ndarray,
    boxes: np.ndarray,
    confs: List[float],
    out: Path,
    max_draw: int = 500,
):
    out_img = img.copy()
    H, W = out_img.shape[:2]
    for i, box in enumerate(boxes[:max_draw]):
        x1, y1, x2, y2 = map(int, box.tolist())
        color = (0, 255, 0)
        cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
        # label = f"{confs[i]:.2f}" if i < len(confs) else ""
        # if label:
        # cv2.putText(out_img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    ok = cv2.imencode(".png", cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))[1]
    out.write_bytes(ok.tobytes())


def draw_polygons_on_image(img: np.ndarray, polys: List[np.ndarray], out: Path):
    out_img = img.copy()
    H, W = out_img.shape[:2]
    stem = out.stem
    for i, poly in enumerate(polys):
        if poly is None or len(poly) < 3:
            continue
        pts = np.asarray(poly).reshape(-1, 2)
        pts_i = np.clip(pts.round().astype(np.int32), 0, max(W, H))
        color = _seeded_color_for_instance(stem, i)
        cv2.fillPoly(out_img, [pts_i], color)
        cv2.polylines(out_img, [pts_i], True, (0, 0, 0), 1)
    alpha = 0.6
    blended = (img * (1 - alpha) + out_img * alpha).astype(np.uint8)
    ok = cv2.imencode(".png", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))[1]
    out.write_bytes(ok.tobytes())


def draw_sam_masks_on_image(img: np.ndarray, masks: List[np.ndarray], out: Path):
    out_img = img.copy()
    H, W = out_img.shape[:2]
    stem = out.stem
    for i, m in enumerate(masks):
        if m is None:
            continue
        # ensure mask shape H,W
        mask = m.astype(bool).astype(np.uint8)
        color = _seeded_color_for_instance(stem, i)
        colored = np.zeros_like(out_img, dtype=np.uint8)
        colored[:, :, 0] = color[2]
        colored[:, :, 1] = color[1]
        colored[:, :, 2] = color[0]
        out_img = np.where(
            mask[..., None],
            ((0.6 * colored) + (0.4 * out_img)).astype(np.uint8),
            out_img,
        )
        # Draw black contour around mask (match YOLO polygon styling)

        polys = mask_to_polygon_contours(mask)
        for poly in polys:
            if poly is None or poly.size == 0:
                continue
            pts_i = np.clip(poly.round().astype(np.int32), 0, max(W, H))
            cv2.polylines(out_img, [pts_i], True, (0, 0, 0), 1)

    ok = cv2.imencode(".png", cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))[1]
    out.write_bytes(ok.tobytes())


def mask_to_polygon_contours(mask: np.ndarray) -> List[np.ndarray]:
    """
    Convert a binary mask to a list of polygon contours (pixel coords).
    We’ll choose the largest contour for YOLO export to keep one line per instance.
    """
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = [c.squeeze(1).astype(np.float32) for c in contours if len(c) >= 3]
    return polys


def xyxy_tensor_to_numpy(xyxy: torch.Tensor) -> np.ndarray:
    return xyxy.detach().cpu().numpy().astype(np.float32)


def save_mask_png(mask: np.ndarray, path: Path):
    # path parent is ensured by caller
    m = (mask > 0).astype(np.uint8) * 255
    ok = cv2.imencode(".png", m)[1]
    path.write_bytes(ok.tobytes())


def to_device(model, device: str):
    try:
        return model.to(device)
    except Exception:
        return model


# ---------------- main worker ----------------
class YOLOAndSAMRunner:
    def __init__(
        self,
        images_dir: Path | list[str],
        yolo_weights: Path,
        out_yolo_txt_dir: Path,
        use_sam: bool = True,
        sam_checkpoint: Path | None = None,
        sam_model_type: str | None = "vit_b",
        out_sam_png_dir: Path | None = None,
        out_sam_npy_dir: Path | None = None,
        yolo_imgsz: int = 640,  # default from ultralytics = 640
        yolo_conf: float = 0.25,  # default from ultralytics = 0.25
        yolo_iou: float = 0.7,  # default from ultralytics = 0.7
        device: str | None = None,
        # result saving
        save_results_dir: Path | None = None,
        save_yolo_boxes_dir: Path | None = None,
        save_yolo_boxes_format: str = "pixel",
        result_sample_count: int = 10,
    ):
        self.images_dir = images_dir
        self.yolo_weights = str(yolo_weights)
        self.out_yolo_txt_dir = Path(out_yolo_txt_dir)
        self.use_sam = use_sam
        self.sam_checkpoint = str(sam_checkpoint) if sam_checkpoint else None
        self.sam_model_type = sam_model_type
        self.out_sam_png_dir = Path(out_sam_png_dir) if out_sam_png_dir else None
        self.out_sam_npy_dir = Path(out_sam_npy_dir) if out_sam_npy_dir else None
        self.yolo_imgsz = yolo_imgsz
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # result save options
        self.save_results_dir = Path(save_results_dir) if save_results_dir else None
        self.save_yolo_boxes_dir = (
            Path(save_yolo_boxes_dir) if save_yolo_boxes_dir else None
        )
        # 'pixel' (x1 y1 x2 y2 in pixels) or 'yolo' (class conf x_c y_c w h normalized)
        self.save_yolo_boxes_format = save_yolo_boxes_format
        self.result_sample_count = int(result_sample_count)
        self.sampled_set: set[str] = set()

        if self.save_results_dir:
            ensure_dir(self.save_results_dir)
        if self.save_yolo_boxes_dir:
            ensure_dir(self.save_yolo_boxes_dir)

        ensure_dir(self.out_yolo_txt_dir)
        if self.use_sam:
            if not self.sam_checkpoint:
                raise ValueError("SAM is enabled but no --sam_checkpoint provided.")
            if self.out_sam_png_dir is None and self.out_sam_npy_dir is None:
                raise ValueError(
                    "SAM is enabled but neither --out_sam_png_dir nor --out_sam_npy_dir is set."
                )
            if self.out_sam_png_dir:
                ensure_dir(self.out_sam_png_dir)
            if self.out_sam_npy_dir:
                ensure_dir(self.out_sam_npy_dir)

        # Load models
        self.yolo = YOLO(self.yolo_weights)
        # Ensure proper device
        to_device(self.yolo.model, self.device)

        if self.use_sam:
            self._init_sam()
        else:
            self.sam_predictor = None

    def _init_sam(self):
        checkpoint = torch.load(
            self.sam_checkpoint,
            map_location=torch.device(self.device),
            weights_only=True,
        )

        # Initialize the model from the registry without loading the checkpoint
        sam = sam_model_registry[self.sam_model_type]()

        # Load the state_dict from the checkpoint
        sam.load_state_dict(checkpoint)

        # Move the model to the appropriate device
        sam.to(device=self.device)

        self.sam_predictor = SamPredictor(sam)

    def run(self):
        images = list_images(self.images_dir)
        if not images:
            raise RuntimeError(f"No images found in {self.images_dir}")

        # choose sample images for result overlays
        if self.save_results_dir:
            sampled = random.sample(images, min(len(images), self.result_sample_count))
            self.sampled_set = {p.name for p in sampled}
        else:
            self.sampled_set = set()

        for img_path in tqdm(images, desc="Predicting"):
            self._process_image(
                img_path, save_results=(img_path.name in self.sampled_set)
            )

        print("\nDone. Predictions written to:")
        print(
            f"  YOLO bboxes: {self.save_yolo_boxes_dir}"
            if self.save_yolo_boxes_dir
            else "  (no YOLO boxes saved)"
        )
        print(f"  YOLO polygons: {self.out_yolo_txt_dir}")
        if self.use_sam and self.out_sam_png_dir:
            print(f"  SAM PNG masks: {self.out_sam_png_dir}")
        if self.use_sam and self.out_sam_npy_dir:
            print(f"  SAM NPY stacks: {self.out_sam_npy_dir}")

    @torch.inference_mode()
    def _process_image(self, img_path: Path, save_results: bool = False):
        # --- Load image
        rgb = imread_rgb(img_path)
        H, W = rgb.shape[:2]

        # --- YOLO prediction (seg)
        # returns a list (one Result per image)
        results = self.yolo.predict(
            source=str(img_path),
            imgsz=640,  # same as val default
            device=0,  # pick your GPU (or "cpu")
            save=False,  # don't save visuals unless you want them
            verbose=False,  # no per-image output
        )
        if not results:
            return
        res = results[0]

        # Collect polygons and boxes from YOLO result
        yolo_polys_px: List[np.ndarray] = []
        yolo_boxes_xyxy: np.ndarray = np.zeros((0, 4), dtype=np.float32)
        yolo_box_confs: List[float] = []

        if res.masks is not None and res.masks.xy:
            # res.masks.xy is a list of (K_i, 2) polygons in PIXEL coords (float)
            for i, poly in enumerate(res.masks.xy):
                if poly is None or len(poly) < 3:
                    continue
                yolo_polys_px.append(np.asarray(poly, dtype=np.float32))
                yolo_box_confs.append(float(res.boxes[i].conf))
        else:
            # No masks (e.g., bbox-only model) → skip YOLO polygon output

            raise ValueError(f"No masks from YOLO for image {img_path.name}")

        if res.boxes is not None and res.boxes.xyxy is not None:
            yolo_boxes_xyxy = xyxy_tensor_to_numpy(res.boxes.xyxy)
            # clip boxes to image
            yolo_boxes_xyxy[:, 0::2] = np.clip(yolo_boxes_xyxy[:, 0::2], 0, W - 1)
            yolo_boxes_xyxy[:, 1::2] = np.clip(yolo_boxes_xyxy[:, 1::2], 0, H - 1)
        else:
            raise ValueError(f"No boxes from YOLO for image {img_path.name}")

        # --- Write YOLO polygon predictions (.txt)
        stem = img_path.stem
        out_txt = self.out_yolo_txt_dir / f"{stem}.txt"
        if len(yolo_polys_px) > 0:
            lines = [
                polygon_to_yolo_line(p, c, W, H)
                for p, c in zip(yolo_polys_px, yolo_box_confs)
            ]
            out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            # If no predictions, write an empty file or remove existing one
            if out_txt.exists():
                out_txt.unlink(missing_ok=True)

        # --- Optionally save YOLO boxes in pixel coords
        if self.save_yolo_boxes_dir and yolo_boxes_xyxy.shape[0] > 0:
            boxes_out = self.save_yolo_boxes_dir / f"{stem}_boxes.txt"
            if self.save_yolo_boxes_format == "pixel":
                save_boxes_txt(yolo_boxes_xyxy, yolo_box_confs, boxes_out)
            else:
                # write normalized YOLO-format boxes
                save_boxes_yolo_norm_txt(
                    yolo_boxes_xyxy, yolo_box_confs, boxes_out, W, H
                )

        # --- SAM prompted by YOLO boxes
        sam_masks: List[np.ndarray] = []
        if self.use_sam and yolo_boxes_xyxy.shape[0] > 0:
            # Set image once per frame
            self.sam_predictor.set_image(rgb)
            for box in yolo_boxes_xyxy:
                # SAM expects XYXY in pixel coords
                # Choose single best mask per box (multimask_output=False)
                masks, scores, _ = self.sam_predictor.predict(
                    box=box.astype(np.float32),
                    multimask_output=False,
                )
                # masks: (1, H, W) bool
                m = masks[0].astype(np.uint8)
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                if m.sum() == 0:
                    continue
                sam_masks.append(m)

            # Save PNGs and/or NPY stack
            if sam_masks:
                if self.out_sam_png_dir:
                    img_dir = self.out_sam_png_dir / stem
                    ensure_dir(img_dir)
                    for i, m in enumerate(sam_masks):
                        save_mask_png(m, img_dir / f"mask_{i:03d}.png")
                if self.out_sam_npy_dir:
                    arr = np.stack(sam_masks, axis=0)  # (N,H,W)
                    np.save(self.out_sam_npy_dir / f"{stem}.npy", arr)

        # --- Optionally save result overlays for selected images
        if save_results and self.save_results_dir:
            res_dir = self.save_results_dir / stem
            ensure_dir(res_dir)
            # 1) YOLO boxes overlay
            if yolo_boxes_xyxy.shape[0] > 0:
                out_path = res_dir / f"{stem}_yolo_boxes.png"
                draw_boxes_on_image(rgb, yolo_boxes_xyxy, yolo_box_confs, out_path)
            # 2) YOLO polygons overlay
            if len(yolo_polys_px) > 0:
                out_path = res_dir / f"{stem}_yolo_polys.png"
                draw_polygons_on_image(rgb, yolo_polys_px, out_path)
            # 3) SAM masks overlay
            if sam_masks:
                out_path = res_dir / f"{stem}_sam_masks.png"
                draw_sam_masks_on_image(rgb, sam_masks, out_path)

        # --- Save result overlays if requested for this image
        # Draw three images: yolo boxes, yolo polygons, sam masks
        if (
            getattr(self, "save_results_dir", None)
            and getattr(self, "sampled_set", None) is not None
        ):
            pass
        # else: SAM disabled or no boxes → do nothing


# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        "Run Ultralytics YOLO (seg) + SAM (box-prompted) and export predictions"
    )
    ap.add_argument(
        "--images", type=Path, required=True, help="Folder with input images"
    )
    ap.add_argument(
        "--yolo_weights",
        type=Path,
        required=True,
        help="Ultralytics weights (e.g., yolov8s-seg.pt)",
    )
    ap.add_argument(
        "--out_yolo_txt_dir",
        type=Path,
        required=True,
        help="Output dir for YOLO-format polygons (.txt)",
    )
    ap.add_argument("--yolo_conf", type=float, default=0.25, help="YOLO conf threshold")
    ap.add_argument(
        "--yolo_iou", type=float, default=0.7, help="YOLO IoU NMS threshold"
    )

    # SAM options
    ap.add_argument(
        "--use_sam", action="store_true", help="Enable SAM with YOLO boxes as prompts"
    )
    ap.add_argument("--sam_checkpoint", type=Path, help="Path to SAM checkpoint (.pth)")
    ap.add_argument(
        "--sam_model_type",
        type=str,
        default="vit_b",
        choices=["vit_h", "vit_l", "vit_b"],
    )
    ap.add_argument(
        "--out_sam_png_dir",
        type=Path,
        help="If set, write per-image folders of PNG masks",
    )
    ap.add_argument(
        "--out_sam_npy_dir", type=Path, help="If set, write one (N,H,W) .npy per image"
    )

    ap.add_argument(
        "--device", type=str, default=None, help="cuda, cuda:0, or cpu (auto if None)"
    )
    ap.add_argument(
        "--save-results-dir",
        type=Path,
        help="If set, save result overlays for sampled images",
    )
    ap.add_argument(
        "--save-yolo-boxes-dir", type=Path, help="If set, save YOLO boxes per image"
    )
    ap.add_argument(
        "--save-yolo-boxes-format",
        type=str,
        default="yolo",
        choices=["pixel", "yolo"],
        help="Format for saved YOLO boxes: 'pixel' writes x1 y1 x2 y2 in pixels, 'yolo' writes normalized x_center y_center width height",
    )
    args = ap.parse_args()

    runner = YOLOAndSAMRunner(
        images_dir=args.images,
        yolo_weights=args.yolo_weights,
        out_yolo_txt_dir=args.out_yolo_txt_dir,
        use_sam=args.use_sam,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        out_sam_png_dir=args.out_sam_png_dir,
        out_sam_npy_dir=args.out_sam_npy_dir,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        device=args.device,
        save_results_dir=args.save_results_dir,
        save_yolo_boxes_dir=args.save_yolo_boxes_dir,
        save_yolo_boxes_format=args.save_yolo_boxes_format,
    )
    runner.run()

"""
python infer_yolo_and_sam.py 
  --images /data/images 
  --yolo_weights yolov8s-seg.pt 
  --out_yolo_txt_dir /data/preds_yolo_txt 
  --use_sam 
  --sam_checkpoint /models/sam_vit_b.pth 
  --sam_model_type vit_b 
  --out_sam_png_dir /data/preds_sam_png
"""
