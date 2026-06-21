from dataclasses import dataclass
from typing import List, Optional
import os
from scipy.optimize import curve_fit
from ear_traits.scan_time_utils import (
    ImgData,
    CamExtrinsics,
    CamIntrinsics,
)
import numpy as np
import cv2
from scipy.spatial import KDTree
from scipy.integrate import quad
from ultralytics import YOLO


@dataclass
class KernelTraitData:
    ear_id: int
    motor_steps: int
    kernel_id: int
    x: float
    y: float
    z: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    mask_area_yolo_px: Optional[float] = None
    mask_area_sam_px: Optional[float] = None
    mask_iou: Optional[float] = None
    px_to_mm2: Optional[float] = None
    mask_area_yolo_mm2: Optional[float] = None
    mask_area_sam_mm2: Optional[float] = None
    unit_vector_x: Optional[float] = None
    unit_vector_y: Optional[float] = None
    unit_vector_z: Optional[float] = None


def build_yolo_engine(yolo_model_path: str, quant: Optional[str] = None):
    if yolo_model_path.endswith(".engine"):
        engine_path = yolo_model_path
    elif yolo_model_path.endswith(".pt"):
        engine_path = yolo_model_path.replace(".pt", ".engine")
    else:
        raise ValueError(
            f"Invalid model path: {yolo_model_path}. Expected .pt or .engine file."
        )

    if os.path.exists(engine_path):
        return YOLO(engine_path, task="segment")
    elif quant == "int8":
        print(
            f"Exporting YOLO model to TensorRT engine and int8 quantization format. This may take a while..."
        )
        model = YOLO(yolo_model_path, task="segment")
        model.export(
            format="engine",
            dynamic=True,
            batch=16,
            workspace=None,
            int8=True,
            data="/home/geink81/RestoredBackup/Mais/Ears/trainingdata/int8_calibration/PublicationTraining20MP_calib.yaml",  # Path to your dataset
            device="cuda:0",
        )
        return YOLO(engine_path, task="segment")

    elif quant == "fp16":
        print(
            f"Exporting YOLO model to TensorRT engine and fp16 quantization format. This may take a while..."
        )
        model = YOLO(yolo_model_path, task="segment")
        model.export(
            format="engine",
            dynamic=True,
            batch=16,
            workspace=None,
            half=True,
            device="cuda:0",
        )
        return YOLO(engine_path, task="segment")

    elif quant == "fp32":
        print(
            f"Exporting YOLO model to TensorRT engine format. This may take a while..."
        )
        model = YOLO(yolo_model_path, task="segment")
        engine_path = model.export(format="engine")
        return YOLO(engine_path, task="segment")

    elif quant == None:
        return YOLO(yolo_model_path, task="segment")

    else:
        raise ValueError(
            f"Invalid quantization format: {quant}. Supported formats are: int8, fp16, fp32."
        )


class KernelCounting:
    def __init__(
        self,
        roi_imgs: List[ImgData],
        low_cam_extrinsics: List[CamExtrinsics],
        low_cam_intrinsics: CamIntrinsics,
        low_img_poly_pts: List,
        model,
        ear_width: float,
    ):
        """
        Parameters:
            roi_imgs: List of ROI images.
            low_cam_extrinsics: List or array of camera extrinsics with .xyz attribute.
            low_cam_intrinsics: Camera intrinsics object.
            low_img_poly_pts: List of polygon points for each image.
            model: Yolo Model object that supports tracking and prediction (with predictor attribute).
            ear_width: Width of the ear in mm.
        """
        self.roi_imgs = roi_imgs
        self.low_cam_extrinsics = low_cam_extrinsics
        self.low_cam_intrinsics = low_cam_intrinsics
        self.low_img_poly_pts = low_img_poly_pts
        self.model = model
        self.ear_width = ear_width

        # Initialize result lists.
        self.bboxes_per_ear = []
        self.box_centers_per_ear = []
        self.centre_polyn_pts = []
        self.yolo_mask_cnts_per_ear = []
        self.seen_ids = []

    def process(self):
        """
        Process the ROI images.
        """
        # Process each ROI image.
        [self._process_single_image(*imgobj_) for imgobj_ in enumerate(self.roi_imgs)]

        self._reset_tracker()

    def _reset_tracker(self):
        """
        Reset the tracker and clear the seen IDs.
        """
        if self.model.predictor is not None:
            self.model.predictor.trackers[0].reset()
        self.seen_ids = []

    def _append_empty_results(self):
        self.bboxes_per_ear.append([])
        self.box_centers_per_ear.append([])
        self.centre_polyn_pts.append([])
        self.yolo_mask_cnts_per_ear.append([])

    def _process_single_image(self, imgi, imgobj_):
        # If an image caused a problem previously, skip it.
        if imgobj_.error_msg:
            self._append_empty_results()
            return

        # Use centre crop of wide ears only to improve tracking.
        if self.ear_width < 42:
            input_img = imgobj_.img
        else:
            input_img = imgobj_.img_centre

        self.buff_xmin, _, self.buff_ymin, _ = imgobj_.buffered_dims

        # Get pixel-to-mm conversion.
        PXLMM = imgobj_.intrinsics.px_to_mm
        self.stripe_x = 0

        # Instantiate ear center object and parse motor steps.
        ear_center_obj = Earcenter(imgobj_)

        # Get the camera position from extrinsics.
        cam_pos = self.low_cam_extrinsics[imgi].xyz

        # Determine the distance from the previous camera.
        if imgi == 0:
            dist_to_prev_cam = np.linalg.norm(cam_pos - self.low_cam_extrinsics[-1].xyz)
        else:
            dist_to_prev_cam = np.linalg.norm(
                cam_pos - self.low_cam_extrinsics[imgi - 1].xyz
            )

        # If the camera distance is too small,
        # record empty results and skip.
        if dist_to_prev_cam < 5:
            self._append_empty_results()
            return

        # For large jumps, adjust self.stripe_x and reset the model tracker.
        elif dist_to_prev_cam > self.low_cam_intrinsics.cam_distance * 1.1:
            self.stripe_x = PXLMM * (
                dist_to_prev_cam / self.low_cam_intrinsics.cam_distance
            )
            self._reset_tracker()

        # Perform object tracking on the image center.
        yolo_results_img = self.model.track(
            input_img,
            persist=True,
            imgsz=640,
            retina_masks=False,
            verbose=False,
            device="cuda:0",
        )

        # If no kernels are detected, record empty results and skip.
        if len(yolo_results_img[0].boxes) == 0:
            self._append_empty_results()
            return

        # Extract boxes, segmentation masks, and track IDs.
        boxes = yolo_results_img[0].boxes.xywh.cpu().numpy()
        sam_boxes = yolo_results_img[0].boxes.xyxy.cpu()
        maskscnts = yolo_results_img[0].masks.xy

        # if tracker looses track, reset the tracker
        if yolo_results_img[0].boxes.id is None:
            self._reset_tracker()
            track_ids = np.arange(-1, -len(boxes) - 1, -1).tolist()
        else:
            track_ids = yolo_results_img[0].boxes.id.int().cpu().tolist()

        # Compute the ear center line using the polygon points (adjusted by the image buffer).
        cropped_center_pts = (
            self.low_img_poly_pts[imgi][0] - self.buff_xmin,
            self.low_img_poly_pts[imgi][1] - self.buff_ymin,
        )
        centerpts, lines = ear_center_obj.getcenterline(
            polynom2=True, old_eq=[], pts=cropped_center_pts
        )
        self.centre_polyn_pts.append(centerpts)
        eq, _ = lines
        self._process_kernel(eq, boxes, sam_boxes, track_ids, maskscnts)

    def _process_kernel(self, eq, boxes, sam_boxes, track_ids, maskscnts):
        a, b, c = eq
        # Process each detected box.
        bboxes_per_img = []
        box_centers_per_img = []
        yolo_mask_cnts_per_img = []
        for i in range(len(boxes)):
            box = boxes[i]
            sam_box = sam_boxes[i]
            kernel_id = track_ids[i]
            mcnt = maskscnts[i]

            x, y, w, h = box
            x1, y1, x2, y2 = sam_box

            # Check if the detected curve is within the expected region and that the object is new.
            if (
                self.curveinbox((a, b, c), (x, y), w, stripex=self.stripe_x)
                and kernel_id not in self.seen_ids
            ):
                bboxes_per_img.append([x1, y1, x2, y2])
                box_centers_per_img.append([y + self.buff_ymin, x + self.buff_xmin])
                self.seen_ids.append(kernel_id)
                yolo_mask_cnts_per_img.append(mcnt)

        self.bboxes_per_ear.append(bboxes_per_img)
        self.box_centers_per_ear.append(box_centers_per_img)
        self.yolo_mask_cnts_per_ear.append(yolo_mask_cnts_per_img)

    def get_bboxes(self):
        return self.bboxes_per_ear

    def get_box_centers(self):
        return self.box_centers_per_ear

    def get_polyn_pts(self):
        return self.centre_polyn_pts

    def get_yolo_mask_cnts(self):
        return self.yolo_mask_cnts_per_ear

    def get_seen_ids(self):
        return self.seen_ids

    def curveinbox(self, abc, center, w, stripex=0):
        a, b, c = abc
        cx, cy = center
        boxleftx = int(cx - 0.5 * w)
        boxrightx = int(cx + 0.5 * w)

        if not a and not b:  # in case of vertical line
            if c <= boxrightx and c >= boxleftx:
                print("vertical line in box")
                return True
            else:
                return False

        xline = a * cy**2 + b * cy + c

        if xline <= boxrightx and xline >= boxleftx - stripex:
            return True
        return False


class Earcenter:  # TODO clean up functions
    def __init__(self, img: ImgData):
        self.img = img.img
        self.mask = img.mask

    def simple_sample_pts(self, start, sample_n, deltay):
        stop = start + deltay
        return np.linspace(start, stop, sample_n, dtype=np.int32)

    def getcenterline(self, polynom2=False, old_eq=[], sample_pt_n=12, pts=None):
        ptsnp = None
        if pts is None:
            mask = self.mask
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            roi_cnt_size = 0
            roi_cnt = None

            for cnt in cnts:
                cnt_size = cv2.contourArea(cnt)
                if cnt_size > roi_cnt_size:
                    roi_cnt = cnt
                    roi_cnt_size = cnt_size

            roi_mask0 = np.zeros_like(mask)
            roi_mask = cv2.drawContours(roi_mask0, [roi_cnt], -1, 255, -1)
            y_indces, x_indces = np.where(roi_mask == 255)

            pts = []
            equations = []
            ylimits = []
            ymin, ymax = np.amin(y_indces), np.amax(y_indces)
            deltay = ymax - ymin

            for y in self.simple_sample_pts(ymin, sample_pt_n, deltay):

                xi = x_indces[y_indces == y]
                if len(xi) > 0:
                    xmin, xmax = np.amin(xi), np.amax(xi)

                if len(pts) == 0 or polynom2:
                    xcenter = int((xmin + xmax) / 2)
                    point = xcenter, int(y)
                else:
                    xcenter = int(
                        (xmin + xmax + 1 * pts[-1][0][0]) / 3
                    )  # enter some bias to previous point
                    point = xcenter, int(y)
                    eq = self.calculate_line_equation(point, pts[-1][0])
                    equations.append(eq)
                    ylimits.append((pts[-1][0][1], point[1]))

                pts.append([point])

            ptsnp = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        if polynom2:
            if ptsnp is not None:
                x_data, y_data = ptsnp[:, :, 0].flatten(), ptsnp[:, :, 1].flatten()
            else:
                x_data, y_data = np.array(pts[0], dtype=np.int32), np.array(
                    pts[1], dtype=np.int32
                )
            popt, pcov = curve_fit(
                self.quadratic_func,
                y_data,
                x_data,
                bounds=([-0.5, -np.inf, -np.inf], [0.5, np.inf, np.inf]),
            )
            if len(old_eq) > 0:
                popt = self.weighted_mean_function(old_eq, popt, 0.5)
            y_fit = np.linspace(
                y_data.min(), y_data.max(), 100, dtype=np.int32
            ).reshape((-1, 1))
            x_fit = np.array(self.quadratic_func(y_fit, *popt), dtype=np.int32).reshape(
                (-1, 1)
            )

            ptsnp = np.zeros((len(x_fit), 1, 2), dtype=np.int32)
            ptsnp[:, :, 0] = x_fit
            ptsnp[:, :, 1] = y_fit
            equations = popt
            ylimits = (0, np.inf)

        return ptsnp, (equations, ylimits)

    def quadratic_func(self, x, a, b, c):
        return a * x**2 + b * x + c

    def calculate_line_equation(self, point1, point2):
        """Calculates the slope-intercept form (y = mx + b) of the line passing through the given points.

        Args:
            point1: A tuple (x1, y1) representing the first point.
            point2: A tuple (x2, y2) representing the second point.

        Returns:
            A tuple (m, b) representing the slope and y-intercept of the line.
        """

        x1, y1 = point1
        x2, y2 = point2

        # Handle vertical lines
        if x1 == x2:
            return None, x1  # Equation is x = x1 (slope is infinite)

        slope = (y2 - y1) / (x2 - x1)
        y_intercept = y1 - slope * x1

        return slope, y_intercept

    def weighted_mean_function(self, f, g, weight_f):
        """
        Calculates the weighted mean of two second-order functions.

        Args:
            f: A list of coefficients representing the first function (ax^2 + bx + c).
            g: A list of coefficients representing the second function (dx^2 + ex + f).
            weight_f: The weight of the first function (between 0 and 1).

        Returns:
            A list of coefficients representing the weighted mean function.
        """

        # Ensure valid weight
        if not (0 <= weight_f <= 1):
            raise ValueError("Weight must be between 0 and 1")

        weight_g = 1 - weight_f  # Calculate the weight of the second function

        mean_coeffs = []
        for i in range(3):
            mean_coeffs.append(f[i] * weight_f + g[i] * weight_g)

        return mean_coeffs


def mean_curvature(a, b, c, x1, x2):
    curvature_func = lambda x: curvature(a, b, x)
    integral, error = quad(curvature_func, x1, x2)
    mean_curv = integral / (x2 - x1)
    return mean_curv


def kernel_duplicate(query_point, points, bbox):
    points = points.tolist()
    if len(points) == 0:
        return False
    # print(points)
    distthresh = bbox / 4
    tree = KDTree(points)
    distance, index = tree.query(query_point)
    mindist = np.amin(distance)
    # print(mindist,' < ', distthresh,"?")
    # print(distance[index])

    if mindist < distthresh:
        return True
    else:
        return False


def curveinbox(abc, center, w, stripex=0):
    a, b, c = abc
    cx, cy = center
    boxleftx = int(cx - 0.5 * w)
    boxrightx = int(cx + 0.5 * w)

    if not a and not b:  # in case of vertical line
        if c <= boxrightx and c >= boxleftx:
            print("vertical line in box")
            return True
        else:
            return False

    xline = a * cy**2 + b * cy + c

    if xline <= boxrightx and xline >= boxleftx - stripex:
        return True
    return False


def curvature(a, b, x):
    return np.abs(2 * a) / (1 + (2 * a * x + b) ** 2) ** (3 / 2)
