from dataclasses import dataclass, field

from apriltag import apriltag
import cv2
import numpy as np
import pandas as pd
import os
import json
from typing import Optional
from dataclasses import dataclass
from typing import List, Tuple
from typing import Optional, Union, Tuple, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


@dataclass
class CamIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float
    w: int
    h: int
    px_to_mm: Optional[float] = None
    cam_name: Optional[str] = None
    cam_distance: Optional[float] = (
        None  # The expected distance between neighbouring camera positions
    )
    origin_distance: Optional[float] = (
        None  # The expected distance between the cameras and the origin (spike)
    )
    undist_cam_mtx: Optional[np.ndarray] = None

    @property
    def dist_coeffs(self):
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])

    @property
    def cam_mtx(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])


@dataclass
class CamExtrinsics:
    rotation_vector: np.ndarray
    translation_vector: np.ndarray
    img_path: Optional[str] = None
    cam_name: Optional[str] = None

    @property
    def rotation_matrix(self):
        return cv2.Rodrigues(self.rotation_vector)[0]

    @property
    def transformation_matrix(self):
        return np.hstack((self.rotation_matrix, self.translation_vector))

    @property
    def xyz(self):
        rotation_matrix_inv = np.transpose(self.rotation_matrix)
        # Calculate the camera position in world coordinates
        return -np.dot(rotation_matrix_inv, self.translation_vector)

    @property
    def extrinsic_matrix(self):
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :4] = self.transformation_matrix
        return extrinsic_matrix


@dataclass
class MarkerData:
    img: np.ndarray
    x1: int
    y1: int
    x2: int
    y2: int
    spikey: int


@dataclass
class ImgData:
    img: np.ndarray
    mask: np.ndarray
    imgp: str
    extrinsics: "CamExtrinsics"
    intrinsics: "CamIntrinsics"
    buffered_dims: Tuple[int, int, int, int] = None
    img_centre: np.ndarray = None
    mask_centre: np.ndarray = None
    error_msg: str = ""

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.img.shape


@dataclass
class EarViewDimData:
    xmin: List[int] = field(default_factory=list)
    xmax: List[int] = field(default_factory=list)
    ymin: List[int] = field(default_factory=list)
    ymax: List[int] = field(default_factory=list)
    ear_height: List[float] = field(default_factory=list)
    ear_width: List[float] = field(default_factory=list)


@dataclass
class ArchiveImgDims:
    cam_name: str
    orig_resolutions: Tuple[int, int]
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass
class ArchiveImgData:
    imgs: List[np.ndarray]
    img_names: List[str]
    img_dims: ArchiveImgDims


class CamTrinsics:
    def __init__(
        self,
        intrinsic_p: str,
        real_world_points_p: str,
        ee: bool = False,
    ) -> None:

        self.cam_intrinsic_df = pd.read_csv(intrinsic_p)
        self.real_world_points_df = pd.read_csv(real_world_points_p)
        self.marker_detector = apriltag("tagStandard41h12")
        self.cam_pos_list: List[float] = []
        self.cam_dist_list: List[float] = []
        self.orig_dist_list: List[float] = []

        self.marker_roi: MarkerData = None
        self.cam_intrinsics: CamIntrinsics = None
        self.cam_extrinsics: CamExtrinsics = None

        self._ee = ee

    """def px_to_mm(self, px_value) -> float:
        if self.cam_intrinsic_df["px_to_mm"].values[0] is None:
            raise ValueError("No pixel to mm conversion factor")
        return px_value / self.cam_intrinsic_df["px_to_mm"].values[0]"""

    def get_cam_name(self, imgp: str, img: np.ndarray) -> str:
        if "low" in imgp:
            pos = "low"
        elif "up" in imgp:
            pos = "up"
        else:
            print(imgp)
            raise ValueError("Could not detect known camera")

        if self._ee:
            prefix = "P Trial "
        else:
            prefix = ""

        if img.shape[:2] == (5376, 3672):
            return f"{prefix}20MP {pos}"
        elif img.shape[:2] == (2592, 1944):
            return f"{prefix}5MP {pos}"
        else:
            print(imgp)
            raise ValueError("Could not detect known camera")

    def get_marker_roi(self, img: np.ndarray, camname: str) -> MarkerData:
        df = self.cam_intrinsic_df
        dfcam = df.loc[df["Cam"] == f"{camname}"]
        x1 = dfcam["marker_x1"].values[0]
        y1 = dfcam["marker_y1"].values[0]
        x2 = dfcam["marker_x2"].values[0]
        y2 = dfcam["marker_y2"].values[0]
        spikey = dfcam["spike_y"].values[0]
        if len(img.shape) == 3:
            imgroi = cv2.cvtColor(img[y1:y2, x1:x2].copy(), cv2.COLOR_BGR2GRAY)
        else:
            imgroi = img[y1:y2, x1:x2].copy()

        return MarkerData(imgroi, x1, y1, x2, y2, spikey)

    def get_intrinsics(self, camname: str, img: np.ndarray) -> CamIntrinsics:
        df = self.cam_intrinsic_df

        dfcam = df.loc[df["Cam"] == f"{camname}"]

        calibdata = CamIntrinsics(
            dfcam["fx"].values[0],
            dfcam["fy"].values[0],
            dfcam["cx"].values[0],
            dfcam["cy"].values[0],
            dfcam["k1"].values[0],
            dfcam["k2"].values[0],
            dfcam["p1"].values[0],
            dfcam["p2"].values[0],
            dfcam["k3"].values[0],
            w=img.shape[1],
            h=img.shape[0],
            cam_name=camname,
        )
        if "low" in camname:
            calibdata.px_to_mm = dfcam["pixel_to_mm"].values[0]
        calibdata.cam_distance = dfcam["expected_cam_dist"].values[0]
        calibdata.origin_distance = dfcam["expected_origin_dist"].values[0]
        return calibdata

    def marker_prepro(
        self,
        img_input: Union[str, np.ndarray],
    ):
        if isinstance(img_input, str):
            img_path = img_input
            img, cam_name = read_ear_img(img_path, gray_scale=True)
            if cam_name is not None:
                self.cam_name = cam_name
            else:
                self.cam_name = self.get_cam_name(img_input, img)
        elif isinstance(img_input, np.ndarray):
            img_path = None
            img = img_input
        else:
            raise ValueError("Image path or image array must be provided")

        self.cam_intrinsics = self.get_intrinsics(self.cam_name, img)
        self.marker_roi = self.get_marker_roi(img, self.cam_name)

        return self.marker_roi, self.cam_intrinsics, img_path, self.real_world_points_df

    def reset(self):
        if len(self.cam_pos_list) != 50:
            raise RuntimeError("Less or more than 50 images have been taken")
        self.cam_pos_list = []
        self.cam_dist_list = []
        self.orig_dist_list = []

        self.marker_roi: MarkerData = None
        self.cam_intrinsics: CamIntrinsics = None
        self.cam_extrinsics: CamExtrinsics = None


def _visualize_marker_detection(
    img: np.ndarray,
    tx: Optional[int] = None,
    ty: Optional[int] = None,
    tag_id: Optional[int] = None,
    img_path: Optional[str] = None,
) -> np.ndarray:
    if len(img.shape) == 2:
        vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = img

    circle_cond = tx is not None and ty is not None and tag_id is not None
    if circle_cond:
        cv2.circle(vis_img, (int(tx), int(ty)), 6, (0, 255, 0), 2)
        cv2.putText(
            vis_img,
            f"{tag_id}",
            (int(tx), int(ty)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    if img_path is not None:
        cv2.putText(
            vis_img,
            os.path.basename(img_path),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
    return vis_img


def detect_markers_and_orient_cams(
    marker_roi: "MarkerData",
    cam_intrinsics: "CamIntrinsics",
    img_path: str,
    real_world_df: pd.DataFrame,
    use_corners: bool = True,
    use_vert_corners: bool = True,
    p_dataset: bool = False,
) -> Union["CamExtrinsics", np.ndarray]:
    detector = apriltag("tagStandard41h12")
    tags = detector.detect(marker_roi.img)
    if len(tags) <= 1:
        print("Warning: Not enough tags detected")
        return _visualize_marker_detection(marker_roi.img.copy(), img_path=img_path)

    world_points = []
    image_points = []

    for ti, tag in enumerate(tags):
        if use_corners:
            ti *= 5
        tx, ty = tag["center"]
        tag_id = tag["id"]
        if tag_id > 7:
            print("Warning: Wrong tag detected")
            return _visualize_marker_detection(
                marker_roi.img.copy(), tx, ty, tag_id, img_path
            )
        world_tag = real_world_df.loc[real_world_df["id"] == tag_id]

        world_points.append(
            world_tag.loc[real_world_df["corner"] == -1, ["x", "y", "z"]].values
        )
        image_points.append([tx + marker_roi.x1, ty + marker_roi.y1])
        swap_m0 = False

        if (
            tag_id == 0
            and tag["lb-rb-rt-lt"][0][1] > tag["lb-rb-rt-lt"][2][1]
            and p_dataset
        ):
            # During the P experiment, tag 0 was rotated
            print("Having to Swap corners for tag 0")
            swap_m0 = True

        if use_corners:
            if use_vert_corners or tag_id >= 4:
                for ci, corner in enumerate(tag["lb-rb-rt-lt"]):
                    ti += 1
                    cx, cy = corner
                    if swap_m0:
                        wpi = ci - 1
                        if wpi < 0:
                            wpi = 3
                    else:
                        wpi = ci

                    world_pt = world_tag.loc[
                        real_world_df["corner"] == wpi,
                        ["x", "y", "z"],
                    ].values

                    world_points.append(world_pt)
                    image_points.append([cx + marker_roi.x1, cy + marker_roi.y1])

    image_points = np.array(image_points, dtype=np.float32)
    world_points = np.array(world_points, dtype=np.float32)
    if len(image_points) >= 6:
        f = cv2.SOLVEPNP_ITERATIVE
    elif len(image_points) >= 4:
        f = cv2.SOLVEPNP_EPNP
    else:
        raise ValueError(f"Not enough points to solve PnP. {img_path}")

    success, rotation_vector, translation_vector = cv2.solvePnP(
        world_points,
        image_points,
        cam_intrinsics.cam_mtx,
        cam_intrinsics.dist_coeffs,
        flags=f,
    )
    cam_extrinsics = CamExtrinsics(
        rotation_vector, translation_vector, img_path, cam_intrinsics.cam_name
    )

    if not success:
        raise RuntimeError(
            f"Could not solve PnP for marker based cameras orientation. {img_path}"
        )

    return cam_extrinsics


def read_ear_img(
    img_path: str, gray_scale: bool = False
) -> Tuple[np.ndarray, Optional[str]]:
    cam_name = None
    if gray_scale:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_path)

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    if "_low_" in img_name:
        cam_pos = "low"
    elif "_up_" in img_name:
        cam_pos = "up"
    else:
        raise ValueError(
            "Img name does not follow naming convention. Looking for '_low_' or '_up_' in img name"
        )

    if img_name.endswith("_crp"):
        ear_dir = os.path.dirname(img_path)
        json_files = [f for f in os.listdir(ear_dir) if f.endswith(".json")]
        if len(json_files) != 2:
            raise FileNotFoundError("Two json files are expected in cropped directory")
        json_name = [json_f for json_f in json_files if cam_pos in json_f][0]
        json_p = os.path.join(ear_dir, json_name)
        with open(json_p, "r") as f:
            crop_dim_data = json.load(f)
        cam_name = crop_dim_data["cam_name"]

        # pad the cropped image back to original size
        if gray_scale:
            padding_shape = (
                crop_dim_data["orig_resolutions"][0],
                crop_dim_data["orig_resolutions"][1],
            )
        else:
            padding_shape = (
                crop_dim_data["orig_resolutions"][0],
                crop_dim_data["orig_resolutions"][1],
                3,
            )
        img_padded = np.zeros(padding_shape, dtype=np.uint8)
        img_padded[
            crop_dim_data["ymin"] : crop_dim_data["ymax"],
            crop_dim_data["xmin"] : crop_dim_data["xmax"],
        ] = img

        assert list(img_padded.shape[:2]) == crop_dim_data["orig_resolutions"]
        return img_padded, cam_name

    return img, cam_name


class ImgDimensions:
    def __init__(
        self,
        img: np.ndarray,
        cam_intrinsics: CamIntrinsics,
        marker_roi: MarkerData,
        hsv_thresh_path: str,
    ):
        self.img = img
        self.cam_intrinsics = cam_intrinsics
        self.marker_roi = marker_roi
        self.hsv_thresh_path = hsv_thresh_path

        self.undist_cam_mtx = None
        self.dist_cam_mtx = None
        self.dist_array = None

    def undistort(
        self,
        fg_img: np.ndarray,
        mask: np.ndarray,
        ear_dims: List,
        marker_dims: List,
    ):

        fx, fy, cx, cy = (
            self.cam_intrinsics.fx,
            self.cam_intrinsics.fy,
            self.cam_intrinsics.cx,
            self.cam_intrinsics.cy,
        )
        self.dist_array = np.array(
            [
                self.cam_intrinsics.k1,
                self.cam_intrinsics.k2,
                self.cam_intrinsics.p1,
                self.cam_intrinsics.p2,
                self.cam_intrinsics.k3,
            ]
        )

        self.dist_cam_mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        w, h, _ = fg_img.shape

        self.undist_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(
            self.dist_cam_mtx, self.dist_array, (w, h), 0, (w, h)
        )

        undist_fg_img = cv2.undistort(
            fg_img, self.dist_cam_mtx, self.dist_array, None, self.undist_cam_mtx
        )
        undist_mask = cv2.undistort(
            mask, self.dist_cam_mtx, self.dist_array, None, self.undist_cam_mtx
        )
        # undistort points:
        undist_ear_dims = cv2.undistortPoints(
            ear_dims, self.dist_cam_mtx, self.dist_array, P=self.undist_cam_mtx
        )
        undist_ear_dims = np.squeeze(undist_ear_dims.astype(np.int32), axis=1)
        undist_marker_dims = cv2.undistortPoints(
            marker_dims, self.dist_cam_mtx, self.dist_array, P=self.undist_cam_mtx
        )
        undist_marker_dims = np.squeeze(undist_marker_dims.astype(np.int32), axis=1)

        # undist_imgs = (undist_orig_img, undist_fg_img)

        return (
            undist_fg_img,
            undist_mask,
            undist_ear_dims,
            undist_marker_dims,
        )

    def redistort_points(self, pts: np.ndarray):
        if isinstance(pts, Tuple):
            xmin, xmax, ymin, ymax = pts
            pts = np.array([[xmin, ymin], [xmax, ymax]], dtype=np.float32)

        redist_pts = cv2.undistortPoints(
            pts.astype(np.float32),
            self.undist_cam_mtx,
            self.dist_array * -1,
            P=self.dist_cam_mtx,
        )
        return np.squeeze(redist_pts.astype(np.int32), axis=1)

    def background_masking(self, img: np.ndarray):
        try:
            with open(self.hsv_thresh_path, "r") as f:
                hsv = json.load(f)
        except:
            raise FileNotFoundError("HSV Threshholds for background not found")

        hmin, hmax, smin, smax, vmin, vmax = (
            hsv["hmin"],
            hsv["hmax"],
            hsv["smin"],
            hsv["smax"],
            hsv["vmin"],
            hsv["vmax"],
        )
        imgBLUR = cv2.GaussianBlur(img, (3, 3), 0)
        imgHSV = cv2.cvtColor(imgBLUR, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([hmin, smin, vmin])
        higher_hsv = np.array([hmax, smax, vmax])
        mask = cv2.inRange(imgHSV, lower_hsv, higher_hsv)
        mask = cv2.erode(mask, (3, 3), mask, iterations=1)
        mask = cv2.dilate(mask, (3, 3), mask, iterations=1)
        # Find the largest external contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a new mask with the same dimensions as the original mask
        filled_mask = np.zeros_like(mask)

        # Draw the largest contour filled on the new mask
        cv2.drawContours(filled_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Update the mask with the filled mask
        mask = filled_mask
        foreground_img = cv2.bitwise_and(img, img, mask=mask).copy()
        return mask, foreground_img

    def get_ear_dims(
        self,
        ear_dims: Tuple[EarViewDimData, EarViewDimData],
        marker_dims: Tuple[EarViewDimData, EarViewDimData],
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        Tuple[EarViewDimData, EarViewDimData],
        Tuple[EarViewDimData, EarViewDimData],
    ]:
        ear_dims_dist, ear_dims_undist = ear_dims
        marker_dims_dist, marker_dims_undist = marker_dims

        raw_mask, fg_img = self.background_masking(self.img.copy())
        mask = raw_mask.copy()
        mask = cv2.erode(mask, (3, 3), iterations=3)
        mask = cv2.dilate(mask, (3, 3), iterations=3)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_size = 0
        biggest_cnt = None
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area > max_size:
                max_size = area
                biggest_cnt = cnt
        xidcs, yidcs = biggest_cnt[:, :, 0], biggest_cnt[:, :, 1]

        ear_dims_xmin = np.amin(xidcs)
        ear_dims_xmax = np.amax(xidcs)
        ear_dims_ymin = np.amin(yidcs)
        ear_dims_ymax = np.amax(yidcs)

        ear_dims_dist.xmin.append(ear_dims_xmin)
        ear_dims_dist.ymin.append(ear_dims_ymin)
        ear_dims_dist.xmax.append(ear_dims_xmax)
        ear_dims_dist.ymax.append(ear_dims_ymax)
        ear_dims_dist_new = np.array(
            [[ear_dims_xmin, ear_dims_ymin], [ear_dims_xmax, ear_dims_ymax]],
            dtype=np.float32,
        )

        xmin_crop, xmax_crop = np.amin(xidcs), np.amax(xidcs)
        # check whether x extends of either markers or maize ear are larger
        if self.marker_roi.x1 < xmin_crop:
            xmin_crop = self.marker_roi.x1
        if self.marker_roi.x2 > xmax_crop:
            xmax_crop = self.marker_roi.x2

        ymin_crop, ymax_crop = np.amin(yidcs), self.marker_roi.y2
        marker_dims_dist.xmin.append(xmin_crop)
        marker_dims_dist.ymin.append(ymin_crop)
        marker_dims_dist.xmax.append(xmax_crop)
        marker_dims_dist.ymax.append(ymax_crop)
        marker_dims_dist_new = np.array(
            [[xmin_crop, ymin_crop], [xmax_crop, ymax_crop]],
            dtype=np.float32,
        )

        undist_fg_img, undist_mask, ear_dims_undist_new, marker_dims_undist_new = (
            self.undistort(
                self.img.copy(), mask, ear_dims_dist_new, marker_dims_dist_new
            )
        )

        if self.cam_intrinsics.px_to_mm is not None:  # low camera only
            cnt_hw_dist = biggest_cnt[biggest_cnt[:, :, 1] <= self.marker_roi.spikey]
            cnt_hw_undist = cv2.undistortPoints(
                cnt_hw_dist.astype(np.float32),
                self.dist_cam_mtx,
                self.dist_array,
                P=self.undist_cam_mtx,
            )
            cnt_hw_undist = cnt_hw_undist.astype(np.int32)

            rect = cv2.minAreaRect(cnt_hw_undist)
            width, height = rect[1]

            if width > height:
                width, height = height, width

            width = width / self.cam_intrinsics.px_to_mm
            height = height / self.cam_intrinsics.px_to_mm
            ear_dims_undist.ear_width.append(width)
            ear_dims_undist.ear_height.append(height)

        ear_dims_undist.xmin.append(ear_dims_undist_new[0][0])
        ear_dims_undist.ymin.append(ear_dims_undist_new[0][1])
        ear_dims_undist.xmax.append(ear_dims_undist_new[1][0])
        ear_dims_undist.ymax.append(ear_dims_undist_new[1][1])

        marker_dims_undist.xmin.append(marker_dims_undist_new[0][0])
        marker_dims_undist.ymin.append(marker_dims_undist_new[0][1])
        marker_dims_undist.xmax.append(marker_dims_undist_new[1][0])
        marker_dims_undist.ymax.append(marker_dims_undist_new[1][1])

        ear_dims_new = (ear_dims_dist, ear_dims_undist)
        marker_dims_new = (marker_dims_dist, marker_dims_undist)

        return undist_fg_img, undist_mask, ear_dims_new, marker_dims_new

    def get_original_img(self):
        return self.img


class PreImageAnalyzer:
    def __init__(
        self,
        cam_extrinsics: List[CamExtrinsics],
        cam_intrinsic: CamIntrinsics,
        marker_data: MarkerData,
        hsv_thresh_path: str,
        img_n: int = 50,
    ):
        # covering the ear and marker dimensions:
        self.marker_crop_data = (
            EarViewDimData(),
            EarViewDimData(),
        )  # distorted, undistorted
        self.ear_crop_data = (
            EarViewDimData(),
            EarViewDimData(),
        )  # distorted, undistorted
        self.undist_ear_crop_data = None
        self.cam_extrinsics = cam_extrinsics
        self.cam_intrinsics = cam_intrinsic
        self.marker_data = marker_data
        self.hsv_thresh_path = hsv_thresh_path
        self.img_n = img_n
        self.cam_name = cam_extrinsics[0].cam_name

        self.foreground_imgs: List[ImgData] = []
        # self.undist_imgs: List[np.ndarray] = []
        self.orig_dist_list: List[str] = []
        self.cam_dist_list: List[str] = []
        self.err_lst: List[str] = []
        self.img = None
        self.common_ear_crop_dims: Optional[Tuple[int, int, int, int]] = (
            None  # (xmin, xmax, ymin, ymax)
        )
        self.outlier_indices: List[int] = []

    def compute_foreground(self, do_archive_imgs: bool = False):

        archive_lst = []
        for idx in range(self.img_n):
            img_path = self.cam_extrinsics[idx].img_path
            self.img, _ = read_ear_img(img_path)
            imgred = ImgDimensions(
                self.img, self.cam_intrinsics, self.marker_data, self.hsv_thresh_path
            )
            foreground_img, mask, self.ear_crop_data, self.marker_crop_data = (
                imgred.get_ear_dims(self.ear_crop_data, self.marker_crop_data)
            )
            self.undist_ear_crop_data = self.ear_crop_data[1]
            self.cam_intrinsics.undist_cam_mtx = imgred.undist_cam_mtx
            # undist_img, foreground_img = undist_imgs
            fg_img_obj = ImgData(
                foreground_img,
                mask,
                self.cam_extrinsics[idx].img_path,
                self.cam_extrinsics[idx],
                self.cam_intrinsics,
            )
            self.foreground_imgs.append(fg_img_obj)
            # self.undist_imgs.append(undist_img)

            if do_archive_imgs:
                # distorted(original)
                archive_lst.append(imgred.get_original_img())

        self.common_ear_crop_dims, common_dist_marker_crops, self.outlier_indices = (
            self.filter_irregular_ear_dims()
        )
        processed_errors = self.process_errors(self.outlier_indices)

        if not do_archive_imgs:
            return processed_errors, None

        else:
            archive_data = self.get_cropped_archive_imgs(
                archive_lst, common_dist_marker_crops
            )
            return processed_errors, archive_data

    def process_errors(self, outlier_indices: List[int]):
        fg_e_msgs = []
        if len(outlier_indices) == 0:
            return fg_e_msgs

        for i in outlier_indices:
            self.undist_ear_crop_data.xmin[i] = None
            self.undist_ear_crop_data.xmax[i] = None
            self.undist_ear_crop_data.ymin[i] = None
            self.undist_ear_crop_data.ymax[i] = None
            if "low" in self.cam_name:
                self.undist_ear_crop_data.ear_height[i] = None
                self.undist_ear_crop_data.ear_width[i] = None
            e_msg: str = (
                f"An irregular object might have been detected. The following image shows irregular ear dimensions and will be skipped:\n{self.foreground_imgs[i].imgp}"
            )
            print(e_msg)
            self.foreground_imgs[i].error_msg = e_msg
            fg_e_msgs.append(e_msg)

        return fg_e_msgs

    def filter_irregular_ear_dims(
        self, iqr_multiplier: float = 3, resolution_factor: float = 0.03
    ):
        unfiltered = np.array(
            [
                self.undist_ear_crop_data.xmin,
                self.undist_ear_crop_data.xmax,
                self.undist_ear_crop_data.ymin,
                self.undist_ear_crop_data.ymax,
            ]
        ).T

        resolution_offset = int(resolution_factor * self.cam_intrinsics.w)
        # compute IQR thresholds per column
        Q1 = np.percentile(unfiltered, 25, axis=0)
        Q3 = np.percentile(unfiltered, 75, axis=0)
        IQR = Q3 - Q1
        lower = Q1 - iqr_multiplier * (IQR + resolution_offset)
        upper = Q3 + iqr_multiplier * (IQR + resolution_offset)

        # find any coordinate outside [lower, upper]
        is_outlier = np.any((unfiltered < lower) | (unfiltered > upper), axis=1)
        outlier_indices = np.nonzero(is_outlier)[0].tolist()
        inlier_indices = np.nonzero(~is_outlier)[0].tolist()

        xmin_crop, xmax_crop = np.amin(
            np.array(self.undist_ear_crop_data.xmin)[inlier_indices]
        ), np.amax(np.array(self.undist_ear_crop_data.xmax)[inlier_indices])
        ymin_crop, ymax_crop = np.amin(
            np.array(self.undist_ear_crop_data.ymin)[inlier_indices]
        ), np.amax(np.array(self.undist_ear_crop_data.ymax)[inlier_indices])

        dist_marker_crop = self.marker_crop_data[0]
        xmin_marker_crop, xmax_marker_crop = np.amin(
            np.array(dist_marker_crop.xmin)[inlier_indices]
        ), np.amax(np.array(dist_marker_crop.xmax)[inlier_indices])
        ymin_marker_crop, ymax_marker_crop = np.amin(
            np.array(dist_marker_crop.ymin)[inlier_indices]
        ), np.amax(np.array(dist_marker_crop.ymax)[inlier_indices])

        return (
            (
                xmin_crop,
                xmax_crop,
                ymin_crop,
                ymax_crop,
            ),
            (
                xmin_marker_crop,
                xmax_marker_crop,
                ymin_marker_crop,
                ymax_marker_crop,
            ),
            outlier_indices,
        )

    def get_cropped_archive_imgs(
        self, archive_imgs_raw, dist_marker_dims, buffer_ratio: float = 0.05
    ):
        xmin0, xmax0, ymin0, ymax0 = dist_marker_dims

        img_res = archive_imgs_raw[0].shape[:2]
        buffer_px = int(buffer_ratio * img_res[0])
        xmin = int(xmin0 - buffer_px) if xmin0 - buffer_px >= 0 else 0
        ymin = int(ymin0 - buffer_px) if ymin0 - buffer_px >= 0 else 0
        xmax = int(xmax0 + buffer_px) if xmax0 + buffer_px <= img_res[1] else img_res[1]
        ymax = int(ymax0 + buffer_px) if ymax0 + buffer_px <= img_res[0] else img_res[0]

        archive_img_dims = ArchiveImgDims(
            self.cam_intrinsics.cam_name, img_res, xmin, ymin, xmax, ymax
        )
        archive_imgs = [img[ymin:ymax, xmin:xmax].copy() for img in archive_imgs_raw]
        archive_img_names = [
            os.path.basename(ce.img_path) for ce in self.cam_extrinsics
        ]
        archive_data = ArchiveImgData(archive_imgs, archive_img_names, archive_img_dims)
        return archive_data

    def get_img_data_list(self):
        return self.foreground_imgs.copy()

    def get_ear_crop_dims(self):
        return self.common_ear_crop_dims

    def get_ear_wh(self):
        if len(self.undist_ear_crop_data.ear_width) == self.img_n:
            ear_width_cleaned = [
                w for w in self.undist_ear_crop_data.ear_width if w is not None
            ]
            ear_width = np.median(ear_width_cleaned)
            ear_height_cleaned = [
                h for h in self.undist_ear_crop_data.ear_height if h is not None
            ]
            ear_height = np.median(ear_height_cleaned)
        else:
            ear_width = None
            ear_height = None
        return ear_width, ear_height

    def get_voxel_grid_dims(
        self, buffer_factor=1.2
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        Get the voxel grid dimensions depending on the cropped ear images.
        Tilted and large ears require a larger voxel grid while smaller and straight ears require a smaller one.
        """
        ear_xmin_cleaned = [x for x in self.undist_ear_crop_data.xmin if x is not None]
        ear_xmax_cleaned = [x for x in self.undist_ear_crop_data.xmax if x is not None]
        ear_ymin_cleaned = [y for y in self.undist_ear_crop_data.ymin if y is not None]
        ear_ymax_cleaned = [y for y in self.undist_ear_crop_data.ymax if y is not None]

        xmin_crop, xmax_crop = np.amin(ear_xmin_cleaned), np.amax(ear_xmax_cleaned)
        ymin_crop, ymax_crop = np.amin(ear_ymin_cleaned), np.amax(ear_ymax_cleaned)
        xy_dim_0 = (xmax_crop - xmin_crop) / self.cam_intrinsics.px_to_mm
        z_dim_0 = (ymax_crop - ymin_crop) / self.cam_intrinsics.px_to_mm
        xy_dim_half = (xy_dim_0 * buffer_factor) / 2
        z_dim = z_dim_0 * buffer_factor
        return (
            (-xy_dim_half, xy_dim_half),
            (-xy_dim_half, xy_dim_half),
            (0.0, z_dim),
        )

    def get_camera_distances(self, cam_positions: List[np.ndarray], idx: int):
        origin_distance = np.linalg.norm(cam_positions[idx])
        self.orig_dist_list.append(origin_distance)

        if idx >= 1:
            cam_distance = np.linalg.norm(cam_positions[idx - 1] - cam_positions[idx])
            self.cam_dist_list.append(cam_distance)

        if idx == 49:
            final_distance = np.linalg.norm(cam_positions[0] - cam_positions[idx])
            self.cam_dist_list.append(final_distance)

            orig_msgs, orig_distr = self.dist_check(
                self.orig_dist_list, 0.1, mode="origin"
            )

            camdist_msgs, camdist_distr = self.dist_check(
                self.cam_dist_list, 0.2, mode="cam_sequence"
            )
            self.err_lst.extend(orig_msgs)
            self.err_lst.extend(camdist_msgs)
            return orig_distr, camdist_distr

        elif idx >= 50:
            raise ValueError("More than 50 images have been taken")

    ''' def _plot_camera_positions(self, cam_positions: List[np.ndarray]):
        cam_positions = np.array(cam_positions)
        x_positions = cam_positions[:, 0]
        y_positions = cam_positions[:, 1]

        plt.figure(figsize=(8, 6))
        plt.plot(x_positions, y_positions, marker="o", label="Camera Positions")
        plt.xlabel("X Position (mm)")
        plt.ylabel("Y Position (mm)")
        plt.title("Camera Positions in X-Y Plane")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        canvas = plt.gca().figure.canvas
        canvas.draw()
        plot_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return plot_img'''
    def _plot_camera_positions(self, cam_positions: np.ndarray) -> np.ndarray[np.uint8]:
        """
        Plot camera XY positions and return the rendered plot as an RGB uint8 image (H, W, 3).
        Works on Python 3.12 and in headless environments.
        """
        arr = np.asarray(cam_positions, dtype=float)

        if arr.size == 0:
            raise ValueError("cam_positions is empty.")

        # Accept [(x,y), ...], np.array([[x,y],...]) or flat [x0,y0,x1,y1,...]
        if arr.ndim == 1:
            if arr.size % 2 != 0:
                raise ValueError("1D cam_positions must contain an even number of elements (x,y pairs).")
            arr = arr.reshape(-1, 2)
        elif arr.shape[1] < 2:
            raise ValueError("Each camera position must have at least two components (x, y).")

        x, y = arr[:, 0], arr[:, 1]

        # Object-oriented Matplotlib (no need for a GUI backend)
        fig = Figure(figsize=(8, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot(x, y, marker="o", label="Camera Positions")
        ax.set_xlabel("X Position (mm)")
        ax.set_ylabel("Y Position (mm)")
        ax.set_title("Camera Positions in X-Y Plane")
        ax.grid(True)
        ax.legend()
        ax.set_aspect("equal", adjustable="box")  # optional, but often desirable
        fig.tight_layout()

        # Render and extract as RGB
        canvas.draw()
        w, h = canvas.get_width_height()

        # Prefer RGB if available; otherwise convert RGBA → RGB
        if hasattr(canvas, "tostring_rgb"):
            img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        else:
            rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            img = rgba[..., :3].copy()

        # Cleanup
        fig.clear()
        return img

    def camera_position_control(self):
        self.cam_pos_plot = None
        cam_pos_list = [ce.xyz for ce in self.cam_extrinsics]
        if len(cam_pos_list) != 50:
            raise RuntimeError(
                f"{len(cam_pos_list)} images have been provided, but 50 are expected"
            )
        orig_distr, camdist_distr = [
            self.get_camera_distances(cam_pos_list, i) for i in range(len(cam_pos_list))
        ][-1]
        if len(self.err_lst) > 0:
            self.cam_pos_plot = self._plot_camera_positions(cam_pos_list)
        return orig_distr, camdist_distr

    def dist_check(
        self, lst: List[float], error_perc=0.2, mode="cam_sequence"
    ) -> Tuple[List[str], Tuple[float, float, float]]:
        min_dist = np.amin(lst)
        max_dist = np.amax(lst)
        if mode == "origin":
            expected_dist = self.cam_intrinsics.origin_distance
        elif mode == "cam_sequence":
            expected_dist = self.cam_intrinsics.cam_distance

        if expected_dist is None:
            expected_dist = np.median(lst)

        dist_distr_tuple = (min_dist, max_dist, expected_dist)
        max_dev = error_perc * expected_dist
        msgs = []
        for i, dist in enumerate(lst):
            if abs(expected_dist - dist) > max_dev:
                img2 = i + 1
                if img2 == len(lst):
                    img2 = 0
                if mode == "cam_sequence":
                    msg = f"Distance between image {i} and image {img2} for camera {self.cam_name} is {dist:.2f} mm while the expected distance is {expected_dist:.2f} mm"
                elif mode == "origin":
                    msg = f"Distance to origin for camera {self.cam_name} is {dist:.2f} mm while the expected distance is {expected_dist:.2f} mm"
                msgs.append(msg)
        return msgs, dist_distr_tuple

    def reset(self):
        self.undist_imgs = []
        self.foreground_imgs = []
        self.img = None


class RoiImgGenerator:
    def __init__(
        self,
        foreground_imgs: List[ImgData],
        common_ear_crop_dims: Tuple[int, int, int, int],
    ):
        self.roi_imgs_overlap = self.add_overlap(foreground_imgs)
        self.all_cropped_objs: List[ImgData] = []
        self.common_ear_crop_dims = common_ear_crop_dims

    def add_overlap(self, imgs: List[ImgData], overalp_n: int = 0):
        for i in range(overalp_n):
            imgs.append(imgs[i])
        return imgs

    def __len__(self):
        return len(self.roi_imgs_overlap)

    def get_all_cropped_objs(self) -> List[ImgData]:
        return self.all_cropped_objs

    def gen_roi_img_cropped(self):
        for i in range(len(self.roi_imgs_overlap)):
            img_obj = self.roi_imgs_overlap.pop(0)
            img_crop, buff_crop_dims = self.get_ear_crop(img_obj.img)
            mask_crop, _ = self.get_ear_crop(img_obj.mask)
            if img_obj.error_msg:
                img_centre = None
                mask_centre = None
            else:
                img_centre, mask_centre = self.get_centre_crop(img_crop, mask_crop)

            cropped_obj = ImgData(
                img_crop,
                mask_crop,
                img_obj.imgp,
                img_obj.extrinsics,
                img_obj.intrinsics,
                buff_crop_dims,
                img_centre=img_centre,
                mask_centre=mask_centre,
            )
            self.all_cropped_objs.append(cropped_obj)
            yield cropped_obj, i

    def get_centre_crop(self, img: np.ndarray, mask: np.ndarray, ratio: float = 0.6):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            raise ValueError("No contours found in mask")
        largest_contour = max(contours, key=cv2.contourArea)
        x1, y1, w, h = cv2.boundingRect(largest_contour)
        center_x = x1 + w // 2
        crop_width = int(w * ratio)
        x_start = max(center_x - crop_width // 2, 0)
        x_end = min(center_x + crop_width // 2, img.shape[1])
        centre_img = img.copy()
        centre_mask = mask.copy()
        centre_img[:, :x_start] = 0
        centre_img[:, x_end:] = 0
        centre_mask[:, :x_start] = 0
        centre_mask[:, x_end:] = 0
        return centre_img, centre_mask

    def get_ear_crop(self, img: np.ndarray):
        img_crop, buff_crop_dims = self._get_img_crop(img, self.common_ear_crop_dims)
        return img_crop, buff_crop_dims

    def _get_img_crop(
        self,
        img: np.ndarray,
        crop_dims: Tuple[int, int, int, int],
        buffer_px: int = 20,
    ):
        xmin_crop, xmax_crop, ymin_crop, ymax_crop = crop_dims
        buffered_xmin = xmin_crop - buffer_px if xmin_crop - buffer_px >= 0 else 0
        buffered_xmax = (
            xmax_crop + buffer_px
            if xmax_crop + buffer_px <= img.shape[1]
            else img.shape[1]
        )
        buffered_ymin = ymin_crop - buffer_px if ymin_crop - buffer_px >= 0 else 0
        buffered_ymax = (
            ymax_crop + buffer_px
            if ymax_crop + buffer_px <= img.shape[0]
            else img.shape[0]
        )
        img_crop = img[buffered_ymin:buffered_ymax, buffered_xmin:buffered_xmax].copy()
        return img_crop, (buffered_xmin, buffered_xmax, buffered_ymin, buffered_ymax)

    def __iter__(self):
        return self.gen_roi_img_cropped()
