from dataclasses import dataclass, field
from apriltag import apriltag
import cv2
import numpy as np
import pandas as pd
from typing import Union, List, Tuple



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
    px_to_mm: float = None

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
class EarViewDimData:
    xmin: List[int] = field(default_factory=list)
    xmax: List[int] = field(default_factory=list)
    ymin: List[int] = field(default_factory=list)
    ymax: List[int] = field(default_factory=list)
    ear_height: List[float] = field(default_factory=list)
    ear_width: List[float] = field(default_factory=list)


class CamTrinsics:
    def __init__(
        self,
        intrinsic_p: str,
        real_world_points_p: str,
        camname: str = None,
        campos_n: int = 50,
    ) -> None:

        self.cam_intrinsic_df = pd.read_csv(intrinsic_p)
        self.real_world_points_df = pd.read_csv(real_world_points_p)
        '''self.marker_detector = Detector(
            searchpath=["apriltags"],
            families="tagStandard41h12",
            nthreads=4,
            quad_decimate=3.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )'''
        self.marker_detector = apriltag("tagStandard41h12")
        self.campos_n = campos_n
        self.cam_name = camname
        self.cam_pos_list: List[float] = []
        self.cam_dist_list: List[float] = []
        self.orig_dist_list: List[float] = []

        self.marker_roi: MarkerData = None
        self.cam_intrinsics: CamIntrinsics = None
        self.cam_extrinsics: CamExtrinsics = None

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

        if img.shape == (5376, 3672, 3):
            return f"20MP {pos}"
        elif img.shape == (2592, 1944, 3):
            return f"5MP {pos}"
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
        imgroi = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

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
        )
        if "low" in camname:
            calibdata.px_to_mm = dfcam["pixel_to_mm"].values[0]
        return calibdata

    def get_trinsics(
        self, imgp: Union[str, np.ndarray], camname: str = None
    ) -> Tuple[CamIntrinsics, CamExtrinsics, MarkerData]:
        if isinstance(imgp, str):
            img = cv2.imread(imgp)
            self.cam_name = self.get_cam_name(imgp, img)
        elif isinstance(imgp, np.ndarray):
            img = imgp
        else:
            raise ValueError("Image path or image array must be provided")

        camname = self.cam_name

        self.cam_intrinsics = self.get_intrinsics(camname, img)
        dist_coeffs, camera_matrix = (
            self.cam_intrinsics.dist_coeffs,
            self.cam_intrinsics.cam_mtx,
        )

        self.marker_roi = self.get_marker_roi(img, camname)
        # imgshow = img.copy()

        tags = self.marker_detector.detect(
            self.marker_roi.img)
        if len(tags) < 2:
            raise ValueError("Not enough tags detected")
        world_points = np.zeros(
            (len(tags) * 5, 3), dtype=np.float32
        )  # 1 center + 4 corners
        image_points = np.zeros((len(tags) * 5, 2), dtype=np.float32)
        for ti, tag in enumerate(tags):
            ti *= 5
            tx, ty = tag["center"]
            tag_id = tag["id"]
            if tag_id > 7:
                print("Warning: Wrong tag detected")
                # cv2.circle(imgshow, (int(tx), int(ty)), 6, (0,255,0), 2)
                # ShowMe([imgshow],.3)
                # raise ValueError("Wrong tag detected")
            world_tag = self.real_world_points_df.loc[
                self.real_world_points_df["id"] == tag_id
            ]
            world_points[ti] = world_tag.loc[
                self.real_world_points_df["corner"] == -1, ["x", "y", "z"]
            ].values
            image_points[ti] = [tx + self.marker_roi.x1, ty + self.marker_roi.y1]
            for ci, corner in enumerate(tag["lb-rb-rt-lt"]):
                ti += 1
                cx, cy = corner
                world_points[ti] = world_tag.loc[
                    self.real_world_points_df["corner"] == ci, ["x", "y", "z"]
                ].values
                image_points[ti] = [cx + self.marker_roi.x1, cy + self.marker_roi.y1]
                # cv2.circle(imgshow, (int(cx), int(cy)), 6, (0, 255, 0), 2)
                # ShowMe([imgshow],.3)
            # cv2.circle(imgshow, (int(px), int(py)), 6, (0,255,0), 2)
            # ShowMe([imgshow],.3)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            world_points, image_points, camera_matrix, dist_coeffs
        )
        self.cam_extrinsics = CamExtrinsics(rotation_vector, translation_vector)

        if success:
            pass

        else:
            raise RuntimeError(
                "Could not solve PnP for marker based cameras orientation"
            )

        self.cam_pos_list.append(self.cam_extrinsics.xyz)

        # print("Marker time:", time()-marker_start)
        return self.cam_intrinsics, self.marker_roi, img

    def camera_position_control(self):
        origin_distance = np.linalg.norm(self.cam_pos_list[-1])
        self.orig_dist_list.append(origin_distance)
        # print(f"Distance to origin: {origin_distance}")

        if len(self.cam_pos_list) >= 2:
            cam_distance = np.linalg.norm(self.cam_pos_list[-2] - self.cam_pos_list[-1])
            self.cam_dist_list.append(cam_distance)
            # print(f"Distance to previous camera: {cam_distance}")
        if len(self.cam_pos_list) == self.campos_n:
            final_distance = np.linalg.norm(
                self.cam_pos_list[0] - self.cam_pos_list[-1]
            )
            self.cam_dist_list.append(final_distance)

            if not self.dist_check(self.orig_dist_list, 0.1) and self.campos_n > 2:
                raise RuntimeWarning(
                    f"Distance to origin is not stable ({self.cam_name}){self.cam_dist_list}"
                )
            if not self.dist_check(self.cam_dist_list, 0.2):
                raise RuntimeWarning(
                    f"Distance to previous camera is not stable ({self.cam_name}){self.cam_dist_list}"
                )
        elif len(self.cam_pos_list) > self.campos_n:
            raise ValueError("More than 50 images have been taken")

    def dist_check(self, lst: List[float], error_perc=0.2) -> None:

        min_dist = np.amin(lst)
        max_dist = np.amax(lst)
        mean_dist = np.mean(lst)

        if len(lst) > 2:
            max_dev = error_perc * mean_dist

            if mean_dist - min_dist > max_dev:
                return False
            elif max_dist - mean_dist > max_dev:
                return False
            else:
                return True
        elif len(lst) == 2:
            if max_dist > 51:
                return False
            else:
                return True

    def reset(self):
        # if len(self.cam_pos_list) != 50:
        # raise RuntimeError(f"{len(self.cam_pos_list)} images have been taken for {self.cam_name}")
        self.cam_pos_list = []
        self.cam_dist_list = []
        self.orig_dist_list = []

        self.marker_roi: MarkerData = None
        self.cam_intrinsics: CamIntrinsics = None
        self.cam_extrinsics: CamExtrinsics = None


class ImgSizeReduction:
    def __init__(
        self, img: np.ndarray, cam_intrinsics: CamIntrinsics, marker_roi: MarkerData
    ):

        self.img = img
        self.cam_intrinsics = cam_intrinsics
        self.marker_roi = marker_roi

        self.undist_img: np.ndarray = None
        self.foreground_img: np.ndarray = None
        self.mask: np.ndarray = None

    def undistort(self):

        fx, fy, cx, cy = (
            self.cam_intrinsics.fx,
            self.cam_intrinsics.fy,
            self.cam_intrinsics.cx,
            self.cam_intrinsics.cy,
        )
        dist = np.array(
            [
                self.cam_intrinsics.k1,
                self.cam_intrinsics.k2,
                self.cam_intrinsics.p1,
                self.cam_intrinsics.p2,
                self.cam_intrinsics.k3,
            ]
        )

        cam_mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        w, h, _ = self.img.shape

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            cam_mtx, dist, (w, h), 0, (w, h)
        )
        self.undist_img = cv2.undistort(self.img, cam_mtx, dist, None, newcameramtx)

        return self.undist_img

        # dstroi = cv2.bitwise_and(self.undist_img, self.undist_img, mask=self.background_masking())

    def background_masking(self, hsvname="/home/mais/Desktop/EarScanJetson/Maize"):

        try:
            hsv = np.load(hsvname + "a.npy")

        except:
            raise FileNotFoundError("HSV Threshholds for background not found")

        hmin, hmax, smin, smax, vmin, vmax = hsv
        imgBLUR = cv2.GaussianBlur(self.undist_img, (3, 3), 0)

        imgHSV = cv2.cvtColor(imgBLUR, cv2.COLOR_BGR2HSV)

        lower_hsv = np.array([hmin, smin, vmin])
        higher_hsv = np.array([hmax, smax, vmax])

        mask = cv2.inRange(imgHSV, lower_hsv, higher_hsv)
        self.mask = cv2.erode(mask, (3, 3), mask, iterations=3)
        self.foreground_img = cv2.bitwise_and(
            self.undist_img, self.undist_img, mask=self.mask
        )
        return self.mask

    def get_ear_dims(self, crop_dims: EarViewDimData, ear_dims: EarViewDimData):
        self.undistort()
        self.background_masking()
        mask = self.mask.copy()
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
        cnt_hw = biggest_cnt[biggest_cnt[:, :, 1] <= self.marker_roi.spikey]
        rect = cv2.minAreaRect(cnt_hw)

        width, height = rect[1]

        if width > height:
            width, height = height, width

        if self.cam_intrinsics.px_to_mm is not None:
            width = width / self.cam_intrinsics.px_to_mm
            height = height / self.cam_intrinsics.px_to_mm
            ear_dims.ear_width.append(width)
            ear_dims.ear_height.append(height)

        ear_dims.xmin.append(np.amin(xidcs))
        ear_dims.ymin.append(np.amin(yidcs))
        ear_dims.xmax.append(np.amax(xidcs))
        ear_dims.ymax.append(np.amax(yidcs))

        xmin_crop, xmax_crop = np.amin(xidcs), np.amax(xidcs)
        # check whether x extends of either markers or maize ear are lager
        if self.marker_roi.x1 < xmin_crop:
            xmin_crop = self.marker_roi.x1
        if self.marker_roi.x2 > xmax_crop:
            xmax_crop = self.marker_roi.x2

        ymin_crop, ymax_crop = np.amin(yidcs), self.marker_roi.y2
        crop_dims.xmin.append(xmin_crop)
        crop_dims.ymin.append(ymin_crop)
        crop_dims.xmax.append(xmax_crop)
        crop_dims.ymax.append(ymax_crop)
        # self.undist_img[ymin_crop-buffer_px:ymax_crop+buffer_px, xmin_crop-buffer_px:xmax_crop+buffer_px]

        return crop_dims, ear_dims


class PreImageAnalyzer:
    def __init__(self, trinsics: CamTrinsics, img_n=50):
        self.marker_crop_data = EarViewDimData()
        self.ear_crop_data = EarViewDimData()
        self.trinsics = trinsics
        self.img_n = img_n

        self.foreground_imgs: List[np.ndarray] = []

        self.output_paths: List[str] = []
        self.err_lst: List[str] = []
        self.img = None

    def __call__(self, img: Union[str, np.ndarray]) -> np.ndarray:
        intr_obj, marker_obj, img = self.trinsics.get_trinsics(img)
        self.img = img
        try:
            self.trinsics.camera_position_control()
        except RuntimeWarning as rw:
            print(rw)
            self.err_lst.append(rw)
        except ValueError as ve:
            print(ve)
            self.err_lst.append(ve)

        imgred = ImgSizeReduction(img, intr_obj, marker_obj)
        self.marker_crop_data, self.ear_crop_data = imgred.get_ear_dims(
            self.marker_crop_data, self.ear_crop_data
        )
        self.foreground_imgs.append(imgred.foreground_img)

    """def save_cropped(self):
        if len(self.output_paths) != 50:
            raise ValueError("Less or more than 50 images have been taken per camera")
        for imgp in self.output_paths:
            img = cv2.imread(imgp)
            img_crop, _, _ = self.get_img_crop(img, self.marker_crop_data)
            cv2.imwrite(imgp, img_crop)"""

    def get_marker_crop(
        self,
        img: np.ndarray,
        marker_dims: EarViewDimData,
    ):
        # TODO: Implement this function
        return

    def get_ear_crop(self, img: np.ndarray):
        img_crop = self._get_img_crop(img, self.ear_crop_data)

        return img_crop

    def _get_img_crop(
        self, img: np.ndarray, crop_dims: EarViewDimData, buffer_px: int = 20
    ):

        xmin_crop, xmax_crop = np.amin(crop_dims.xmin), np.amax(crop_dims.xmax)
        ymin_crop, ymax_crop = np.amin(crop_dims.ymin), np.amax(crop_dims.ymax)

        img_crop = img[
            ymin_crop - buffer_px : ymax_crop + buffer_px,
            xmin_crop - buffer_px : xmax_crop + buffer_px,
        ]

        return img_crop

    def get_ear_wh(self):
        if len(self.ear_crop_data.ear_width) == self.img_n:
            ear_width = np.median(
                self.ear_crop_data.ear_width,
            )
            ear_height = np.median(self.ear_crop_data.ear_height)
        else:
            ear_width = None
            ear_height = None

        return ear_width, ear_height

    def roi_img_generator(self):
        if len(self.foreground_imgs) != self.img_n:
            raise ValueError(
                f"Less or more than 50 images have been taken per camera ({len(self.foreground_imgs)})"
            )
        for img in self.foreground_imgs:
            yield self.get_ear_crop(img)

    def __iter__(self):
        return self.roi_img_generator()

    def reset_trinsics(self):
        self.trinsics.reset()



