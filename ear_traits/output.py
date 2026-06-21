import os
import cv2
import json
import numpy as np
import time
import pandas as pd
from dataclasses import dataclass, asdict
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from ear_traits.scan_time_utils import ImgData, ArchiveImgDims, ArchiveImgData
from ear_traits.count_kernels import KernelTraitData, KernelCounting
from ear_traits.voxel_carving_jax import TrackData, VoxelCarvingJAX
from ear_traits.scan_time_utils import CamExtrinsics
from ear_traits.configs import EarTraitConfigs
from ear_traits.kernel_row_kalman import KernelRowAssignment
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class CameraDistanceDistributions:
    """
    Camera distance distributions for low and up camera.
    ----------------
    Parameters:
        orig_distr_low: Tuple[float, float, float] (min, max, median)
            Distances from low cameras position to origin (0, 0, 0)
        camdist_distr_low: Tuple[float, float, float] (min, max, median)
            Distances from camera to camera
        orig_distr_up: Tuple[float, float, float] (min, max, median)
            Distances from up cameras position to origin (0, 0, 0)
        camdist_distr_up: Tuple[float, float, float] (min, max, median)
            Distances from camera to camera
    """

    orig_distr_low: Tuple[float, float, float]
    camdist_distr_low: Tuple[float, float, float]
    orig_distr_up: Tuple[float, float, float]
    camdist_distr_up: Tuple[float, float, float]


@dataclass
class EarTraits:
    ear_id: str
    ear_width_pred: float
    ear_length_pred: float
    ear_width_ref: Optional[float] = None
    ear_length_ref: Optional[float] = None
    ear_volume_pred: Optional[float] = None
    kernel_n_pred: Optional[int] = None
    ear_volume_ref: Optional[float] = None
    kernel_n_ref: Optional[int] = None
    kernel_row_n_pred: Optional[int] = None
    kernel_row_n_flex_pred: Optional[int] = None
    kernel_row_n_ref: Optional[int] = None
    max_kernel_row_len_pred: Optional[int] = None
    max_kernel_row_len_ref: Optional[int] = None


@dataclass
class OutputDirs:
    root: str
    cropped: str
    diagnostics: str
    issues: str


class DataCompilation:
    def __init__(
        self, out_dirs: OutputDirs, configs: EarTraitConfigs, verbose: bool = True
    ):
        self.out_dirs = out_dirs
        self.configs = configs
        self.verbose = verbose

        self.ear_id = None
        self.voxelcarvingjax = None
        self.kernel_counter = None

        self.roi_imgs: Optional[List[ImgData]] = None
        self.tracked_data_lst: Optional[List[TrackData]] = None
        self.kernel_ids: Optional[List[List[int]]] = None
        self.kernel_ids_flattened: Optional[List[int]] = None
        self.kernel_pos_list_flattened: Optional[List] = None
        self.valid_yolo_mask_cnts: Optional[List[List]] = None
        self.valid_bboxes_per_ear: Optional[List[List]] = None
        self.low_cam_extrinsics: Optional[List[CamExtrinsics]] = None
        self.kerneldf: pd.DataFrame = None
        self.resultdf: pd.DataFrame = None

        self.ear_dict_list: List[dict] = []
        self.kernel_dict_list: List[dict] = []
        self.issue_id: int = 0

    def _reset_issue_id(self):
        self.issue_id = 0

    def set_ear_id(self, ear_id: str):
        self.ear_id = ear_id

    def save_general_error(self, error, ear_id):
        direct = os.path.join(self.out_dirs.issues, ear_id)
        if not os.path.exists(direct):
            os.makedirs(direct)
        path = os.path.join(direct, f"General_error.txt")
        with open(path, "w") as f:
            f.write(f"{error}")

    def save_marker_detection_issues(self, error, ear_id):
        direct = os.path.join(self.out_dirs.issues, ear_id)
        if not os.path.exists(direct):
            os.makedirs(direct)
        if isinstance(error, np.ndarray):
            path = os.path.join(direct, f"marker_detection_{self.issue_id}.jpg")
            cv2.imwrite(path, error)
        else:
            path = os.path.join(direct, f"camera_orientation_{self.issue_id}.txt")
            with open(path, "w") as f:
                f.write(f"{error}")
        self.issue_id += 1

    def save_cam_distance_issues(self, errors, cam_pos_plots, ear_id):
        if len(errors) == 0:
            return
        low_plot, top_plot = cam_pos_plots

        direct = os.path.join(self.out_dirs.issues, ear_id)
        if not os.path.exists(direct):
            os.makedirs(direct)
        path = os.path.join(direct, f"Cam_distance_irregularities.txt")
        print("Unstable camera distances detected. See details at:\n", path)
        if low_plot is not None:
            cv2.imwrite(
                os.path.join(direct, f"cam_pos_low.jpg"),
                low_plot,
            )
        if top_plot is not None:
            cv2.imwrite(
                os.path.join(direct, f"cam_pos_up.jpg"),
                top_plot,
            )
        with open(path, "w") as f:
            for error in errors:
                f.write(f"{error}\n")

    def save_ear_crop_dims_issues(self, errors, ear_id):
        if len(errors) == 0:
            return
        direct = os.path.join(self.out_dirs.issues, ear_id)
        if not os.path.exists(direct):
            os.makedirs(direct)
        path = os.path.join(direct, f"Ear_crop_dims_irregularities.txt")
        with open(path, "w") as f:
            for error in errors:
                f.write(f"{error}\n")

    def save_archive_imgs(self, archive_data: ArchiveImgData, ear_id: str):
        direct = os.path.join(self.out_dirs.cropped, ear_id)
        if not os.path.exists(direct):
            os.makedirs(direct)

        json_data = asdict(archive_data.img_dims)
        json_name = f"cropped_dims{archive_data.img_dims.cam_name}.json"

        with open(
            os.path.join(direct, json_name),
            "w",
        ) as f:
            json.dump(json_data, f)

        for img, name in zip(archive_data.imgs, archive_data.img_names):
            short_name = os.path.splitext(name)[0]
            dest_path = os.path.join(direct, f"{short_name}_crp.jpg")
            cv2.imwrite(dest_path, img)

    def save_krows_in_space(
        self,
        krow_assignment: KernelRowAssignment,
        ear_id: str,
        kernel_in_view_ds: list[np.ndarray],
        low_3d_pts: np.ndarray,
        img_idx: int = 0,
    ):
        if not self.configs.safe_results:
            return

        pt_to_poly_in_view_d = kernel_in_view_ds[img_idx]
        if len(pt_to_poly_in_view_d) == 0:
            raise ValueError("No kernelin_view_mask for the given img_idx")

        direct = os.path.join(self.out_dirs.diagnostics, ear_id)
        if not os.path.exists(direct):
            os.makedirs(direct)

        path = os.path.join(direct, f"kernel_rows_in_space_{img_idx}.png")

        krow_assignment.save_connected_3d_points_plot(
            path,
            cam_extrinsics=self.low_cam_extrinsics[img_idx],
            pt_to_poly_in_view_d=pt_to_poly_in_view_d,
            centre_poly_pts=low_3d_pts,
        )

    def save_failed_kernel_orientation(self, errors: List[str], ear_id: str):
        if len(errors) == 0:
            return
        direct = os.path.join(self.out_dirs.issues, ear_id)
        if not os.path.exists(direct):
            os.makedirs(direct)
        path = os.path.join(direct, f"Failed_kernel_orientation.txt")
        with open(path, "w") as f:
            for error in errors:
                f.write(f"{error}\n")

    def compile_jax_output(
        self,
        voxelcarvingjax: VoxelCarvingJAX,
        kernel_counter: KernelCounting,
        roi_imgs: List[ImgData],
        low_cam_extrinsics: List[CamExtrinsics],
    ):
        self.voxelcarvingjax = voxelcarvingjax
        self.kernel_counter = kernel_counter
        self.roi_imgs = roi_imgs
        self.low_cam_extrinsics = low_cam_extrinsics

        self.tracked_data_lst, failed_kernel_msgs = (
            voxelcarvingjax.get_unseen_kernel_xyz(
                kernel_counter.get_box_centers(), low_cam_extrinsics
            )
        )
        self.save_failed_kernel_orientation(failed_kernel_msgs, self.ear_id)
        self.kernel_ids = self._assign_kernel_ids(self.tracked_data_lst)
        self.kernel_ids_flattened = [
            id_ for sublist in self.kernel_ids for id_ in sublist
        ]
        self.kernel_pos_list_flattened = [
            xyz
            for tracked_data in self.tracked_data_lst
            for xyz in tracked_data.tracked_kernel_xyz
        ]
        self.valid_yolo_mask_cnts = [
            [
                kernel_counter.get_yolo_mask_cnts()[i][j]
                for j in tracked_data.valid_bbox_indices
            ]
            for i, tracked_data in enumerate(self.tracked_data_lst)
        ]
        self.valid_bboxes_per_ear = [
            [kernel_counter.get_bboxes()[i][j] for j in tracked_data.valid_bbox_indices]
            for i, tracked_data in enumerate(self.tracked_data_lst)
        ]

        return self.valid_bboxes_per_ear

    def _find_densest_slab(self, points: np.ndarray, slab_fraction: float = 0.1):
        """
        Finds the Z-interval of thickness slab_fraction*(zmax - zmin)
        that contains the most points, and returns its bounds and indices.

        Parameters:
        - points: (N, 3) array of XYZ coordinates.
        - slab_fraction: fraction of the total Z-range for the slab thickness.

        Returns:
        - best_start: lower Z bound of the densest slab
        - best_end:   upper Z bound of the densest slab
        - idx:        boolean mask of points within that slab
        - max_count:  number of points in the slab
        """
        # Extract Z-values
        z = points[:, 2]
        zmin, zmax = z.min(), z.max()
        thickness = slab_fraction * (zmax - zmin)

        # Sort by Z
        order = np.argsort(z)
        z_sorted = z[order]

        max_count = 0
        best_start = zmin
        i = 0

        # Two-pointer sweep to find densest window
        for j in range(len(z_sorted)):
            # Move start pointer forward until window fits
            while z_sorted[j] - z_sorted[i] > thickness:
                i += 1
            count = j - i + 1
            if count > max_count:
                max_count = count
                best_start = z_sorted[i]

        best_end = best_start + thickness

        return (best_end, best_start)

    def get_kernel_row_n(self):
        """
        Get the number of kernel rows per ear. Technically they are columns.
        """
        # TODO: Calc infertile Zone

        # If there are no kernels, return 0
        if len(self.kernel_pos_list_flattened) == 0:
            return 0
        kernels_xyz = np.array(self.kernel_pos_list_flattened)
        zmax = np.amax(kernels_xyz[:, 2])
        zmin = np.amin(kernels_xyz[:, 2])

        # Get the centre area of the ear (10 percent of the ear)
        zrange = zmax - zmin
        zcenter = zmin + zrange / 2
        zcenter_slab = (zcenter + zrange * 0.05, zcenter - zrange * 0.05)
        return self._cluster_kernel_rows(kernels_xyz, zcenter_slab)

    def get_kernel_row_n_flex(self):
        if len(self.kernel_pos_list_flattened) == 0:
            return 0
        kernels_xyz = np.array(self.kernel_pos_list_flattened)
        z_slab = self._find_densest_slab(kernels_xyz, slab_fraction=0.1)
        return self._cluster_kernel_rows(kernels_xyz, z_slab)

    def _cluster_kernel_rows(self, kernels_xyz, zcenter: Tuple[float, float]):
        xyz_roi = kernels_xyz[
            (kernels_xyz[:, 2] < zcenter[0]) & (kernels_xyz[:, 2] > zcenter[1])
        ]
        xy_roi = xyz_roi[:, :2]

        if len(xy_roi) == 0:
            return 0

        clustering = DBSCAN(eps=3, min_samples=2).fit(xy_roi)
        relevant_lbls = clustering.labels_[clustering.labels_ != -1]
        if len(relevant_lbls) == 0:
            return 0
        kernel_row_n = relevant_lbls.max() + 1

        # kernel row number cannot be uneven due to biological reasons
        if kernel_row_n % 2 == 0:
            return kernel_row_n
        elif kernel_row_n % 2 == 1:
            return kernel_row_n + 1

    def _plot_kernel_row_n_clustering(self, xy_array, labels):
        plt.scatter(xy_array[:, 0], xy_array[:, 1], c=labels, cmap="tab10")
        plt.savefig("_temp_kernel_row_n_clustering.png")

    def append_result_data(
        self,
        ear_traits: EarTraits,
        cam_distances: CameraDistanceDistributions,
        ear_start: float,
        sammasks,
    ):
        self.ear_id = ear_traits.ear_id

        kernel_data_per_ear, kernel_result_imgs = self._get_kernel_data(
            sam_masks=sammasks
        )

        [self.kernel_dict_list.append(asdict(item)) for item in kernel_data_per_ear]
        ear_trait_data = asdict(ear_traits)

        if self.configs.do_kernel_count and len(kernel_data_per_ear) > 0:
            if self.configs.do_sam_seg:
                mean_kernel_area_mm2 = sum(
                    [item.mask_area_sam_mm2 for item in kernel_data_per_ear]
                ) / len(kernel_data_per_ear)
            else:
                mean_kernel_area_mm2 = sum(
                    [item.mask_area_yolo_mm2 for item in kernel_data_per_ear]
                ) / len(kernel_data_per_ear)
        else:
            mean_kernel_area_mm2 = 0
            ear_trait_data["kernel_n_pred"] = 0

        ear_trait_data["mean_kernel_area_mm2"] = mean_kernel_area_mm2
        ear_trait_data = self._append_camera_distance_distributions(
            cam_distances, ear_trait_data
        )

        processing_time = time.time() - ear_start
        ear_trait_data["processing_time"] = processing_time

        self.kerneldf = pd.DataFrame(self.kernel_dict_list)

        if self.configs.krow_assignment:
            current_ear_kernel_df = self.kerneldf.loc[
                self.kerneldf["ear_id"] == self.ear_id
            ]
            self.krow_assignment = KernelRowAssignment(
                current_ear_kernel_df, self.configs
            )
            current_ear_kernel_df, max_kernel_row_len = self.krow_assignment.run()
            # Drop all rows with current ear_id
            self.kerneldf = self.kerneldf.loc[self.kerneldf["ear_id"] != self.ear_id]

            # Concatenate
            self.kerneldf = pd.concat(
                [self.kerneldf, current_ear_kernel_df], ignore_index=True
            )
            # ensure int type for kernel_row_id and order_in_krow
            self.kerneldf = self.kerneldf.astype(
                {
                    "order_in_krow": "int32",
                    "kernel_row_id": "int32",
                }
            )
            self.kernel_dict_list = self.kerneldf.to_dict("records")

            ear_trait_data["max_kernel_row_len_pred"] = max_kernel_row_len
            kernel_sheet = self._create_kernel_sheet(
                kernel_result_imgs,
                self.kerneldf.loc[self.kerneldf["ear_id"] == self.ear_id],
                max_kernel_row_len,
            )
            self._save_kernel_sheet(kernel_sheet, self.ear_id)

        self.ear_dict_list.append(ear_trait_data)
        self._verbose(ear_trait_data)
        self.resultdf = pd.DataFrame(self.ear_dict_list)

        if self.configs.safe_results:
            self.resultdf.to_csv(
                f"{self.out_dirs.root}/{self.configs.exp_name}_EarTraits.csv",
                index=False,
            )
            self.kerneldf.to_csv(
                f"{self.out_dirs.root}/{self.configs.exp_name}_KernelTraits.csv",
                index=False,
            )
        self._reset_issue_id()

    def _pad_to_size_exact(
        self, img: np.ndarray, target_h: int, target_w: int, pad_value=0
    ) -> np.ndarray:
        """
        Symmetrically pad a 2D or 3D image so its shape is exactly
        (target_h, target_w[, C]). Assumes img.height <= target_h
        and img.width  <= target_w.

        Pads top/bottom and left/right as evenly as possible,
        putting any extra pixel on bottom/right.

        Args:
            img:      HxW or HxWxC array
            target_h: desired height (>= H)
            target_w: desired width  (>= W)
            pad_value: constant fill (scalar or tuple for channels)

        Returns:
            Padded image of shape (target_h, target_w[, C]).
        """
        h, w = img.shape[:2]
        # compute total pad needed
        pad_h = target_h - h
        pad_w = target_w - w

        # split evenly, extra pixel to bottom/right
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        # build pad_width spec
        if img.ndim == 3:
            pad_width = ((top, bottom), (left, right), (0, 0))
        else:
            pad_width = ((top, bottom), (left, right))

        return np.pad(
            img, pad_width=pad_width, mode="constant", constant_values=pad_value
        )

    def _create_kernel_sheet(
        self,
        kernel_result_imgs: List[dict[np.ndarray, str]],
        kernel_df: pd.DataFrame,
        max_kernel_row_len: int,
        id_buffer: int = 50,
    ):
        """
        Create a sheet with kernel result images.
        """
        if not self.configs.safe_results:
            return

        # Find the maximum dimensions of the images
        shapes = np.vstack([d["img"].shape for d in kernel_result_imgs])
        max_dims = tuple(shapes.max(axis=0))
        y_max, x_max = max_dims[:2]
        y_with_id = int(y_max + 0.3 * id_buffer)

        # Pad each image to the maximum dimensions
        kernel_result_imgs = [
            {
                "img": self._pad_to_size_exact(d["img"], y_max, x_max),
                "kernel_id": d["kernel_id"],
            }
            for d in kernel_result_imgs
        ]

        # Create a sheet image with the appropriate size
        sheet_col_n = kernel_df["kernel_row_id"].nunique()
        kernel_sheet_h = (
            y_with_id * max_kernel_row_len + id_buffer
        )  # extra space for ids
        kernel_sheet_w = x_max * sheet_col_n
        kernel_sheet_img = np.zeros(
            (kernel_sheet_h, kernel_sheet_w, 3),
            dtype=np.uint8,
        )
        result_img_df = pd.DataFrame(kernel_result_imgs)

        sorted_kernel_df = kernel_df.copy().sort_values(
            by=["kernel_row_id", "z"], ascending=[True, False]
        )

        # Fill the kernel sheet image with the result images
        for krow_idx, krowid in enumerate(sorted_kernel_df["kernel_row_id"].unique()):
            krow_df = sorted_kernel_df[sorted_kernel_df["kernel_row_id"] == krowid]
            for z_idx, row in enumerate(krow_df.itertuples()):
                img = result_img_df.loc[
                    result_img_df["kernel_id"] == row.kernel_id, "img"
                ].values[0]
                if img is not None or img.size != 0:
                    y_offset = z_idx * y_with_id + id_buffer
                    x_offset = krow_idx * x_max
                    kernel_sheet_img[
                        y_offset : y_offset + img.shape[0],
                        x_offset : x_offset + img.shape[1],
                    ] = img
                    text_y = int(y_offset + img.shape[0] + 12)
                    text_x = int(x_offset + img.shape[1] // 2 - 17)
                    cv2.putText(
                        kernel_sheet_img,
                        f"{row.kernel_id}",
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (100, 255, 0),
                        2,
                    )

        # Write krow ids to top row
        for krow_idx, krowid in enumerate(sorted_kernel_df["kernel_row_id"].unique()):
            x_offset = krow_idx * x_max
            cv2.putText(
                kernel_sheet_img,
                f"{krowid}",
                (int(x_offset + 0.5 * x_max - 15), 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
        return kernel_sheet_img

    def _save_kernel_sheet(self, kernel_sheet_img, ear_id: str):
        diagnose_dest = os.path.join(self.out_dirs.diagnostics, ear_id)
        if not os.path.exists(diagnose_dest):
            os.makedirs(diagnose_dest)
        kernel_sheet_img_p = os.path.join(diagnose_dest, "kernel_sheet.jpg")

        cv2.imwrite(kernel_sheet_img_p, kernel_sheet_img)

    def _verbose(self, ear_traits):
        if not self.verbose:
            return
        print("---")
        print(f"Ear ID: {ear_traits['ear_id']}")

        if self.configs.reference:
            print(f"Ear Length")
            print(
                f"Ref: {ear_traits['ear_length_ref']:.2f} Pred: {ear_traits['ear_length_pred']:.2f}"
            )
            print(f"Ear Width")
            print(
                f"Ref: {ear_traits['ear_width_ref']:.2f} Pred: {ear_traits['ear_width_pred']:.2f}"
            )
            print(f"Ear Volume")
            print(
                f"Ref: {ear_traits['ear_volume_ref']:.2f} Pred: {ear_traits['ear_volume_pred']:.2f}"
            )
            print(f"Kernel N")
            print(
                f"Ref: {int(ear_traits['kernel_n_ref'])} Pred: {int(ear_traits['kernel_n_pred'])}"
            )
            print(f"Kernel Row N")
            print(
                f"Ref: {int(ear_traits['kernel_row_n_ref'])} Pred: {int(ear_traits['kernel_row_n_pred'])} Flex: {int(ear_traits['kernel_row_n_flex_pred'])}"
            )
            if ear_traits["max_kernel_row_len_pred"] is not None:
                print(f"Max Kernel Row Length")
                print(
                    f"Ref: {int(ear_traits['max_kernel_row_len_ref'])} Pred: {int(ear_traits['max_kernel_row_len_pred'])}"
                )
        else:
            print(f"Ear Length")
            print(f"Pred: {ear_traits['ear_length_pred']:.2f}")
            print(f"Ear Width")
            print(f"Pred: {ear_traits['ear_width_pred']:.2f}")
            print(f"Ear Volume")
            print(f"Pred: {ear_traits['ear_volume_pred']:.2f}")
            print(f"Kernel N")
            print(f"Pred: {int(ear_traits['kernel_n_pred'])}")
            print(f"Kernel Row N")
            print(f"Pred: {int(ear_traits['kernel_row_n_pred'])}")
            print(f"Kernel Row N Flex")
            print(f"Pred: {int(ear_traits['kernel_row_n_flex_pred'])}")

        print("====================================")
        print(f"Image processing time: {ear_traits['processing_time']:.2f} s")
        print("====================================")

    def _append_camera_distance_distributions(
        self,
        cam_distances: CameraDistanceDistributions,
        ear_traits_dict: dict,
    ):
        min_orig_dist_low, max_orig_dist_low, median_orig_dist_low = (
            cam_distances.orig_distr_low
        )
        min_cam_dist_low, max_cam_dist_low, median_cam_dist_low = (
            cam_distances.camdist_distr_low
        )
        min_orig_dist_up, max_orig_dist_up, median_orig_dist_up = (
            cam_distances.orig_distr_up
        )
        min_cam_dist_up, max_cam_dist_up, median_cam_dist_up = (
            cam_distances.camdist_distr_up
        )

        ear_traits_dict["min_origin_distance_low"] = min_orig_dist_low
        ear_traits_dict["max_origin_distance_low"] = max_orig_dist_low
        ear_traits_dict["median_origin_distance_low"] = median_orig_dist_low
        ear_traits_dict["min_camera_distance_low"] = min_cam_dist_low
        ear_traits_dict["max_camera_distance_low"] = max_cam_dist_low
        ear_traits_dict["median_camera_distance_low"] = median_cam_dist_low
        ear_traits_dict["min_origin_distance_up"] = min_orig_dist_up
        ear_traits_dict["max_origin_distance_up"] = max_orig_dist_up
        ear_traits_dict["median_origin_distance_up"] = median_orig_dist_up
        ear_traits_dict["min_camera_distance_up"] = min_cam_dist_up
        ear_traits_dict["max_camera_distance_up"] = max_cam_dist_up
        ear_traits_dict["median_camera_distance_up"] = median_cam_dist_up
        return ear_traits_dict

    def diagnostic_visualization(self, sammasks, ear_id, low_3d_pts):
        krow_dfs = None
        if not self.configs.safe_results:
            return
        if not self.configs.do_kernel_count:
            return

        start_diagnostic_visualizations = time.time()
        from_3d_to_2d_imgpos, pt_ids, kernels_in_view_ds = (
            self.voxelcarvingjax.get_low_img_tracked_kernel_xy(
                self.kernel_pos_list_flattened, self.kernel_ids_flattened, nth=5
            )
        )
        if self.configs.krow_assignment:
            # TODO put this into a function
            new_pt_ids = []
            new_from_3d_to_2d_imgpos = []
            krow_dfs = []

            for ids, xy_arr in zip(pt_ids, from_3d_to_2d_imgpos):
                if len(xy_arr) == 0:
                    new_pt_ids.append([])
                    new_from_3d_to_2d_imgpos.append([])
                    krow_dfs.append(None)

                    continue
                imgpos_2d_df = pd.DataFrame(xy_arr, columns=["2dx", "2dy"])
                imgpos_2d_df["kernel_id"] = ids
                ear_df = self.kerneldf.loc[self.kerneldf["ear_id"] == ear_id].copy()
                krow_df = pd.merge(imgpos_2d_df, ear_df, on="kernel_id", how="inner")
                assert (
                    len(krow_df) == len(krow_df["kernel_id"].unique()) == len(ids)
                ), "Some kernel ids not found in kerneldf"
                # order krows first by krow_id, then by z
                # krow_df = krow_df.sort_values(
                #    by=["kernel_row_id", "z"], ascending=[True, False]
                # )
                new_pt_ids.append(krow_df["kernel_id"].values)
                new_from_3d_to_2d_imgpos.append(krow_df[["2dx", "2dy"]].values)
                # krow_ids.append(krow_df["kernel_row_id"].values)
                krow_dfs.append(
                    krow_df[["kernel_id", "kernel_row_id", "order_in_krow"]].copy()
                )

            pt_ids = new_pt_ids
            from_3d_to_2d_imgpos = new_from_3d_to_2d_imgpos

            self.save_krows_in_space(
                self.krow_assignment,
                self.ear_id,
                kernels_in_view_ds,
                low_3d_pts,
                img_idx=0,
            )

        diagnose_dest = os.path.join(self.out_dirs.diagnostics, ear_id)
        if not os.path.exists(diagnose_dest):
            os.makedirs(diagnose_dest)

        final_kernel_selection_visualization(
            self.roi_imgs,
            from_3d_to_2d_imgpos,
            pt_ids,
            output_dir=diagnose_dest,
            krow_dfs=krow_dfs,
        )
        if not self.configs.tracking_vis or not self.configs.do_sam_seg:
            return

        tarcking_dest = os.path.join(diagnose_dest, "tracking_vis")
        if not os.path.exists(tarcking_dest):
            os.makedirs(tarcking_dest)

        kernel_tracking_visualization(
            self.kernel_counter.get_box_centers(),
            self.tracked_data_lst,
            self.kernel_counter.get_polyn_pts(),
            sammasks,
            self.roi_imgs,
            output_dir=tarcking_dest,
        )
        print(
            f"Diagnostic visualizations done in: {time.time() - start_diagnostic_visualizations:.2f} seconds"
        )

    def _get_kernel_crop(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Get the kernel result img.
        """
        y_idcs, x_idcs = np.where(mask > 0)
        xmin, xmax = x_idcs.min(), x_idcs.max()
        ymin, ymax = y_idcs.min(), y_idcs.max()
        result_large = cv2.bitwise_and(img, img, mask=mask)
        result_img = result_large[ymin : ymax + 1, xmin : xmax + 1]
        return result_img

    def _get_kernel_data(
        self, sam_masks
    ) -> Tuple[List[KernelTraitData], List[dict[np.ndarray, str]]]:

        kernel_traits_per_ear = []
        kernel_result_imgs = []
        for i, tracked_data in enumerate(self.tracked_data_lst):
            kernel_positions_per_img = tracked_data.tracked_kernel_xyz
            if len(kernel_positions_per_img) == 0:
                continue
            img = self.roi_imgs[i].img
            imgp = self.roi_imgs[i].imgp
            imgname = os.path.splitext(os.path.basename(imgp))[0]
            name_ending = imgname.split("_")[-1]
            if name_ending == "crp":
                motor_steps = int(imgname.split("_")[-2])
            else:
                motor_steps = int(imgname.split("_")[-1])
            for j, kernel_position in enumerate(kernel_positions_per_img):

                cnt = self.valid_yolo_mask_cnts[i][j]
                kernel_id = self.kernel_ids[i][j]
                bbox = self.valid_bboxes_per_ear[i][j]
                px_to_mm2_scale = tracked_data.tracked_scales[j].item()
                if len(cnt) < 3:
                    yolo_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                else:
                    yolo_mask = cv2.fillPoly(
                        np.zeros(img.shape[:2], dtype=np.uint8),
                        np.array(cnt, dtype=np.int32).reshape((1, -1, 2)),
                        255,
                    )

                if sam_masks is not None:
                    sam_mask = sam_masks[i][j]
                    if self.configs.krow_assignment:
                        kernel_result_imgs.append(
                            {
                                "img": self._get_kernel_crop(img, sam_mask),
                                "kernel_id": kernel_id,
                            }
                        )
                else:
                    sam_mask = None
                    if self.configs.krow_assignment:
                        kernel_result_imgs.append(
                            {
                                "img": self._get_kernel_crop(img, yolo_mask),
                                "kernel_id": kernel_id,
                            }
                        )

                yolo_mask_size_px, sam_mask_size_px, iou = self._maskcomp(
                    yolo_mask, sam_mask
                )
                yolo_mask_size_mm2 = yolo_mask_size_px / px_to_mm2_scale
                sam_mask_size_mm2 = (
                    (sam_mask_size_px / px_to_mm2_scale)
                    if sam_masks is not None
                    else None
                )
                unit_x, unit_y, unit_z = (
                    self.voxelcarvingjax.unit_direction_vector_at_z(
                        kernel_position[2].item()
                    )
                )

                kernel_traits_per_ear.append(
                    KernelTraitData(
                        ear_id=self.ear_id,
                        motor_steps=motor_steps,
                        kernel_id=kernel_id,
                        x=kernel_position[0].item(),
                        y=kernel_position[1].item(),
                        z=kernel_position[2].item(),
                        bbox_x1=bbox[0].item(),
                        bbox_y1=bbox[1].item(),
                        bbox_x2=bbox[2].item(),
                        bbox_y2=bbox[3].item(),
                        mask_area_yolo_px=yolo_mask_size_px,
                        mask_area_sam_px=sam_mask_size_px,
                        mask_iou=iou,
                        px_to_mm2=px_to_mm2_scale,
                        mask_area_yolo_mm2=yolo_mask_size_mm2,
                        mask_area_sam_mm2=sam_mask_size_mm2,
                        unit_vector_x=unit_x.item(),
                        unit_vector_y=unit_y.item(),
                        unit_vector_z=unit_z.item(),
                    )
                )
        return kernel_traits_per_ear, kernel_result_imgs

    def _maskcomp(self, mask1, mask2=None):
        mask1_size = np.count_nonzero(mask1)
        mask2_size = None
        iou = None

        if mask2 is not None:
            mask2_size = np.count_nonzero(mask2)
            intersection = np.logical_and(mask1, mask2)
            union = np.logical_or(mask1, mask2)
            iou = np.sum(intersection) / np.sum(union)
        return mask1_size, mask2_size, iou

    def _assign_kernel_ids(self, tracked_data: List[TrackData]):
        self.kernel_ids = []
        id_ = 0
        for tracked_data_per_img in tracked_data:
            inner_kernel_ids = []
            for kernel_pos in tracked_data_per_img.tracked_kernel_xyz:
                inner_kernel_ids.append(id_)
                id_ += 1
            self.kernel_ids.append(inner_kernel_ids)
        return self.kernel_ids


def _draw_bboxes(
    paint_img,
    boxes_centers,
    idcs,
    idcs_with_dupl,
    xyzs,
    xyzs_dupl,
    buff_dims,
    mask_mm2=None,
):
    buff_xmin, _, buff_ymin, _ = buff_dims
    xyzs = np.array(xyzs)
    xyzs_dupl = np.array(xyzs_dupl)
    idcs = np.array(idcs)
    idcs_with_dupl = np.array(idcs_with_dupl)
    mask_mm2 = np.array(mask_mm2) if mask_mm2 is not None else None

    for i, center in enumerate(boxes_centers):
        y, x = center

        if i in idcs:
            colour = (0, 255, 0)
            kernelx, kernely, kernelz = xyzs[np.where(idcs == i)][0]
            if mask_mm2 is not None:
                scale = mask_mm2[np.where(idcs == i)][0]
                cv2.putText(
                    paint_img,
                    f"{scale:.2f} pxtomm2",
                    (int(x - buff_xmin - 25), int(y - buff_ymin)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    colour,
                    1,
                )
            else:
                cv2.putText(
                    paint_img,
                    f"{int(kernelx)},{int(kernely)},{int(kernelz)}",
                    (int(x - buff_xmin - 25), int(y - buff_ymin)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    colour,
                    1,
                )
        elif i in idcs_with_dupl:
            colour = (0, 0, 255)
            kernelx, kernely, kernelz = xyzs_dupl[np.where(idcs_with_dupl == i)][0]
            cv2.putText(
                paint_img,
                f"{int(kernelx)},{int(kernely)},{int(kernelz)}",
                (int(x - buff_xmin - 25), int(y - buff_ymin)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                colour,
                1,
            )
        else:
            colour = (0, 0, 255)
            cv2.circle(
                paint_img, (int(x - buff_xmin), int(y - buff_ymin)), 2, colour, -1
            )

    return paint_img


def _draw_fitted_center_polynom(paint_img, center_polyn_pts):
    for pt in center_polyn_pts:
        pt = pt[0]
        if np.any(pt < 0):
            continue
        cv2.circle(paint_img, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)
    return paint_img


def _draw_masks(paint_img, masks):
    rand_colours = np.random.randint(0, 255, (len(masks), 3))
    for i, mask in enumerate(masks):
        paint_img[mask > 0] = rand_colours[i]
    return paint_img


def kernel_tracking_visualization(
    boxes_centers,
    tracked_lst: List[TrackData],
    centre_polyn_pts,
    sammasks,
    img_objs: List[ImgData],
    output_dir,
):
    for imgi, imgobj_ in enumerate(img_objs):
        img = imgobj_.img
        buff_dims = imgobj_.buffered_dims
        paint_img = img.copy()
        paint_img = _draw_masks(paint_img, sammasks[imgi])
        paint_img = _draw_fitted_center_polynom(paint_img, centre_polyn_pts[imgi])
        paint_img = _draw_bboxes(
            paint_img,
            boxes_centers[imgi],
            tracked_lst[imgi].valid_bbox_indices,
            tracked_lst[imgi].valid_bbox_indices_with_duplicates,
            tracked_lst[imgi].tracked_kernel_xyz,
            tracked_lst[imgi].tracked_kernel_xyz_with_duplicates,
            buff_dims,
            # mask_mm2=tracked_lst[imgi].tracked_scales,
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(f"{output_dir}/img_{imgi}.jpg", paint_img)


def _draw_all_selected_kernels(
    paint_img,
    pts: np.ndarray,
    kernel_ids: np.ndarray,
    buff_dims,
    krow_df: pd.DataFrame | None = None,
):
    buff_xmin, _, buff_ymin, _ = buff_dims
    # draw connections of detected kernel rows in order of order_in_krow
    if krow_df is not None:
        assert (
            len(pts) == len(kernel_ids) == len(krow_df)
        ), "Length of pts, kernel_ids and krow_df must be the same"
        for krow_id in krow_df["kernel_row_id"].unique():
            single_row_df = krow_df[krow_df["kernel_row_id"] == krow_id]
            for order_i in single_row_df["order_in_krow"].unique():
                id1 = single_row_df.loc[
                    single_row_df["order_in_krow"] == order_i, ["kernel_id"]
                ].values
                id2 = single_row_df.loc[
                    single_row_df["order_in_krow"] == order_i + 1, ["kernel_id"]
                ].values

                # in case a line is discontinued at the ear edges. Such lines are continued later when contious connections are present.
                # e.g. order in krow=[0,1,2,3,7,8,9] -> two lines 0-1-2-3 and 7-8-9
                if len(id2) == 0:
                    continue

                assert len(id1) == 1 and len(id2) == 1, "Kernel IDs should be unique"
                pt1 = pts[kernel_ids == id1[0]]
                pt2 = pts[kernel_ids == id2[0]]
                if len(pt1) == 0 or len(pt2) == 0:
                    continue

                pt1 = pt1[0]
                pt2 = pt2[0]
                cv2.line(
                    paint_img,
                    (int(pt1[0] - buff_xmin), int(pt1[1] - buff_ymin)),
                    (int(pt2[0] - buff_xmin), int(pt2[1] - buff_ymin)),
                    (100, 100, 100),  # (168, 50, 121), #purple
                    1,
                    lineType=cv2.LINE_AA,
                )
    for pt, k_id in zip(pts, kernel_ids):

        cv2.circle(
            paint_img,
            (int(pt[0] - buff_xmin), int(pt[1] - buff_ymin)),
            2,
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            paint_img,
            f"{k_id}",
            (int(pt[0] - buff_xmin - 25), int(pt[1] - buff_ymin + 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    return paint_img


def final_kernel_selection_visualization(
    img_objs: List[ImgData], all_pts, all_ids, output_dir, krow_dfs: list | None = None
):
    for imgi, imgobj_ in enumerate(img_objs):
        pts = all_pts[imgi]
        krow_df = krow_dfs[imgi] if krow_dfs is not None else None
        if len(pts) == 0:
            continue
        kernel_ids = all_ids[imgi]
        img = imgobj_.img
        buff_dims = imgobj_.buffered_dims
        paint_img = img.copy()
        paint_img = _draw_all_selected_kernels(
            paint_img, pts, kernel_ids, buff_dims, krow_df
        )
        cv2.imwrite(f"{output_dir}/all_kernels_{imgi}.jpg", paint_img)
