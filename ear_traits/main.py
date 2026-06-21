from ear_traits.scan_time_utils import (
    PreImageAnalyzer,
    CamTrinsics,
    RoiImgGenerator,
    detect_markers_and_orient_cams,
)
from ear_traits.count_kernels import *
from ear_traits.output import (
    DataCompilation,
    EarTraits,
    OutputDirs,
    CameraDistanceDistributions,
)
from ear_traits.voxel_carving_jax import VoxelCarvingJAX
from ear_traits.configs import EarTraitConfigs, arg_parsing
import warnings

warnings.filterwarnings(
    "ignore", message=".*os\\.fork\\(\\) was called.*", category=RuntimeWarning
)
import glob
import pandas as pd
import os
from natsort import natsorted
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import traceback
import torch


class ReferenceData:
    def __init__(
        self,
        root_path: str,
        ref_csv_path: str,
        ref_id_column: str = "Ear ID",
        ref_traits: List[str] = [
            "Kernel number",
            "Ear length",
            "Ear width",
            "Ear volume in ml",
            "Kernel columns",
            "Kernel rows",
        ],
    ):
        self._root_path: str = root_path
        self._ref_csv_path: str = ref_csv_path
        self._ref_id_column: str = ref_id_column
        self._ref_traits: List[str] = ref_traits
        self._ref_df: Optional[pd.DataFrame] = None
        self._input_paths: Optional[List[str]] = None

    @property
    def ref_df(self) -> pd.DataFrame:
        """Lazily load the reference DataFrame."""
        if self._ref_df is None:
            if not os.path.exists(self._ref_csv_path):
                raise FileNotFoundError(
                    f"Reference CSV not found: {self._ref_csv_path}"
                )
            df = pd.read_csv(self._ref_csv_path)
            for trait in self._ref_traits:
                if trait not in df.keys():
                    raise KeyError(
                        f"Column '{trait}' not found in reference CSV. Possible keys: {df.keys()}"
                    )
            self._ref_df = df.loc[:, [self._ref_id_column, *self._ref_traits]]
        return self._ref_df

    @property
    def input_paths(self) -> List[str]:
        """Generate input paths based on reference IDs."""
        if not os.path.exists(self._root_path):
            raise FileNotFoundError(f"Root path not found: {self._root_path}")
        if self._input_paths is None:
            if self._ref_id_column not in self.ref_df.columns:
                raise KeyError(
                    f"Column '{self._ref_id_column}' not found in reference CSV."
                )
            ref_ids: List[str] = self.ref_df[self._ref_id_column].tolist()
            self._input_paths = [self._check_input_paths(rid) for rid in ref_ids]
        return self._input_paths

    def _check_input_paths(self, ear_id: str) -> str:
        p: str = os.path.join(self._root_path, str(ear_id))
        if os.path.exists(p):
            return p
        else:
            raise FileNotFoundError(
                f"Ear folder not found: {p}. Check if the correct reference file is selected."
            )

    def __getitem__(self, idx: int):
        traits = self.ref_df.iloc[idx].to_dict()
        return self.input_paths[idx], traits

    def __len__(self) -> int:
        return len(self.input_paths)


class CamPairGenerator:
    def __init__(self, ear_path: str):
        self.ear_path = ear_path
        self.paths = natsorted(glob.glob(ear_path + "/*"))
        self.pairs = self._match_pairs()

    def _match_pairs(self):
        """Generate pairs of matching images."""
        for path in self.paths:
            if "_low_" in path:
                low_path = path
                up_path = path.replace("_low_", "_up_")
                if not os.path.exists(up_path):
                    raise FileNotFoundError(f"Matching image not found: {up_path}")
                yield low_path, up_path

    def __iter__(self):
        return self.pairs


def plot_kernel_xy(kernel_pos_list: List[Tuple[float, float, float]]) -> None:
    kernel_pos_xy = np.array([pos[:2] for pos in kernel_pos_list])
    plt.figure(figsize=(10, 10), dpi=300)
    plt.scatter(kernel_pos_xy[:, 0], kernel_pos_xy[:, 1])
    plt.savefig("kernel_xy.png")


def pooled_marker_detection(marker_prepro_res: Tuple) -> Tuple:
    low_data, up_data = marker_prepro_res
    markers_low = detect_markers_and_orient_cams(*low_data, p_dataset=ee)
    markers_up = detect_markers_and_orient_cams(*up_data, p_dataset=ee)
    return (markers_low, markers_up)


def check_marker_detection(
    extrinsics: List[Tuple], idx: int, data_compiler: DataCompilation, ear_id: str
) -> List:
    extr_list = []
    cancel_ear_processing = False
    for extr in extrinsics:
        if not isinstance(extr[idx], CamExtrinsics):
            data_compiler.save_marker_detection_issues(extr[idx], ear_id)
            cancel_ear_processing = True
        extr_list.append(extr[idx])
    return extr_list if not cancel_ear_processing else False


def build_output_structure(output_dir: str) -> OutputDirs:
    i = 1
    while os.path.exists(output_dir):
        if i > 1:
            output_dir = output_dir[:-3]
        if i > 10:
            output_dir = output_dir[:-1]
        output_dir += f"({i})"
        i += 1

    os.makedirs(output_dir)
    cropped_dir = os.path.join(output_dir, "cropped")
    if not os.path.exists(cropped_dir):
        os.makedirs(cropped_dir)
    diagnostics_dir = os.path.join(output_dir, "diagnostics_images")
    if not os.path.exists(diagnostics_dir):
        os.makedirs(diagnostics_dir)
    issues_dir = os.path.join(output_dir, "issues")
    if not os.path.exists(issues_dir):
        os.makedirs(issues_dir)

    return OutputDirs(output_dir, cropped_dir, diagnostics_dir, issues_dir)


def check_root_path(root_path: str) -> Tuple[str, bool]:
    allow_crop = True
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"Root input path not found: {root_path}")
    cropped_dir = os.path.join(root_path, "cropped")
    list_of_dirs = os.listdir(root_path)
    if (
        "cropped" in list_of_dirs
        and "diagnostics_images" in list_of_dirs
        and "issues" in list_of_dirs
    ):
        allow_crop = False
        return cropped_dir, allow_crop
    else:
        return root_path, allow_crop


def process_ear(p, reftraits, ear_id):

    # memory
    tcm_free, tcm_total = torch.cuda.mem_get_info(torch.device("cuda:0"))
    tcm_used_MB = (tcm_total - tcm_free) / 1024**2

    ear_width_ref = (
        float(reftraits["Ear width"]) * 10 if configs.reference else None
    )  # to mm
    ear_length_ref = float(reftraits["Ear length"]) * 10 if configs.reference else None
    ear_volume_pred = None
    kernel_n_pred = None
    ear_volume_ref = float(reftraits["Ear volume in ml"]) if configs.reference else None
    kernel_n_ref = int(reftraits["Kernel number"]) if configs.reference else None
    kernel_row_n_pred = None
    kernel_row_n_flex_pred = None
    # in reference data kernel rows and columns are named like in an excel sheet
    kernel_row_n_ref = int(reftraits["Kernel columns"]) if configs.reference else None
    max_kernel_row_len_pred = None
    max_kernel_row_len_ref = (
        int(reftraits["Kernel rows"]) if configs.reference else None
    )

    data_compiler.set_ear_id(ear_id)

    # if not ear_id == "39320223414029":
    # return

    print(ear_id)
    print("Memory GPU used: %.0f MB" % (tcm_used_MB))
    ear_start = time.time()
    cam_paths = CamPairGenerator(p)

    marker_prepro_res = [
        (camtrins_low.marker_prepro(low_path), camtrins_up.marker_prepro(up_path))
        for low_path, up_path in cam_paths
    ]
    low_cam_intrinsics = marker_prepro_res[0][0][1]
    low_marker = marker_prepro_res[0][0][0]
    up_cam_intrinsics = marker_prepro_res[0][1][1]
    up_marker = marker_prepro_res[0][1][0]

    with Pool(
        processes=cpu_count() - 2,
    ) as pool:
        extrinsics = pool.map(pooled_marker_detection, marker_prepro_res)

    low_cam_extrinsics = check_marker_detection(extrinsics, 0, data_compiler, ear_id)
    up_cam_extrinsics = check_marker_detection(extrinsics, 1, data_compiler, ear_id)

    if not low_cam_extrinsics or not up_cam_extrinsics:
        print("Error in Marker detection. Continuing with next ear.")
        return

    print(f"Marker detection done in: {time.time() - ear_start:.2f} seconds")

    campos_start = time.time()

    scantime_low = PreImageAnalyzer(
        low_cam_extrinsics, low_cam_intrinsics, low_marker, configs.hsv_thresh_p
    )

    orig_distr_low, camdist_distr_low = scantime_low.camera_position_control()
    scantime_up = PreImageAnalyzer(
        up_cam_extrinsics, up_cam_intrinsics, up_marker, configs.hsv_thresh_p
    )

    orig_distr_up, camdist_distr_up = scantime_up.camera_position_control()
    cam_distances = CameraDistanceDistributions(
        orig_distr_low, camdist_distr_low, orig_distr_up, camdist_distr_up
    )

    errorlists = scantime_low.err_lst + scantime_up.err_lst
    error_plots = scantime_low.cam_pos_plot, scantime_up.cam_pos_plot
    data_compiler.save_cam_distance_issues(errorlists, error_plots, ear_id)

    print(f"Camera position control done in: {time.time() - campos_start:.2f} seconds")
    fg_start = time.time()

    fg_errors_low, archive_data_low = scantime_low.compute_foreground(
        do_archive_imgs=configs.archive_imgs
    )
    fg_errors_up, archive_data_up = scantime_up.compute_foreground(
        do_archive_imgs=configs.archive_imgs
    )

    fg_errors = fg_errors_low + fg_errors_up
    data_compiler.save_ear_crop_dims_issues(fg_errors, ear_id)

    ear_width, ear_length = scantime_low.get_ear_wh()

    roi_imgs = RoiImgGenerator(
        scantime_low.get_img_data_list(), scantime_low.get_ear_crop_dims()
    )

    [i for i, img in roi_imgs]
    cropped_objs = roi_imgs.get_all_cropped_objs()
    if configs.jetson:
        del roi_imgs
    print(f"Foreground extraction done in: {time.time() - fg_start:.2f} seconds")

    if configs.archive_imgs:
        data_compiler.save_archive_imgs(archive_data_low, ear_id)
        data_compiler.save_archive_imgs(archive_data_up, ear_id)

    if configs.do_voxel_carving:

        start_jax_voxel_carving = time.time()
        voxelcarvingjax = VoxelCarvingJAX(configs, scantime_low.get_voxel_grid_dims())
        all_img_data = (
            scantime_low.get_img_data_list(),
            scantime_up.get_img_data_list(),
        )
        ear_volume_pred = voxelcarvingjax.get_ear_volume(
            all_img_data,
            (low_cam_extrinsics, up_cam_extrinsics),
            (low_cam_intrinsics, up_cam_intrinsics),
        )
        print(
            f"Voxel carving with JAX done in: {time.time() - start_jax_voxel_carving:.2f} seconds",
        )
        low_img_poly_pts, low_3d_pts = voxelcarvingjax.get_low_img_center_polynoms()

    # Free up memory
    if configs.jetson:
        scantime_low.reset()
        scantime_up.reset()

    if configs.do_kernel_count:

        kernel_counting_start = time.time()

        kernel_counter = KernelCounting(
            cropped_objs,
            low_cam_extrinsics,
            low_cam_intrinsics,
            low_img_poly_pts,
            yolo_model,
            ear_width,
        )
        kernel_counter.process()

        valid_bboxes_per_ear = data_compiler.compile_jax_output(
            voxelcarvingjax,
            kernel_counter,
            cropped_objs,
            low_cam_extrinsics,
        )
        kernel_row_n_pred = data_compiler.get_kernel_row_n()
        kernel_row_n_flex_pred = data_compiler.get_kernel_row_n_flex()
        kernel_n_pred = sum(
            [len(bboxes_per_img) for bboxes_per_img in valid_bboxes_per_ear]
        )
        print(
            f"Kernel counting done in: {time.time() - kernel_counting_start:.2f} seconds"
        )
        if configs.do_sam_seg:
            sam_seg_start = time.time()

            if kernel_n_pred > 500:
                sam_batch_size = 2
            else:
                sam_batch_size = 3
            sammasks = sam_model.seg_from_bbox_batch(
                cropped_objs, valid_bboxes_per_ear, sam_batch_size
            )
            print(
                f"SAM segmentation done in: {time.time() - sam_seg_start:.2f} seconds"
            )
        else:
            sammasks = None

    ear_id_res = str(reftraits["Ear ID"] if configs.reference else ear_id).replace(
        ".0", ""
    )
    ear_trait_results = EarTraits(
        ear_id=ear_id_res,
        ear_width_pred=ear_width,
        ear_length_pred=ear_length,
        ear_width_ref=ear_width_ref,
        ear_length_ref=ear_length_ref,
        ear_volume_pred=ear_volume_pred,
        kernel_n_pred=kernel_n_pred,
        ear_volume_ref=ear_volume_ref,
        kernel_n_ref=kernel_n_ref,
        kernel_row_n_pred=kernel_row_n_pred,
        kernel_row_n_flex_pred=kernel_row_n_flex_pred,
        kernel_row_n_ref=kernel_row_n_ref,
        max_kernel_row_len_ref=max_kernel_row_len_ref,
        max_kernel_row_len_pred=max_kernel_row_len_pred,
    )
    data_compiler.append_result_data(
        ear_trait_results, cam_distances, ear_start, sammasks
    )
    data_compiler.diagnostic_visualization(sammasks, ear_id, low_3d_pts)


if __name__ == "__main__":
    args = arg_parsing()

    # Path to the root directory containing the ear folders:
    if args.root_dir:
        ROOTP = args.root_dir
    else:
        ROOTP = ""

    # Path to the manual reference data (Optional if REF is set to False):
    REFP = "EarReference/BreedersEarsManMeasurementsCSV.csv"
    # Paths to the camera settings and real world points CSV files:
    INTRINSIC_P = "ScannerCamSettings.csv"
    REALWORLD_POINTS_P = "Cps2401_pluscorners.csv"
    # Path to the SAM model checkpoint (Optional if DOSAM_SEG is set to False):
    SAM_P = args.sam
    # SAM model type (Optional if DOSAM_SEG is set to False):
    SAM_TYPE = "vit_b"
    # Path to the YOLO model checkpoint (Optional if configs.do_kernel_count: is set to False):
    YOLO_P = args.yolo

    configs = EarTraitConfigs(
        exp_name=args.exp_name,  # Name of the experiment. Will be used to name the output files
        reference=args.reference,  # Whether to use reference data for comparison. Only files with the same IDs as the reference data will be processed.
        do_voxel_carving=args.no_voxel_carving,  # Whether to perform voxel carving for ear volume estimation.
        do_kernel_count=args.no_kernel_count,  # Whether to count kernels (with YOLO).
        do_sam_seg=args.no_sam_seg,  # Whether to perform kernel segmentation with SAM. Otherwise lower quality Segmentation will be done by YOLO directly.
        safe_results=args.no_saving_results,  # Whether to save the results to CSV files.
        tracking_vis=args.show,  # Whether to save kernel tracking progress.
        hsv_thresh_p=args.hsv_thresh,  # Path to json file storing hsv thresholds
        archive_imgs=args.archive_imgs,  # Whether to save crops of the original images.
        jetson=args.jetson,  # Whether to adapt the pipeline for Jetson Orin Nano.
        mm_per_voxelside=args.mm_per_voxelside,  # Size of the voxel side in mm.
        yolo_engine=args.yolo_engine,  # Whether to compile yolo to engine format.
        debug=args.debug,  # Whether to enable debug mode.
        krow_assignment=args.krow_assignment,  # Whether to assign kernels to rows.
    )
    if args.output_dir:
        root_output_dir = f"{args.output_dir}/{args.exp_name}"
    else:
        root_output_dir = f"Results/{args.exp_name}"

    out_dirs = build_output_structure(root_output_dir)

    ee = True if "Esteban" in ROOTP else False
    if ee:
        REALWORLD_POINTS_P = "EE_markers_with_corners.csv"

    if configs.do_sam_seg:
        from ear_traits.sam_seg import SamSeg

        sam_model = SamSeg(SAM_P, SAM_TYPE)

    if configs.do_kernel_count:
        yolo_model = build_yolo_engine(YOLO_P, quant=configs.yolo_engine)

    # Check whether images were alredy cropped and undistorted
    ROOTP, allow_crop = check_root_path(ROOTP)

    # No archiving images when archived images are already read
    if not allow_crop:
        configs.archive_imgs = False

    if configs.reference:
        path_obj = ReferenceData(ROOTP, REFP)
    else:
        path_obj = sorted(glob.glob(ROOTP + "/*"))

    camtrins_low = CamTrinsics(INTRINSIC_P, REALWORLD_POINTS_P, ee)
    camtrins_up = CamTrinsics(INTRINSIC_P, REALWORLD_POINTS_P, ee)

    data_compiler = DataCompilation(out_dirs, configs)

    for i, pref in enumerate(path_obj):

        # restrict loop (optional)
        if args.first_i:
            if (i + 1) < args.first_i:
                continue
        if args.last_i:
            if (i + 1) > args.last_i:
                break

        print(f"Processing ear {i+1}/{len(path_obj)}")
        if configs.reference:
            p, reftraits = pref
        else:
            p = pref
            reftraits = None
        ear_id = os.path.basename(p)

        if not configs.debug:
            try:
                process_ear(p, reftraits, ear_id)
            except Exception as e:
                traced_e = traceback.format_exc()
                print(
                    f"Unexpected error occured during processing of {p}:\n{traced_e}\n"
                )
                data_compiler.save_general_error(traced_e, ear_id)
        else:
            process_ear(p, reftraits, ear_id)
