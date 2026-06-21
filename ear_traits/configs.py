from dataclasses import dataclass
import argparse
from typing import Optional


@dataclass
class EarTraitConfigs:
    exp_name: str
    reference: bool
    do_voxel_carving: bool
    do_kernel_count: bool
    do_sam_seg: bool
    safe_results: bool
    tracking_vis: bool
    hsv_thresh_p: str
    archive_imgs: bool
    jetson: bool
    mm_per_voxelside: float  # e.g: 2.0 mm * 2.0 mm * 2.0 mm = 8.0 mm^3
    yolo_engine: Optional[str]
    debug: bool
    krow_assignment: bool

    def __post_init__(self):
        if self.do_kernel_count and not self.do_voxel_carving:
            print(
                "Kernel counting requires voxel carving to be enabled. Setting do_voxel_carving to True"
            )
            self.do_voxel_carving = True
        if self.do_sam_seg and not self.do_kernel_count:
            print(
                "SAM segmentation requires kernel counting to be enabled. Setting do_kernel_count to True"
            )
            self.do_kernel_count = True
        if self.tracking_vis and not self.do_kernel_count:
            print(
                "Kernel tracking visualization requires kernel counting to be enabled. Setting do_kernel_count to True"
            )
            self.do_kernel_count = True
        if self.tracking_vis and not self.do_sam_seg:
            print(
                "Kernel tracking visualization requires SAM segmentation to be enabled. Setting do_sam_seg to True"
            )
            self.do_sam_seg = True


def arg_parsing():
    parser = argparse.ArgumentParser(description="Ear traits analysis")
    parser.add_argument(
        "-d",
        "--root_dir",
        type=str,
        help="Path to the root directory containing the ear folders",
    )
    parser.add_argument(
        "-e",
        "--exp_name",
        type=str,
        help="Identifier for naming results and error files",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Enable showing the kernel tracking process in the diagnostic_images folder (Default: False)",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Enable loading reference data (Default: False)",
    )
    parser.add_argument(
        "--no_voxel_carving",
        action="store_false",
        help="Disable voxel carving for ear volume estimation (Default: True)",
    )
    parser.add_argument(
        "--no_kernel_count",
        action="store_false",
        help="Disable kernel counting (Default: True)",
    )
    parser.add_argument(
        "--no_sam_seg",
        action="store_false",
        help="Disable kernel segmentation with SAM (Default: True)",
    )
    parser.add_argument(
        "--no_saving_results",
        action="store_false",
        help="Disable saving the results to CSV files (Default: True)",
    )
    parser.add_argument(
        "--archive_imgs",
        action="store_true",
        help="Disable saving errors as file (Default: False)",
    )
    parser.add_argument(
        "--hsv_thresh",
        type=str,
        default="background_hsv.json",
        help="Path to json file storing hsv thresholds (Default: background_hsv.json)",
    )
    parser.add_argument(
        "--yolo",
        type=str,
        default="KernelYOLO8x.pt",
        help="Path to Yolo Model (Default: KernelYOLO8x.pt)",
    )
    parser.add_argument(
        "--sam",
        type=str,
        default="KernelSAM_b.pth",
        help="Path to SAM (Default: KernelSAM_b.pth)",
    )
    parser.add_argument(
        "--jetson",
        action="store_true",
        help="Adapted pipeline for Jetson Orin Nano (Default: False)",
    )
    parser.add_argument(
        "--mm_per_voxelside",
        type=float,
        default=0.5,
        help="Defines the voxel grid resolution in mm (Default: 0.5 mm per voxel side)",
    )
    parser.add_argument(
        "--yolo_engine",
        type=str,
        default=None,
        help="Whether to built an use yolo engine. This might speed up the processing (Default: None). Quantizazion options: 'int8', 'fp16', 'fp32'.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Disable error catching (Default: False)",
    )
    parser.add_argument(
        "--krow_assignment",
        action="store_true",
        help="Whether to assign detected kernels to rows (Default: False)",
    )
    parser.add_argument(
        "--first_i",
        type=int,
        default=None,
        help="Start loop at given loop number.",
    )
    parser.add_argument(
        "--last_i",
        type=int,
        default=None,
        help="Stop loop at given loop number.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory",
    )
    args = parser.parse_args()
    return args
