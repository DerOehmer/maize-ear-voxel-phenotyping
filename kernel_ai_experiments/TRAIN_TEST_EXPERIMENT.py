from yv8train import YOLOv8Trainer
from apply_yolo_and_sam import YOLOAndSAMRunner
from kernel_test_torchmetrics import InferenceTesting
import os
import yaml
from pathlib import Path


class TrainTestExperiment:
    def __init__(self, backbones: list[str], dataset_yamls: list[str]):
        self.backbones = backbones
        self.dataset_yamls = dataset_yamls

    def run(self):
        for backbone in self.backbones:
            if not os.path.exists(backbone):
                raise FileNotFoundError(f"Backbone model file {backbone} not found.")
            for dataset_yaml in self.dataset_yamls:
                if not os.path.exists(dataset_yaml):
                    raise FileNotFoundError(
                        f"Dataset YAML file {dataset_yaml} not found."
                    )

                print(f"Training with backbone: {backbone} on dataset: {dataset_yaml}")
                trainer = YOLOv8Trainer(
                    model_path=backbone, training_data_config_path=dataset_yaml
                )
                backbone_name = self._path_to_name(backbone)
                dataset_name = self._path_to_name(dataset_yaml)
                dataset_dir = os.path.dirname(dataset_yaml)
                project_name = f"{backbone_name}_{dataset_name}"
                training_project = (
                    f"kernel_ai_experiments/TrainingResults/{project_name}"
                )
                if not os.path.exists(training_project):
                    os.makedirs(training_project)
                trainer.train(project=training_project, epochs=300)

                test_img_ps, test_mask_ps = self._get_test_img_and_mask_paths_from_yaml(
                    dataset_yaml
                )

                inference_project = (
                    f"kernel_ai_experiments/InferenceTestResults/{project_name}"
                )
                if not os.path.exists(inference_project):
                    os.makedirs(inference_project)

                yolo_mask_res_p = os.path.join(inference_project, "yolo")
                sam_mask_res_p = os.path.join(inference_project, "sam")
                yolo_bbox_res_p = os.path.join(inference_project, "yolo_bboxes")

                yolo_weight_p = os.path.join(
                    training_project, "train", "weights", "best.pt"
                )
                if not "complete" in dataset_name:
                    inf_runner = YOLOAndSAMRunner(
                        images_dir=test_img_ps,
                        out_yolo_txt_dir=yolo_mask_res_p,
                        use_sam=True,
                        sam_checkpoint="sam_vit_b_01ec64.pth",
                        sam_model_type="vit_b",
                        out_sam_png_dir=sam_mask_res_p,
                        save_yolo_boxes_dir=yolo_bbox_res_p,
                        yolo_weights=yolo_weight_p,
                        save_results_dir=os.path.join(
                            inference_project, "result_overlays"
                        ),
                        save_yolo_boxes_format="yolo",
                    )
                    inf_runner.run()

                    inf_tester = InferenceTesting(
                        gt_dir=Path(os.path.join(dataset_dir, "test")),
                        pred_dir=Path(inference_project),
                        backbone=backbone_name,
                        dataset=dataset_name,
                    )
                    inf_tester.run_test()

                # If dataset is "complete", also run inference and testing on the 5MP and 20MP subsets
                else:
                    for single_dataset_yaml in self.dataset_yamls:
                        if "complete" in single_dataset_yaml:
                            continue
                        single_dataset_name = self._path_to_name(single_dataset_yaml)
                        single_inference_project = (
                            f"{inference_project}_{single_dataset_name}"
                        )
                        yolo_mask_res_p = os.path.join(single_inference_project, "yolo")
                        sam_mask_res_p = os.path.join(single_inference_project, "sam")
                        yolo_bbox_res_p = os.path.join(
                            single_inference_project, "yolo_bboxes"
                        )

                        inf_runner = YOLOAndSAMRunner(
                            images_dir=self._get_test_img_and_mask_paths_from_yaml(
                                single_dataset_yaml
                            )[0],
                            out_yolo_txt_dir=yolo_mask_res_p,
                            use_sam=True,
                            sam_checkpoint="sam_vit_b_01ec64.pth",
                            sam_model_type="vit_b",
                            out_sam_png_dir=sam_mask_res_p,
                            save_yolo_boxes_dir=yolo_bbox_res_p,
                            yolo_weights=yolo_weight_p,
                            save_results_dir=os.path.join(
                                single_inference_project, "result_overlays"
                            ),
                            save_yolo_boxes_format="yolo",
                        )
                        inf_runner.run()
                        single_dataset_dir = os.path.dirname(single_dataset_yaml)
                        inf_tester = InferenceTesting(
                            gt_dir=Path(os.path.join(single_dataset_dir, "test")),
                            pred_dir=Path(single_inference_project),
                            backbone=backbone_name,
                            dataset=single_dataset_name + "_c",
                        )
                        inf_tester.run_test()

    def _path_to_name(self, path: str) -> str:
        base = os.path.basename(path)
        name, _ = os.path.splitext(base)
        return name

    def _get_test_img_and_mask_paths_from_yaml(self, yaml_path: str):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        if "test" not in data or not data["test"]:
            raise ValueError(f"No test set defined in {yaml_path}")

        yaml_dir = os.path.dirname(yaml_path)
        test_paths = data["test"]
        if isinstance(test_paths, str):
            test_paths = [os.path.join(yaml_dir, test_paths)]
        elif isinstance(test_paths, list):
            test_paths = [os.path.join(yaml_dir, tp) for tp in test_paths]
        else:
            raise ValueError(f"Unexpected format for test paths in {yaml_path}")

        img_paths = []
        mask_paths = []
        for test_p in test_paths:
            if not os.path.exists(test_p):
                raise FileNotFoundError(
                    f"Test path {test_p} from {yaml_path} does not exist."
                )

            for root, _, files in os.walk(test_p):
                for file in files:
                    if file.endswith((".jpg", ".jpeg", ".png")):
                        img_paths.append(os.path.join(root, file))
                    elif file.endswith(".txt"):
                        mask_paths.append(os.path.join(root, file))

            if not img_paths:
                raise ValueError(f"No image files found in test path {test_p}")
            if not mask_paths:
                raise ValueError(f"No mask files found in test path {test_p}")

        return img_paths, mask_paths


def initial_dir_check(dirs: list[str]):
    for d in dirs:
        if not os.path.exists(d):
            raise FileNotFoundError(f"Required directory {d} does not exist.")


def main():
    BACKBONES = [
        "yolov8m-seg.pt"
    ]  # ["yolov8n-seg.pt", "yolov8m-seg.pt", "yolov8x-seg.pt"]
    DATASET_YAMLS = [
        "kernel_ai_experiments/trainingdata/5MP_TVT/5MP_TVT.yaml",
    ]
    initial_dir_check(BACKBONES + DATASET_YAMLS)

    exp_runner = TrainTestExperiment(backbones=BACKBONES, dataset_yamls=DATASET_YAMLS)
    exp_runner.run()


if __name__ == "__main__":
    main()
