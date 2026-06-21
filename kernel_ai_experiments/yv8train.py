import argparse
from ultralytics import YOLO
import torch


class YOLOv8Trainer:
    def __init__(self, model_path: str, training_data_config_path: str):
        self.model_path = model_path
        self.training_data_config_path = training_data_config_path
        self.device = [0, 1] if torch.cuda.device_count() == 2 else 0
        self.batch = 16 if torch.cuda.device_count() == 2 else 0
        self.model = YOLO(self.model_path, verbose=False)

    def train(self, project: str, epochs: int = 300):
        results = self.model.train(
            data=self.training_data_config_path,
            epochs=epochs,
            device=self.device,
            project=project,
            batch=self.batch,
            verbose=False,
        )
        return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 segmentation model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="yolov8n-seg.pt",
        help="Path to the pretrained YOLO model. Default is 'yolov8n-seg.pt'.",
    )
    args = parser.parse_args()

    trainer = YOLOv8Trainer(
        model_path=args.model_path,
        training_data_config_path="trainingdata/_test_5MP_TVT/_test_5MP_TVT.yaml",
    )
    trainer.train(project="training_results", epochs=200)

    test_metrics = trainer.model.val(split="test")
    print("map50-95", test_metrics.box.map)  # map50-95
    print("map50", test_metrics.box.map50)  # map50
    print("map75", test_metrics.box.map75)  # map75


if __name__ == "__main__":
    main()
