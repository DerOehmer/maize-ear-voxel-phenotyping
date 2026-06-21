import segment_anything
from segment_anything.utils.transforms import ResizeLongestSide
import numpy as np
import torch
from typing import List, Any
import cv2


class SamSeg:
    def __init__(self, sam_checkpoint, weight_type="vit_l", device="cuda"):
        self.device = device

        checkpoint = torch.load(
            sam_checkpoint, map_location=torch.device(device), weights_only=True
        )

        # Initialize the model from the registry without loading the checkpoint
        self.sam = segment_anything.sam_model_registry[weight_type]()

        # Load the state_dict from the checkpoint
        self.sam.load_state_dict(checkpoint)

        # Move the model to the appropriate device
        self.sam.to(device=self.device)

        predictor = segment_anything.SamPredictor(self.sam)
        self.predictor = predictor

    def _prep_image(self, imgobj: Any, transform: Any):
        img = transform.apply_image(cv2.cvtColor(imgobj.img, cv2.COLOR_BGR2RGB))
        img = torch.tensor(img, device=self.device).permute(2, 0, 1).contiguous()
        return img

    def _postprocess_mask(self, mask):
        mask = mask.cpu().numpy().astype(np.uint8) * 255
        return np.squeeze(mask, 0)  # Shape (1, H, W) -> (H, W)

    def compile_img_batch(self, img_objs: List[Any], bboxes_xyxy: List[List]) -> tuple:
        resize_transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        batched_input = []
        invalid_indcs = []
        for i, bboxes_img in enumerate(bboxes_xyxy):
            if len(bboxes_img) == 0:
                invalid_indcs.append(i)
                continue
            sam_img = self._prep_image(img_objs[i], resize_transform)
            sam_boxes = self.bboxtransforming(
                img_objs[i].shape, torch.tensor(bboxes_img)
            )
            batched_input.append(
                {
                    "image": sam_img,
                    "boxes": sam_boxes,
                    "original_size": img_objs[i].shape[:2],
                }
            )

        return batched_input, invalid_indcs

    def seg_from_bbox_batch(
        self, img_objs: List[Any], bboxes_xyxy: List[List], batch_size: int = 4
    ) -> List[List[Any]]:
        batched_input, invalid_idcs = self.compile_img_batch(img_objs, bboxes_xyxy)
        batched_output = []
        for i in range(0, len(batched_input) + 1, batch_size):
            sub_batched_input = batched_input[i : i + batch_size]
            if len(sub_batched_input) == 0:
                continue
            with torch.inference_mode():
                sub_batched_output = self.sam(sub_batched_input, multimask_output=False)
            batched_output.extend(sub_batched_output)

        all_masks_lst = []
        for i, output in enumerate(batched_output):
            masks_img = output["masks"]
            masks_img_lst = []
            for mask in masks_img:
                masks_img_lst.append(self._postprocess_mask(mask))
            all_masks_lst.append(masks_img_lst)

        for i in sorted(invalid_idcs):
            all_masks_lst.insert(i, [])

        del batched_input
        del batched_output
        torch.cuda.empty_cache()

        return all_masks_lst

    def bboxtransforming(self, img_shape, bboxes):
        transf_bboxes = self.predictor.transform.apply_boxes_torch(
            bboxes.to(device=self.device), img_shape[:2]
        )
        return transf_bboxes
