import json
import pickle as pkl

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from helpers.helper import Helper
from models.affordance_reasoning.myeval_mask import ImageProcessor


class AffordanceHelper(Helper):
    def __init__(self, aff_cfg, logdir, seed=0, device="cuda:0"):
        super().__init__(aff_cfg, f"{logdir}/affordance")

        self.device = device
        self.seed = seed
        self.img_proc = ImageProcessor(self.cfg, self.device, self.seed)

    def deal_image(self, image_np: np.array, text: str, depth=None, req_mask=False, mask=False):
        img = Image.fromarray(image_np).convert("RGB")
        # breakpoint()

        if depth is not None:
            img, image_processed, mask, box, scores = self.img_proc.process_image_directly([text], img, depth, req_mask=req_mask, mask=mask)
        else:
            img, image_processed, mask, box, scores = self.img_proc.process_image_wo_depth([text], img, req_mask=req_mask, mask=mask)
        return img, image_processed, mask, box, scores

    def affordance_query(self, image: np.array, prompt: str, depth=None, quiet=False, require_img=False):
        img, image_processed, mask, box, scores = self.deal_image(image, prompt, depth, req_mask=self.cfg.req_mask, mask=self.cfg.masks)

        if not quiet:
            image_save_path = self.get_export_path("rgb", ext="png")
            depth_save_path = self.get_export_path("depth", ext="pkl")
            Image.fromarray(image.astype("uint8"), "RGB").save(image_save_path)
            if depth is not None:
                pkl.dump(depth.tolist(), open(depth_save_path, "wb"))

            proc_image_save_path = self.get_export_path("rgb_proc", ext="png")
            cv2.imwrite(proc_image_save_path, cv2.cvtColor(image_processed, cv2.COLOR_BGR2RGB))

            mask_save_path = self.get_export_path("aff_mask", ext="png")
            mask = mask.squeeze()
            Image.fromarray(mask.astype("uint8") * 255).save(mask_save_path)

            json_save_path = self.get_export_path("aff_res", ext="json")
            json.dump(
                {"bbox": box.tolist(), "scores": scores[0].tolist()},
                open(json_save_path, "w"),
            )

            logger.info(f"Affordance query: {prompt}. Results saved to {self.logdir}")

        if require_img:
            return image_processed, box.tolist(), mask, scores
        else:
            return box.tolist(), mask, scores
