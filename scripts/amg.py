# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
import json
import os
from typing import Any, Dict, List
from PIL import Image  # 用于处理图像
import numpy as np  # 用于数组操作

# 解析命令行参数
parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument("--input", type=str, required=True, help="Path to either a single input image or folder of images.")
parser.add_argument("--output", type=str, required=True, help="Path to the directory where masks will be output.")
parser.add_argument("--model-type", type=str, required=True, help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']")
parser.add_argument("--checkpoint", type=str, required=True, help="The path to the SAM checkpoint to use for mask generation.")
parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")
parser.add_argument("--convert-to-rle", action="store_true", help="Save masks as COCO RLEs in a single json instead of PNGs.")

amg_settings = parser.add_argument_group("AMG Settings")
amg_settings.add_argument("--points-per-side", type=int, default=None)
amg_settings.add_argument("--points-per-batch", type=int, default=None)
amg_settings.add_argument("--pred-iou-thresh", type=float, default=None)
amg_settings.add_argument("--stability-score-thresh", type=float, default=None)
amg_settings.add_argument("--stability-score-offset", type=float, default=None)
amg_settings.add_argument("--box-nms-thresh", type=float, default=None)
amg_settings.add_argument("--crop-n-layers", type=int, default=None)
amg_settings.add_argument("--crop-nms-thresh", type=float, default=None)
amg_settings.add_argument("--crop-overlap-ratio", type=int, default=None)
amg_settings.add_argument("--crop-n-points-downscale-factor", type=int, default=None)
amg_settings.add_argument("--min-mask-region-area", type=int, default=None)

def apply_mask_and_save(input_image_path: str, mask: np.ndarray, output_path: str) -> None:
    """应用掩码并保存为透明背景的 PNG 图像"""
    image = Image.open(input_image_path).convert("RGBA")
    image_np = np.array(image)

    mask_np = (mask * 255).astype(np.uint8)  # 将掩码转换为 Alpha 通道
    image_np[:, :, 3] = mask_np  # 替换 Alpha 通道

    result_image = Image.fromarray(image_np)
    result_image.save(output_path, format="PNG")

def write_single_mask(input_image_path: str, masks: List[Dict[str, Any]], output_path: str) -> None:
    """只使用面积最大的掩码来生成一个抠像图"""
    if not masks:
        print("No valid masks found.")
        return

    # 筛选面积最大的掩码
    largest_mask = max(masks, key=lambda x: x["area"])

    # 应用掩码并保存为 PNG
    mask = largest_mask["segmentation"]
    apply_mask_and_save(input_image_path, mask, output_path)

    print(f"Saved the cutout image to: {output_path}")

def get_amg_kwargs(args):
    """获取 AMG 配置参数"""
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    return {k: v for k, v in amg_kwargs.items() if v is not None}

def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)

    generator = SamAutomaticMaskGenerator(sam)

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [os.path.join(args.input, f) for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))]

    os.makedirs(args.output, exist_ok=True)

    for t in targets:
        print(f"Processing '{t}'...")
        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = generator.generate(image)

        output_file = os.path.join(args.output, f"{os.path.basename(t).split('.')[0]}_cutout.png")
        write_single_mask(t, masks, output_file)

    print("Done!")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)