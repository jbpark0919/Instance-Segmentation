# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from PIL import Image
from pathlib import Path

# Cityscapes 클래스 매핑 (19개 주요 클래스)
CITYSCAPES_CATEGORIES = [
    {"id": 1, "name": "road"},
    {"id": 2, "name": "sidewalk"},
    {"id": 3, "name": "building"},
    {"id": 4, "name": "wall"},
    {"id": 5, "name": "fence"},
    {"id": 6, "name": "pole"},
    {"id": 7, "name": "traffic light"},
    {"id": 8, "name": "traffic sign"},
    {"id": 9, "name": "vegetation"},
    {"id": 10, "name": "terrain"},
    {"id": 11, "name": "sky"},
    {"id": 12, "name": "person"},
    {"id": 13, "name": "rider"},
    {"id": 14, "name": "car"},
    {"id": 15, "name": "truck"},
    {"id": 16, "name": "bus"},
    {"id": 17, "name": "train"},
    {"id": 18, "name": "motorcycle"},
    {"id": 19, "name": "bicycle"},
]

CATEGORY_ID_MAP = {cat["name"]: cat["id"] for cat in CITYSCAPES_CATEGORIES}

def convert_to_coco(cityscapes_root, split, out_dir):
    """
    Cityscapes 데이터셋을 COCO 형식으로 변환
    """
    leftImg8bit_dir = os.path.join(cityscapes_root, "leftImg8bit", split)
    gtFine_dir = os.path.join(cityscapes_root, "gtFine", split)

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": CITYSCAPES_CATEGORIES,
    }
    
    image_id = 0
    annotation_id = 0

    for city in tqdm(os.listdir(leftImg8bit_dir), desc=f"Processing {split}"):
        city_img_dir = os.path.join(leftImg8bit_dir, city)
        city_gt_dir = os.path.join(gtFine_dir, city)

        for img_file in os.listdir(city_img_dir):
            if not img_file.endswith("_leftImg8bit.png"):
                continue
            
            img_path = os.path.join(city_img_dir, img_file)
            img_name = img_file.replace("_leftImg8bit.png", "")

            # GT 파일 경로
            gt_json_path = os.path.join(city_gt_dir, f"{img_name}_gtFine_polygons.json")

            if not os.path.exists(gt_json_path):
                continue
            
            with open(gt_json_path, "r") as f:
                gt_data = json.load(f)

            # 이미지 정보 추가
            img = Image.open(img_path)
            width, height = img.size
            coco_data["images"].append({
                "id": image_id,
                "file_name": os.path.join(split, city, img_file),
                "width": width,
                "height": height,
            })

            # 어노테이션 추가
            for obj in gt_data["objects"]:
                category_name = obj["label"]
                if category_name not in CATEGORY_ID_MAP:
                    continue
                
                category_id = CATEGORY_ID_MAP[category_name]
                segmentation = [np.ravel(obj["polygon"]).tolist()]  # COCO 형식 맞추기
                
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "iscrowd": 0,
                    "area": 0,  # Placeholder (실제 영역 계산 가능)
                    "bbox": [],  # Placeholder (bounding box 필요 시 추가)
                })
                annotation_id += 1
            
            image_id += 1

    # JSON 저장
    out_path = os.path.join(out_dir, f"instancesonly_filtered_gtFine_{split}.json")
    os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(coco_data, f, indent=4)
    
    print(f"✅ {split} 변환 완료! 저장 위치: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert Cityscapes to COCO format")
    parser.add_argument("cityscapes_root", type=str, help="Path to Cityscapes dataset root")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for COCO JSON files")
    parser.add_argument("--nproc", type=int, default=8, help="Number of parallel processes")

    args = parser.parse_args()

    # 병렬 처리 실행
    splits = ["train", "val", "test"]
    with mp.Pool(args.nproc) as pool:
        pool.starmap(convert_to_coco, [(args.cityscapes_root, split, args.out_dir) for split in splits])

if __name__ == "__main__":
    main()
