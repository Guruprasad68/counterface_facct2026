import argparse
import json
import os
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from llava.mm_utils import get_model_name_from_path, process_images
from llava.model.builder import load_pretrained_model


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_clip_model(device: str):
    model_path = "liuhaotian/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    _, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device_map=device,
        device=device,
    )
    vision_tower = model.get_vision_tower()
    vision_tower.select_feature = "cls_patch"  # Needed or cls embeddings get truncated
    return model, image_processor, vision_tower


def get_clip_embeddings(
    storage_base_dir,
    images_path_list,
    model,
    image_processor,
    vision_tower,
    device,
    batch_size=16,
):
    """
    Generate and save CLIP embeddings for a list of images.

    Args:
        storage_base_dir (str): Base directory to store embeddings.
        images_path_list (list): List of image file paths.
        model: Loaded LLaVA model.
        image_processor: LLaVA image processor.
        vision_tower: Vision tower submodule.
        device (str): Torch device string (e.g. cuda:0).
        batch_size (int): Number of images to process in a single batch.

    Returns:
        None
    """
    num_images = len(images_path_list)

    for i in range(0, num_images, batch_size):
        batch_images = []
        batch_paths = images_path_list[i : i + batch_size]

        for image_path in batch_paths:
            batch_images.append(Image.open(image_path))

        image_processed = process_images(batch_images, image_processor, model.config).to(device)
        embeddings = vision_tower(image_processed)

        for image_path, embedding in zip(batch_paths, embeddings):
            parts = image_path.split("/")
            attribute = parts[-4]
            demo = parts[-3]
            image_name = parts[-2]
            image_id = parts[-1].split(".")[0]

            Path(f"{storage_base_dir}/{attribute}/{demo}/{image_name}").mkdir(parents=True, exist_ok=True)

            save_path = f"{storage_base_dir}/{attribute}/{demo}/{image_name}/{image_id}.pt"
            torch.save(embedding.squeeze(0), save_path)

    print(f"Embeddings saved at {storage_base_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_base_dir", type=str, required=True)
    parser.add_argument("--paths_file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    storage_base_dir = args.storage_base_dir
    paths_file = args.paths_file
    device = args.device

    if not os.path.exists(storage_base_dir):
        Path(storage_base_dir).mkdir(parents=True, exist_ok=True)

    model, image_processor, vision_tower = load_clip_model(device)
    attr_demo_wise_paths = load_json(paths_file)
    for attr in tqdm(attr_demo_wise_paths, desc="Attributes"):
        for demo in attr_demo_wise_paths[attr]:
            get_clip_embeddings(
                storage_base_dir,
                attr_demo_wise_paths[attr][demo],
                model,
                image_processor,
                vision_tower,
                device,
            )


if __name__ == "__main__":
    main()

## Example code for creating the image_paths file (paths below are placeholders)
# import glob
#
# diffusion_attributes = ['sunglasses', 'mustache', ...]
# DEMOGRAPHICS = ['asian_man', 'asian_woman', ...]
# TRANSFORMED_FACES_DIR = "/path/to/datasets/transformed_faces"
#
# attribute_demo_wise_paths_for_distortion_embeddings = {}
# ...
#             all_paths = glob.glob(f"{TRANSFORMED_FACES_DIR}/{attr}/{demo}/**/*.jpg", recursive=True)
#             ...
# save_json(attribute_demo_wise_paths_for_distortion_embeddings, "/path/to/attribute_demo_wise_paths.json")
