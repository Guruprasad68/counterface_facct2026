"""
Generate modified faces using SEGA (Semantic Guidance) with Stable Diffusion.
Applies facial attribute edits to source face images.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import torch
from diffusers import SemanticStableDiffusionPipeline
from tqdm import tqdm

# 18 diffusion/SEGA attributes (excludes bald, blond_hair which are GAN-only)
SEGA_ATTRIBUTES = [
    "facemask",
    "sunglasses",
    "mustache",
    "pigtails",
    "scarf",
    "blue_hair",
    "head_band",
    "dark_colored_skin_tone",
    "light_colored_skin_tone",
    "buzz_cut",
    "glasses",
    "heavy_makeup",
    "red_hair",
    "shoulder_length_hair",
    "goatee",
    "red_lipstick",
    "smile",
    "thick_beard",
]

NEGATIVE_PROMPTS = """(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly,
duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured,
gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"""


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate modified faces using SEGA")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for modified faces")
    parser.add_argument("--edit_caption_json", type=str, required=True, help="Attribute to edit caption mapping JSON")
    parser.add_argument("--hyperparams_men_json", type=str, required=True, help="SEGA hyperparameters for men")
    parser.add_argument("--hyperparams_women_json", type=str, required=True, help="SEGA hyperparameters for women")
    parser.add_argument("--demographics2names_json", type=str, required=True, help="Demographics to names mapping JSON")
    parser.add_argument("--seeds_json", type=str, required=True, help="Seeds per identity JSON")
    parser.add_argument("--attributes", type=str, default=None, help="Comma-separated attributes to process (default: all SEGA attributes)")
    parser.add_argument("--demographics", type=str, default=None, help="Comma-separated demographics to process (default: all)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    return parser.parse_args()


def main():
    args = parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load configs
    edit_captions = load_json(args.edit_caption_json)
    hyperparams_men = load_json(args.hyperparams_men_json)
    hyperparams_women = load_json(args.hyperparams_women_json)
    demographics2names = load_json(args.demographics2names_json)
    source_seeds = load_json(args.seeds_json)

    # Determine which attributes and demographics to process
    if args.attributes:
        attributes = [a.strip() for a in args.attributes.split(",")]
    else:
        attributes = SEGA_ATTRIBUTES

    if args.demographics:
        demographics = [d.strip() for d in args.demographics.split(",")]
    else:
        demographics = list(demographics2names.keys())

    # Load pipeline
    pipe = SemanticStableDiffusionPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V4.0", safety_checker=None
    ).to(args.device)

    prompt_head = "A photo of the face of"

    for attr in tqdm(attributes, desc="Attributes"):
        if attr not in edit_captions:
            print(f"Warning: {attr} not in edit captions, skipping")
            continue

        for demo in tqdm(demographics, desc="Demographics", leave=False):
            if demo not in demographics2names:
                continue

            # Select hyperparameters based on gender
            if demo.split("_")[1] == "man":
                hyperparams = hyperparams_men.get(attr, {})
            else:
                hyperparams = hyperparams_women.get(attr, {})

            names = demographics2names[demo]
            for name in names:
                output_path = f"{args.output_dir}/{attr}/{demo}/{name}"

                # Skip if already done
                if len(glob.glob(f"{output_path}/*.jpg")) == 6:
                    continue

                if name not in source_seeds:
                    print(f"Warning: {name} not in seeds file, skipping")
                    continue

                prompt = f"{prompt_head} {' '.join(name.split('_'))}"
                seeds = source_seeds[name]
                generators = [torch.Generator(device=args.device).manual_seed(int(seed)) for seed in seeds]

                out = pipe(
                    height=512,
                    width=512,
                    prompt=[prompt] * len(seeds),
                    generator=generators,
                    guidance_scale=7.5,
                    num_inference_steps=50,
                    negative_prompt=[NEGATIVE_PROMPTS] * len(seeds),
                    edit_mom_beta=0.6,
                    editing_prompt=edit_captions[attr],
                    **hyperparams,
                ).images

                Path(output_path).mkdir(parents=True, exist_ok=True)
                for i, img in enumerate(out):
                    img.save(f"{output_path}/{name}_000{i}.jpg")

    print(f"Modified faces saved to {args.output_dir}")


if __name__ == "__main__":
    main()
