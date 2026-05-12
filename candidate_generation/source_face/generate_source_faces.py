"""
Generate source faces using batched self-similarity approach.
Generates multiple face images per identity and selects the most consistent subset.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from diffusers import SemanticStableDiffusionPipeline
from torch.functional import F
from tqdm import tqdm

from embedding import FaceNetEmbedding


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def cosine_similarity(embed_1, embed_2, dim=1):
    return F.cosine_similarity(embed_1, embed_2, dim=dim).item()


def get_embeddings(facenet, images):
    """Takes a list of generated images and returns their FaceNet embeddings."""
    embeddings = {}
    for i, out_i in enumerate(images):
        embeddings[f"gen_{i}"] = facenet.get_embedding(out_i)
    return embeddings


def get_pairwise_sim(embeddings):
    """Compute pairwise cosine similarities between all embeddings."""
    pairwise_sim = {}
    num_seeds = len(embeddings)
    for i in range(num_seeds):
        for j in range(i + 1, num_seeds):
            pairwise_sim[f"{i}_{j}"] = cosine_similarity(embeddings[f"gen_{i}"], embeddings[f"gen_{j}"])
    return pairwise_sim


def get_best_combination(pairwise_sim, identity: str, num_to_keep=6, num_seeds=20):
    """
    Returns the best combination of seeds based on the average similarity score.

    Args:
        pairwise_sim: Dict containing the pairwise similarity scores
        identity: Identity of the person
        num_to_keep: Number of seeds to keep
        num_seeds: Number of seeds

    Returns:
        best_combination_df: DataFrame containing the best combination of seeds
        best_combination: Tuple containing the best combination of seed indices
        max_avg_similarity: Maximum average similarity score
    """
    max_avg_similarity = 0
    best_combination = None
    ncr = num_to_keep * (num_to_keep - 1) / 2
    combinations = itertools.combinations(range(num_seeds), num_to_keep)

    for combination in combinations:
        avg_similarity = (
            sum(pairwise_sim[f"{i}_{j}"] for idx, i in enumerate(combination) for j in combination[idx + 1 :]) / ncr
        )
        if avg_similarity > max_avg_similarity:
            best_combination = combination
            max_avg_similarity = avg_similarity

    # Create summary dataframe
    df_cols = ["Identity", "Gen_Face_ID1", "Gen_Face_ID2", "FaceNetCosineSimilarityScore"]
    best_combination_df = pd.DataFrame(columns=df_cols)
    id_dict = {v: k for k, v in enumerate(best_combination)}

    for idx_i, i in enumerate(best_combination):
        for j in best_combination[idx_i + 1 :]:
            data = {
                "Identity": identity,
                "Gen_Face_ID1": f"{identity}_000{id_dict[i]}",
                "Gen_Face_ID2": f"{identity}_000{id_dict[j]}",
                "FaceNetCosineSimilarityScore": pairwise_sim[f"{i}_{j}"],
            }
            best_combination_df = pd.concat([best_combination_df, pd.DataFrame([data])], ignore_index=True)

    return best_combination_df, best_combination, max_avg_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Generate source faces with self-similarity selection")
    parser.add_argument("--demo_wise_names_file", type=str, required=True, help="Demographics to names mapping JSON")
    parser.add_argument("--similarity_table_output_path", type=str, required=True, help="Output path for similarity tables")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated faces")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--num_seeds", type=int, default=20, help="Number of seeds to generate per identity")
    parser.add_argument("--num_to_keep", type=int, default=6, help="Number of best seeds to keep")
    return parser.parse_args()


def main():
    args = parse_args()

    demographic_wise_names = load_json(args.demo_wise_names_file)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    negative_prompts = """(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly,
        duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured,
        gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, exposing body"""

    pipe = SemanticStableDiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V4.0").to(device)
    facenet = FaceNetEmbedding()
    pipe.set_progress_bar_config(disable=True)

    demo_wise_avg_similarity = {}
    used_seeds = {}

    for demo in tqdm(demographic_wise_names, desc="Generating Demographics"):
        names = demographic_wise_names[demo]
        print(f"Demographic: {demo}")
        used_seeds[demo] = {}

        Path(f"{args.similarity_table_output_path}/{demo}").mkdir(parents=True, exist_ok=True)
        Path(f"{args.output_dir}/{demo}").mkdir(parents=True, exist_ok=True)

        avg_similarities = {}
        for name in tqdm(names, desc="Generating Names", leave=False):
            name_for_prompt = " ".join(name.split("_"))
            prompt = f"A photo of the face of {name_for_prompt}"
            all_seeds = [int(seed) for seed in np.random.randint(0, 1000000, args.num_seeds)]
            generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in all_seeds]

            out = pipe(
                height=512,
                width=512,
                prompt=[prompt] * args.num_seeds,
                generator=generators,
                guidance_scale=7.5,
                num_inference_steps=50,
                negative_prompt=[negative_prompts] * args.num_seeds,
            ).images

            embeddings = get_embeddings(facenet=facenet, images=out)
            pairwise_sim = get_pairwise_sim(embeddings)
            best_df, best_combination, avg_similarity = get_best_combination(
                pairwise_sim, name, num_to_keep=args.num_to_keep, num_seeds=args.num_seeds
            )

            avg_similarities[name] = avg_similarity
            best_df.to_csv(f"{args.similarity_table_output_path}/{demo}/{name}.csv", index=False)
            used_seeds[demo][name] = [all_seeds[k] for k in best_combination]

            Path(f"{args.output_dir}/{demo}/{name}").mkdir(parents=True, exist_ok=True)
            for idx, k in enumerate(best_combination):
                out[k].save(f"{args.output_dir}/{demo}/{name}/{name}_000{idx}.jpg")

        demo_wise_avg_similarity[demo] = np.average(list(avg_similarities.values()))
        save_json(avg_similarities, f"{args.output_dir}/{demo}/name_wise_avg_similarity.json")

    save_json(used_seeds, f"{args.output_dir}/all_used_seeds.json")
    save_json(demo_wise_avg_similarity, f"{args.output_dir}/demo_wise_avg_similarity.json")

    print(f"Source faces saved to {args.output_dir}")


if __name__ == "__main__":
    main()
