import argparse
import json

import joblib
import numpy as np
import pandas as pd
import torch


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths_file", type=str, required=True, help="Attribute demo-wise paths JSON")
    parser.add_argument("--embedding_dir", type=str, required=True, help="Directory containing the embeddings")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON path")
    parser.add_argument(
        "--detector_dir",
        type=str,
        required=True,
        help="Directory with linear_svm_model_just_cls_token_v4.pkl and group_demo_wise_tuned_tpr_fpr.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths_file = args.paths_file
    embedding_dir = args.embedding_dir
    output_file = args.output_file
    detector_dir = args.detector_dir

    attr_demo_wise_paths = load_json(paths_file)

    model = joblib.load(f"{detector_dir}/linear_svm_model_just_cls_token_v4.pkl")

    df_threshold = pd.read_csv(f"{detector_dir}/group_demo_wise_tuned_tpr_fpr.csv")
    df_threshold.set_index("Unnamed: 0", inplace=True)
    df_threshold.index.name = None

    distortion_responses = {}
    for attr in attr_demo_wise_paths:
        distortion_responses[attr] = {}
        for demo in attr_demo_wise_paths[attr]:
            distortion_responses[attr][demo] = {}
            image_paths = attr_demo_wise_paths[attr][demo]

            embedding_paths = [
                f'{embedding_dir}/{path.split("/")[-4]}/{path.split("/")[-3]}/{path.split("/")[-2]}/{path.split("/")[-1].split(".")[0]}.pt'
                for path in image_paths
            ]
            embeddings = [torch.load(embedding_path)[0].cpu().numpy() for embedding_path in embedding_paths]
            embeddings = np.array(embeddings)

            probabilities = model.decision_function(embeddings)

            threshold = float(df_threshold[demo][attr].split(",")[2])

            labels = [1 if prob > threshold else 0 for prob in probabilities]

            for path, label in zip(image_paths, labels):
                distortion_responses[attr][demo][path] = label

    save_json(distortion_responses, output_file)


if __name__ == "__main__":
    main()
