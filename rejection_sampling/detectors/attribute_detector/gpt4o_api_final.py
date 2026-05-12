import argparse
import base64
import json
import os
import time
from multiprocessing import Pool

import pandas as pd
import requests
from dotenv import load_dotenv
from pathlib import Path
from system_prompts import *
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_detector_assignment_table(csv_path: str) -> pd.DataFrame:
    best_model_df = pd.read_csv(csv_path)
    best_model_df.set_index("Unnamed: 0", inplace=True)
    best_model_df.index.name = None
    return best_model_df


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def parse_json(data_str):
    data_str = data_str.strip("'")
    data_str = data_str.replace("```json\n", "", 1)
    data_str = data_str.rsplit("\n```", 1)[0]
    data_str = data_str.replace("'", '"')

    parsed_json = json.loads(data_str)

    return parsed_json


def get_completion(image_path, prompt):
    base64_image = encode_image(image_path)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gpt-4o-2024-05-13",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.1,
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_with_retry(image1, system_prompt):
    return get_completion(image1, system_prompt)


def get_gpt_responses(image_path, system_prompt):
    response = generate_with_retry(image_path, system_prompt)
    try:
        content = parse_json(response.json()["choices"][0]["message"]["content"])
    except (AttributeError, KeyError, TypeError, json.JSONDecodeError):
        try:
            content = parse_json(json.loads(response.text)["choices"][0]["message"]["content"])
        except Exception:
            content = "Error"
    return (image_path, content)


def get_attributes_for_prompt_based_on_demo(best_model_df: pd.DataFrame, demo: str, model_name: str):
    demo_df = best_model_df[demo]

    attributes_for_prompt = []

    for attr in demo_df.index:
        if demo_df[attr] == model_name:
            attributes_for_prompt.append(attr)

    return attributes_for_prompt


def init_tqdm(total):
    global pbar
    pbar = tqdm(total=total, desc="Processing Images")


def update_tqdm(_):
    pbar.update()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths_file", type=str, required=True)
    parser.add_argument("--output_file_path", type=str, required=True)
    parser.add_argument("--best_detector_csv", type=str, required=True, help="CSV: best detector per attribute x demographic")
    parser.add_argument("--batch_size", type=int, default=25)

    return parser.parse_args()


def main():
    args = parse_args()
    output_file_path = args.output_file_path
    batch_size = args.batch_size
    paths_file = args.paths_file

    best_model_df = load_detector_assignment_table(args.best_detector_csv)
    attr_demo_wise_image_paths = load_json(paths_file)

    if not os.path.exists(os.path.dirname(output_file_path)):
        Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)

    if os.path.exists(output_file_path):
        already_done = load_json(output_file_path)
    else:
        already_done = {}

    caption_dict = already_done
    for attr in attr_demo_wise_image_paths:
        for demo in attr_demo_wise_image_paths[attr]:
            image_paths = attr_demo_wise_image_paths[attr][demo]

            image_paths = [image_path for image_path in image_paths if image_path not in caption_dict.keys()]
            image_paths = [image_path for image_path in image_paths if os.path.exists(image_path)]

            attributes_list = get_attributes_for_prompt_based_on_demo(best_model_df, demo, model_name="gpt4o")
            system_prompt_for_attribute_demo = make_prompt_based_on_attributes(attributes_list)

            for image_paths_batch in tqdm(
                [image_paths[i : i + batch_size] for i in range(0, len(image_paths), batch_size)]
            ):
                with Pool(25, initializer=init_tqdm, initargs=(len(image_paths_batch),)) as pool:
                    args_to_pass = [(image_path, system_prompt_for_attribute_demo) for image_path in image_paths_batch]
                    results = pool.starmap(get_gpt_responses, args_to_pass)
                caption_dict = {**caption_dict, **dict(results)}

                save_json(caption_dict, output_file_path)


if __name__ == "__main__":
    main()
