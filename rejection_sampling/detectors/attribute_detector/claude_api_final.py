import argparse
import base64
import json
import logging
import os
import time
from multiprocessing import Pool

import anthropic
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from system_prompts import *
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv()


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
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def init_tqdm(total):
    global pbar
    pbar = tqdm(total=total, desc="Processing Images")


def update_tqdm(_):
    pbar.update()


def parse_json(data_str):
    data_str = data_str.replace("```json\n", "", 1) if "```json\n" in data_str else data_str
    data_str = data_str.rsplit("\n```", 1)[0]
    data_str = data_str.replace("'", '"')
    parsed_json = json.loads(data_str)
    return parsed_json


def generate(image1_data, prompt_to_model):
    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image1_data,
                        },
                    },
                    {"type": "text", "text": prompt_to_model},
                ],
            }
        ],
    )

    return message.content[0].text


@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(7))
def generate_with_retry(image1, prompt_to_model):
    return generate(image1, prompt_to_model=prompt_to_model)


def process_image(image_path, prompt_to_model):
    try:
        time.sleep(6)
        image1_data = encode_image(image_path)
        response = generate_with_retry(image1_data, prompt_to_model)
        caption = parse_json(response)

        return image_path, caption

    except Exception:
        logging.info(f"Error in processing {image_path}")
        return image_path, "Error"


def get_attributes_for_prompt_based_on_demo(best_model_df: pd.DataFrame, demo: str, model_name: str):
    demo_df = best_model_df[demo]

    attributes_for_prompt = []

    for attr in demo_df.index:
        if demo_df[attr] == model_name:
            attributes_for_prompt.append(attr)

    return attributes_for_prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths_file", type=str, required=True)
    parser.add_argument("--output_file_path", type=str, required=True)
    parser.add_argument("--best_detector_csv", type=str, required=True, help="CSV: best detector per attribute x demographic")
    parser.add_argument("--num_processes", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=15, required=False)

    return parser.parse_args()


def main():
    args = parse_args()

    output_file_path = args.output_file_path
    batch_size = args.batch_size
    num_processes = args.num_processes
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
            attributes_list = get_attributes_for_prompt_based_on_demo(best_model_df, demo, model_name="claude")
            system_prompt_for_attribute_demo = make_prompt_based_on_attributes(attributes_list)
            for image_paths_batch in tqdm(
                [image_paths[i : i + batch_size] for i in range(0, len(image_paths), batch_size)]
            ):
                with Pool(num_processes, initializer=init_tqdm, initargs=(len(image_paths_batch),)) as pool:
                    results = pool.starmap(
                        process_image,
                        [(image_path, system_prompt_for_attribute_demo) for image_path in image_paths_batch],
                    )

                caption_dict = {**caption_dict, **dict(results)}

                save_json(caption_dict, output_file_path)


if __name__ == "__main__":
    main()
