import argparse
import base64
import json
import os
import time
from functools import partial
from multiprocessing import Pool

import requests
from dotenv import load_dotenv
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


gpt4o_attribute_survey_prompt = """
Does this image have {attribute}? (begin your answer with Yes or No)
"""


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


def get_completion(attribute, prompt, few_shots, base64_image):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gpt-4o-2024-05-13",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"You need to analyze face images for presence of {attribute} and respond with yes or no. Below are some references.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"This image has {attribute}."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(few_shots[0])}"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"This image does not have {attribute}."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(few_shots[1])}"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"This image has {attribute}."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(few_shots[2])}"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"This image does not have {attribute}."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(few_shots[3])}"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}\n Make sure to use the examples for reference and answer accordingly.",
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            },
        ],
        "max_tokens": 1000,
        "temperature": 0.1,
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_gpt_responses(image_path, few_shots, attribute):
    response = get_completion(
        attribute=attribute,
        prompt=gpt4o_attribute_survey_prompt.format(attribute=attribute),
        few_shots=few_shots,
        base64_image=encode_image(image_path),
    )
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (AttributeError, KeyError, TypeError, json.JSONDecodeError):
        try:
            content = parse_json(json.loads(response.text)["choices"][0]["message"]["content"])
        except Exception as e:
            content = "Error:" + str(e)
    return (image_path, content)


def process_path(pair, few_shots, attribute):
    path, transformed_path = pair.split("|")[0], pair.split("|")[1]
    try:
        ans = get_gpt_responses(transformed_path, few_shots, attribute)
        if ans[1].strip().startswith("Yes"):
            return path, "Yes"
        if ans[1].strip().startswith("No"):
            return path, "No"
        return path, "Error"
    except Exception:
        return path, "Error"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file_path", type=str, required=True)
    parser.add_argument(
        "--few_shot_paths_json",
        type=str,
        required=True,
        help="JSON mapping attribute -> demo -> few-shot image paths",
    )
    parser.add_argument(
        "--eval_pairs_json",
        type=str,
        required=True,
        help="JSON mapping attribute -> demo -> list of 'path|transformed_path' pairs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_file_path = args.output_file_path

    gpt_attr_demo_pairs = [
        ("smile", "asian_woman"),
        ("smile", "black_woman"),
        ("smile", "indian_man"),
        ("smile", "indian_woman"),
        ("shoulder_length_hair", "asian_woman"),
        ("shoulder_length_hair", "black_woman"),
        ("shoulder_length_hair", "indian_woman"),
        ("shoulder_length_hair", "white_woman"),
    ]

    few_shots_all = load_json(args.few_shot_paths_json)
    paths_for_few_shot = load_json(args.eval_pairs_json)

    if os.path.exists(output_file_path):
        few_shot_responses = load_json(output_file_path)
    else:
        few_shot_responses = {}

    for attribute, demo in gpt_attr_demo_pairs:
        few_shots = few_shots_all[attribute][demo]
        eval_paths = paths_for_few_shot[attribute][demo]
        eval_paths = [pair for pair in eval_paths if pair.split("|")[0] not in few_shot_responses]

        process_path_partial = partial(process_path, few_shots=few_shots, attribute=attribute)
        with Pool(25) as p:
            results = list(
                tqdm(p.imap(process_path_partial, eval_paths, chunksize=1), total=len(eval_paths), desc="Processing")
            )

        for path, response in results:
            few_shot_responses[path] = {attribute: response}

        save_json(few_shot_responses, output_file_path)


if __name__ == "__main__":
    main()
