import argparse
import json
import os
import time
from multiprocessing import Pool

import pandas as pd
import vertexai
import vertexai.preview.generative_models as generative_models
from pathlib import Path
from system_prompts import *
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from vertexai.generative_models import GenerativeModel, Image, Part


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


def init_tqdm(total):
    global pbar
    pbar = tqdm(total=total, desc="Processing Images")


def update_tqdm(_):
    pbar.update()


def parse_json(data_str):
    data_str = data_str.replace("```json\n", "", 1)
    data_str = data_str.rsplit("\n```", 1)[0]
    data_str = data_str.replace("'", '"')
    parsed_json = json.loads(data_str)
    return parsed_json


def generate(image1, system_prompt, vertex_project: str, vertex_location: str):
    vertexai.init(project=vertex_project, location=vertex_location)

    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 0.1,
        "top_p": 0.55,
    }

    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    model = GenerativeModel(
        "gemini-1.5-pro-001",
        system_instruction=[system_prompt],
    )
    response = model.generate_content(
        [image1, "Annotate the images for the different attributes"],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=False,
    )

    return response


@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def generate_with_retry(image1, system_prompt, vertex_project: str, vertex_location: str):
    return generate(image1, system_prompt, vertex_project, vertex_location)


def process_image(image_path, system_prompt, vertex_project: str, vertex_location: str):
    try:
        time.sleep(5)
        image1 = Part.from_image(Image.load_from_file(image_path))

        response = generate_with_retry(image1, system_prompt=system_prompt, vertex_project=vertex_project, vertex_location=vertex_location)
        caption = parse_json(response.candidates[0].content.parts[0].text.strip())
        return image_path, caption
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return image_path, None


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
    parser.add_argument("--vertex_project", type=str, required=True, help="GCP project id for Vertex AI")
    parser.add_argument("--vertex_location", type=str, default="us-central1")
    parser.add_argument("--num_processes", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=15, required=False)

    return parser.parse_args()


def main():
    args = parse_args()

    paths_file = args.paths_file
    output_file_path = args.output_file_path
    batch_size = args.batch_size
    num_processes = args.num_processes
    vertex_project = args.vertex_project
    vertex_location = args.vertex_location

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
            system_prompt_for_attribute_demo = make_prompt_based_on_attributes(
                get_attributes_for_prompt_based_on_demo(best_model_df, demo, model_name="gemini")
            )

            for image_paths_batch in tqdm(
                [image_paths[i : i + batch_size] for i in range(0, len(image_paths), batch_size)],
                desc="Batch Processing",
            ):
                with Pool(num_processes, initializer=init_tqdm, initargs=(len(image_paths_batch),)) as pool:
                    args_to_pass = [
                        (image_path, system_prompt_for_attribute_demo, vertex_project, vertex_location)
                        for image_path in image_paths_batch
                    ]
                    results = pool.starmap(process_image, args_to_pass)
                caption_dict = {**caption_dict, **dict(results)}

                save_json(caption_dict, output_file_path)


if __name__ == "__main__":
    main()
