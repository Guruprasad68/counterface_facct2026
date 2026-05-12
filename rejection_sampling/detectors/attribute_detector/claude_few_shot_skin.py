import anthropic
import argparse
import base64
import json
import os
from functools import partial
from multiprocessing import Pool

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv()


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


claude_skin_tone_survey_prompt = """
Which of the two images has lighter skin tone? (begin your answer with "Right Face", "Left Face" or "No significant difference")
"""


def encode_image(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def parse_json(data_str):
    """Parses a JSON string into a Python dictionary."""
    data_str = data_str.strip("'")
    data_str = data_str.replace("```json\n", "", 1)
    data_str = data_str.rsplit("\n```", 1)[0]
    data_str = data_str.replace("'", '"')
    return json.loads(data_str)


def generate(inputs):
    """Generate a response from Claude."""
    client = anthropic.Anthropic()
    image1_paths, few_shot_image_paths = inputs
    image1_data = encode_image(image1_paths)
    few_shot_image_data = [encode_image(img_path) for img_path in few_shot_image_paths]
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You need to analyze two face images and respond which of the two has lighter skin tone. Your response should be one of 'Right Face', 'Left Face' or 'No significant difference'. Below are some references.",
                    },
                    {"type": "text", "text": "Following is an example where the right face has lighter skin tone."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": few_shot_image_data[0],
                        },
                    },
                    {"type": "text", "text": "Following is an example where the left face has lighter skin tone."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": few_shot_image_data[1],
                        },
                    },
                    {"type": "text", "text": "Now, an example where there is no significant difference in skin tone."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": few_shot_image_data[2],
                        },
                    },
                    {
                        "type": "text",
                        "text": f"{claude_skin_tone_survey_prompt}\n Make sure to use the examples for reference and answer accordingly.",
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image1_data,
                        },
                    },
                ],
            }
        ],
    )

    return message.content[0].text


@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def generate_with_retry(inputs):
    """Retry wrapper for the generate function."""
    return generate(inputs)


def process_image(image_path, few_shot_paths):
    """Process a single image."""
    try:
        response = generate_with_retry([image_path, few_shot_paths])
        try:
            caption = parse_json(response)
        except Exception:
            caption = response

        return image_path, caption

    except Exception as e:
        return image_path, f"Error: {str(e)}"


def process_path(path, few_shots):
    """Process a path and classify lighter skin tone."""
    ans = process_image(path, few_shots)
    result = ans[1].strip().lower()

    if result.startswith("right"):
        return path, "Right Face"
    if result.startswith("left"):
        return path, "Left Face"
    if result.startswith("no"):
        return path, "No significant difference"
    return path, "Error"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file_path", type=str, required=True, help="Path to the output file")
    parser.add_argument(
        "--skin_tone_paths_json",
        type=str,
        required=True,
        help="JSON: attr -> demo -> list of image paths to evaluate",
    )
    parser.add_argument(
        "--skin_tone_examples_json",
        type=str,
        required=True,
        help="JSON: demo -> few-shot example paths for skin tone task",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_file_path = args.output_file_path

    skin_tone_few_shot_paths = load_json(args.skin_tone_paths_json)
    few_shots = load_json(args.skin_tone_examples_json)

    responses = load_json(output_file_path) if os.path.exists(output_file_path) else {}
    for attr in skin_tone_few_shot_paths:
        for demo in skin_tone_few_shot_paths[attr]:
            eval_paths = skin_tone_few_shot_paths[attr][demo]
            eval_paths = [path for path in eval_paths if path not in responses]

            process_path_partial = partial(process_path, few_shots=few_shots[demo])

            with Pool(processes=25) as pool:
                results = list(
                    tqdm(
                        pool.imap(process_path_partial, eval_paths, chunksize=1),
                        total=len(eval_paths),
                        desc="Processing",
                    )
                )

            for path, response in results:
                responses[path] = {"face_with_lighter_skin_tone": response}

            save_json(responses, output_file_path)


if __name__ == "__main__":
    main()
