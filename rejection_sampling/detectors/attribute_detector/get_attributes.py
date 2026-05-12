"""
Unified attribute detector that queries Claude, GPT-4o, and Gemini for face attribute detection.
Takes attribute-demo-wise concatenated image paths and returns combined responses from all models.
Supports few-shot prompting for specific attribute-demographic combinations.
"""
import argparse
import base64
import json
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anthropic
import requests
import vertexai
import vertexai.preview.generative_models as generative_models
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from vertexai.generative_models import GenerativeModel, Image, Part

# Add utils to path for few_shot_config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from few_shot_config import GPT4O_FEW_SHOT_PAIRS, CLAUDE_SKIN_TONE_FEW_SHOT_DEMOS

load_dotenv()


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def parse_json_response(data_str: str):
    """Parse JSON from model response, handling markdown code blocks."""
    data_str = data_str.strip("'")
    data_str = data_str.replace("```json\n", "", 1) if "```json\n" in data_str else data_str
    data_str = data_str.rsplit("\n```", 1)[0]
    data_str = data_str.replace("'", '"')
    return json.loads(data_str)


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_image_standard_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def make_prompt(attributes: List[str]) -> str:
    """Build the system prompt for attribute detection."""
    api_head_prompt1 = "Assess the two faces in the image for the following attributes:"
    api_head_prompt2 = "Your output should only consist of a JSON that contains the attributes."
    with_skin_tone_head_prompt = "For all attributes except 'face_with_lighter_skin_tone', the output should be a list of two Yes/No responses."
    non_skin_prompt = "The first Yes/No of the list should correspond to the left face and the second Yes/No should correspond to the right face."
    with_skin_prompt = "For face_with_lighter_skin_tone, the output should be either 'Right face', 'No significant difference' or 'Left face'."

    example_with_skin_tone = """
An example output would be:
{
    "<attribute1>":["Yes","No"],
    "<attribute2>":["Yes","Yes"],
    ...
    "face_with_lighter_skin_tone":"Right face"
}
"""

    final_prompt = api_head_prompt1 + "\n"
    for attr in attributes:
        if attr not in ["light_colored_skin_tone", "dark_colored_skin_tone"]:
            final_prompt += f"{attr}\n"

    final_prompt += "face_with_lighter_skin_tone\n"
    final_prompt += api_head_prompt2
    final_prompt += with_skin_tone_head_prompt + "\n"
    final_prompt += non_skin_prompt + "\n"
    final_prompt += with_skin_prompt + "\n"
    final_prompt += example_with_skin_tone + "\n"

    return final_prompt.strip()


# =============================================================================
# Claude API
# =============================================================================
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
def call_claude(image_path: str, prompt: str) -> dict:
    """Call Claude API for attribute detection."""
    client = anthropic.Anthropic()
    image_data = encode_image_standard_base64(image_path)

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    return parse_json_response(message.content[0].text)


@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def call_claude_skin_tone_few_shot(image_path: str, few_shot_paths: List[str]) -> str:
    """Call Claude API for skin tone detection with few-shot examples."""
    client = anthropic.Anthropic()
    image_data = encode_image_standard_base64(image_path)
    few_shot_data = [encode_image_standard_base64(p) for p in few_shot_paths]

    prompt = 'Which of the two images has lighter skin tone? (begin your answer with "Right Face", "Left Face" or "No significant difference")'

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
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": few_shot_data[0]}},
                    {"type": "text", "text": "Following is an example where the left face has lighter skin tone."},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": few_shot_data[1]}},
                    {"type": "text", "text": "Now, an example where there is no significant difference in skin tone."},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": few_shot_data[2]}},
                    {"type": "text", "text": f"{prompt}\n Make sure to use the examples for reference and answer accordingly."},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                ],
            }
        ],
    )

    result = message.content[0].text.strip().lower()
    if result.startswith("right"):
        return "Right Face"
    if result.startswith("left"):
        return "Left Face"
    if result.startswith("no"):
        return "No significant difference"
    return "Error"


# =============================================================================
# GPT-4o API
# =============================================================================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_gpt4o(image_path: str, prompt: str, api_key: str) -> dict:
    """Call GPT-4o API for attribute detection."""
    base64_image = encode_image_base64(image_path)
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
    content = response.json()["choices"][0]["message"]["content"]
    return parse_json_response(content)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_gpt4o_few_shot(image_path: str, attribute: str, few_shot_paths: List[str], api_key: str) -> str:
    """Call GPT-4o API for single attribute with few-shot examples. Returns Yes/No."""
    base64_image = encode_image_base64(image_path)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    prompt = f"Does this image have {attribute}? (begin your answer with Yes or No)"

    payload = {
        "model": "gpt-4o-2024-05-13",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"You need to analyze face images for presence of {attribute} and respond with yes or no. Below are some references."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"This image has {attribute}."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(few_shot_paths[0])}"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"This image does not have {attribute}."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(few_shot_paths[1])}"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"This image has {attribute}."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(few_shot_paths[2])}"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"This image does not have {attribute}."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(few_shot_paths[3])}"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}\n Make sure to use the examples for reference and answer accordingly."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            },
        ],
        "max_tokens": 1000,
        "temperature": 0.1,
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    content = response.json()["choices"][0]["message"]["content"].strip()

    if content.lower().startswith("yes"):
        return "Yes"
    if content.lower().startswith("no"):
        return "No"
    return "Error"


# =============================================================================
# Gemini API
# =============================================================================
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def call_gemini(image_path: str, prompt: str, vertex_project: str, vertex_location: str) -> dict:
    """Call Gemini API for attribute detection."""
    vertexai.init(project=vertex_project, location=vertex_location)

    generation_config = {"max_output_tokens": 8192, "temperature": 0.1, "top_p": 0.55}

    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    model = GenerativeModel("gemini-1.5-pro-001", system_instruction=[prompt])
    image_part = Part.from_image(Image.load_from_file(image_path))
    response = model.generate_content(
        [image_part, "Annotate the images for the different attributes"],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=False,
    )

    return parse_json_response(response.candidates[0].content.parts[0].text.strip())


# =============================================================================
# Unified processing
# =============================================================================
@dataclass
class APIConfig:
    openai_api_key: str
    vertex_project: str
    vertex_location: str
    use_claude: bool = True
    use_gpt4o: bool = True
    use_gemini: bool = True
    rate_limit_sleep: float = 2.0
    # Few-shot config
    attribute: str = ""
    demographic: str = ""
    gpt4o_few_shot_paths: Optional[List[str]] = None
    claude_skin_tone_few_shot_paths: Optional[List[str]] = None


def process_single_image(
    image_path: str,
    prompt: str,
    config: APIConfig,
) -> Tuple[str, Dict]:
    """Process a single image with all enabled APIs and return combined response."""
    combined = {}
    errors = []

    time.sleep(config.rate_limit_sleep)

    # Check if this (attribute, demographic) pair needs few-shot for GPT-4o
    needs_gpt4o_few_shot = (config.attribute, config.demographic) in GPT4O_FEW_SHOT_PAIRS
    # Check if this demographic needs Claude skin tone few-shot
    needs_claude_skin_few_shot = config.demographic in CLAUDE_SKIN_TONE_FEW_SHOT_DEMOS

    if config.use_claude:
        try:
            claude_resp = call_claude(image_path, prompt)
            # If skin tone few-shot is needed, override skin tone result
            if needs_claude_skin_few_shot and config.claude_skin_tone_few_shot_paths:
                try:
                    skin_tone_result = call_claude_skin_tone_few_shot(
                        image_path, config.claude_skin_tone_few_shot_paths
                    )
                    claude_resp["face_with_lighter_skin_tone"] = skin_tone_result
                except Exception as e:
                    errors.append(f"claude_skin_few_shot:{e}")
            combined["claude"] = claude_resp
        except Exception as e:
            errors.append(f"claude:{e}")
            combined["claude"] = None

    if config.use_gpt4o:
        try:
            gpt_resp = call_gpt4o(image_path, prompt, config.openai_api_key)
            # If few-shot is needed for this attribute, override that attribute's result
            if needs_gpt4o_few_shot and config.gpt4o_few_shot_paths:
                try:
                    few_shot_result = call_gpt4o_few_shot(
                        image_path, config.attribute, config.gpt4o_few_shot_paths, config.openai_api_key
                    )
                    # The few-shot returns a single Yes/No, need to format as [Yes/No, Yes/No] for both faces
                    # Since this is for concatenated images, we actually need the transformed image path
                    # For now, store the few-shot result separately
                    gpt_resp[f"{config.attribute}_few_shot"] = few_shot_result
                except Exception as e:
                    errors.append(f"gpt4o_few_shot:{e}")
            combined["gpt4o"] = gpt_resp
        except Exception as e:
            errors.append(f"gpt4o:{e}")
            combined["gpt4o"] = None

    if config.use_gemini:
        try:
            gemini_resp = call_gemini(image_path, prompt, config.vertex_project, config.vertex_location)
            combined["gemini"] = gemini_resp
        except Exception as e:
            errors.append(f"gemini:{e}")
            combined["gemini"] = None

    if errors:
        combined["_errors"] = errors

    return image_path, combined


def process_image_wrapper(args):
    """Wrapper for multiprocessing."""
    image_path, prompt, config_dict = args
    # Handle Optional fields that might be None
    gpt4o_few_shot = config_dict.pop("gpt4o_few_shot_paths", None)
    claude_skin_few_shot = config_dict.pop("claude_skin_tone_few_shot_paths", None)
    config = APIConfig(
        **config_dict,
        gpt4o_few_shot_paths=gpt4o_few_shot,
        claude_skin_tone_few_shot_paths=claude_skin_few_shot,
    )
    return process_single_image(image_path, prompt, config)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified attribute detector using Claude, GPT-4o, and Gemini")
    parser.add_argument("--paths_file", type=str, required=True, help="JSON: attribute -> demo -> [image_paths]")
    parser.add_argument("--attributes", type=str, required=True, help="Comma-separated list of attributes to detect")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON path")
    parser.add_argument("--vertex_project", type=str, required=True, help="GCP project for Vertex AI (Gemini)")
    parser.add_argument("--vertex_location", type=str, default="us-central1")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_processes", type=int, default=5)
    parser.add_argument("--rate_limit_sleep", type=float, default=2.0, help="Sleep between API calls (seconds)")
    parser.add_argument("--skip_claude", action="store_true", help="Skip Claude API calls")
    parser.add_argument("--skip_gpt4o", action="store_true", help="Skip GPT-4o API calls")
    parser.add_argument("--skip_gemini", action="store_true", help="Skip Gemini API calls")
    # Few-shot arguments
    parser.add_argument(
        "--gpt4o_few_shot_json",
        type=str,
        default=None,
        help="JSON: attribute -> demo -> [4 few-shot image paths] for GPT-4o",
    )
    parser.add_argument(
        "--claude_skin_tone_few_shot_json",
        type=str,
        default=None,
        help="JSON: demo -> [3 few-shot image paths] for Claude skin tone",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not args.skip_gpt4o and not openai_api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")

    attributes = [a.strip() for a in args.attributes.split(",")]
    prompt = make_prompt(attributes)

    attr_demo_wise_paths = load_json(args.paths_file)

    # Load few-shot paths if provided
    gpt4o_few_shot_all = load_json(args.gpt4o_few_shot_json) if args.gpt4o_few_shot_json else {}
    claude_skin_few_shot_all = load_json(args.claude_skin_tone_few_shot_json) if args.claude_skin_tone_few_shot_json else {}

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    if os.path.exists(args.output_file):
        results = load_json(args.output_file)
    else:
        results = {}

    for attr in tqdm(attr_demo_wise_paths, desc="Attributes"):
        for demo in tqdm(attr_demo_wise_paths[attr], desc=f"Demographics ({attr})", leave=False):
            image_paths = attr_demo_wise_paths[attr][demo]
            image_paths = [p for p in image_paths if p not in results]
            image_paths = [p for p in image_paths if os.path.exists(p)]

            if not image_paths:
                continue

            # Get few-shot paths for this attribute/demo if available
            gpt4o_few_shot_paths = None
            if (attr, demo) in GPT4O_FEW_SHOT_PAIRS and attr in gpt4o_few_shot_all:
                gpt4o_few_shot_paths = gpt4o_few_shot_all.get(attr, {}).get(demo)

            claude_skin_few_shot_paths = None
            if demo in CLAUDE_SKIN_TONE_FEW_SHOT_DEMOS:
                claude_skin_few_shot_paths = claude_skin_few_shot_all.get(demo)

            config_dict = {
                "openai_api_key": openai_api_key,
                "vertex_project": args.vertex_project,
                "vertex_location": args.vertex_location,
                "use_claude": not args.skip_claude,
                "use_gpt4o": not args.skip_gpt4o,
                "use_gemini": not args.skip_gemini,
                "rate_limit_sleep": args.rate_limit_sleep,
                "attribute": attr,
                "demographic": demo,
                "gpt4o_few_shot_paths": gpt4o_few_shot_paths,
                "claude_skin_tone_few_shot_paths": claude_skin_few_shot_paths,
            }

            for i in tqdm(range(0, len(image_paths), args.batch_size), desc=f"Batches ({attr}/{demo})", leave=False):
                batch = image_paths[i : i + args.batch_size]
                batch_args = [(img_path, prompt, config_dict) for img_path in batch]

                with Pool(args.num_processes) as pool:
                    batch_results = pool.map(process_image_wrapper, batch_args)

                for img_path, response in batch_results:
                    results[img_path] = response

                save_json(results, args.output_file)

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
