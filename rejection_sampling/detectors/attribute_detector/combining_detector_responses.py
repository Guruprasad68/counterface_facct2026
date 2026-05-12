"""
Takes the paths attribute-demo wise paths file, the GPT, claude and gemini responses and then combines them into a single file which is
separated by attribute-demographic combination.
"""
import argparse
import json
import logging
import os
from pathlib import Path


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


logging.basicConfig(
    filename="combining_responses.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths_file", type=str, required=True)
    parser.add_argument("--gpt_responses_file", type=str, required=True)
    parser.add_argument("--claude_responses_file", type=str, required=True)
    parser.add_argument("--gemini_responses_file", type=str, required=True)
    parser.add_argument("--combined_response_file", type=str, required=True)
    parser.add_argument("--reassign_dict", type=str, required=True, help="Primary demographic reassignment mapping JSON")
    parser.add_argument(
        "--reassign_dict_extra",
        type=str,
        default=None,
        help="Optional second JSON merged into reassignment mapping",
    )
    parser.add_argument("--run_id", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    paths_file = args.paths_file
    gpt_responses_file = args.gpt_responses_file
    claude_responses_file = args.claude_responses_file
    gemini_responses_file = args.gemini_responses_file
    output_file_path = args.combined_response_file

    reassign_dict = load_json(args.reassign_dict)
    if args.reassign_dict_extra:
        reassign_dict.update(load_json(args.reassign_dict_extra))

    if not os.path.exists(os.path.dirname(output_file_path)):
        Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)

    gpt_responses = load_json(gpt_responses_file)
    claude_responses = load_json(claude_responses_file)
    gemini_responses = load_json(gemini_responses_file)
    attr_demo_wise_combined_responses = {}
    attribute_demo_wise_image_paths = load_json(paths_file)

    for attr in attribute_demo_wise_image_paths:
        attr_demo_wise_combined_responses[attr] = {}
        for demo in attribute_demo_wise_image_paths[attr]:
            print(f"Processing {attr}-{demo}")
            image_paths = attribute_demo_wise_image_paths[attr][demo]

            combined_responses = {}
            for path in image_paths:
                if path in gpt_responses:
                    gpt_response = gpt_responses[path]
                    if not isinstance(gpt_response, dict):
                        gpt_response = {}
                        logging.info(f"Path {path}: Improper GPT response")
                else:
                    logging.info(f"Path {path} not found in GPT responses")
                    gpt_response = {}

                if path in gemini_responses:
                    gemini_response = gemini_responses[path]
                    if not isinstance(gemini_response, dict):
                        gemini_response = {}
                        logging.info(f"Path {path}: Improper Gemini response")
                else:
                    logging.info(f"Path {path} not found in Gemini responses")
                    gemini_response = {}

                if path in claude_responses:
                    claude_response = claude_responses[path]
                    if not isinstance(claude_response, dict):
                        claude_response = {}
                        logging.info(f"Path {path}: Improper Claude response")
                else:
                    logging.info(f"Path {path} not found in Claude responses")
                    claude_response = {}

                combined_response = {}
                combined_response.update(gpt_response)
                combined_response.update(claude_response)
                combined_response.update(gemini_response)

                combined_responses[path] = combined_response

            attr_demo_wise_combined_responses[attr][demo] = combined_responses

    for attr in attr_demo_wise_combined_responses:
        for demo in list(attr_demo_wise_combined_responses[attr].keys()):
            paths_to_remove = []
            for path in list(attr_demo_wise_combined_responses[attr][demo].keys()):
                demo_path = path.split("/")[-3]
                name_path = path.split("/")[-2]

                if f"{demo_path}/{name_path}" in reassign_dict:
                    if reassign_dict[f"{demo_path}/{name_path}"] != "REMOVE":
                        new_demo = reassign_dict[f"{demo_path}/{name_path}"]
                        if new_demo not in attr_demo_wise_combined_responses[attr]:
                            attr_demo_wise_combined_responses[attr][new_demo] = {}
                        attr_demo_wise_combined_responses[attr][new_demo][path] = attr_demo_wise_combined_responses[attr][
                            demo
                        ][path]
                        paths_to_remove.append(path)
                    elif reassign_dict[f"{demo_path}/{name_path}"] == "REMOVE":
                        paths_to_remove.append(path)

            for path in paths_to_remove:
                print(f"Removing {path}")
                del attr_demo_wise_combined_responses[attr][demo][path]

    save_json(attr_demo_wise_combined_responses, output_file_path)
    logging.info(f"For run_id {args.run_id} saved the combined responses to {output_file_path}")


if __name__ == "__main__":
    main()
