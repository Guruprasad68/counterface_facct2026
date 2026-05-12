"""
Generate counterfactual data by filtering detector responses based on specificity criteria.
Filters out images where unintended attributes changed during the editing process.
"""
import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

DEMOGRAPHICS = [
    "asian_man",
    "asian_woman",
    "black_man",
    "black_woman",
    "white_man",
    "white_woman",
    "indian_man",
    "indian_woman",
]

# 20 edit attributes (from attribute-method mapping)
# Diffusion + SEGA: facemask, sunglasses, mustache, pigtails, scarf, blue_hair, head_band
# GAN + StyleCLIP: bald, dark_colored_skin_tone, light_colored_skin_tone, blond_hair
# Both: buzz_cut, glasses, heavy_makeup, red_hair, shoulder_length_hair, goatee, red_lipstick, smile, thick_beard
EDIT_ATTRIBUTES = [
    "facemask",
    "sunglasses",
    "mustache",
    "pigtails",
    "scarf",
    "blue_hair",
    "head_band",
    "bald",
    "dark_colored_skin_tone",
    "light_colored_skin_tone",
    "blond_hair",
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

# Non-relative attributes (excludes skin tone which is handled separately)
NON_RELATIVE_ATTRIBUTES = [a for a in EDIT_ATTRIBUTES if a not in ["dark_colored_skin_tone", "light_colored_skin_tone"]]


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def split_combined_response_to_src_tgt(combined_response: Dict, attributes: list):
    """Split concatenated image response into source and target responses."""
    src_response = {}
    tgt_response = {}

    for attr in attributes:
        if attr not in combined_response:
            src_response[attr] = "NA"
            tgt_response[attr] = "NA"
            continue
        src_response[attr] = combined_response[attr][0]
        tgt_response[attr] = combined_response[attr][1]

    return src_response, tgt_response


def check_specificity(row, attr_edit, df_confusion, non_relative_attributes):
    """Check if only the intended attribute changed (specificity check)."""
    for attr in non_relative_attributes:
        confusion_val = int(df_confusion[attr][attr_edit])
        if confusion_val == -2:
            continue
        elif confusion_val == 1 and row[f"{attr}_tgt"] == 1:
            continue
        elif confusion_val == -1 and row[f"{attr}_tgt"] == row[f"{attr}_src"]:
            continue
        elif confusion_val == 0 and row[f"{attr}_tgt"] == 0:
            continue
        else:
            return False
    return True


def organize_paths_by_attr_demo(paths: set):
    """Organize paths by attribute and demographic."""
    attribute_demo_wise_paths = {}
    for path in paths:
        attribute_image = path.split("/")[-4]
        demo = path.split("/")[-3]

        if attribute_image not in attribute_demo_wise_paths:
            attribute_demo_wise_paths[attribute_image] = {}
        if demo not in attribute_demo_wise_paths[attribute_image]:
            attribute_demo_wise_paths[attribute_image][demo] = []
        attribute_demo_wise_paths[attribute_image][demo].append(path)

    return attribute_demo_wise_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Generate counterfactual data via rejection sampling")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--detector_responses_json", type=str, required=True, help="Combined detector responses JSON")
    parser.add_argument("--confusion_matrix_csv", type=str, required=True, help="Attribute conditioning matrix CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load confusion matrix for attribute conditioning
    df_confusion = pd.read_csv(args.confusion_matrix_csv)
    df_confusion.set_index("Unnamed: 0", inplace=True)
    df_confusion.index.name = None

    # Load detector responses
    attribute_demo_wise_combined_responses = load_json(args.detector_responses_json)

    # Parse responses into source/target format
    data = []
    for attribute_edit in EDIT_ATTRIBUTES:
        if attribute_edit not in attribute_demo_wise_combined_responses:
            print(f"Warning: {attribute_edit} not found in detector responses")
            continue
        for demo in attribute_demo_wise_combined_responses[attribute_edit]:
            for path in attribute_demo_wise_combined_responses[attribute_edit][demo]:
                response = attribute_demo_wise_combined_responses[attribute_edit][demo][path]
                if not isinstance(response, Dict):
                    print(f"Error in processing {path}")
                    continue

                row = {"transformed_image_path": path}
                row["name"] = path.split("/")[-2]

                source_response, transformed_response = split_combined_response_to_src_tgt(
                    response, NON_RELATIVE_ATTRIBUTES
                )

                for attr in transformed_response:
                    row[f"{attr}_tgt"] = 1 if transformed_response[attr] == "Yes" else 0
                    row[f"{attr}_src"] = 1 if source_response[attr] == "Yes" else 0

                skin_tone_response = response.get("face_with_lighter_skin_tone", "")
                if "right" in skin_tone_response.lower():
                    row["face_with_lighter_skin_tone"] = 1
                elif "left" in skin_tone_response.lower():
                    row["face_with_lighter_skin_tone"] = 0
                elif "no" in skin_tone_response.lower():
                    row["face_with_lighter_skin_tone"] = -1
                else:
                    row["face_with_lighter_skin_tone"] = -1

                row["demographic"] = demo
                row["attribute_edit"] = attribute_edit
                data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(f"{output_dir}/response_file.csv", index=False)

    # Main filtering: check specificity for each attribute edit
    attribute_wise_demo_counts = {}
    all_correct_paths = []
    all_not_selected_paths = []

    for demo in DEMOGRAPHICS:
        attribute_wise_demo_counts[demo] = {}
        df_demo = df[df["demographic"] == demo]

        for attr_edit in EDIT_ATTRIBUTES:
            selected_paths = []
            not_selected_paths = []

            if attr_edit not in ["dark_colored_skin_tone", "light_colored_skin_tone"]:
                df_attr = df_demo[df_demo["attribute_edit"] == attr_edit]
                overall_count = len(df_attr)

                if overall_count == 0:
                    attribute_wise_demo_counts[demo][attr_edit] = "NA"
                    continue

                count = 0
                attr_applied_count = 0

                for _, row in df_attr.iterrows():
                    changed = 0
                    # Skip if attribute already present in source
                    if row[f"{attr_edit}_src"] == 1:
                        not_selected_paths.append(row["transformed_image_path"])
                        overall_count -= 1
                        continue
                    if row[f"{attr_edit}_src"] == 0 and row[f"{attr_edit}_tgt"] == 1:
                        attr_applied_count += 1

                    # Check specificity
                    for attr in NON_RELATIVE_ATTRIBUTES:
                        confusion_val = int(df_confusion[attr][attr_edit])
                        if confusion_val == -2:
                            continue
                        elif confusion_val == 1 and row[f"{attr}_tgt"] == 1:
                            continue
                        elif confusion_val == -1 and row[f"{attr}_tgt"] == row[f"{attr}_src"]:
                            continue
                        elif confusion_val == 0 and row[f"{attr}_tgt"] == 0:
                            continue
                        else:
                            changed = 1
                            not_selected_paths.append(row["transformed_image_path"])
                            break

                    if changed == 0:
                        count += 1
                        selected_paths.append(row["transformed_image_path"])
                        all_correct_paths.append(row["transformed_image_path"])

            else:  # skin tone attributes
                df_attr = df_demo[df_demo["attribute_edit"] == attr_edit]
                overall_count = len(df_attr)

                if overall_count == 0:
                    attribute_wise_demo_counts[demo][attr_edit] = "NA"
                    continue

                count = 0
                attr_applied_count = 0

                for _, row in df_attr.iterrows():
                    changed = 0

                    if attr_edit == "dark_colored_skin_tone":
                        if row["face_with_lighter_skin_tone"] == 0:
                            attr_applied_count += 1
                        else:
                            changed = 1
                            not_selected_paths.append(row["transformed_image_path"])

                    elif attr_edit == "light_colored_skin_tone":
                        if row["face_with_lighter_skin_tone"] == 1:
                            attr_applied_count += 1
                        else:
                            changed = 1
                            not_selected_paths.append(row["transformed_image_path"])

                    # Check specificity for other attributes
                    if changed == 0:
                        for attr in NON_RELATIVE_ATTRIBUTES:
                            confusion_val = int(df_confusion[attr][attr_edit])
                            if confusion_val == -2:
                                continue
                            elif confusion_val == 1 and row[f"{attr}_tgt"] == 1:
                                continue
                            elif confusion_val == -1 and row[f"{attr}_tgt"] == row[f"{attr}_src"]:
                                continue
                            elif confusion_val == 0 and row[f"{attr}_tgt"] == 0:
                                continue
                            else:
                                changed = 1
                                not_selected_paths.append(row["transformed_image_path"])
                                break

                    if changed == 0:
                        count += 1
                        selected_paths.append(row["transformed_image_path"])
                        all_correct_paths.append(row["transformed_image_path"])

            all_not_selected_paths.extend(not_selected_paths)

            if overall_count > 0:
                ratio = count / overall_count
                print(f"{demo} {attr_edit}: {count}/{overall_count} ({ratio:.2%}) applied={attr_applied_count}")
                attribute_wise_demo_counts[demo][attr_edit] = (count, overall_count, ratio, attr_applied_count)

    # Save path lists
    save_json(all_correct_paths, f"{output_dir}/all_correct_paths.json")
    save_json(all_not_selected_paths, f"{output_dir}/all_not_selected_paths.json")
    save_json(attribute_wise_demo_counts, f"{output_dir}/attribute_wise_demo_counts.json")

    # Organize by attribute and demographic
    attribute_demo_wise_correct = organize_paths_by_attr_demo(set(all_correct_paths))
    attribute_demo_wise_incorrect = organize_paths_by_attr_demo(set(all_not_selected_paths))

    # Create summary DataFrames
    df_correct_counts = pd.DataFrame(index=EDIT_ATTRIBUTES, columns=DEMOGRAPHICS)
    df_incorrect_counts = pd.DataFrame(index=EDIT_ATTRIBUTES, columns=DEMOGRAPHICS)

    for attr in attribute_demo_wise_correct:
        for demo in attribute_demo_wise_correct[attr]:
            df_correct_counts.loc[attr, demo] = len(attribute_demo_wise_correct[attr][demo])

    for attr in attribute_demo_wise_incorrect:
        for demo in attribute_demo_wise_incorrect[attr]:
            df_incorrect_counts.loc[attr, demo] = len(attribute_demo_wise_incorrect[attr][demo])

    # Save outputs
    save_json(attribute_demo_wise_correct, f"{output_dir}/attribute_demo_wise_correct_paths.json")
    save_json(attribute_demo_wise_incorrect, f"{output_dir}/attribute_demo_wise_incorrect_paths.json")
    df_correct_counts.to_csv(f"{output_dir}/correct_paths_count.csv")
    df_incorrect_counts.to_csv(f"{output_dir}/incorrect_paths_count.csv")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
