import requests
import argparse
import os
import json
from dotenv import load_dotenv
from tqdm import tqdm
import time

load_dotenv()


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


API_KEY = os.environ['FACEPP_APIKEY']
API_SECRET = os.environ['FACEPP_APISECRET']

FACEPP_ENDPOINT = 'https://api-us.faceplusplus.com/facepp/v3/compare'

def facepp_compare_images(image_url1, image_url2):
    data = {
        'api_key': API_KEY,
        'api_secret': API_SECRET,
        'image_url1': image_url1,
        'image_url2': image_url2
    }

    # Make the POST request to the Face++ API
    try:
        response = requests.post(FACEPP_ENDPOINT, data=data)
        result = response.json()
    except Exception as e:
        result='Error'

    return result


def parse_args():
    parser = argparse.ArgumentParser(description='Face++ API for comparing images')
    parser.add_argument('--fr_comparison_paths_file', type=str, required=True, help='Path to JSON file with comparison pairs')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--image_base_url', type=str, required=True, help='Base URL for images (e.g., https://your-bucket.s3.region.amazonaws.com)')
    return parser.parse_args()

def main():
    args = parse_args()
    fr_comparison_paths_file = args.fr_comparison_paths_file
    output_dir = args.output_dir
    image_base_url = args.image_base_url.rstrip('/')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if the response_dict and similarity_dict already exist
    if os.path.exists(f'{output_dir}/facepp_compare_faces_response.json'):
        response_dict = load_json(f'{output_dir}/facepp_compare_faces_response.json')
        similarity_dict = load_json(f'{output_dir}/facepp_compare_faces_similarity.json')
    else:
        response_dict = {}
        similarity_dict = {}

    fr_comparisons = load_json(fr_comparison_paths_file)

    comparisons_to_process = []
    for attr in fr_comparisons:
        for demo in fr_comparisons[attr]:
            comparisons_to_process += [(p1, p2) for p1, p2 in fr_comparisons[attr][demo] if f"{p1}|{p2}" not in response_dict]

    for p1, p2 in tqdm(comparisons_to_process):
        key = f"{p1}|{p2}"

        p1_parts = p1.split('/')
        p2_parts = p2.split('/')

        p1_key = '/'.join(p1_parts[-6:])
        p2_key = '/'.join(p2_parts[-6:])

        p1_url = f"{image_base_url}/{p1_key}"
        p2_url = f"{image_base_url}/{p2_key}"

        response = facepp_compare_images(p1_url, p2_url)

        if 'confidence' in response:
            similarity_dict[key] = response['confidence']
        else:
            similarity_dict[key] = 'Error'

        response_dict[key] = response
        save_json(response_dict, f'{output_dir}/facepp_compare_faces_response.json')
        save_json(similarity_dict, f'{output_dir}/facepp_compare_faces_similarity.json')

        time.sleep(1)
        
if __name__ == '__main__':
    main()