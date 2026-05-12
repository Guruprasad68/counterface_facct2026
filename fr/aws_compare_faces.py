import requests
import argparse
import os
import json
from dotenv import load_dotenv
from tqdm import tqdm
import time
import boto3
from functools import partial
import multiprocessing

load_dotenv()


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


# These will be initialized in main() after parsing args
rekognition_client = None
s3_bucket_name = None


def aws_compare(path1,path2):
    response = rekognition_client.compare_faces(
    SourceImage={
        'S3Object': {
            'Bucket': s3_bucket_name,
            'Name': path1
        }
    },
    TargetImage={
        'S3Object': {
            'Bucket': s3_bucket_name,
            'Name': path2
        }
    },SimilarityThreshold=0
)
    return response


def process_comparison(p, concat_face_dir_name):
    p1, p2 = p
    # p1_concate_face_dir_name= p1.split('/')[-6]
    # p2_concate_face_dir_name= p2.split('/')[-6]

    # p1_attr, p1_demo, p1_name, p1_image_num,p1_with_or_wo = p1.split('/')[-5], p1.split('/')[-4], p1.split('/')[-3], p1.split('/')[-2], p1.split('/')[-1]
    # p2_attr, p2_demo, p2_name, p2_image_num,p2_with_or_wo = p2.split('/')[-5], p2.split('/')[-4], p2.split('/')[-3], p2.split('/')[-2], p2.split('/')[-1]

    # p1_url = f"{p1_concate_face_dir_name}/{p1_attr}/{p1_demo}/{p1_name}/{p1_image_num}/{p1_with_or_wo}"
    # p2_url = f"{p2_concate_face_dir_name}/{p2_attr}/{p2_demo}/{p2_name}/{p2_image_num}/{p2_with_or_wo}"

    p1_parts = p1.split('/')
    p2_parts = p2.split('/')

    p1_url = '/'.join(p1_parts[-6:])
    p2_url = '/'.join(p2_parts[-6:])

    try:
        response = aws_compare(p1_url, p2_url)
        sim = response['FaceMatches'][0]['Similarity'] if 'FaceMatches' in response and response['FaceMatches'] else 'Error'
    except Exception as e:
        response = str(e)
        sim = 'Error'
    time.sleep(1)
    
    return (f"{p1}|{p2}", response, sim)

def parse_args():
    parser = argparse.ArgumentParser(description='AWS Rekognition API for comparing images')
    parser.add_argument('--fr_comparison_paths_file', type=str, required=True, help='Path to JSON file with comparison pairs')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--s3_bucket', type=str, required=True, help='S3 bucket name containing the images')
    parser.add_argument('--aws_region', type=str, default='us-east-2', help='AWS region')
    return parser.parse_args()

def main():
    global rekognition_client, s3_bucket_name
    
    args = parse_args()
    fr_comparison_paths_file = args.fr_comparison_paths_file
    output_dir = args.output_dir
    s3_bucket_name = args.s3_bucket
    
    session = boto3.Session(
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        region_name=args.aws_region
    )
    rekognition_client = session.client('rekognition')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if the response_dict and similarity_dict already exist
    if os.path.exists(f'{output_dir}/aws_compare_faces_response.json'):
        response_dict = load_json(f'{output_dir}/aws_compare_faces_response.json')
        similarity_dict = load_json(f'{output_dir}/aws_compare_faces_similarity.json')
    else:
        response_dict = {}
        similarity_dict = {}

    fr_comparisons = load_json(fr_comparison_paths_file)

    comparisons_to_process = []
    for attr in fr_comparisons:
        for demo in fr_comparisons[attr]:
            comparisons_to_process += [(p1, p2) for p1, p2 in fr_comparisons[attr][demo] if f"{p1}|{p2}" not in response_dict]
    

    # Define the worker function with partial to include additional arguments
    worker = partial(process_comparison, concat_face_dir_name=None)

    # Use multiprocessing to process the comparisons
    with multiprocessing.Pool(processes=25) as pool:
        results = list(tqdm(pool.imap(worker, comparisons_to_process), total=len(comparisons_to_process)))

    # Update the response and similarity dictionaries with results
    for key, response, sim in results:
        response_dict[key] = response
        similarity_dict[key] = sim

    # Save updated dictionaries
    save_json(response_dict, f'{output_dir}/aws_compare_faces_response.json')
    save_json(similarity_dict, f'{output_dir}/aws_compare_faces_similarity.json')
        
if __name__ == '__main__':
    main()