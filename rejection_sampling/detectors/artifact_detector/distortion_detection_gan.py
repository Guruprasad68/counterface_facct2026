import sys 
sys.path.append(f'/home/viswanathanr/workspace/face_benchmark')
from helper import save_json, load_json
import os 
import glob 
import random 
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from PIL import Image 
from tqdm import tqdm
import json 

STYLE_CLIP_DIR=f'/mnt/datasets/viswanathanr/semantic_fairness/data/CVPR25/STYLE_CLIP'
UTILS_DIR=f'/mnt/datasets/viswanathanr/semantic_fairness/data/CVPR25/UTILS'
CONCATENATED_FACES_DIR=f'/mnt/datasets/viswanathanr/semantic_fairness/data/CVPR25/STYLE_CLIP/concatenated_faces_all'

def run_molmo(model_id='allenai/Molmo-72B-0924', device=f'cuda:0'): #Adapted from vllm batch inference code
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device.split(':')[-1]}"
    model_name = model_id

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["NCCL_P2P_DISABLE"]="1" 
    llm=LLM(model=model_id, tensor_parallel_size=4,
                    trust_remote_code=True,
                    dtype="bfloat16")
    stop_token_ids = None

    return llm, stop_token_ids


llm, stop_token_ids = run_molmo(f'allenai/Molmo-72B-0924', device=f'cuda:0')
sampling_params = SamplingParams(temperature=0.1,
                                    max_tokens=10,
                                    stop_token_ids=stop_token_ids)
batch_size=30
distortion_assessment_prompt_molmo=f"Carefully examine the two faces in the image.\
    Compare the face on the right and the left. Consider the face from the head including hair. \
    The right face is obtained by editing the left such as changing hairstyle, facial hair, hair-color, skin-tone etc. Does the editing cause \
    any patches of perturbations, blurring, distortions, or other unnatural artifacts?  Answer just 'yes' or 'no'.\
    "

attributes_list=load_json(f'{UTILS_DIR}/cvpr_added_attribute_list.json')

distortion_responses={}
for attr in attributes_list:
    images_all=glob.glob(f'{CONCATENATED_FACES_DIR}/{attr}/**/*.jpg', recursive=True)
    print(f'Processing {attr} with {len(images_all)} images')


    for i in tqdm(range(0,len(images_all),batch_size),desc=f'Checking for distortion'):
            image_paths_batch=images_all[i:i+batch_size]
            inputs_batch=[]
            for img_path in image_paths_batch:
                inputs = {
                    "prompt": distortion_assessment_prompt_molmo,
                    "multi_modal_data": {
                        'image' : Image.open(img_path),
                    },
                }
                inputs_batch.append(inputs)
            outputs = llm.generate(inputs_batch, sampling_params=sampling_params)
            for img_path, output in zip(image_paths_batch, outputs):
                generated_text = output.outputs[0].text
                try:
                    distortion_responses[img_path]=json.loads(generated_text.replace("\n", "").replace(" ", "").replace(",}","}"))
                except:
                    distortion_responses[img_path]=generated_text
                    continue
            save_json(distortion_responses, f'{UTILS_DIR}/distortion_responses_molmo_GAN_images.json')