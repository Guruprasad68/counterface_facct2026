"""File contains different helper functions in storing, retrieving embeddings, scores, tables etc."""

import pickle
from PIL import Image
from typing import List
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch

def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as fin:
        pickle_obj = pickle.load(fin)

    return pickle_obj


def dump_to_pickle(obj_to_dump, pickle_path):
    with open(pickle_path, 'wb') as fout:
        pickle.dump(obj_to_dump, fout)


def read_image(image_path):
    return Image.open(image_path).convert('RGB').resize((512, 512))


def read_images_batch(image_paths: List[str]):
    images = []
    to_tensor = transforms.ToTensor()
    for img_path in image_paths:
        images.append(to_tensor(Image.open(img_path).convert('RGB').resize((512, 512))))
    return images

def cosine_similarity(embed_1, embed_2,dim=1):
    return F.cosine_similarity(embed_1, embed_2,dim=dim)

def l1_distance(embed_1,embed_2):
    return torch.cdist(embed_1, embed_2, p=1).item()

def l2_distance(embed_1,embed_2):
    return torch.cdist(embed_1, embed_2, p=2).item()

def tensor_to_pil_image(tensor):
    """
    Converts a pytorch tensor to PIL Image
    :param tensor: Pytorch tensor
    :return: PIL Image
    """
    tensor = tensor.cpu().float()
    tensor = (tensor + 1) / 2
    numpy_array = (tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()
    pil_image = Image.fromarray(numpy_array)

    return pil_image


def normalize_arcface_embeddings(embed: List):
    embed_tensor = torch.Tensor(embed)

    embed_norm = torch.norm(embed_tensor, p=2)

    return embed_tensor / embed_norm

def fixed_image_standardization(image_tensor):
    """
    Image tensor normalization code
    Copied from https://github.com/timesler/facenet-pytorch/blob/b5aaef5e552aa5b29e8f0f6500ed92953fc7f953/models/mtcnn.py#L508
    :param image_tensor: takes image pytorch tensor
    :return: processed_tensor : normalized version of the image tensor
    """
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor