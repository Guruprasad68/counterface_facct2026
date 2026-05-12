"""
Handles different classes for facenet and arcface embedding generation
FaceNet-Pytorch https://github.com/timesler/facenet-pytorch model for calculating the facenet embedding
David Sandberg's MTCNN implementation used for MTCNN https://github.com/davidsandberg/facenet
"""
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision.transforms as transforms
from deepface import DeepFace
from PIL import Image
import PIL
import numpy as np
from helper import fixed_image_standardization


class FaceNetEmbedding:
    """
    Class for calculating facenet embedding from facenet-pytorch. MTCNN is also using the same repo
    """
    def __init__(self, image_size=512, margin=0, pretrained='vggface2'):
        self.mtcnn = MTCNN(image_size=image_size, margin=margin)
        self.resnet = InceptionResnetV1(pretrained=pretrained).eval()
        self.transform = transforms.ToTensor()
        
    def get_embedding(self, image_pil: PIL.Image, use_mtcnn=True):
        if use_mtcnn:
            img_cropped = self.mtcnn(image_pil)
            if img_cropped is not None:
                return self.resnet(img_cropped.unsqueeze(0))
        return self.resnet(self.transform(image_pil).unsqueeze(0))


class FaceNetHarrisonApproach:
    """
    Class for calculating facenet embedding from facenet-pytorch using MTCNN from David Sandberg
    """
    def __init__(self, pretrained='vggface2'):
        self.resnet = InceptionResnetV1(pretrained=pretrained).eval()
        self.trans = transforms.Compose([
            transforms.Resize(160),
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization
        ])

    def get_embedding(self, image_pil: PIL.Image):
        trans_image_pil = self.trans(image_pil)

        return self.resnet(trans_image_pil.unsqueeze(0))


def arcface_embedding(img_path: str,
                      model_name="ArcFace",
                      detector_backend='skip'):
    embedding = DeepFace.represent(img_path=img_path,
                                   model_name=model_name,
                                   detector_backend=detector_backend
                                   )
    return embedding[0]['embedding']
