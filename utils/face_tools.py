from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import transforms
from itertools import combinations
import numpy as np

from utils.tensor_process import reverse_standard_norm, reverse_imagenet_norm, reverse_itnet_norm

_image_size = 160
mtcnn = MTCNN(image_size=_image_size)
facenet = InceptionResnetV1(pretrained="vggface2",
                            device="cuda" if torch.cuda.is_available() else "cpu")
facenet.eval()

to_tensor_trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(_image_size),])
resize_trans = transforms.Resize(_image_size)

blank_image = torch.zeros((3, _image_size, _image_size)).to("cpu")



@torch.no_grad()
def match_face(img1:torch.tensor, img2:torch.Tensor)->bool:
     """
     True means match, otherwise False

     The metric used to measure distance between faces is cosine simiarity.
     If cos(theta) > 0.5, the two faces belong to the same person, and return True
     """
     device = facenet.device
     if img1.shape[0] == 0 or img2.shape[0] == 0:
         return torch.zeros((0,)).cpu()
     
     if img1.__class__ is not torch.Tensor:
          img1 = to_tensor_trans(img1).unsqueeze(0).to(device)
          img2 = to_tensor_trans(img2).unsqueeze(0).to(device)

     features1 = facenet(img1)
     features2 = facenet(img2)
     cos = torch.cosine_similarity(features1, features2, dim=1)

     return (cos>0.5).cpu()


@torch.no_grad()
def inner_cos_score(imgs:torch.Tensor)->bool:
     """
     True means match, otherwise False

     The metric used to measure distance between faces is cosine simiarity.
     If cos(theta) > 0.5, the two faces belong to the same person, and return True
     """
     cos_score_list = []

     if imgs.shape[0] == 0:
         return cos_score_list
         
     features_list = []
     features = facenet(imgs)
     features_list = [f for f in features]
     f_comb = list(combinations(features_list, r=2))

     for f_tuple in f_comb:
          cos = torch.cosine_similarity(f_tuple[0], f_tuple[1], dim=0)
          cos_score_list.append(max(0, cos.cpu().numpy()))

     return cos_score_list


@torch.no_grad()
def inter_cos_score(id1_imgs:torch.Tensor, id2_imgs:torch.Tensor)->bool:
     """
     True means match, otherwise False

     The metric used to measure distance between faces is cosine simiarity.
     If cos(theta) > 0.5, the two faces belong to the same person, and return True
     """
     if id1_imgs.shape[0] == 0 or id2_imgs.shape[0] == 0:
         return []
     id1_features = facenet(id1_imgs)
     id2_features = facenet(id2_imgs)
     cos_scores = torch.cosine_similarity(id1_features, id2_features, dim=1).cpu().numpy()
     return [np.maximum(np.zeros_like(cos_score), cos_score) for cos_score in cos_scores]


def detect_face_mtcnn_idx(images:torch.Tensor, norm_type:str=None):

    if norm_type == 'imagenet':
        images = reverse_imagenet_norm(images)
    elif norm_type == 'standard':
        images = reverse_standard_norm(images)
    elif norm_type == "itnet":
        images = reverse_itnet_norm(images)
    
    images_permuted = images.permute(0, 2, 3, 1).mul(255).add_(0.5).to("cpu", torch.uint8)
    none_face_idx = torch.zeros((images.shape[0],))

    faces = mtcnn(images_permuted)
    
    for idx, face in enumerate(faces):
        if face is None:
            faces[idx] = blank_image
            none_face_idx[idx] =  1

    faces = torch.stack(faces)
    return faces, none_face_idx


def detect_face_mtcnn(images:torch.Tensor, norm_type:str=None):
    assert norm_type in [None, "imagenet", 'standard']

    if norm_type == 'imagenet':
        images = reverse_imagenet_norm(images)
    elif norm_type == 'standard':
        images = reverse_standard_norm(images)

    images_permuted = images.permute(0, 2, 3, 1).mul(255).add_(0.5).to("cpu", torch.uint8)

    faces = mtcnn(images_permuted)
    for idx, face in enumerate(faces):
        if face is None:
            faces[idx] = resize_trans(images[idx]).to("cpu")

    return faces
