from dataclasses import dataclass
import torch

@dataclass
class SDGFAConfig:
    num_epochs: int = 20
    early_stop_epochs: int = 5
    warmup_epochs: int = 1
    lr_decay_weight: float = 0.9
    lr_decay_step = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    face_model:str = "facenet-vggface2"
    gt_stylef_dist_path = "data/gt_stylef_dist.pth"
    styleswin_weights_path = "data/CelebAHQ_256.pt"
    log_dir = "log/sdgfa"
    batch_size: int = 2
    mps_lower_bound = 0.05

    # optimised parameters
    grad_accum_steps: int = 8 # (2, 4, 8)
    lr: float = 5e-4 # (1e-3, 5e-4, 1e-4)


@dataclass
class GenerationConfig:
    batch_size: int = 10
    img_size = 256
    num_gen_img = 100
    input_norm: str = "imagenet"
    sdgfa_weights_path: str = "log/sdgfa/sdgfa.pth"
    styleswin_weights_path = "data/CelebAHQ_256.pt"
    log_dir: str = "log/generation"
    device = "cuda" if torch.cuda.is_available() else "cpu"
