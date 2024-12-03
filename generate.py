import argparse
import torch
import wandb

from configs.algo_config import GenerationConfig
from configs.dataset_config import DatasetConfig
from dataset.datasets import GenerationDataset
from models.sdgfa import StyleDiverseGFA
from run_fn.generate import genenerate_fn


def get_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate images using SDGFA model.")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the SDGFA model weights.")
    parser.add_argument("--log_dir", type=str, default="log/sdgfa_gen", help="Directory to save generated images.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for generation.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for generation (e.g., 'cuda', 'cpu').")
    parser.add_argument("--img_size", type=int, default=256, help="Image size for generation.")
    parser.add_argument("--num_gen_img", type=int, default=10, help="Number of images to generate.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # Load configs
    algo_config = GenerationConfig(
        log_dir=args.log_dir,
        batch_size=args.batch_size
    )
    algo_config.device = args.device
    algo_config.num_gen_img = args.num_gen_img
    dataset_config = DatasetConfig()

    # Load the model
    model = StyleDiverseGFA(img_size=args.img_size)
    model_state_dict = torch.load(args.weights_path, map_location=algo_config.device, weights_only=True)
    model.load_state_dict(model_state_dict)
    model = model.to(algo_config.device)
    model.eval()

    # Load the test dataset
    test_dataset = GenerationDataset(dataset_config)
    genenerate_fn(model, test_dataset, algo_config)
