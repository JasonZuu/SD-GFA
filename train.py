import argparse
import torch
import wandb

from configs.algo_config import SDGFAConfig
from configs.dataset_config import DatasetConfig
from dataset.datasets import ImageTripletDataset
from models.sdgfa import StyleDiverseGFA
from run_fn.train_sdgfa import sdgfa_train_fn
from utils.misc import set_seed, load_gt_stylef_dist


def get_args():
    parser = argparse.ArgumentParser(description='Train the SD-GFA on CelebA-HQ dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--entity', type=str, default="mingchengzhu250",
                        help='wandb entity')
    parser.add_argument('--project', type=str, default="FaceAnonymization",
                        help='wandb project name')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help='device to use for training')
    parser.add_argument("--use_best_hparams", action="store_true", help="whether to use best hparams")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--img_size", type=int, default=256, help="image size")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    
    # initialize wandb
    run = wandb.init(entity=args.entity, project=args.project, name="SDGFA_train")
    
    # load config
    dataset_config = DatasetConfig()
    dataset_config.img_size = args.img_size

    algo_config = SDGFAConfig()
    algo_config.device = args.device

    # load dataset
    train_dataset = ImageTripletDataset(dataset_config, dataset_type="train")
    val_dataset = ImageTripletDataset(dataset_config, dataset_type="val")
    
    # load model
    model = StyleDiverseGFA(img_size=args.img_size, styleswin_weights_path=algo_config.styleswin_weights_path)
    
    # load gt_stylef_dist
    gt_stylef_dist = load_gt_stylef_dist(algo_config.gt_stylef_dist_path, device=algo_config.device)

    # train the model
    best_metrics = sdgfa_train_fn(model=model,
                                  train_dataset=train_dataset,
                                  val_dataset=val_dataset,
                                  gt_stylef_dist=gt_stylef_dist,
                                  algo_config=algo_config,
                                  run=run)
    wandb.finish()
