import torch
from torchvision.utils import save_image
from pathlib import Path
from tqdm import tqdm
import os            
from torch.utils.data import DataLoader                                                                                                                   

from configs.algo_config import GenerationConfig


@torch.no_grad()
def genenerate_fn(model, test_dataset:str, algo_config:GenerationConfig):
    Path(algo_config.log_dir).mkdir(exist_ok=True, parents=True)

    loader = DataLoader(test_dataset, batch_size=algo_config.batch_size, shuffle=False)
    
    pbar = tqdm(total=len(loader), desc=f'SD-GFA Generation', unit='batch')
    img_count = 0
    for i_batch, (img, key) in enumerate(loader):
        img, key = img.to(algo_config.device), key.to(algo_config.device)
        img = model(img, key)
        
        # export the generated images
        for i_img in range(img.shape[0]):
            out_img_path = os.path.join(algo_config.log_dir, f'out_{img_count}.png')
            save_image(img[i_img],out_img_path,
                        nrow=1, padding=0)
            img_count += 1
            if img_count >= algo_config.num_gen_img:
                break

        pbar.update()
        if img_count >= algo_config.num_gen_img:
            break
        
