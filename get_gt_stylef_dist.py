import torch
from tqdm import tqdm

from styleswin.generator import StyleSwinGenerator


@torch.no_grad()
def get_styleswin256_style_dist(num_samples, device:str, batch_size:int,\
                                        weights_path:str, gt_stylef_export_path:str,):
     generator = StyleSwinGenerator(size=256, style_dim=512, n_mlp=8).eval()
     generator.load_state_dict(torch.load(weights_path)['g'])
     generator = generator.to(device)

     gt_styles = None
     pbar = tqdm(range(0, num_samples, batch_size), desc="Getting style latents")
     for i in range(0, num_samples, batch_size):
          noise = torch.randn((batch_size, 512), device=device)
          style_latents = generator.get_style_latents(noise)
          if gt_styles is None:
               gt_styles = style_latents.detach().cpu()
          else:
               gt_styles = torch.concat((gt_styles, style_latents.detach().cpu()), dim=0)
          pbar.update()
     
     gt_mean_style = torch.mean(gt_styles, dim=0)
     gt_std_style = torch.std(gt_styles, dim=0)

     gt_stylef_dict = {"mean": gt_mean_style,
                      "std": gt_std_style}
     torch.save(gt_stylef_dict, gt_stylef_export_path)


if __name__ == "__main__":
     num_samples = 100000
     device = 'cpu'

     styleswin_weights_path = 'data/CelebAHQ_256.pt'
     gt_stylef_export_path = 'data/gt_stylef_dist.pth'

     get_styleswin256_style_dist(num_samples, device, batch_size=128,\
                                   gt_stylef_export_path=gt_stylef_export_path,\
                                   weights_path=styleswin_weights_path)
