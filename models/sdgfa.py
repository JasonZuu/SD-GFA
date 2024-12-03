import torch.nn.functional as F
import torch
import torch.nn as nn
import os
from facenet_pytorch import InceptionResnetV1

from styleswin import StyleSwinGenerator


class EqualizedControlModule(nn.Module):
     def __init__(self, 
                  key_dim, 
                  hidden_dim=512,
                  output_dim=512):
          """
          params:
               key_dim: the dimenstion of the key
               hidden_dim: the hidden dimenstion of the key encoder
               output_dim: the output dimenstion of the key encoder
          """
          super().__init__()
          self.key_dim=key_dim
          self.key_encoder = nn.Sequential(nn.Linear(self.key_dim, hidden_dim),
                                             nn.LeakyReLU(0.2),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.LeakyReLU(0.2),)
          self.condition_encoder = nn.Sequential(nn.Linear(1024, hidden_dim),
                                                  nn.LeakyReLU(0.2),
                                                  nn.Linear(hidden_dim, output_dim),
                                                  nn.LeakyReLU(0.2),)

     def forward(self, img_emb, key):
          key_emb = self.key_encoder(key)
          condition_emb = self.condition_encoder(torch.concat([img_emb, key_emb], dim=1))
          return condition_emb


class StyleDiverseGFA(nn.Module):
     def __init__(self, 
                  img_size:int,
                  styleswin_weights_path:str=None,
                  key_dim=8,
                  style_dim=512):
          """
          params:
            key_dim: the dimenstion of the key
            image_size: the size of the input image 
          """
          super().__init__()
          self.key_dim=key_dim
          self.img_encoder = InceptionResnetV1(pretrained='vggface2') # Output: (B, 512)
          self.control_module = EqualizedControlModule(key_dim=key_dim, output_dim=style_dim)

          # Load the StyleSwinGenerator weights
          self.decoder = StyleSwinGenerator(size=img_size, style_dim=style_dim, n_mlp=8)
          if styleswin_weights_path is not None:
               decoder_state_dict = torch.load(styleswin_weights_path)
               self.decoder.load_state_dict(decoder_state_dict["g"])

          self.img_encoder.eval()
          self.decoder.eval()
          self._freeze_params()

          self.control_module.train()

     def get_style_latents(self, img, key):
          img_emb = self.encoder(img) # Output: (B, 512)
          style_latents = self.control_module(img_emb, key) # Output: (B, 512)
          return style_latents
     

     def forward_with_hidden(self, img, key):
          img_emb = self.img_encoder(img) # Output: (B, 512)
          style_latents = self.control_module(img_emb, key) # Output: (B, 512)
          
          output = self.decoder.forward(style_latents, reverse_norm=True)
          info = {'style_latents': style_latents}
          return output, info
     
     def forward(self, img, key):
          img, info = self.forward_with_hidden(img, key)
          return img  
     
     def load_control_module_state_dict(self, control_module_state_dict):
          self.control_module.load_state_dict(control_module_state_dict)
     
     def load_decoder_state_dict(self, decoder_state_dict):
          self.decoder.load_state_dict(decoder_state_dict["g"])

     def _freeze_params(self):
          for para in self.img_encoder.parameters():
               para.requires_grad_(False)
          for para in self.decoder.parameters():
               para.requires_grad_(False)


def test_SDGFA():
     key_dim = 8
     img_size = 256
     batch_size = 6
     model = StyleDiverseGFA(key_dim=key_dim, img_size=img_size)
     img = torch.randn(batch_size, 3, img_size, img_size)
     key = torch.randn(batch_size, key_dim)
     output = model(img, key)
     print(output.shape)


if __name__ == "__main__":
    test_SDGFA()
    