import requests
import torch
from PIL import Image

from einops import repeat
from einops.layers.torch import Rearrange
from torch import nn
from torchvision import transforms

from model.modalities import SimpleTransformer


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class VisionHead(nn.Module):  # its just ViT
    # https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
    def __init__(self, out_proj_dim: int):
        super().__init__()
        image_size = 224
        patch_size = 16
        hidden_size = 768
        num_layers = 12
        num_heads = 12
        channels = 3
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, ('Image dimensions must be '
                                                                                     'divisible by the patch size.')

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, hidden_size))

        self.transformer = SimpleTransformer(hidden_size, num_heads, num_layers)
        self.proj = nn.Linear(hidden_size, out_proj_dim)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        x += self.pos_embedding

        x = self.transformer(x)
        return self.proj(x)


if __name__ == '__main__':
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    image = transforms.Resize((224, 224))(image)
    image = transforms.ToTensor()(image).unsqueeze(0).cuda().bfloat16()
    vh = VisionHead(768).cuda().bfloat16()
    out = vh(image)
    print(out.shape)
