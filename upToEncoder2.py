

#my version torch summery had issues because some tensors in cuda and some are in cpu so chat gpt changed that.
#refer my code first because thats how I understand

import torch
import torch.nn as nn
from torchinfo import summary

# Determining device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Creating the input tensor on the same device as the model
x = torch.rand(3, 224, 224).unsqueeze(0).to(device)
print(x.shape)

class UpToEncoder(nn.Module):
    def __init__(self, in_chanel=3, embed_dim=768, patch_size=16) -> None:
        super().__init__()

        # Defining the model layers
        self.patcher = nn.Conv2d(in_channels=in_chanel, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2, 3)

    def forward(self, x):
        x = self.patcher(x)
        x = self.flatten(x).permute(0, 2, 1)

        cls_token = nn.Parameter(torch.randn([1, 1, 768], device=device), requires_grad=True)
        x = torch.cat((x, cls_token.expand(-1, x.size(1), -1)), 1)

        pos_token = nn.Parameter(torch.randn([1, x.size(1), 768], device=device), requires_grad=True)
        x = x + pos_token

        return x

# Creating the model on the same device as the input tensor
asd = UpToEncoder().to(device)

# Running the input tensor through the model
x = asd(x)

print(x.shape)
print(x)

# Generating the model summary
summary(model=asd, input_size=(1, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20, row_settings=["var_names"])
