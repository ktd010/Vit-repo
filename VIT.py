import torch
import torch.nn as nn


from Encoder import Encoder
from Encoder import Attention
from Encoder import MLP
from upToEncoder2 import UpToEncoder



from torchinfo import summary

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

x=torch.rand(3,224,224).unsqueeze(0).to(device)
print(x.shape)

class VIT(nn.Module):
    def __init__(self,
                 num_heads=8,
                 in_chanel=3,
                 embed_dim=768,
                 output_mlp=3072,
                 patch_size=16,
                 num_classes=10) -> None:
        super().__init__()

        self.uptoencoder=UpToEncoder(in_chanel=in_chanel, embed_dim=embed_dim, patch_size=patch_size)
        self.encoder=Encoder(num_heads=num_heads, embed_dim=embed_dim, output_mlp=output_mlp,patch_size=patch_size)
        self.classifier=nn.Sequential(nn.LayerNorm(normalized_shape=[197,768]),
                                      nn.Linear(in_features=embed_dim, out_features=num_classes))

        

    def forward(self,x):

        x=self.uptoencoder(x)
        x=self.encoder(x)
        x=self.classifier(x)
        


        

        return x
    
vit=VIT().to(device)

x=vit(x)
    
print(x.shape)

summary(model=vit, input_size=(1, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20, row_settings=["var_names"])