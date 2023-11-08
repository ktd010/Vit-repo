import torch
import torch.nn as nn

from torchinfo import summary


# Determining device but did not put model to device probably need to update
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

x=torch.rand(1,197,768).to(device)
print(x.shape)

#writing this for batch size=1
class Attention(nn.Module):
    def __init__(self,
                 num_heads=8,
                 embed_dim=768,
                 patch_size=16) -> None:
        super().__init__()

        self.msablock=nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self,x):

        #multihead_attn =self.msablock(x)  # put this to class parameters
        output, attn_weights = self.msablock(x, x, x)

        return output
    

MSA=Attention().to(device)
x=MSA(x)
print(x.shape)

class MLP(nn.Module):
    def __init__(self,
                 embed_dim=768,
                 output_mlp=3072) -> None:
        super().__init__()

        self.linear=nn.Linear(in_features=embed_dim, out_features=output_mlp)
        self.activation=nn.ReLU()
        self.linear2=nn.Linear(in_features=output_mlp, out_features=embed_dim)

        
        


    def forward(self,x):

        x=self.linear(x)
        x=self.activation(x)
        x=self.linear2(x)

        return x
    
mlp=MLP().to(device)

x=mlp(x)

print(x.shape)


#layer_norm = nn.LayerNorm(normalized_shape=[y.shape[1], y.shape[2]])
#N = layer_norm(y) 

class Encoder(nn.Module):
    def __init__(self, num_heads=8, embed_dim=768, output_mlp=3072, patch_size=16) -> None:
        super().__init__()

        self.layernorm = nn.LayerNorm(normalized_shape=[197, 768]).to(device)
        self.msa = Attention(num_heads=num_heads, embed_dim=embed_dim, patch_size=patch_size).to(device)
        self.layernorm2 = nn.LayerNorm(normalized_shape=[197, 768]).to(device)
        self.mlp = MLP(embed_dim=embed_dim, output_mlp=output_mlp).to(device)

    def forward(self, x):
        x = self.layernorm(x)
        y = self.msa(x)
        x = x + y
        x = self.layernorm2(x)
        x = self.mlp(x)

        return x

enc = Encoder().to(device)
x = enc(x)

print(x.shape)

summary(model=enc, input_size=(1, 197, 768), col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20, row_settings=["var_names"])