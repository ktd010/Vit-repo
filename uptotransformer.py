import torch
import torch.nn as nn


#very important
from torchinfo import summary

#run this and test code 

from PIL import Image
from torchvision.transforms import transforms

# put cuda available part
device="cuda" if torch.cuda.is_available()  else "cpu"
print(device)

#create a class to get patch embedding image come in (batch size, channel, h ,w)

embedding_dim=768
batch_size=1
class_token=nn.Parameter(torch.ones(batch_size,1,embedding_dim), requires_grad=True)
print(class_token.shape)






#Path_embedding_class_token= torch.cat((class_token, patch_embeddings), dim=1) 

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embedding_dim=768) -> None:
        super().__init__()



        #convert image to patches using con2d

        self.patcher=nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size)  # comment stride is needed to create patch
        self.flatten=nn.Flatten(start_dim=2,end_dim=3)
        self.clas_embed=nn.Parameter(torch.randn(1,1,embedding_dim),requires_grad=True)
        #UP TO Class token


    def forward(self, x):
        x = self.patcher(x)
        x = self.flatten(x)
        x = x.permute(0, 2, 1)
        print(x.shape)

        # Use the clas_embed parameter as a tensor, not as a callable
        cls_tok = self.clas_embed.repeat(batch_size, 1, 1)
        x = torch.cat((cls_tok, x), dim=1)
        #ADDED CLASS TOKEN practice upto here 

        return x
    

    
image=torch.ones(3,224,224).unsqueeze(0)
print(image.shape)

patch_embedding = PatchEmbedding()

# Forward pass the image through the PatchEmbedding module to obtain patch embeddings
patch_embeddings = patch_embedding(image)

# Print the shape of patch_embeddings
print(patch_embeddings.shape)

