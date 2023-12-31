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

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embedding_dim=768) -> None:
        super().__init__()



        #convert image to patches using con2d

        self.patcher=nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size)  # comment stride is needed to create patch
        self.flatten=nn.Flatten(start_dim=2,end_dim=3)


    def forward(self,x):
        
        x=self.patcher(x)
        x=self.flatten(x)
        x=x.permute(0,2,1)

        return x
    





image_path = r"C:\Users\kdtar\OneDrive\Desktop\Kasun\pytorch learn\Implement ViT-08\dog.jpg"
image = Image.open(image_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = transform(image).unsqueeze(0)

print(image.shape)

# Create an instance of the PatchEmbedding class
patch_embedding = PatchEmbedding()

# Forward pass the image through the PatchEmbedding module to obtain patch embeddings
patch_embeddings = patch_embedding(image)

# Print the shape of patch_embeddings
print(patch_embeddings.shape)


summary(model=patch_embedding, input_size=(32, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"], 
        col_width=20, row_settings=["var_names"])

# add batch dim

Patch_with_batch=patch_embeddings.unsqueeze(0)

print(Patch_with_batch.shape)

#add class embedding to([1, 1, 196, 768])  196 








