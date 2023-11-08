import torch
import torch.nn as nn

from torchinfo import summary

#without cuda


x=torch.rand(3,224,224).unsqueeze(0)
print(x.shape)

#writing this for batch size=1
class UpToEncoder(nn.Module):
    def __init__(self,
                 in_chanel=3,
                 embed_dim=768,
                 patch_size=16) -> None:
        super().__init__()

        self.patcher=nn.Conv2d(in_channels=in_chanel, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten=nn.Flatten(2,3)


    def forward(self,x):

        x=self.patcher(x)
        x=self.flatten(x).permute(0,2,1)

        cls_token=nn.Parameter(torch.randn([1,1,768]),requires_grad=True)
        x=torch.cat((x, cls_token), 1)
        pos_token=nn.Parameter(torch.randn([1,197,768]),requires_grad=True)
        x=x+pos_token

        return x
    

asd=UpToEncoder()
x=asd(x)

print(x.shape)

#print(x)




        







