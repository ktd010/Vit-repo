import torch
import torch.nn as nn



x=torch.rand(3,224,224).unsqueeze(0)
print(x.shape)


conv=nn.Conv2d(3,768,16,16)

y=conv(x)
print(y.shape)
flatten=nn.Flatten(2,3)
y=flatten(y).permute(0,2,1)
print(y.shape)

class_token=nn.Parameter(torch.randn([1,1,768]),requires_grad=True)
print(class_token.shape)


y=torch.cat((y,class_token),1)
print(y.shape)

pos_token=nn.Parameter(torch.randn([1,197,768]),requires_grad=True)
print(class_token.shape)


y=y+pos_token

print(y.shape)
#print(y)

#now done upto transformer encoder write a class, start from here and write  class

layer_norm = nn.LayerNorm(normalized_shape=[y.shape[1], y.shape[2]])
N = layer_norm(y) 

print(N.shape)

#good

multihead_attn = nn.MultiheadAttention(embed_dim=768, num_heads=8)  # put this to class parameters
output, attn_weights = multihead_attn(N, N, N)

print(output)
print(output.shape)
#print(attn_weights)

y=output+N

print(y.shape)

#2nd layer norm
layer_norm = nn.LayerNorm(normalized_shape=[y.shape[1], y.shape[2]])
N = layer_norm(y)

print(N.shape)

print(attn_weights.shape) #what to do with these attention weights , understand self attention more , code it from scratch

#need mlp layer create a block

linear=nn.Linear(768,3072)
output=linear(N)
print(output.shape)
nonL1=nn.ReLU()
y=nonL1(output)
print(y.shape)
linear2=nn.Linear(3072,768)
y=linear2(y)
print(y.shape)

#need classifier, say it has 10 classes, i am not sure this step correct  this should be incorrecdt
#put a layernorm before
linear3=nn.Linear(768,10)
y=linear3(y)
print(y.shape)