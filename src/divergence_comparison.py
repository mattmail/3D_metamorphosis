import torch
from torch.nn import Conv3d
from kornia.filters import filter3d

from utils import get_3d_sobel

conv = Conv3d(3,1,3,padding=1, padding_mode='replicate', bias=False)
div_w = get_3d_sobel()
conv.weight = torch.nn.Parameter(div_w, requires_grad=False)

def div3D(x, div_w):
    x_sobel = div_w * 32
    field_x_dx = filter3d(x[:,0].unsqueeze(1),
                              x_sobel[:,0]/x_sobel[:,0].abs().sum())
    field_y_dy = filter3d(x[:,1].unsqueeze(1),
                             x_sobel[:,1]/x_sobel[:,1].abs().sum()) # TODO : might be a kind of transposition of the thing
    field_z_dz = filter3d(x[:,2].unsqueeze(1),
                              x_sobel[:,2]/x_sobel[:,2].abs().sum())

    return field_x_dx + field_y_dy + field_z_dz


test_tensor = torch.rand((1,3,10,20,30))

print(torch.sum(torch.abs(div3D(test_tensor, div_w) - conv(test_tensor))))
print(torch.sum(torch.abs(div3D(test_tensor, div_w))))
print(torch.sum(torch.abs(conv(test_tensor))))
