from typing import Tuple, List
import torch
from torch.nn.functional import pad
import kornia
# from kornia.filters.kernels import normalize_kernel2d
import kornia.filters.filter as flt

from fft_conv import fft_conv


def fft_filter(input: torch.Tensor, kernel: torch.Tensor,
             border_type: str = 'reflect',
             normalized: bool = False) -> torch.Tensor:
    r"""Function that convolves a tensor with a kernel. This function is almost
    the function filter2d from kornia, adapted to work with 2d and 3d tensors.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)` or :math:`(B, C, D, H, W)`
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)` or :math:`(1, kD, kH, kW)`.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): If True, kernel will be L1 normalized.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Input kernel type is not a torch.Tensor. Got {}"
                        .format(type(kernel)))

    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    len_input_shape = len(input.shape)
    len_kernel_shape = len(kernel.shape)
    #print(kernel.shape,"shape =",len_kernel_shape)
    if not len_input_shape in [4,5] :
        raise ValueError("Invalid input shape, we expect BxCxHxW or BxCxDxHxW. Got: {}"
                         .format(input.shape))

    if not len_kernel_shape in [3,4]:
        raise ValueError("Invalid kernel shape, we expect 1xkHxkW or 1xkDxkHxkW. Got: {}"
                         .format(kernel.shape))

    if  len_input_shape != len_kernel_shape + 1:
        raise ValueError("Size of input and kernel do not match,"
                        +"we expect input.Size() == kernel.Size() + 1, "
                        +"got {} and {}".format(input.shape,kernel.shape) )

    borders_list: List[str] = ['constant', 'reflect', 'replicate', 'circular']
    if border_type not in borders_list:
        raise ValueError("Invalid border_type, we expect the following: {0}."
                         "Got: {1}".format(borders_list, border_type))

    # prepare kernel
    # b, c, h, w = input.shape
    c = input.shape[1]
    tmp_kernel: torch.Tensor = kernel.unsqueeze(0).to(input.device).to(input.dtype)
    if normalized and len_kernel_shape == 3:
        tmp_kernel = kornia.filters.kernels.normalize_kernel2d(tmp_kernel)[None]
    elif normalized and len_kernel_shape == 4:
        norm = tmp_kernel.abs().sum([2,3,4])
        tmp_kernel = tmp_kernel / norm[...,None,None]

    # pad the input tensor
    padding_shape: List[int] = flt._compute_padding(tmp_kernel.shape[2:])
    input_pad: torch.Tensor = pad(input, padding_shape, mode=border_type)
    #b, c, hp, wp = input_pad.shape
    expand_list = len_input_shape * [-1]
    expand_list[0] = c
    return fft_conv(input_pad,tmp_kernel.expand(expand_list),
                    groups=c, padding=0, stride=1)

class GaussianRKHS(torch.nn.Module):
    """ Is equivalent to a gaussian blur. This function support 2d and 3d images in the
    PyTorch convention

    Args :


        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): If True, kernel will be L1 normalized.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.

    """
    def __init__(self,sigma : Tuple,
                 border_type: str = 'replicate'):
        """

        :param sigma: (Tuple[float,float] or [float,float,float])
        :border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``,
          ``'replicate'`` or ``'circular'``.
          the ``'reflect'`` one is not implemented yet by pytorch
        """
        big_odd = lambda val : max(6,int(val*6)) + (1 - max(6,int(val*6)) %2)
        kernel_size = tuple([big_odd(s) for s in sigma])

        super().__init__()
        if len(sigma) == 2:
            self.kernel = kornia.filters.get_gaussian_kernel2d(kernel_size,sigma)[None]
            self.filter = flt.filter2d
        elif len(sigma) == 3:
            self.kernel = self.get_gaussian_kernel3d(kernel_size,sigma)
            #self.filter = flt.filter3d
            self.filter = fft_filter
        else:
            raise ValueError("Sigma is expected to be a tuple of size 2 or 3 same as the input dimension,"
                             +"len(sigma) == {}".format(len(sigma)))
        self.border_type = border_type

        # TODO : define better the condition for using the fft filter
        # this filter works in 2d and 3d
        if max(kernel_size) > 7:
            self.filter = fft_filter

    def forward(self, input: torch.Tensor):
        """

        :param input: (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)` or :math:`(B, C, D, H, W)`
        :return:
        """
        return self.filter(input,self.kernel,self.border_type)

    def get_gaussian_kernel3d(self,kernel_size,sigma):
        if not isinstance(kernel_size, tuple) or len(kernel_size) != 3:
            raise TypeError(f"kernel_size must be a tuple of length three. Got {kernel_size}")
        if not isinstance(sigma, tuple) or len(sigma) != 3:
            raise TypeError(f"sigma must be a tuple of length three. Got {sigma}")
        ksize_d, ksize_h, ksize_w= kernel_size
        sigma_d, sigma_h, sigma_w = sigma
        kernel_d: torch.Tensor = kornia.filters.kernels.get_gaussian_kernel1d(ksize_d, sigma_d)
        kernel_h: torch.Tensor = kornia.filters.kernels.get_gaussian_kernel1d(ksize_h, sigma_h)
        kernel_w: torch.Tensor = kornia.filters.kernels.get_gaussian_kernel1d(ksize_w, sigma_w)
        kernel_2d: torch.Tensor = torch.matmul(kernel_d.unsqueeze(-1),
                                               kernel_h.unsqueeze(-1).t())
        kernel_3d: torch.Tensor = torch.matmul(kernel_2d.unsqueeze(-1),
                                               kernel_w.unsqueeze(-1).t())
        return kernel_3d[None]

class GaussianRKHS2d(GaussianRKHS):

    def __init__(self,
                 sigma: Tuple[float, float],
                 border_type: str = 'reflect') -> None:

        super(GaussianRKHS2d, self).__init__(sigma,border_type)

