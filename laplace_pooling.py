import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia
from kornia.filters import get_gaussian_kernel2d, get_laplacian_kernel2d
        
        
class Laplacian_Pooling(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=5, stride=2, pad_off=0, sigma=0.8):
        super(Laplacian_Pooling, self).__init__()
        self.channels = channels
        self.pad_off = pad_off
        self.filt_size = filt_size
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        
        
        self.gauss_filt = get_gaussian_kernel2d(kernel_size=(filt_size,filt_size), sigma=((sigma),(sigma)))
        self.lap_filt = get_laplacian_kernel2d(filt_size)
        self.stride = stride

        self.register_buffer('gaussian_filt', self.gauss_filt[None,None,:,:].repeat((self.channels,1,1,1)))
        self.register_buffer('laplacian_filt', self.lap_filt[None,None,:,:].repeat((self.channels,1,1,1)))
        
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)
        
    def forward(self, x):
        if(self.filt_size==1):
            print('Filt1')
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            s = F.conv2d(self.pad(x) , self.gaussian_filt, stride=self.stride, groups=x.shape[1])
            l = F.conv2d(self.pad(x) , self.laplacian_filt, stride=self.stride, groups=x.shape[1])
            out = torch.cat((s, l), 1)
            out = F.conv2d(channels*2, channels, kernel_size=1, stride=1, bias=False)
        return out
    
    
def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer
        
    
    
