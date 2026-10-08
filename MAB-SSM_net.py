import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from mamba_ssm import Mamba
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from torch_dwt.functional import dwt3
import copy

class UpsampleLayer(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            pool_op_kernel_size,
            mode='nearest'
        ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x

class BiPixelMambaLayer(nn.Module):
    def __init__(self, dim, p, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)


        self.mamba_forw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                use_fast_path=False,
        )

     
        self.out_proj = copy.deepcopy(self.mamba_forw.out_proj)


        
        self.mamba_backw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                use_fast_path=False,

        )

      
        # adjust the window size here to fit the feature map
        self.p = p*5
        self.p1 = 5*p
        self.p2 = 7*p
        self.p3 = 6*p
        self.mamba_forw.out_proj = nn.Identity()
        self.mamba_backw.out_proj = nn.Identity()
       


    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        ll = len(x.shape)

        B, C = x.shape[:2]

        assert C == self.dim
        img_dims = x.shape[2:]

        if ll == 5: #3d
         
            Z,H,W = x.shape[2:]

            if Z%self.p1==0 and H%self.p2==0 and W%self.p3==0:
                x_div = x.reshape(B, C, Z//self.p1, self.p1, H//self.p2, self.p2, W//self.p3, self.p3)
                x_div = x_div.permute(0, 3, 5, 7, 1, 2, 4, 6).contiguous().view(B*self.p1*self.p2*self.p3, C, Z//self.p1, H//self.p2, W//self.p3)
            else:
                x_div = x

        elif ll == 4: #2d
            H,W = x.shape[2:]

            if H%self.p==0 and W%self.p==0:                
                x_div = x.reshape(B, C, H//self.p, self.p, W//self.p, self.p).permute(0, 3, 5, 1, 2, 4).contiguous().view(B*self.p*self.p, C, H//self.p, W//self.p)            
            else:
                x_div = x
        

        NB = x_div.shape[0]
        if ll == 5: #3d
            NZ,NH,NW = x_div.shape[2:]
        else:
            NH,NW = x_div.shape[2:]

        n_tokens = x_div.shape[2:].numel()
   

        x_flat = x_div.reshape(NB, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        y_norm = torch.flip(x_norm, dims=[1])

        x_mamba = self.mamba_forw(x_norm)
        y_mamba = self.mamba_backw(y_norm)

        x_out = self.out_proj(x_mamba + torch.flip(y_mamba, dims=[1]))

        if ll == 5:
            if Z%self.p1==0 and H%self.p2==0 and W%self.p3==0:
                x_out = x_out.transpose(-1, -2).reshape(B, self.p1, self.p2, self.p3, C, NZ, NH, NW).permute(0, 4, 5, 1, 6, 2, 7, 3).contiguous().reshape(B, C, *img_dims)
            else:
                x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
        if ll == 4:
            if H%self.p==0 and W%self.p==0:
                x_out = x_out.transpose(-1, -2).reshape(B, self.p, self.p, C, NH, NW).permute(0, 3, 4, 1, 5, 2).contiguous().reshape(B, C, *img_dims)
            else:
                x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
        out = x_out + x

        return out


class BiWindowMambaLayer(nn.Module):
    def __init__(self, dim, p, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        self.p = p
        self.mamba_forw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                use_fast_path=False,
        )

     
        self.out_proj = copy.deepcopy(self.mamba_forw.out_proj)

        

        
        self.mamba_backw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                use_fast_path=False,

        )

      

       
        self.mamba_forw.out_proj = nn.Identity()
        self.mamba_backw.out_proj = nn.Identity()
       


    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        ll = len(x.shape)

        B, C = x.shape[:2]

        assert C == self.dim
   
        img_dims = x.shape[2:]

        if ll == 5: #3d
            
            Z,H,W = x.shape[2:]

            if Z%self.p==0 and H%self.p==0 and W%self.p==0:
                pool_layer = nn.AvgPool3d(self.p, stride=self.p)
                x_div = pool_layer(x)
            else:
                x_div = x

        elif ll == 4: #2d

            H,W = x.shape[2:]
            if self.p==0:
                self.p = 1
            if H%self.p==0 and W%self.p==0:
                pool_layer = nn.AvgPool2d(self.p, stride=self.p)
                x_div = pool_layer(x)
            else:
                x_div = x
           
        if ll == 5: #3d
            NZ,NH,NW = x_div.shape[2:]
        else:
            NH,NW = x_div.shape[2:]

        n_tokens = x_div.shape[2:].numel()
   

        x_flat = x_div.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        y_norm = torch.flip(x_norm, dims=[1])

        x_mamba = self.mamba_forw(x_norm)
        y_mamba = self.mamba_backw(y_norm)

        x_out = self.out_proj(x_mamba + torch.flip(y_mamba, dims=[1]))

        if ll == 5:
            if Z%self.p==0 and H%self.p==0 and W%self.p==0:
                unpool_layer = nn.Upsample(scale_factor=self.p, mode='nearest')
                x_out = x_out.transpose(-1, -2).reshape(B, C, NZ, NH, NW)
                x_out = unpool_layer(x_out)
            else:
                x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
        if ll == 4:
            if self.p==0:
                self.p = 1
            if H%self.p==0 and W%self.p==0:
                unpool_layer = nn.Upsample(scale_factor=self.p, mode='nearest')
                x_out = x_out.transpose(-1, -2).reshape(B, C, NH, NW)
                x_out = unpool_layer(x_out)
            else:
                x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
                
        out = x_out + x

        return out

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
    
    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        return out

class BasicResBlock(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
        ):
        super().__init__()
        
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        
        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        
        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
                  
    def forward(self, x):
        
        y = self.conv1(x)
        y = self.act1(self.norm1(y))  
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)

class SeparateConv3DBlock(nn.Module):
    def __init__(self, input_channels, kernel_size, padding_size):
        super().__init__()
        reduced_channels = max(1, input_channels // 4)

        # 先用 1×1×1 卷积降维
        self.reduce_conv = nn.Conv3d(input_channels, reduced_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.reduce_bn = nn.BatchNorm3d(reduced_channels)
        self.reduce_relu = nn.ReLU(inplace=True)
        
        self.conv_n_1_1 = nn.Conv3d(reduced_channels, reduced_channels, (kernel_size,1,1), stride=1, padding=(padding_size,0,0), bias=False)
        self.conv_1_n_1 = nn.Conv3d(reduced_channels, reduced_channels, (1,kernel_size,1), stride=1, padding=(0,padding_size,0), bias=False)
        self.conv_1_1_n = nn.Conv3d(reduced_channels, reduced_channels, (1,1,kernel_size), stride=1, padding=(0,0,padding_size), bias=False)
        self.conv_1_1_1 = nn.Conv3d(reduced_channels, reduced_channels, (1,1,1), stride=1, padding=(0,0,0), bias=False)
        
        self.bn = nn.BatchNorm3d(reduced_channels)
        self.relu = nn.ReLU(inplace=False)

	# 输出恢复通道
        self.expand_conv = nn.Conv3d(reduced_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.expand_bn = nn.BatchNorm3d(input_channels)
    
    def forward(self, x):
	# 降维
        x_reduced = self.reduce_conv(x)
        x_reduced = self.reduce_bn(x_reduced)
        x_reduced = self.reduce_relu(x_reduced)

        x_x = self.conv_n_1_1(x_reduced)
        x_x = self.conv_1_n_1(x_x)
        x_x = self.conv_1_1_1(x_x)
        x_x = self.bn(x_x)
        x_x = self.relu(x_x)
        
        x_y = self.conv_1_n_1(x_reduced)
        x_y = self.conv_n_1_1(x_y)
        x_y = self.conv_1_1_1(x_y)
        x_y = self.bn(x_y)
        x_y = self.relu(x_y)
        
        x_z = self.conv_1_1_n(x_reduced)
        x_z = self.conv_1_1_1(x_z)
        x_z = self.bn(x_z)
        x_z = self.relu(x_z)
        
        x_sum = x_x + x_y + x_z

	# 恢复通道
        out = self.expand_conv(x_sum)
        out = self.expand_bn(out)
        
        return out
   
    
class SkipMultiscaleAnisotropy3DBlock(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        
        self.initial_conv = nn.Conv3d(input_channels, input_channels, kernel_size=(1,1,1), stride=1, padding=0)
        self.initial_bn = nn.BatchNorm3d(input_channels)
        self.initial_relu = nn.ReLU(inplace=False)
        self.avgpool3d = nn.AvgPool3d(kernel_size=(2,2,2), stride = 1, padding=(1,1,1))
        
        self.separateconv3Dblock_1 = SeparateConv3DBlock(input_channels, kernel_size=3, padding_size=1)
        self.separateconv3Dblock_2 = SeparateConv3DBlock(input_channels, kernel_size=9, padding_size=4)
        self.separateconv3Dblock_3 = SeparateConv3DBlock(input_channels, kernel_size=15, padding_size=7)
        
        self.attention = nn.Sequential(
            nn.Conv3d(4*input_channels, 1, kernel_size=(1,1,1)),
            nn.Softmax(dim=1)
        )
        
        self.end_conv = nn.Conv3d(6*input_channels, input_channels, kernel_size=(1,1,1), stride=1, padding=0)
        
    def compute_gradient(self, x):
        if not x.requires_grad:
            return torch.zeros_like(x)
        
        x = torch.autograd.Variable(x, requires_grad=True)
        gradient_x = torch.autograd.grad(outputs=x.sum(), inputs=x, create_graph=True)[0]
        gradient_x = torch.abs(gradient_x)
        
        return gradient_x
    
    def forward(self, x):
        
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)
        
        x_pool = self.avgpool3d(x)
        
        if self.training:
            x_1 = self.separateconv3Dblock_1(x)
            x1_gradient = self.compute_gradient(x_1)
            x1_with_gradient = torch.cat((x_1, x1_gradient), dim=1)

            x_2 = self.separateconv3Dblock_2(x)
            x2_gradient = self.compute_gradient(x_2)
            x2_with_gradient = torch.cat((x_2, x2_gradient), dim=1)

            x_3 = self.separateconv3Dblock_3(x)
            x3_gradient = self.compute_gradient(x_3)
            x3_with_gradient = torch.cat((x_3, x3_gradient), dim=1)
        else:
            x_1 = self.separateconv3Dblock_1(x)
            x1_gradient = torch.zeros_like(x_1)
            x1_with_gradient = torch.cat((x_1, x1_gradient), dim=1)

            x_2 = self.separateconv3Dblock_2(x)
            x2_gradient = torch.zeros_like(x_2)
            x2_with_gradient = torch.cat((x_2, x2_gradient), dim=1)

            x_3 = self.separateconv3Dblock_3(x)
            x3_gradient = torch.zeros_like(x_3)
            x3_with_gradient = torch.cat((x_3, x3_gradient), dim=1)
            
        
        
        x1_2_attention_weight = self.attention(torch.cat((x1_with_gradient, x2_with_gradient), dim=1))
        x2_3_attention_weight = self.attention(torch.cat((x2_with_gradient, x3_with_gradient), dim=1))
        
        x1_weight = x1_with_gradient * x1_2_attention_weight[:, :, :x_1.size(2), :x_1.size(3), :x_1.size(4)]
        x2_weight = x2_with_gradient * x1_2_attention_weight[:, :, :x_2.size(2), :x_2.size(3), :x_2.size(4)]
        x3_weight = x3_with_gradient * x2_3_attention_weight[:, :, :x_3.size(2), :x_3.size(3), :x_3.size(4)]
        
        concat_out = torch.cat((x1_weight, x2_weight, x3_weight), dim=1)
        # concat_out = torch.cat((x_1, x_2, x_3), dim=1)
        
        out = self.end_conv(concat_out)
        out = self.initial_bn(out)
        out = self.initial_relu(out)
        
        return out
        

class BasicRes_DWT_Block(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
        ):
        super().__init__()
        
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        
        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        
        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
                  
    def forward(self, x):
        lll, hll, lhl, hhl, llh, hlh, lhh, hhh = dwt3(x, "haar").unbind(1)
        # x = torch.cat([lll, hll, lhl, hhl, llh, hlh, lhh, hhh], dim=1)
        y = self.conv1(lll)
        
        # y = self.conv1(x)
        y = self.act1(self.norm1(y))  
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)
    
class UNetResEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        self.conv_pad_sizes = []
        for krnl in kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        stem_channels = features_per_stage[0]

        self.stem = nn.Sequential(
            BasicRes_DWT_Block(
                conv_op = conv_op,
                input_channels = input_channels,
                output_channels = stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0],
                stride=1,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True
            ), 
            *[
                BasicBlockD(
                    conv_op = conv_op,
                    input_channels = stem_channels,
                    output_channels = stem_channels,
                    kernel_size = kernel_sizes[0],
                    stride = 1,
                    conv_bias = conv_bias,
                    norm_op = norm_op,
                    norm_op_kwargs = norm_op_kwargs,
                    nonlin = nonlin,
                    nonlin_kwargs = nonlin_kwargs,
                ) for _ in range(n_blocks_per_stage[0] - 1)
            ]
        )


        input_channels = stem_channels

        stages = []
        for s in range(n_stages):

            stage = nn.Sequential(
                BasicRes_DWT_Block(
                    conv_op = conv_op,
                    norm_op = norm_op,
                    norm_op_kwargs = norm_op_kwargs,
                    input_channels = input_channels,
                    output_channels = features_per_stage[s],
                    kernel_size = kernel_sizes[s],
                    padding=self.conv_pad_sizes[s],
                    stride=strides[s],
                    use_1x1conv=True,
                    nonlin = nonlin,
                    nonlin_kwargs = nonlin_kwargs
                ),
                *[
                    BasicBlockD(
                        conv_op = conv_op,
                        input_channels = features_per_stage[s],
                        output_channels = features_per_stage[s],
                        kernel_size = kernel_sizes[s],
                        stride = 1,
                        conv_bias = conv_bias,
                        norm_op = norm_op,
                        norm_op_kwargs = norm_op_kwargs,
                        nonlin = nonlin,
                        nonlin_kwargs = nonlin_kwargs,
                    ) for _ in range(n_blocks_per_stage[s] - 1)
                ]
            )


            stages.append(stage)
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs

        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output


class UNetResDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 num_classes,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):

        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        stages = []
        upsample_layers = []
        mamba_layers = []
        skipmultiscaleanisotropy_layers = []

        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]


            upsample_layers.append(UpsampleLayer(
                conv_op = encoder.conv_op,
                input_channels = input_features_below,
                output_channels = input_features_skip,
                pool_op_kernel_size = stride_for_upsampling,
                mode='nearest'
            ))
            
            skipmultiscaleanisotropy_layers.append(SkipMultiscaleAnisotropy3DBlock(
                input_channels = input_features_skip
            ))

            stages.append(nn.Sequential(
                BasicRes_DWT_Block(
                    conv_op = encoder.conv_op,
                    norm_op = encoder.norm_op,
                    norm_op_kwargs = encoder.norm_op_kwargs,
                    nonlin = encoder.nonlin,
                    nonlin_kwargs = encoder.nonlin_kwargs,
                    input_channels = 2 * input_features_skip,
                    output_channels = input_features_skip,
                    kernel_size = encoder.kernel_sizes[-(s + 1)],
                    padding=encoder.conv_pad_sizes[-(s + 1)],
                    stride=1,
                    use_1x1conv=True
                ),
                *[
                    BasicBlockD(
                        conv_op = encoder.conv_op,
                        input_channels = input_features_skip,
                        output_channels = input_features_skip,
                        kernel_size = encoder.kernel_sizes[-(s + 1)],
                        stride = 1,
                        conv_bias = encoder.conv_bias,
                        norm_op = encoder.norm_op,
                        norm_op_kwargs = encoder.norm_op_kwargs,
                        nonlin = encoder.nonlin,
                        nonlin_kwargs = encoder.nonlin_kwargs,
                    ) for _ in range(n_conv_per_stage[s-1] - 1)

                ]
            ))
            
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))
            # mamba_layers.append(BiPixelMambaLayer(input_features_skip, 2**( (n_stages_encoder-s+1)//2-1) ))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.skipmultiscaleanisotropy_layers = nn.ModuleList(skipmultiscaleanisotropy_layers)
        # self.mamba_layers = nn.ModuleList(mamba_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            
            x_skipmultiscaleanisotropy = self.skipmultiscaleanisotropy_layers[s](skips[-(s+2)])
            # print("x.shape:", x.shape)
            # print("skips[-(s+2)].shape:", skips[-(s+2)].shape)
            
            # m_x = self.mamba_layers[s](skips[-(s+2)])
            
            x = torch.cat((x, x_skipmultiscaleanisotropy), 1)
            # x = torch.cat((x, skips[-(s+2)]), 1)
            # x = torch.cat((x, m_x), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output
    
class MABSSM(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 stem_channels: int = None
                 ):
        super().__init__()
        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        

        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks_per_stage[s] = 1    

        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_per_stage_decoder[s] = 1

        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = UNetResEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            stem_channels=stem_channels
        )

        # self.mamba_layer = MambaLayer(dim = features_per_stage[-1])
        self.bipixelmamba_layer = BiPixelMambaLayer(dim = features_per_stage[-1], p = 1)
        self.biwindowmamba_layer = BiWindowMambaLayer(dim = features_per_stage[-1], p = 1)

        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        # skips[-1] = self.mamba_layer(skips[-1])
        skips[-1] = self.bipixelmamba_layer(skips[-1])
        skips[-1] = self.biwindowmamba_layer(skips[-1])
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


def get_mab_ssm_3d_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True
    ):
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = 'MABSSM'
    network_class = MABSSM
    kwargs = {
        'MABSSM': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }

    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))

    return model
