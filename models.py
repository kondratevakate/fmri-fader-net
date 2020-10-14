import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Identity(nn.Module):
    def __init__(self,):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class Reshape(nn.Module):
    def __init__(self, shape):
        nn.Module.__init__(self)
        self.shape = shape
    def forward(self, input):
        return input.view((-1,) + self.shape)
    
    
class VoxFCN(nn.Module):
    """
    Model with only convolutional layers organised in blocks. 
    Each block contains 2 conv layers with BN and ReLU followed by maxpooling and channelwise dropout.
    Blocks are followed by globalmaxpooling aggregating values of each feature map (activation) and a single fully-connected layer.
    """
    def __init__(self, input_shape=(64, 64, 64), n_outputs=2, n_filters=16, n_blocks=4, dropout_conv=0):
        super(self.__class__, self).__init__()
        # Global{Max/Avg}Pool instead of Flatten layer
        self.model = nn.Sequential()
        
        for i in range(n_blocks):
            print("Activation size before {} block:".format(i), 
                  np.array(input_shape) // (2 ** i))
            if i == 0:
                self.model.add_module("conv3d_{}".format(i * 2 + 1), 
                                      nn.Conv3d(1, 
                                                (2 ** i) * n_filters, 
                                                kernel_size=3, padding=1))
            else:
                self.model.add_module("conv3d_{}".format(i * 2 + 1), 
                                      nn.Conv3d((2 ** (i - 1)) * n_filters, 
                                                (2 ** i) * n_filters, 
                                                kernel_size=3, padding=1))
            self.model.add_module("batch_norm_{}".format(i * 2 + 1),
                                  nn.BatchNorm3d((2 ** i) * n_filters))
            self.model.add_module("activation_{}".format(i * 2 + 1),
                                  nn.ReLU(inplace=True))
            self.model.add_module("conv3d_{}".format(i * 2 + 2), 
                                  nn.Conv3d((2 ** i) * n_filters, 
                                            (2 ** i) * n_filters, 
                                            kernel_size=3, padding=1))
            self.model.add_module("batch_norm_{}".format(i * 2 + 2),
                                  nn.BatchNorm3d((2 ** i) * n_filters))
            self.model.add_module("activation_{}".format(i * 2 + 2),
                                  nn.ReLU(inplace=True))
            self.model.add_module("max_pool3d_{}".format(i * 2 + 1),
                                  nn.MaxPool3d(kernel_size=2))
            self.model.add_module("dropout_conv_{}".format(i * 2 + 1),
                                  nn.Dropout(dropout_conv))
        
        print("Activation size before global pooling:", 
              np.array(input_shape) // (2 ** n_blocks)) # ?
        print("n flatten units:", (2 ** (n_blocks - 1)) * n_filters)
        self.model.add_module("global_pooling_1", 
                              nn.AdaptiveMaxPool3d(output_size=(1, 1, 1)))
        self.model.add_module("flatten_1", 
                              Flatten())
        self.model.add_module("fully_conn_1", 
                              nn.Linear((2 ** (n_blocks - 1)) * n_filters,
                                        n_outputs))

    def forward(self, x):
        return self.model(x)


class VoxCNN(nn.Module):
    def __init__(self, input_shape=(64, 64, 64), n_outputs=2, 
                 n_filters=16, n_blocks=4, dropout_conv=0,
                 n_flatten_units=None, dropout=0):
        super(self.__class__, self).__init__()
        # Flatten layer instead of  Global{Max/Avg}Pool
        self.model = nn.Sequential()
        
        for i in range(n_blocks):
            print("Activation size before {} block:".format(i), 
                  np.array(input_shape) // (2 ** i))
            if i == 0:
                self.model.add_module("conv3d_{}".format(i * 2 + 1), 
                                      nn.Conv3d(1, 
                                                (2 ** i) * n_filters, 
                                                kernel_size=3, padding=1))
            else:
                self.model.add_module("conv3d_{}".format(i * 2 + 1), 
                                      nn.Conv3d((2 ** (i - 1)) * n_filters, 
                                                (2 ** i) * n_filters, 
                                                kernel_size=3, padding=1))
            self.model.add_module("batch_norm_{}".format(i * 2 + 1),
                                  nn.BatchNorm3d((2 ** i) * n_filters))
            self.model.add_module("activation_{}".format(i * 2 + 1),
                                  nn.ReLU(inplace=True))
            self.model.add_module("conv3d_{}".format(i * 2 + 2), 
                                  nn.Conv3d((2 ** i) * n_filters, 
                                            (2 ** i) * n_filters, 
                                            kernel_size=3, padding=1))
            self.model.add_module("batch_norm_{}".format(i * 2 + 2),
                                  nn.BatchNorm3d((2 ** i) * n_filters))
            self.model.add_module("activation_{}".format(i * 2 + 2),
                                  nn.ReLU(inplace=True))
            self.model.add_module("max_pool3d_{}".format(i * 2 + 1),
                                  nn.MaxPool3d(kernel_size=2))
            self.model.add_module("dropout_conv_{}".format(i * 2 + 1),
                                  nn.Dropout(dropout_conv))
            
        # after blocks we just flatten the activations
        # and put it through fc layer to get output (with dropout or not?)
        final_n_filters = (2 ** (n_blocks - 1)) * n_filters
        print("Img tensor size before flatten:", 
              (final_n_filters,) + tuple(np.array(input_shape) // (2 ** n_blocks)))
        
        n_flatten_units = np.prod(np.array(input_shape) // (2 ** n_blocks)) * final_n_filters if n_flatten_units is None else n_flatten_units
        print("n flatten units:", n_flatten_units)
        
        self.model.add_module("flatten_1", 
                              Flatten())
        self.model.add_module("fully_conn_1", 
                              nn.Linear(n_flatten_units, n_outputs))
        self.model.add_module("dropout_1", 
                              nn.Dropout(dropout))
        
    def forward(self, x):
        logits = self.model(x)
        return logits


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out