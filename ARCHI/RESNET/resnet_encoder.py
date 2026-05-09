import torch
import torch.nn as nn

class ResNetEncoder(nn.Module):

    def __init__(self, configs, resnet_norm="gn"):

        super(ResNetEncoder, self).__init__()
        if len(configs) != 4:
            raise ValueError("Only 4 layers can be configued")

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64) if resnet_norm == "gn" else nn.BatchNorm2d(num_features=64),
            nn.PReLU(),
        )

        self.conv2 = EncoderResidualBlock(in_channels=64,  hidden_channels=64,  layers=configs[0], downsample_method="pool", resnet_norm=resnet_norm)
        self.conv3 = EncoderResidualBlock(in_channels=64,  hidden_channels=128, layers=configs[1], downsample_method="conv", resnet_norm=resnet_norm)
        self.conv4 = EncoderResidualBlock(in_channels=128, hidden_channels=256, layers=configs[2], downsample_method="conv", resnet_norm=resnet_norm)
        self.conv5 = EncoderResidualBlock(in_channels=256, hidden_channels=512, layers=configs[3], downsample_method="conv", resnet_norm=resnet_norm)


    def forward(self, x0, inference=False):

        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        
        return x5

class EncoderResidualBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, layers, resnet_norm, downsample_method="conv"):
        super(EncoderResidualBlock, self).__init__()

        if downsample_method == "conv":

            for i in range(layers):

                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels, downsample=True, resnet_norm=resnet_norm)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels, downsample=False, resnet_norm=resnet_norm)
                
                self.add_module('%02d EncoderLayer' % i, layer)
        
        elif downsample_method == "pool":

            # maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            maxpool = nn.AvgPool2d(kernel_size=2, stride=2)

            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):

                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels, downsample=False, resnet_norm=resnet_norm)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels, downsample=False, resnet_norm=resnet_norm)
                
                self.add_module('%02d EncoderLayer' % (i+1), layer)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x


class EncoderResidualLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, downsample, resnet_norm):
        super(EncoderResidualLayer, self).__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=2, padding=1),
                # nn.BatchNorm2d(num_features=hidden_channels),
                nn.GroupNorm(num_groups=32, num_channels=hidden_channels) if resnet_norm == "gn" else nn.BatchNorm2d(num_features=hidden_channels),
                nn.PReLU(),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(num_features=hidden_channels),
                nn.GroupNorm(num_groups=32, num_channels=hidden_channels) if resnet_norm == "gn" else nn.BatchNorm2d(num_features=hidden_channels),
                nn.PReLU(),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_features=hidden_channels),
            nn.GroupNorm(num_groups=32, num_channels=hidden_channels) if resnet_norm == "gn" else nn.BatchNorm2d(num_features=hidden_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2, padding=0),
                # nn.BatchNorm2d(num_features=hidden_channels),
                nn.GroupNorm(num_groups=32, num_channels=hidden_channels) if resnet_norm == "gn" else nn.BatchNorm2d(num_features=hidden_channels),
            )
        else:
            self.downsample = None

        self.relu = nn.Sequential(
            nn.PReLU(),
        )
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity

        x = self.relu(x)

        return x


