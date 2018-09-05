import torch 
import torch.nn as nn


class FRE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
                                nn.Conv2d(3, 64, 3, 1, padding=1),
                                nn.ReLU(True),
                                nn.Conv2d(64, 64, 3, 1, padding=1),
                                nn.ReLU(True))
        self.side1 = nn.ModuleList(side_layer(64, 'side1'))

        self.conv2 = nn.Sequential(
                                nn.Conv2d(64, 128, 3, 1, padding=1),
                                nn.ReLU(True),
                                nn.Conv2d(128, 128, 3, 1, padding=1),
                                nn.ReLU(True))
        self.side2 = nn.ModuleList(side_layer(128, 'side2', scale_factor=2))

        self.conv3 = nn.Sequential(
                                nn.Conv2d(128, 256, 3, 1, padding=1),
                                nn.ReLU(True),
                                nn.Conv2d(256, 256, 3, 1, padding=1),
                                nn.ReLU(True),
                                nn.Conv2d(256, 256, 3, 1, padding=1),
                                nn.ReLU(True))
        self.side3 = nn.ModuleList(side_layer(256, 'side3', scale_factor=4))

        self.conv4 = nn.Sequential(
                                nn.Conv2d(256, 512, 3, 1, padding=1),
                                nn.ReLU(True),
                                nn.Conv2d(512, 512, 3, 1, padding=1),
                                nn.ReLU(True),
                                nn.Conv2d(512, 512, 3, 1, padding=1),
                                nn.ReLU(True))
        self.side4 = nn.ModuleList(side_layer(512, 'side4', scale_factor=8))

        self.conv5 = nn.Sequential(
                                nn.Conv2d(512, 512, 3, 1, padding=2, dilation=2),
                                nn.ReLU(True),
                                nn.Conv2d(512, 512, 3, 1, padding=1),
                                nn.ReLU(True),
                                nn.Conv2d(512, 512, 3, 1, padding=1),
                                nn.ReLU(True))
        self.side5 = nn.ModuleList(side_layer(512, 'side5', scale_factor=8))

        self.fuse = nn.Conv2d(5, 1, 1, 1)

            
    def forward(self, x):
        x = self.conv1(x)
        side1 = self.side1[0](x)
        x = nn.MaxPool2d(2, 2)(x)
        
        # print(side1.size())
        x = self.conv2(x)
        side2 = self.side2[2](self.side2[0](x) + self.side2[1](x))
        x = nn.MaxPool2d(2, 2)(x)
        # print(side2.size())

        x = self.conv3(x)
        side3 = self.side3[2](self.side3[0](x) + self.side3[1](x))
        x = nn.MaxPool2d(2, 2)(x)
        # print(side3.size())

        x = self.conv4(x)
        side4 = self.side4[2](self.side4[0](x) + self.side4[1](x))
        x = nn.MaxPool2d(1, 1)(x)
        # print(side4.size())

        # print(x.size())
        x = self.conv5(x)
        side5 = self.side5[2](self.side5[0](x) + self.side5[1](x))
        # print(side5.size())

        fuse =  torch.cat([side1, side2, side3, side4, side5], dim=1)
        fuse = self.fuse(fuse)
        return [side1, side2, side3, side4, side5, fuse]
    

# print(net.state_dict().keys())
def side_layer(in_channel, name, scale_factor=None):
    if name == 'side1':
        fre_block = [nn.Conv2d(64, 1, 1, 1)]
        return fre_block
    else:

        short_cut = nn.Sequential(nn.Conv2d(in_channel, 128, 1, 1),
                    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1),
                    nn.ReLU(True))

        fre_block = nn.Sequential(nn.Conv2d(in_channel, 32, 1, 1),
                   nn.BatchNorm2d(32, eps=1e-05, momentum=0.1),
                   nn.ReLU(True),
   
                   nn.Conv2d(32, 32, 3, 1, padding=1),
                   nn.BatchNorm2d(32, eps=1e-05, momentum=0.1),
                   nn.ReLU(True),
   
                   nn.Conv2d(32,128, 1, 1),
                   nn.BatchNorm2d(128, eps=1e-05, momentum=0.1),
                   nn.ReLU(True))

        output = nn.Sequential(nn.Conv2d(128, 1, 1, 1),
                    # nn.Upsample(mode='bilinear',
                    #    dd         scale_factor=scale_factor,
                    #             align_corners=True),
                    nn.ConvTranspose2d(1,1,kernel_size=2*scale_factor,
                                        padding=scale_factor // 2,
                                        stride=scale_factor)
                    # nn.Conv2d(1,1,1,1)
                )

        return short_cut, fre_block, output

# inputs = torch.ones(size=[4,3,320,320]).cuda()


# net = FRE().cuda()
# outputs = net(inputs)




