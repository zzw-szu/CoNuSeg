from torch import nn
from .densenet import densenet121,densenet169,densenet201,densenet161
import torch.nn.functional as F
import torch

class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()

class UnetBlock_(nn.Module):
    def __init__(self, up_in1,up_in2,up_out):
        super().__init__()

        self.x_conv = nn.Conv2d(up_in1, up_out, kernel_size=3, padding=1)
        self.x_conv_ = nn.Conv2d(up_in2, up_in1, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(up_out)


        # self.deconv = nn.ConvTranspose2d(2208, 2208, 3, stride=2, padding=1, output_padding=1)
        # nn.init.xavier_normal_(self.deconv.weight)

        #  init my layers
        nn.init.xavier_normal_(self.x_conv.weight)
        nn.init.xavier_normal_(self.x_conv_.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, up_p, x_p):

        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        # up_p = self.deconv(up_p)

        x_p = self.x_conv_(x_p)
        cat_p = torch.add(up_p, x_p)
        cat_p = self.x_conv(cat_p)
        cat_p = F.relu(self.bn(cat_p))

        return cat_p


class UnetBlock(nn.Module):
    def __init__(self, up_in1, up_out, size):
        super().__init__()

        self.x_conv = nn.Conv2d(up_in1, up_out, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(up_out)

        # self.deconv = nn.ConvTranspose2d(size, size, 3, stride=2, padding=1, output_padding=1)
        # nn.init.xavier_normal_(self.deconv.weight)

        #  init my layers
        nn.init.xavier_normal_(self.x_conv.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, up_p, x_p):
        # up_p = F.upsample(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        # print ("up_p", up_p.shape)

        # up_p = self.deconv(up_p)
        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)

        # cat_p = torch.cat([up_p, x_p], dim=1)
        cat_p = torch.add(up_p, x_p)

        cat_p = self.x_conv(cat_p)
        cat_p = F.relu(self.bn(cat_p))

        return cat_p

class DenseUnet_2d(nn.Module):

    def __init__(self, densenet='densenet161'):
        super().__init__()

        if densenet == 'densenet121':
            base_model = densenet121
        elif densenet == 'densenet169':
            base_model = densenet169
        elif densenet == 'densenet201':
            base_model = densenet201
        elif densenet == 'densenet161':
            base_model = densenet161
        else:
            raise Exception('The Densenet Model only accept densenet121, densenet169, densenet201 and densenet161')

        layers = list(base_model(pretrained=True).children())
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers[0]

        self.sfs = [SaveFeatures(base_layers[0][2])]
        self.sfs.append(SaveFeatures(base_layers[0][4]))
        self.sfs.append(SaveFeatures(base_layers[0][6]))
        self.sfs.append(SaveFeatures(base_layers[0][8]))

        # self.up1 = UnetBlock_(2208,2112,768)
        # self.up2 = UnetBlock(768,384,768)
        # self.up3 = UnetBlock(384,96, 384)
        # self.up4 = UnetBlock(96,96, 96)

        self.up1 = UnetBlock_(2208,2112,768)
        self.up2 = UnetBlock(768,384,768)
        self.up3 = UnetBlock(384,96, 384)
        self.up4 = UnetBlock(96,96, 96)


        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(96, 64, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(64, 2, kernel_size=1, padding=0)

        # self.deconv = nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1)
        # nn.init.xavier_normal_(self.deconv.weight)


        #  init my layers
        nn.init.xavier_normal_(self.conv1.weight)
        # nn.init.xavier_normal_(self.conv2.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

    def forward(self, x, dropout=True):
        fea = F.relu(self.rn(x))
        x = self.up1(fea, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)


        # x_fea = self.deconv(x)
        x_fea = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_fea = self.conv1(x_fea)
        if dropout:
            x_fea = F.dropout2d(x_fea, p=0.3)
        x_fea = F.relu(self.bn1(x_fea))
        # x_out = self.conv2(x_fea)

        return x_fea, fea

    def close(self):
        for sf in self.sfs: sf.remove()



if __name__ == "__main__":
    model1 = DenseUnet_2d()
    input = torch.rand(1, 3, 512, 512)
    output, enc = model1(input)
    print()