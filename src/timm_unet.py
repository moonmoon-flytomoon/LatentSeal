import torch
from torch import nn
import torch.nn.functional as F
import timm

class ConvSilu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,name='effnet',gn_group=16):
        super(ConvSilu, self).__init__()
        if '_gn' in name:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
                nn.GroupNorm(num_groups=gn_group, num_channels=out_channels),
                nn.SiLU(inplace=True)
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )

    def forward(self, x):
        return self.layer(x)

class Timm_Unet(nn.Module):
    def __init__(self, name='resnet34', pretrained=True, inp_size=3, otp_size=1, decoder_filters= [16, 32, 64, 128, 256],output_size = [256,256],gn_group=16,
                 **kwargs):
        super(Timm_Unet, self).__init__()
        self.name = name
        self.output_size = output_size

        encoder = timm.create_model(name, features_only=True, pretrained=pretrained, in_chans=inp_size)


        encoder_filters = [f['num_chs'] for f in encoder.feature_info]
        self.edecoder_filters = decoder_filters
        self.encoder_filters = encoder_filters

        self.conv6 = ConvSilu(decoder_filters[-1], decoder_filters[-2],name=self.name,gn_group=gn_group)
        self.conv6_2 = ConvSilu(decoder_filters[-2] + encoder_filters[-2], decoder_filters[-2],name=self.name,gn_group=gn_group)
        self.conv7 = ConvSilu(decoder_filters[-2], decoder_filters[-3],name=self.name,gn_group=gn_group)
        self.conv7_2 = ConvSilu(decoder_filters[-3] + encoder_filters[-3], decoder_filters[-3],name=self.name,gn_group=gn_group)
        self.conv8 = ConvSilu(decoder_filters[-3], decoder_filters[-4],name=self.name,gn_group=gn_group)
        self.conv8_2 = ConvSilu(decoder_filters[-4] + encoder_filters[-4], decoder_filters[-4],name=self.name,gn_group=gn_group)
        self.conv9 = ConvSilu(decoder_filters[-4], decoder_filters[-5], name=self.name, gn_group=gn_group)

        if len(encoder_filters) == 4:
            self.conv9_2 = None
        else:
            self.conv9_2 = ConvSilu(decoder_filters[-5] + encoder_filters[-5], decoder_filters[-5],gn_group=gn_group)

        self.conv10 = ConvSilu(decoder_filters[-5], decoder_filters[-5],gn_group=gn_group)
        self.res = nn.Conv2d(decoder_filters[-5], otp_size, 1, stride=1, padding=0)

        self.tanh = nn.Tanh()

        if pretrained == True:
            self._initialize_weights()
            self.encoder = encoder
        else:
            self.encoder = encoder
            self._initialize_weights()


    def forward(self, x):
        sub_tensors = [x[..., i * 256:(i + 1) * 256, j * 256:(j + 1) * 256] for i in range(3) for j in
                       range(3)]
        x = torch.cat(sub_tensors, dim=1)

        if self.conv9_2 is None:
            enc2, enc3, enc4, enc5 = self.encoder(x)
        else:
            enc1, enc2, enc3, enc4, enc5 = self.encoder(x)


        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3], 1))

        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))

        if self.conv9_2 is not None:
            dec9 = self.conv9_2(torch.cat([dec9,enc1], 1))


        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))  # F.interpolate(dec9, scale_factor=2))

        output = self.res(dec10)
        # output = F.relu(output)
        output = self.tanh(output)
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


