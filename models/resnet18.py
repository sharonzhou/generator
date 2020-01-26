import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, generator, pretrained=False):
        super(ResNet18, self).__init__()
        self.pretrained = pretrained
        self.generator = generator

        if 'BigGAN' in generator:
            in_features = 51200
            out_features = 128
        elif 'WGAN-GP' in generator:
            in_features = 512
            out_features = 128
        elif 'BEGAN' in generator:
            in_features = 512
            out_features = 64
        elif 'ProGAN' in generator:
            in_features = 512
            out_features = 512
        else:
            raise Exception(f'{generator} not found')

        self.in_features = in_features
        self.out_features = out_features

        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(in_features, out_features).cuda()

    def forward(self, X):
        return self.model(X)
    
    def args_dict(self):
        model_args = {
                        'generator': self.generator,
                        'pretrained': self.pretrained,
                     }
        return model_args

