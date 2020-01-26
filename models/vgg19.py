import torch
import torch.nn as nn
import torchvision.models as models


class VGG19(nn.Module):
    def __init__(self, generator, pretrained=False):
        super(VGG19, self).__init__()
        self.pretrained = pretrained
        self.generator = generator

        if 'BigGAN' in generator:
            self.out_features = 128
        elif 'WGAN-GP' in generator:
            self.out_features = 128
        elif 'BEGAN' in generator:
            self.out_features = 64
        elif 'ProGAN' in generator:
            self.out_features = 512
        else:
            raise Exception(f'{model} not found')

        self.model = models.vgg19(pretrained=pretrained, num_classes=self.out_features)

    def forward(self, X):
        return self.model(X)
    
    def args_dict(self):
        model_args = {
                        'generator': self.generator,
                        'pretrained': self.pretrained,
                     }
        return model_args

