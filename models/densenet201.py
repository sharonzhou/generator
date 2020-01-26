import torch
import torch.nn as nn
import torchvision.models as models


class DenseNet201(nn.Module):
    def __init__(self, generator, pretrained=False):
        super(DenseNet201, self).__init__()
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

        self.model = models.densenet201(pretrained=pretrained)
        self.in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.in_features, self.out_features).cuda()

    def forward(self, X):
        return self.model(X)
    
    def args_dict(self):
        model_args = {
                        'generator': self.generator,
                        'pretrained': self.pretrained,
                     }
        return model_args

