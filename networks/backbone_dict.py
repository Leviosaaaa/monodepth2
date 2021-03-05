# import torchvision.models as models
from networks.senet import se_resnet50, senet154

backbone_dict = {
    'se_resnet50': se_resnet50,
     'senet154': senet154
     }
