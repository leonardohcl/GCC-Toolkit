from enum import Enum

class ConvNeuralNetwork(Enum):
    RESNET_50 = 'resnet50'
    DENSENET_121 = "densenet121"
    EFFICIENTNET_B2 = "efficientnet_b2"
    INCEPTION_V3 = "inception_v3"
    VGG19 = "vgg19"