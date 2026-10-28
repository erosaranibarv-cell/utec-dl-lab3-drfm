"""Encoder de caracteristicas: ResNet-50 preentrenada en ImageNet (como en el paper)."""
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

MEDIA_IMAGENET = (0.485, 0.456, 0.406)
STD_IMAGENET = (0.229, 0.224, 0.225)


class EncoderReID(nn.Module):
    """ResNet-50 que convierte una imagen (3x128x64) en un vector de 2048 numeros.

    Dos ajustes tipicos de re-identificacion:
      * layer4 con stride 1: el mapa final queda 8x4 en vez de 4x2, conserva mas detalle
        (importante con imagenes tan chicas);
      * se promedia todo el mapa (average pooling global) para obtener un solo vector.
    Recibe floats en [0,1]; la normalizacion de ImageNet se hace aqui adentro.
    """

    def __init__(self, preentrenado=True):
        super().__init__()
        pesos = ResNet50_Weights.IMAGENET1K_V1 if preentrenado else None
        r = resnet50(weights=pesos)
        # layer4 con stride 1 (primer bloque: conv 3x3 y atajo 1x1)
        r.layer4[0].conv2.stride = (1, 1)
        r.layer4[0].downsample[0].stride = (1, 1)
        self.cuerpo = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool,
                                    r.layer1, r.layer2, r.layer3, r.layer4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.register_buffer("media", torch.tensor(MEDIA_IMAGENET).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD_IMAGENET).view(1, 3, 1, 1))
        self.dim = 2048

    def forward(self, x):
        x = (x - self.media) / self.std
        return self.pool(self.cuerpo(x)).flatten(1)  # (B, 2048)
