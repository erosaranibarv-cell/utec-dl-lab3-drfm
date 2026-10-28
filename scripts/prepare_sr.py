"""Preproceso del paper (paso "SR"): genera las imagenes super-resueltas (SHR).

Cada imagen LR de 32x16 pasa por Swin2SR x4 y sale de 128x64. Se hace UNA vez y se guarda
en cache para no repetirlo en cada epoca de entrenamiento.

Que se genera:
  shr_train.npy    SR de TODAS las imagenes de entrenamiento reducidas x4
                   (luego, segun la camara, solo se usan las que juegan el rol "LR")
  shr_val_q.npy    SR de las 100 consultas LR de validacion (las que trae el dataset)
  shr_val_g.npy    SR de la galeria de validacion reducida x4 (solo se usa en el entrenamiento final)
  shr_test_q.npy   SR de las 3368 consultas LR de prueba (las que trae el dataset)
Tambien mide la calidad (PSNR contra la imagen original): SR frente a una ampliacion bicubica.
"""
import time

import numpy as np
import torch
from transformers import Swin2SRForImageSuperResolution

from src.data.h5data import ampliar_x4_bicubica, cargar_entrenamiento, cargar_particion, reducir_x4
from src.utils import RAIZ, cargar_config, dispositivo


@torch.no_grad()
def superresolver(modelo, lr_uint8, dev, bs=128):
    """(N,3,32,16) uint8 -> (N,3,128,64) uint8 usando Swin2SR."""
    salidas = []
    for i in range(0, len(lr_uint8), bs):
        x = torch.from_numpy(lr_uint8[i:i + bs]).to(dev).float() / 255.0
        y = modelo(pixel_values=x).reconstruction.clamp(0, 1)
        salidas.append((y * 255).round().byte().cpu())
    return torch.cat(salidas).numpy()


def psnr(a_uint8, b_uint8):
    """PSNR medio (dB) entre dos lotes de imagenes. Mas alto = mas parecidas."""
    mse = ((a_uint8.astype(np.float32) - b_uint8.astype(np.float32)) ** 2).mean(axis=(1, 2, 3))
    return float(np.mean(10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-8))))


def main():
    cfg = cargar_config()
    dev = dispositivo()
    cache = RAIZ / cfg["cache_dir"]
    cache.mkdir(exist_ok=True)
    ruta = RAIZ / cfg["data_dir"]

    modelo = Swin2SRForImageSuperResolution.from_pretrained(cfg["sr_model"]).to(dev).eval()
    print("Swin2SR cargada en", dev)

    train = cargar_entrenamiento(ruta)["images"]
    val = cargar_particion(ruta, "validation")
    test = cargar_particion(ruta, "testing")

    t = time.time()
    lr_train = reducir_x4(train)
    shr_train = superresolver(modelo, lr_train, dev)
    np.save(cache / "shr_train.npy", shr_train)
    print(f"train: {shr_train.shape} en {time.time() - t:.0f}s")

    np.save(cache / "shr_val_q.npy", superresolver(modelo, val["q_images_lr"], dev))
    np.save(cache / "shr_val_g.npy", superresolver(modelo, reducir_x4(val["g_images"]), dev))
    np.save(cache / "shr_test_q.npy", superresolver(modelo, test["q_images_lr"], dev))

    # --- Calidad: ¿la SR se parece mas a la original que una ampliacion bicubica? ---
    bic = ampliar_x4_bicubica(lr_train)
    print(f"\nPSNR frente al original (train, {len(train)} imagenes):")
    print(f"  ampliacion bicubica : {psnr(bic, train):.2f} dB")
    print(f"  Swin2SR x4          : {psnr(shr_train, train):.2f} dB")
    q_ref = psnr(np.load(cache / "shr_test_q.npy"), test["q_images"])
    q_bic = psnr(ampliar_x4_bicubica(test["q_images_lr"]), test["q_images"])
    print(f"PSNR consultas de prueba: bicubica {q_bic:.2f} dB | Swin2SR {q_ref:.2f} dB")
    print("cache guardada en", cache)


if __name__ == "__main__":
    main()
