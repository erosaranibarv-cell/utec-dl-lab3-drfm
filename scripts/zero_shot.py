"""Referencia sin entrenar: ResNet-50 de ImageNet tal cual, sin ver nunca Market-1501.

Es el "piso" de comparacion: cualquier metodo entrenado debe superarlo.
Se evalua en las dos condiciones del laboratorio con el mismo modelo:
  * original: consulta y galeria en resolucion original (128x64)
  * baja resolucion: consulta reducida x4 (32x16) y vuelta a ampliar con bicubica, galeria original
Uso:  python -m scripts.zero_shot --particion validation
"""
import argparse
import json

from src.data.h5data import ampliar_x4_bicubica, cargar_particion
from src.eval.features import extraer
from src.eval.reid_metrics import evaluar_reid, similitud_coseno
from src.models.encoder import EncoderReID
from src.utils import RAIZ, cargar_config, dispositivo, fijar_semilla


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particion", default="validation", choices=["validation", "testing"])
    args = ap.parse_args()

    cfg = cargar_config()
    fijar_semilla(cfg["seed"])
    dev = dispositivo()
    print("dispositivo:", dev)

    datos = cargar_particion(RAIZ / cfg["data_dir"], args.particion)
    enc = EncoderReID(preentrenado=True).to(dev).eval()

    gf = extraer(enc, datos["g_images"], dev)                                 # galeria (siempre original)
    qf_orig = extraer(enc, datos["q_images"], dev)                            # consultas originales
    q_lr_up = ampliar_x4_bicubica(datos["q_images_lr"])                       # 32x16 -> 128x64 borroso
    qf_lr = extraer(enc, q_lr_up, dev)                                        # consultas de baja resolucion

    args_m = (datos["q_pids"], datos["g_pids"], datos["q_cams"], datos["g_cams"])
    res_orig = evaluar_reid(similitud_coseno(qf_orig, gf), *args_m)
    res_lr = evaluar_reid(similitud_coseno(qf_lr, gf), *args_m)
    caida = res_orig["mAP"] - res_lr["mAP"]

    print(f"\n[{args.particion}] ResNet-50 ImageNet sin entrenar")
    print("  original      :", {k: round(v, 2) for k, v in res_orig.items()})
    print("  baja resol.   :", {k: round(v, 2) for k, v in res_lr.items()})
    print(f"  caida de mAP  : {caida:.2f} puntos")

    salida = RAIZ / cfg["results_dir"]
    salida.mkdir(exist_ok=True)
    with open(salida / f"zero_shot_{args.particion}.json", "w") as f:
        json.dump({"original": res_orig, "baja_resolucion": res_lr, "caida_mAP": caida}, f, indent=2)


if __name__ == "__main__":
    main()
