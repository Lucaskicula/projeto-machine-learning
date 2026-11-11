"""Extrai amostras variadas dos vídeos em videos/<classe>/ para data/<classe>/.

Uso: python extrair_frames.py
"""
import os
import cv2
import numpy as np
from utils import letterbox, VIDEO_EXT

VIDEOS_DIR = "videos"
DATA_DIR = "data"
FRAMES_POR_SEGUNDO = 4
MAX_POR_VIDEO = 120
DIFF_MINIMA = 6.0

if not os.path.isdir(VIDEOS_DIR):
    raise SystemExit("Crie videos/ com uma subpasta por classe e adicione vídeos.")

classes = sorted(d for d in os.listdir(VIDEOS_DIR)
                 if os.path.isdir(os.path.join(VIDEOS_DIR, d)))
if not classes:
    raise SystemExit("Não encontrei subpastas de classes em videos/.")
print("Classes:", classes)
resumo = {}
for cls in classes:
    origem = os.path.join(VIDEOS_DIR, cls)
    destino = os.path.join(DATA_DIR, cls)
    os.makedirs(destino, exist_ok=True)
    videos = sorted(f for f in os.listdir(origem) if f.lower().endswith(VIDEO_EXT))
    total = 0
    for vid in videos:
        nome = os.path.splitext(vid)[0].replace("__", "_")
        cap = cv2.VideoCapture(os.path.join(origem, vid))
        if not cap.isOpened():
            print(f"  AVISO: não foi possível abrir {cls}/{vid}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        passo = max(1, int(round(fps / FRAMES_POR_SEGUNDO)))
        i, salvos, ultimo = 0, 0, None
        while salvos < MAX_POR_VIDEO:
            ok, frame = cap.read()
            if not ok:
                break
            if i % passo == 0:
                img = letterbox(frame)
                pequeno = cv2.resize(img, (32, 32)).astype(np.float32)
                # Capturas de fundo podem ser telas estáticas. Manter as amostras
                # espaçadas evita que a classe fundo fique com só um frame por vídeo.
                if cls == "fundo" or ultimo is None or np.abs(pequeno - ultimo).mean() >= DIFF_MINIMA:
                    path = os.path.join(destino, f"{nome}__{i:06d}.jpg")
                    if cv2.imwrite(path, img):
                        salvos += 1
                        ultimo = pequeno
            i += 1
        cap.release()
        total += salvos
        print(f"  {cls}/{vid}: {salvos} frames")
    resumo[cls] = total
print("Totais:", resumo)
if resumo and max(resumo.values()) > 1.5 * max(min(resumo.values()), 1):
    print("ATENÇÃO: classes desbalanceadas; grave mais vídeos para as classes menores.")
print("Extração concluída. Agora rode: python train.py")
