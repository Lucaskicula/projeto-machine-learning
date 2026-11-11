"""Classifica vídeo/webcam com média móvel, limiar e linha do tempo.

Uso: python detectar.py teste.mp4 | python detectar.py 0
"""
import os
import sys
import json
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from utils import letterbox

CLASSE_VAZIA = "fundo"
JANELA = 8
LIMIAR_CONFIANCA = 0.60
DURACAO_MINIMA = 0.5

model = tf.keras.models.load_model("modelo.keras")
with open("class_names.json", encoding="utf-8") as f:
    class_names = json.load(f)
fonte = sys.argv[1] if len(sys.argv) > 1 else "0"
webcam = fonte.isdigit()
cap = cv2.VideoCapture(int(fonte) if webcam else fonte)
if not cap.isOpened():
    raise SystemExit(f"Não consegui abrir: {fonte}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
writer = None
if not webcam:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    saida = f"saida_{os.path.splitext(os.path.basename(fonte))[0]}.mp4"
    writer = cv2.VideoWriter(saida, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# A média reduz oscilações; abaixo do limiar exibimos '?' como classe desconhecida.
historico = deque(maxlen=JANELA)
rotulos = []
n = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if webcam:
        frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(letterbox(frame), cv2.COLOR_BGR2RGB).astype("float32")
    probs = model.predict(rgb[None], verbose=0)[0]
    historico.append(probs)
    media = np.mean(historico, axis=0)
    idx = int(np.argmax(media))
    rotulo = class_names[idx] if media[idx] >= LIMIAR_CONFIANCA else "?"
    rotulos.append(rotulo)

    texto = "nenhum objeto" if rotulo == CLASSE_VAZIA else rotulo
    cor = (0, 200, 0) if rotulo not in (CLASSE_VAZIA, "?") else (0, 200, 255)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, f"{texto} ({media[idx]:.0%})", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)
    for j, cls in enumerate(class_names):
        y = 55 + j * 22
        cv2.rectangle(frame, (10, y), (10 + int(media[j] * 150), y + 14), (0, 200, 255), -1)
        cv2.putText(frame, cls, (170, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
    cv2.putText(frame, f"t = {n / fps:5.1f}s", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if writer is not None:
        writer.write(frame)
    if webcam or os.environ.get("MOSTRAR", "1") == "1":
        cv2.imshow("Reconhecimento Pokémon", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    n += 1

cap.release()
if writer is not None:
    writer.release()
    print(f"Vídeo anotado salvo em: {saida}")
cv2.destroyAllWindows()

# Agrupa rótulos consecutivos e ignora fundo, desconhecidos e trechos curtos.
trechos = []
if rotulos:
    inicio, atual = 0, rotulos[0]
    for i, label in enumerate(rotulos[1:], start=1):
        if label != atual:
            trechos.append((inicio, i, atual))
            inicio, atual = i, label
    trechos.append((inicio, len(rotulos), atual))

visiveis = [(a / fps, b / fps, label) for a, b, label in trechos
            if label not in ("?", CLASSE_VAZIA) and (b - a) / fps >= DURACAO_MINIMA]
encontrados = sorted({label for _, _, label in visiveis})
print("Linha do tempo (Pokémon identificados):")
for inicio, fim, label in visiveis:
    print(f"  {inicio:5.1f}s a {fim:5.1f}s  ->  {label}")
print("Pokémon encontrados no vídeo:", encontrados)
