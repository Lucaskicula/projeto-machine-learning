"""Treina MobileNetV2 com validação separada por vídeo e fine-tuning."""
import os
import json
import random
from collections import defaultdict

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

DATA_DIR = "data"
IMG_SIZE = 224
BATCH = 16
EPOCHS_CABeca = 8
EPOCHS_FINE = 8
VAL_FRACTION = 0.2
SEED = 42
# Melhoria 3: na segunda fase, libera somente as 30 camadas finais da base.
FINE_TUNE_CAMADAS = 30
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

if not os.path.isdir(DATA_DIR):
    raise SystemExit("Pasta data/ ausente. Grave vídeos e rode extrair_frames.py.")
class_names = sorted(d for d in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, d)))
if len(class_names) < 4 or "fundo" not in class_names:
    raise SystemExit("Adicione pelo menos 3 classes Pokémon e a classe fundo em data/.")

# Agrupar por nome de vídeo impede que frames quase idênticos vazem à validação.
train_paths, train_y, val_paths, val_y = [], [], [], []
for idx, cls in enumerate(class_names):
    por_video = defaultdict(list)
    for filename in sorted(os.listdir(os.path.join(DATA_DIR, cls))):
        if filename.lower().endswith(".jpg"):
            por_video[filename.split("__")[0]].append(
                os.path.join(DATA_DIR, cls, filename))
    videos = sorted(por_video)
    if len(videos) < 4:
        raise SystemExit(f"{cls}: encontrei {len(videos)} vídeos. Grave pelo menos 4 por classe.")
    random.Random(SEED).shuffle(videos)
    n_val = max(1, round(len(videos) * VAL_FRACTION))
    val_videos = set(videos[:n_val])
    for video in videos:
        paths, labels = ((val_paths, val_y) if video in val_videos
                         else (train_paths, train_y))
        paths.extend(por_video[video])
        labels.extend([idx] * len(por_video[video]))
    print(f"{cls}: vídeos de validação = {sorted(val_videos)}")
if not train_paths or not val_paths:
    raise SystemExit("Não há imagens suficientes; confira data/.")

def load(path, label):
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

def make_ds(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(train_paths, train_y, True)
val_ds = make_ds(val_paths, val_y, False)
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomZoom(0.20),
    tf.keras.layers.RandomTranslation(0.10, 0.10),
    tf.keras.layers.RandomBrightness(0.20),
    tf.keras.layers.RandomContrast(0.20),
])
base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
# Primeiro congelamos toda a MobileNetV2: na fase inicial só a nova camada
# Dense (a "cabeça" que escolhe entre as classes) aprende com nossos vídeos.
base.trainable = False
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = augment(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                        restore_best_weights=True)

# Melhoria 3 - fase 1 do fine-tuning: treina a cabeça classificadora enquanto
# mantém os filtros pré-treinados da MobileNetV2 congelados. A taxa 1e-3 é maior
# porque a cabeça começa com pesos novos.
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hist1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_CABeca,
                  callbacks=[stop])

# Melhoria 3 - fase 2: descongela a base e mantém treináveis apenas as últimas
# camadas. Elas se ajustam às imagens dos Pokémon sem alterar toda a rede.
base.trainable = True
for layer in base.layers[:-FINE_TUNE_CAMADAS]:
    layer.trainable = False
# BatchNormalization permanece congelada para preservar suas estatísticas e
# deixar o fine-tuning estável, especialmente com um conjunto pequeno de vídeos.
for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
# Depois de mudar quais camadas treinam, recompilamos obrigatoriamente o modelo.
# A taxa 1e-5 é baixa para ajustar os filtros pré-treinados gradualmente.
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hist2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE,
                  callbacks=[stop])
model.save("modelo.keras")
with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

# Combina as épocas das duas fases. A linha vertical no gráfico indica o ponto
# em que terminou o treino da cabeça e começou o fine-tuning da base.
split = len(hist1.history["loss"])
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, metric, title in zip(axes, ("accuracy", "loss"), ("Acurácia", "Loss")):
    a = hist1.history[metric] + hist2.history[metric]
    b = hist1.history["val_" + metric] + hist2.history["val_" + metric]
    ax.plot(range(1, len(a) + 1), a, label="treino")
    ax.plot(range(1, len(b) + 1), b, label="validação")
    ax.axvline(split + 0.5, color="red", linestyle="--", label="início do fine-tuning")
    ax.set_title(title)
    ax.set_xlabel("época")
    ax.legend()
fig.tight_layout()
fig.savefig("curvas.png", dpi=160)
plt.close(fig)

y_true = np.asarray(val_y)
y_pred = np.argmax(model.predict(val_ds, verbose=0), axis=1)
print("Matriz de confusão (linhas=real, colunas=previsto):")
print(confusion_matrix(y_true, y_pred, labels=range(len(class_names))))
print(classification_report(y_true, y_pred, labels=range(len(class_names)),
                            target_names=class_names, zero_division=0))

# PDF de contato: amostra representativa das imagens em cada classe.
try:
    from PIL import Image, ImageDraw
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    pdf = canvas.Canvas("dataset_imagens.pdf", pagesize=A4)
    page_w, page_h = A4
    for cls in class_names:
        files = sorted(f for f in os.listdir(os.path.join(DATA_DIR, cls)) if f.lower().endswith(".jpg"))
        vids = {f.split("__")[0] for f in files}
        # Amostra espaçada de até 300 imagens, distribuída em páginas legíveis.
        n_amostra = min(300, len(files))
        indices = np.linspace(0, len(files) - 1, n_amostra, dtype=int) if files else []
        sample = [files[i] for i in indices]
        por_pagina = 54  # 6 colunas x 9 linhas
        for inicio in range(0, max(1, len(sample)), por_pagina):
            lote = sample[inicio:inicio + por_pagina]
            pdf.setFont("Helvetica-Bold", 18)
            sufixo = "" if inicio == 0 else " (continuação)"
            pdf.drawString(36, page_h - 40, cls + sufixo)
            pdf.setFont("Helvetica", 10)
            if inicio == 0:
                pdf.drawString(36, page_h - 58, f"{len(files)} imagens | {len(vids)} vídeos | amostra: {len(sample)}")
            cols, cell, top = 6, 78, page_h - 80
            for j, filename in enumerate(lote):
                with Image.open(os.path.join(DATA_DIR, cls, filename)) as src:
                    thumb = src.convert("RGB")
                    thumb.thumbnail((64, 64))
                    buf = __import__("io").BytesIO()
                    thumb.save(buf, format="JPEG")
                    buf.seek(0)
                col, row = j % cols, j // cols
                x, y = 36 + col * cell, top - (row + 1) * cell
                pdf.drawImage(ImageReader(buf), x, y, width=64, height=64)
            pdf.showPage()
    pdf.save()
except ImportError:
    print("AVISO: instale pillow e reportlab para gerar dataset_imagens.pdf.")
print("Salvos: modelo.keras, class_names.json, curvas.png e dataset_imagens.pdf")
