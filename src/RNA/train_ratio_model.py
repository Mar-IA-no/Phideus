#!/usr/bin/env python3
"""
train_ratio_model.py

Entrena una red neuronal convolucional (CNN) para predecir histogramas de ratios de frecuencia
extraídos de grabaciones WAV. Cada sección del script está ampliamente comentada para
facilitar la personalización y comprensión de cada parámetro.
Asegurate de tener un entorno con Python 3.8+ y las librerías instaladas:
pip install torch torchvision librosa scipy
"""

import os
import json
import librosa              # para carga de audio y CQT
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# ------------------------------
# 1. CONFIGURACIÓN GENERAL
# ------------------------------
# Directorio con los archivos WAV que usaremos para calcular el espectrograma en tiempo real.
WAV_DIR = "./wavs"                              # <-- Modificar si tus WAVs están en otra ruta

# Ruta al JSON previamente generado con generate_ratios_json.py
JSON_PATH = "ratios_dataset.json"               # <-- Modificar si guardaste el JSON con otro nombre o ruta

# Parámetros de CQT
HOP_LENGTH = 512                                # Salto en muestras entre columnas
FMIN_NOTE = 'C1'                                # Nota mínima para CQT (puede cambiarse a 'A0', 'C2', etc.)
N_BINS = 7 * 120                                # 7 octavas × 120 bins por octava para microtonalidad
BINS_PER_OCTAVE = 120                           # Resolución spectro (120 bins = 10 cents de resolución)

# Parámetros de DataLoader
BATCH_SIZE = 16                                 # Número de ejemplos por batch
NUM_WORKERS = 2                                 # Subprocesos para cargar datos (ajustar según CPU)

# Parámetros de entrenamiento
LEARNING_RATE = 1e-3                            # Tasa de aprendizaje del optimizador Adam
NUM_EPOCHS = 30                                 # Número de épocas de entrenamiento

# Ruta donde se guardará el modelo entrenado
MODEL_SAVE_PATH = "ratio_cnn.pth"               # <-- Cambiar si quieres otro nombre

# ------------------------------
# 2. DATASET CUSTOMIZADO
# ------------------------------
class RatioDataset(Dataset):
    """
    Dataset de PyTorch que, dado un JSON con histograma de ratios y los WAV originales,
    genera pares (espectrograma, histograma de ratios) para entrenamiento.
    """
    def __init__(self, json_path, wav_dir):
        # Carga el JSON completo en memoria
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        # Lista de nombres de archivo para indexación
        self.files = list(self.data.keys())
        self.wav_dir = wav_dir

    def __len__(self):
        # Retorna el número de ejemplos disponibles
        return len(self.files)

    def __getitem__(self, idx):
        # Nombre del archivo actual
        fname = self.files[idx]
        info  = self.data[fname]
        path  = os.path.join(self.wav_dir, fname)

        # 1) Carga WAV (mono, frecuencia original)
        y, sr = librosa.load(path, sr=None, mono=True)

        # 2) Cálculo de CQT según parámetros globales
        cqt = librosa.cqt(
            y, sr=sr,
            hop_length=HOP_LENGTH,
            fmin=librosa.note_to_hz(FMIN_NOTE),
            n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE
        )
        mag = np.abs(cqt)

        # 3) Convertir magnitudes a escala log (decibeles) por estabilidad
        cqt_db = librosa.amplitude_to_db(mag, ref=np.max)

        # 4) Convertir a tensor PyTorch con shape (1, n_bins, n_frames)
        spec = torch.from_numpy(cqt_db).unsqueeze(0).float()

        # 5) Cargar histograma de ratios etiquetado (tensor unidimensional)
        ratio_hist = np.array(info['ratio_hist'], dtype=np.float32)
        target = torch.from_numpy(ratio_hist)   # shape = (n_ratio_bins,)

        return spec, target

# ------------------------------
# 3. DEFINICIÓN DEL MODELO
# ------------------------------
class RatioCNN(nn.Module):
    """
    CNN simple que extrae características del espectrograma y predice un histograma de ratios.
    """
    def __init__(self, n_ratio_bins):
        super().__init__()
        # Bloque conv1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                               kernel_size=3, padding=1)
        # Bloque pool
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        # Bloque conv2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, padding=1)
        # Adaptative pooling para obtener tamaño fijo (8×8)
        self.adapt = nn.AdaptiveAvgPool2d((8, 8))
        # Capa final densa que mapea a n_ratio_bins salidas
        self.fc    = nn.Linear(32 * 8 * 8, n_ratio_bins)

    def forward(self, x):
        # x: (batch_size, 1, n_bins, n_frames)
        x = F.relu(self.conv1(x))   # -> (batch,16,n_bins,n_frames)
        x = self.pool(x)            # -> (batch,16,n_bins/2,n_frames/2)
        x = F.relu(self.conv2(x))   # -> (batch,32,n_bins/2,n_frames/2)
        x = self.pool(x)            # -> (batch,32,n_bins/4,n_frames/4)
        x = self.adapt(x)           # -> (batch,32,8,8)
        x = x.flatten(1)            # -> (batch,32*8*8)
        x = self.fc(x)              # -> (batch,n_ratio_bins)
        # Softmax para que la salida sea una distribución de probabilidad
        return F.softmax(x, dim=1)

# ------------------------------
# 4. FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ------------------------------
def train():
    # 4.1) Crear Dataset y DataLoader
    dataset = RatioDataset(JSON_PATH, WAV_DIR)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True   # acelera si tienes GPU
    )

    # 4.2) Instanciar modelo, optimizador y criterio de pérdida
    n_ratio_bins = len(next(iter(dataset.data.values()))['ratio_hist'])
    model = RatioCNN(n_ratio_bins=n_ratio_bins)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()   # regresión de histograma

    # 4.3) Ciclo de entrenamiento
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for specs, targets in loader:
            optimizer.zero_grad()
            outputs = model(specs)          # forward pass
            loss    = criterion(outputs, targets)
            loss.backward()                 # backward pass
            optimizer.step()                # actualizar pesos
            total_loss += loss.item() * specs.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} — Loss: {avg_loss:.6f}")

    # 4.4) Guardar pesos del modelo entrenado
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Modelo guardado en: {MODEL_SAVE_PATH}")

# ------------------------------
# 5. ENTRY POINT
# ------------------------------
if __name__ == "__main__":
    train()

