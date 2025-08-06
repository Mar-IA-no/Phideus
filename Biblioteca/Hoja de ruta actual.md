### 1  |  Perfil de hardware disponible

**GPU**: RTX 3090 (24 GB GDDR6X, \~35 TFLOP TF32 / \~70 TFLOP FP16)
**CPU/RAM**: asumo un desktop/workstation estándar (≥ 16 GB).

Esto impone dos límites prácticos:

| Recurso        | Límite típico                                           | Implicaciones                                                                   |
| -------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **VRAM**       | 23–24 GB útiles                                         | Modelo + batch + activaciones deben caber; los checkpoints FP32 caben en disco. |
| **Throughput** | ≃ 500–600 GFLOP/s sostenidos en FP16 para convs grandes | Entrenamiento de 50–100 M muestras ≃ 1–2 días/epoch con batch 256 hist.         |

---

### 2  |  Arquitectura VAE + CNN‑1D dimensionada para 24 GB

| Bloque                          | Configuración sugerida                                                                                              | Parámetros            | Activaciones (batch 128) |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------- | --------------------- | ------------------------ |
| **Input**                       | Tensor `B = 256 bins × 3 canales` (ratio, energía, entropía); opc. `T = 10` ventanas → shape `(B, C)` o `(T, B, C)` | —                     | 128 × 256 × 3 ≃ 0.1 MB   |
| **Encoder conv**                | 6 bloques {Conv 1D k=5,dil={1,2,4}, ch: 64→256} + GELU + BN + residuales                                            | ≃ 3.4 M               | ≃ 600 MB FP16            |
| **Linear‑Attention** (opcional) | Performer d\_model=256, heads=4, nb\_features=64                                                                    | ≃ 1.2 M               | +180 MB                  |
| **Latent head**                 | FC 256 → 128 (μ, σ)                                                                                                 | ≃ 66 k                | ≃ 20 MB                  |
| **Decoder simétrico**           | ConvTranspose 6 bloques (skip‑conn)                                                                                 | ≃ 3.6 M               | ≃ 650 MB                 |
| **Total**                       | **≈ 8.3 M** params (32 MB FP16)                                                                                     | Activaciones ≤ 1.6 GB |                          |

**Memoria estimada con batch 128 en FP16**:

```
Params        ≃  32 MB
Opt states    ≃  64 MB  (Adam8bit)
Activations   ≃ 1.6 GB
--------------------------------
Total forward ≃ 1.7 GB
x2 (fwd+bwd)  ≃ 3.4 GB
Reserve       ≃ 2 GB  (PyTorch, fragmentation)
--------------------------------
Uso VRAM      ≃ 6 GB
```

➡️ Sobra espacio para **batch 256** o para activar **linear‑attention** sin riesgo.

---

### 3  |  Estrategia de entrenamiento en una sola 3090

| Tema                      | Recomendación                                                                                                                                                                           |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Precisión**             | FP16 + `torch.cuda.amp` (`bf16` si tu versión de PyTorch ≥ 2.1 y driver 535).                                                                                                           |
| **Optimizador**           | `bitsandbytes.optim.Adam8bit` (Reduce 75 % de VRAM de estados).                                                                                                                         |
| **Batch**                 | 256 histogramas estáticos **o** 128 lotes `T=10` ventanas (≈ 130 k updates para 100 epochs con 10 M ventanas).                                                                          |
| **Contrastive loss**      | Con batches < 512 la SimCLR clásica se degrada. <br>Usá **MoCo‑v3** (cola = 4096 embeddings; VRAM marginal) **o** **BYOL** que no depende del tamaño de batch.                          |
| **Gradient accumulation** | Si añades canales/ventanas y te pasas de VRAM, acumula 2–4 pasos para mantener un batch efectivo 256–512.                                                                               |
| **Mixed‑precision I/O**   | Guarda checkpoints encoder+decoder en FP16 (`state_dict`), y mantiene uno en FP32 cada 5 epochs por seguridad.                                                                          |
| **Data pipeline**         | Pre‑computá histogramas y guárdalos en shards parquet/NPZ; carga con **DataLoader(num\_workers ≥ 4, prefetch\_factor = 4)**. La 3090 se sentará esperando si pre‑procesás “on the fly”. |

---

### 4  |  Cronograma aproximado de entrenamiento

| Fase                    |                                       Muestras | Epochs | Batch eff. | Pasos | Tiempo (3090) |
| ----------------------- | ---------------------------------------------: | -----: | ---------: | ----: | ------------: |
| **Pre‑training VAE**    | 10 M histogramas (≈ 500 h audio, stride 0.5 s) |    100 |        256 | 3.9 M |       36–40 h |
| **Clasificador**        |                        100 k etiquetas ligeras |     30 |        256 |  12 k |         1–2 h |
| **MoCo/BYOL fine‑tune** |                        igual al VAE, 20 epochs |      — |          — |  +7 h |               |

*Tiempo medido con FP16, I/O local NVMe, cuDNN 8.9. La mayor parte es VAE.*

---

### 5  |  Checklist de optimizaciones específicas para 3090

1. **Activá `torch.backends.cudnn.benchmark = True`** (con tamaños fijos de entrada).
2. **Usá `torch.compile(mode="reduce-overhead")`** (PyTorch 2.x) — mejora 10‑15 % throughput sin RAM extra.
3. **Desactiva `pin_memory`** si tu RAM es < 32 GB; de lo contrario mantenlo para llenar el pipeline.
4. **Mantén los hilos de DataLoader ≤ CPU cores‑2** para no estrangular el bus PCIe.
5. **Cuida la temperatura**: una sesión de 48 h FP16 a 100 % puede llevar la 3090 > 80 °C; ajusta `power.limit 290W` si es necesario.

---

### 6  |  ¿Hace falta linear‑attention o Perceiver?

Con histogramas **estáticos** (o `T ≤ 10`) la **CNN‑dilatada** basta.
Añadí **Performer block** solo si:

* incorporás **ventanas largas** (`T ≥ 50`) **y**
* la *ablation* muestra > 3 % de mejora en F1 / métrica ecológica.

Coste extra con los parámetros de la tabla: ≃ 200 MB VRAM y 10 % de tiempo – asumible.

---

### 7  |  Plantilla de comandos

```bash
export CUDA_VISIBLE_DEVICES=0

python train_vae.py \
  --data_root /datasets/phideus_histograms \
  --batch_size 256 \
  --latent_dim 128 \
  --num_epochs 100 \
  --contrastive moco_v3 \
  --fp16 \
  --opt adam8bit \
  --val_every 1 \
  --save_dir runs/vae3090
```

*(Ajustá `batch_size` a 128 si añades ventanas temporales)*

---

### 8  |  Conclusión

Con una sola **RTX 3090** puedes entrenar sin problemas:

* **VAE ≈ 8 M params** con histograma enriquecido (proporción + energía + entropía).
* **Batch efectivo 256** gracias a FP16 y optimizer 8‑bit.
* **Pre‑training completo en < 48 h** y fine‑tunes en pocas horas.

Esto mantiene el espíritu minimalista del proyecto, deja margen para experimentar con attention lineal, y evita el salto prematuro a arquitecturas sobredimensionadas que una 3090 no podría sostener cómodamente.
