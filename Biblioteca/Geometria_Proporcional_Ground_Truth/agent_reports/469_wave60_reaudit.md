## Auditoría técnica independiente R469

**Veredicto: REVISE — 0 HIGH, 1 MEDIUM, 0 LOW**

Commit auditado: `fcb949108e5e7050c6f24899d61b35b1540b0dd3`  
Parent directo: `9b2c7e0f65f7e26f33057a2690d4578e4470315e`

### Finding

1. **MEDIUM — Las recuperaciones omiten tiempo durable consumido después de la preparación.**

   `wave60_prior_preparation_elapsed()` en `prepare_wave56_fresh.py:5238-5271` valida y acumula los receipts firmados de preparación de primary y replay, pero devuelve exclusivamente el último `cumulative_duration_seconds` de preparación.

   Un intento puede abortar pre-truth después de ejecutar `source_bind` o `score_apply`. Esas duraciones quedan preservadas en journals ligados por los inventories y attestations terminales. El runner ya reconoce esos tiempos como parte del presupuesto durable en `run_wave60_frozen_policy_transport.py:860-884`, pero el siguiente recovery no los importa.

   Reproducción sobre un paquete real generado por la focal:

   ```text
   pair_terminal: PAIR_ABORTED_PRE_TRUTH
   primary_score_bound: true
   replay_score_bound: true
   signed_preparation_cumulative: 2.0
   authenticated score_apply durations:
     primary: 2.9034597501158714
     replay:  0.00018936395645141602
   wave60_prior_preparation_elapsed(): 2.0
   omitted_score_seconds: 2.903649114072323
   ```

   `validate_pair_failure_package()` aceptó el paquete y confirmó que ambos journals estaban ligados por los inventories terminales. Aun así, vN+1 recibió sólo `2.0 s` de consumo antecedente.

   Esto permite que recuperaciones sucesivas excedan silenciosamente el máximo combinado de `900 s` del plan: cada intento hereda las preparaciones anteriores, pero reinicia el presupuesto de source binding, scoring y demás trabajo pre-truth ya ejecutado.

   Corrección requerida: derivar el prior del recovery desde el paquete terminal autenticado completo, acumulando preparación y todas las fases durables alcanzadas por primary/replay. El nuevo acumulado debe quedar firmado y ser el `prior_elapsed_seconds` de la primary recuperada. Añadir un e2e donde v2 aborte después de `score_apply` y v3 conserve también ese consumo.

### Verificaciones favorables

- La selección de predecessor Git quedó corregida:

  ```text
  config v1 parent = source-law audit
  config v2 parent = amendment v2 audit
  config v3 parent = amendment v3 audit
  ```

- La focal recorrió v1 abortado → v2 primary/replay → v2 abortado → v3 primary/replay mediante transacciones, monitor de presupuesto y firmas reales. No reconstruyó ni reescribió manualmente los receipts de v2/v3.
- Los acumulados de preparación sí encadenaron correctamente:

  ```text
  v1 replay cumulative: 2.0
  v2 primary prior:      2.0
  v2 replay prior:       v2 primary cumulative
  v3 primary prior:      v2 replay cumulative
  v3 replay prior:       v3 primary cumulative
  ```

- La autoridad distingue correctamente:

  - escrow original, cuya config permanece en v1;
  - `config.snapshot.json` inmediata de la root fallida;
  - config previa recuperada desde Git.

- Sin regresión focal de los cierres de R466–R468:

  - validación física de ambos terminales antes de publicar pair failure;
  - rechazo de traversal y symlinks;
  - rechazo de aliases en finalización exitosa;
  - continuidad v1→v2;
  - reanudación `899+2` rechazada antes del rename;
  - manifests y bindings direccionales.

### Pruebas y estado

```text
py_compile: PASS
focal Wave 60: 46 passed in 75.85s
maximum RSS: 887520 KiB
process swaps: 0
git diff --check: PASS
```

No ejecuté la regresión Wave 56–60 porque estaba condicionada a una revisión focal sin findings, igual que en R467.

Estado del host:

```text
RAM disponible inicial: 22859272192 bytes
RAM disponible final:   22885298176 bytes
swap usada inicial:     25012203520 bytes
swap usada final:       25005686784 bytes
```

La swap global era preexistente; el proceso auditado registró cero swaps.

### Identidad física

Los cinco paths son archivos regulares, no symlinks, y coinciden byte a byte con los blobs del commit:

```text
40688554e7d97de9d065b34930303324fdbad6f2f55e4745536751ebac588da0  src/geometria_proporcional/wave60_frozen_policy_transport.py
23fc2029db88c25b8f5b74e3768e4345fb6e8fd8c181e70573e0ae0f54ccba53  experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py
c2ffafbda6234e2d7c92cc829b8f5634c9f5085a7b965f8e280243c31ef5591b  experiments/geometria_proporcional/_wave60_phase_worker.py
93156dc9754ba728433baf11fd3874798e3d945b848eb8a99025796260ea5f1e  experiments/geometria_proporcional/prepare_wave56_fresh.py
2e8d99c8e6f83775a2f7b3ff68efe89a364cd17ec97fe86207b9cfb78960adae  tests/test_wave60_frozen_policy_transport.py
```

El commit modifica exclusivamente el preparador y el test: `425` inserciones y `24` eliminaciones. Worktree inicial y final limpio. El temporal propio de `585 MiB` fue eliminado.

No modifiqué archivos ni commits. No usé ni consulté GPU/CUDA, Colab o Mendieta. Todas las ejecuciones llevaron `CUDA_VISIBLE_DEVICES=''` y cuatro threads.

No emito bloque normativo `wave60-audit-authority-v1` porque el veredicto es `REVISE`.
