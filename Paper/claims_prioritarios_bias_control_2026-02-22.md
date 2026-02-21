# Phideus BIAS_CONTROL — Priorizacion de Aportes para Paper

**Fecha**: 2026-02-22  
**Contexto**: sintesis de discusión interna sobre qué resultados tienen mayor valor publicable.

---

## 1) Aportes con mayor potencial de paper

### 1.1 Claim principal: `d4a4` vs `D0` (valor causal de descriptores)

- **Observacion**:
  - En corto (5ep), `d4a4` supera claramente a `D0`.
  - En largo, `d4a4` consolida ventaja y en 60ep alcanza `S=83.8%` (record actual).
- **Hipotesis**:
  - La inyección ratio same-modality aporta señal informativa real para retrieval, no sólo capacidad extra.
- **Inferencia (provisional)**:
  - Este contraste es el núcleo más sólido para sostener la contribución científica principal.

### 1.2 Claim diferencial: `a4r` como calidad + eficiencia

- **Observacion**:
  - `a4r` rinde muy por encima de `D0` y mantiene tiempos de entrenamiento ~2.6x más rápidos que arquitecturas con secuencia larga estándar.
- **Hipotesis**:
  - Reverse cross-attention (query más corta) actúa como bottleneck útil y reduce costo de self-attention sin perder señal relevante.
- **Inferencia (provisional)**:
  - `a4r` puede convertirse en el argumento más fuerte de eficiencia práctica del paper, no sólo de métrica.

### 1.3 Claim mecanístico secundario: superaditividad en `d4a4`

- **Observacion**:
  - La mejora conjunta de `d4a4` excede la suma simple de mejoras de componentes individuales.
- **Hipotesis**:
  - Hay sinergia no lineal entre ramas descriptoriales (MIDI+Audio), no mera agregación aditiva.
- **Inferencia (provisional)**:
  - Refuerza que el aporte es representacional/mecanístico.

---

## 2) Comparación crítica pendiente: `d4-a4r` vs `a4r`

Esta comparación es clave para delimitar si `D4` agrega valor real cuando ya existe `A4r`.

- **Pregunta central**:
  - ¿`d4-a4r` mejora robustamente a `a4r` o no?

- **Lectura propuesta**:
  - Comparar por scheduler y epochs alineados (`e30`, `e50`, `e60`) con métricas `S`, `A2M`, `M2A`, `hard_neg`.
  - Mantener receta idéntica y, cuando sea posible, multi-seed para robustez.

- **Criterio práctico sugerido**:
  - Si la mejora de `d4-a4r` sobre `a4r` es marginal/no robusta, priorizar `a4r` por parsimonia.
  - Si la mejora es consistente y relevante, sostener claim de aporte adicional de `D4` en ese régimen.

---

## 3) Rol de `t3-wt` en paper

- **Posición recomendada**: incluirlo, pero como contribución secundaria.

- **Justificación**:
  - `t3-wt` muestra señal fuerte y mejora en régimen extendido/hold (`S=81.2%`), útil para narrativa de sensibilidad a scheduler.
  - No desplaza al mejor bloque (`d4a4`) y su costo es mayor que `a4r`.

- **Ubicación sugerida en manuscrito**:
  - Resultados principales como runner-up arquitectural.
  - Profundización de curvas/costo en sección secundaria o apéndice.

---

## 4) Estructura recomendada de narrativa (versión corta)

1. `d4a4 > D0` como claim central de valor descriptorial.
2. `a4r >> D0` en trade-off calidad/costo como claim de eficiencia.
3. `d4-a4r vs a4r` como prueba de necesidad/marginalidad de `D4`.
4. `t3-wt` como evidencia de alternativa arquitectural y de efecto scheduler.
5. Resultados negativos controlados (DANN/cross-modal temprano/MoE corto) como refuerzo de credibilidad experimental.

---

## 5) Nota operativa para Gate 5-B

Si Gate 5-B se ejecuta sólo con `D0` y `d4a4`, el paper ya tiene base fuerte para contribución principal.

Si se agrega `a4r` (aunque sea en subset), aumenta significativamente el valor de publicación por el eje eficiencia computacional + desempeño.
