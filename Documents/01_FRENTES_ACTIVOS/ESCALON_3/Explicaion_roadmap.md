● El roadmap de Escalón 3 quedó completamente reescrito por Codex. Te lo resumo:

  Tesis

  ▎ Escalón 3 es el banco sintético donde Phideus separa experimentalmente storage, retrieval y activation.

  Lissajous es el soporte material, no el núcleo científico. Lo que lo hace único es que la estructura armónica es simultáneamente visible, audible, paramétrica, generable,
  perturbable y evaluable sin ambigüedad.

  Estructura

  Dos arenas transversales:
  - Arena S (Storage): ratios racionales, cierre, recurrencia, equivalence classes. Pregunta: cómo se organiza la estructura en el espacio latente.
  - Arena A (Activation): near-rational, drift, probes irracionales, φ, traversal. Pregunta: cómo se lee o activa la organización almacenada sin relocking.

  Tres niveles geométricos (secuencia obligatoria L0→L1→L2):
  - L0: Flat euclídeo + coseno (baseline)
  - L1: Flat entrenado, leído con coordenadas angulares + φ-traversal (primer test de activación sin reescribir el training)
  - L2: Toroidal explícito (T-VICReg, geodésica, varianza circular)

  8 fases

  ┌───────┬──────────────────────────┬───────────────────────────────────────────┬──────────────────────────┐
  │ Fase  │          Nombre          │             Pregunta central              │          Gate?           │
  ├───────┼──────────────────────────┼───────────────────────────────────────────┼──────────────────────────┤
  │ E3-P0 │ Canonical Generator      │ ¿El dataset es limpio y determinista?     │ —                        │
  ├───────┼──────────────────────────┼───────────────────────────────────────────┼──────────────────────────┤
  │ E3-P1 │ Parameter Recovery       │ ¿El banco es aprendible?                  │ GO si acc>95%            │
  ├───────┼──────────────────────────┼───────────────────────────────────────────┼──────────────────────────┤
  │ E3-P2 │ Flat Cross-Modal         │ ¿Hay señal de retrieval en espacio plano? │ GO si S>60%              │
  ├───────┼──────────────────────────┼───────────────────────────────────────────┼──────────────────────────┤
  │ E3-P3 │ Descriptor × Mechanism   │ ¿Qué familia de descriptor aporta más?    │ Tabla comparativa        │
  ├───────┼──────────────────────────┼───────────────────────────────────────────┼──────────────────────────┤
  │ E3-P4 │ Probe Regime on Flat     │ ¿El método de lectura importa?            │ GATE DECISIVO            │
  ├───────┼──────────────────────────┼───────────────────────────────────────────┼──────────────────────────┤
  │ E3-P5 │ Mixed Geometry           │ ¿Geometría parcial toroidal mejora?       │ Solo si P4 muestra señal │
  ├───────┼──────────────────────────┼───────────────────────────────────────────┼──────────────────────────┤
  │ E3-P6 │ Full T-VICReg            │ ¿La geometría fuerte funciona?            │ Solo si P5 valida        │
  ├───────┼──────────────────────────┼───────────────────────────────────────────┼──────────────────────────┤
  │ E3-P7 │ Dynamic Activation Arena │ ¿Near-rational revela activación?         │ El corazón teórico       │
  ├───────┼──────────────────────────┼───────────────────────────────────────────┼──────────────────────────┤
  │ E3-P8 │ Physical / Beacon        │ ¿Sobrevive fuera del sintético?           │ Convergencia             │
  └───────┴──────────────────────────┴───────────────────────────────────────────┴──────────────────────────┘

  E3-P4 es el gate que decide todo. Si φ-traversal sobre embeddings flat ya muestra señal diferencial vs coseno, entonces vale la pena la geometría toroidal. Si no, el fenómeno no
  existe en este dominio (y eso es un resultado válido).

  Métricas nuevas (Ch10-nativas)

  Además de R@10 y S, el roadmap define métricas de activación:

  - Activation Gain: mejora de φ-retrieval sobre coseno
  - Locking Selectivity: cuánto se pegan probes racionales a clusters específicos
  - Coverage Uniformity: cuán uniformemente recorre el latente un probe en K pasos
  - Relocking Depth: pasos hasta captura estable por un basin racional
  - Basin Exposure: cuántos clusters distintos se vuelven legibles bajo cada probe
  - Probe Sensitivity Spectrum: curva de desempeño por familia de probe (racional simple → φ → noble → random)

  Dataset: 5 tiers

  - Tier 0: Canónico cerrado racional (puro, sin drift, sin modulación)
  - Tier 1: Nuisance-controlled (mismos ratios, variando render/blur/noise)
  - Tier 2: Near-rational dinámico (p:q+ε, drift, phase slip, AM/FM)
  - Tier 3: Noble/φ traversal (probes de lectura, no clases)
  - Tier 4: Captura física (convergencia con Beacon)

  Splits

  4 obligatorios + 4 recomendados:
  - IID, ratio-OOD, scale-OOD, render-OOD (obligatorios)
  - equivalence-OOD, complexity-OOD, closure-OOD, activation-OOD (recomendados)

  Lo que ya está hecho

  E3-P0 completo: 6016 escenas generadas, 6.2 GB, pure Tier 0 en training, equivalencias verificadas (diff=0.00 entre figuras de ratios equivalentes). Roadmap, README, estado
  actual y bitácora ya sincronizados por Codex.

  La frase de cierre de Codex

  ▎ La innovación real no es "meter un toro". La innovación real es que por primera vez tenemos un frente donde la tesis más ambiciosa del libro puede volverse benchmark
  experimental.