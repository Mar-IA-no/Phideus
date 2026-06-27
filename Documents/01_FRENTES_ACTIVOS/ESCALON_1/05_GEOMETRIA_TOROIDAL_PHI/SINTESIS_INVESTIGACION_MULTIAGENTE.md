# Síntesis de investigación multiagente — Extensión geométrica de Escalón 1

> **Fecha de consolidación**: 2026-04-14
> **Alcance**: 7 agentes de investigación en paralelo, >200 fuentes académicas revisadas
> **Objetivo**: extender a Escalón 1 (Audio↔MIDI MAESTRO) la línea que abrió Escalón 3 — geometría no plana, dualidad storage/retrieval/activation, sondaje basado en φ.
> **Estado**: research complete. Sin implementación. Sin plan mode firmado.

---

## 0. Regla cardinal del documento

Este documento **sintetiza** los 7 reportes de investigación, los pone en diálogo con la evidencia interna de Escalón 3, y deja explícita la frontera entre lo que la literatura sostiene y lo que aún es especulación. No propone implementación: para eso, plan mode firmado por el usuario.

Correcciones honestas que se mantienen al frente:

1. **Escalón 3 ya refutó la "hipótesis toro puro".** `P6-flat` fue negativo claro; `P5-mixed` (producto R × T) fue el mejor brazo geométrico. Toda propuesta para E1 debe partir de mixed, no de toro puro.
2. **Silhouette alto no implica mejor retrieval.** `P6-cqtshift` organizó el toro con `sil≈0.985` y aun así perdió en OOD frente a `P5-cqtshift`. Métrica de organización ≠ métrica de transferencia.
3. **Role C de φ (anti-resonancia) puede no transferir desde Lissajous a MAESTRO.** Piano es 12-TET temperado; no es oscilador libre. La anti-resonancia φ se testeó limpio en Escalón 3 porque el forcing era controlado. En E1 hay que reformular qué significa "anti-resonancia" cuando el anchor ya vive en una grilla racional.
4. **No hay paper publicado que combine VICReg + T^N + Audio↔MIDI en MAESTRO.** Ese es el hueco real. Pero el hueco no garantiza mejora; garantiza originalidad.

---

## 1. Los 7 agentes y sus hallazgos clave

| # | Dominio | Fuentes | Entregables centrales |
|---|---------|---------|----------------------|
| 1 | Cross-modal contrastive + geometría | 48 | 3 arquitecturas candidatas (T-VICReg, H-VICReg, Product-Manifold). Recomienda Product-Manifold como diagnóstico. Advierte que no esperemos +10pp por geometría. |
| 2 | Geometrías no-euclídeas en repr learning | 34 | **Smoking gun**: Krumhansl-Kessler 1982 (24 tonalidades sobre T²), Shepard 1982 (pitch helicoidal T⁴). Rotman 2022 como template. |
| 3 | φ-retrieval, golden angle, LDS | 27 | Formaliza 3 roles de φ: (A) máxima no-aproximabilidad, (B) cobertura anytime-uniforme, (C) anti-resonancia. **Objeción honesta**: Role C puede no transferir a MAESTRO. |
| 4 | Storage/retrieval/activation duality | 30 | Modern Hopfield (Ramsauer 2021) regime-metastable ≡ phi-probe HIT. Aubry-André/Ten Martini (Avila-Jitomirskaya 2009): λ=1 + φ-freq → espectro Cantor — formalización matemática fuerte del hallazgo E3. |
| 5 | Audio ML harmónico/periódico | 28 | HCQT (Bittner 2017), Lostanlen 2020 helix via Isomap, PESTO/SPICE transposition-equivariant, SIREN (Sitzmann 2020). **Advertencia**: torus S¹×S¹ como latente VAE es inestable (Davidson 2018). |
| 6 | Toroidal latent spaces en DL | 34+ | Moyo-Chiurunge 2024 (arXiv:2403.00790) harmonic toroidal codes 12×12. PESTO Toeplitz FC como pieza arquitectónica. LSBD-VAE (Tonnaer 2022) métrica D_LSBD. **Advertencia M-flows**: un único flow no aprende topología no-trivial — hay que parametrizar explícito. |
| 7 | Music/pitch geometry en ML | 25+ | Tymoczko orbifold de acordes, Tonnetz neo-Riemanniano, Callender-Quinn-Tymoczko "Generalized voice-leading spaces" (Science 2008), Music Transformer con relative attention. |

**Total estimado**: ~225 referencias académicas. Excede claramente el piso de 150 pedido por el usuario.

---

## 2. Hallazgos convergentes (alta confianza)

Los 7 agentes llegan de rutas independientes a estas mismas conclusiones. La convergencia es el indicador más robusto:

### H-conv-1. El latente natural del par audio-MIDI de piano es un **producto**, no una variedad uniforme
- Agente 2: 24 tonalidades viven en T², pitch en hélice S¹×ℝ
- Agente 5: HCQT separa explícitamente pitch × octava × overtone
- Agente 6: Moyo-Chiurunge 2024 usa T² (quintas × cuartas) + residuo
- Agente 7: Tonnetz es lattice discreto + voice-leading en R^n
- **Evidencia interna E3**: `P5-mixed` > `P6-pure-torus`

Formulación mínima consensuada:

```
latent = T¹_pitchclass  ×  ℝ_octave  ×  ℝ^d_nuisance
         (circular)       (linear)   (timbre, dinámica, sala)
```

Opción enriquecida (más especulativa):

```
latent = T¹_pitchclass × T¹_fifths × ℝ_octave × ℝ^d_nuisance
```

Donde `T¹_fifths` capta circle-of-fifths como eje ortogonal al pitch-class. Moyo-Chiurunge es el soporte más directo.

### H-conv-2. La transposition-equivariance es la palanca fuerte, no la métrica tórica en sí
- Agente 5: PESTO/SPICE loss TE
- Agente 6: Toeplitz FC layers como plantilla concreta (<30k params)
- Agente 7: Music Transformer relative attention ≡ equivariance en escala
- Agente 1: CPSD loss de STONE (SO(12)-equivariant) es pieza prestable

Interpretación: que el encoder audio sea equivariante a shift de pitch (Z/12Z o continuo log₂ f) importa más que la topología del último latente. Es un prior que encaja con el `cqtshift` que ya ganó en E3-P5.

### H-conv-3. Role B de φ (cobertura anytime-uniforme) es el que mejor transfiere a E1
- Agente 3: Roberts R-sequence (φ_d) para query probe
- Agente 4: φ-orbit sobre T^n densa con discrepancia óptima
- Agente 6: φ-spacing en MRI como precedente aplicado

Este rol **sí se traduce** a MAESTRO: sampling quasi-periódico de queries no-locking sobre el latente, indistinto de si el dominio físico es temperado.

### H-conv-4. Role C de φ (anti-resonancia) necesita reformulación, posiblemente no-test en E1
- Agente 3: objeción honesta — piano no es oscilador libre
- Agente 4: Aubry-André requiere oscilador con forcing libre; en MAESTRO el forcing ya está cuantizado a 2^(k/12)

Decisión: Role C se diferirá a Escalón 4 (ECG↔PPG) donde los osciladores SÍ son libres. Forzarlo acá arriesga concluir "φ no funciona" cuando lo que no aplica es el setup.

### H-conv-5. Parametrizar topología explícita > esperar que emerja
- Agente 6: M-flows (Brehmer-Cranmer 2020) teorema clave
- Agente 5: Davidson 2018 inestabilidad práctica de VAE tórico
- Consenso: **no** confiar en que el encoder aprenda la topología por self-organization. Construirla en la proyección final.

---

## 3. Hallazgos divergentes / tensiones (baja confianza)

### T-div-1. Magnitud esperada de mejora
- Agente 1 (VICReg + geom): "no esperar break de +10pp"
- Agente 6 (LSBD lit): "+3 a +6pp si literatura es correcta"
- E3 interno: `P5-flat` perdió IID vs `P2-flat` (-3.1pp)

**Veredicto honesto**: esperar neutral a +3pp en IID, mejora razonable en OOD (scale, transposition). No vender esto como breakthrough pre-experimento.

### T-div-2. ¿Chroma 12-TET o log-continuo?
- Agente 5: advertencia directiva HIT "armonía natural, no perceptual" sugiere pc_sin/cos en log₂(f/f_ref) mod 1, **no mod 12**
- Agente 7: mucha literatura usa 12-TET explícito
- Moyo-Chiurunge: 12×12

**Resolución**: usar representación continua `log₂(f/f_ref) mod 1` como eje del latente. Es coherente con V4-lin y con la directiva del 2026-03-08. 12-TET solo como probe descriptor, no como prior geométrico impuesto.

### T-div-3. ¿Encoder audio cambia o se preserva MERT-lite?
- Agente 5: HCQT front-end es caro pero aportaría información armónica pre-cableada
- Agente 1 y Agente 6: preservar MERTEncoderLite, cambiar solo projection head

**Resolución recomendada**: preservar MERTEncoderLite (~60M, el paper ya lo publicó). Toda la geometría va en la projection head y en la loss. Minimiza riesgo y cost de cómputo.

---

## 4. Arquitectura candidata consensuada (para discusión)

Lo que emerge como consenso entre los 7 reportes — **sujeto a plan mode del usuario**:

### Core
- Encoder audio: MERTEncoderLite sin cambios (backbone congelado o fine-tune mínimo, como ya está).
- Encoder MIDI: el de siempre. Sin cambios.
- **Projection head nueva**: `R^1024 → R^{d_total}` donde `d_total = 2·d_torus + d_euclidean`.

### Latente producto
```
z = (θ_pc, θ_fifths, r_oct, r_nuisance)
    └ T¹ ──┘ └ T¹ ───┘ └ ℝ ─┘ └ ℝ^{d_n} ─┘
      2-d      2-d      1-d    d_n = 32
```

Parametrización: dos heads que devuelven `(cos θ, sin θ)` con norma forzada 1; el resto lineal.

### Loss
- **T-VICReg** en los bloques tóricos (Fisher-Lee circular correlation, varianza circular Mardia-Jupp).
- VICReg estándar en los bloques euclídeos.
- **Transposition-equivariance loss**: augmentation pitch-shift ±{1,3,7} semitonos + constraint de que θ_pc rote proporcionalmente. Plantilla PESTO.

### Métricas de evaluación
- S (global retrieval) — preservar la métrica del paper.
- R@1, R@10 (global + intra-composer + intra-piece) — ya están.
- **D_LSBD** (Tonnaer 2022): cuánta de la varianza del latente es SO(2)-disentangled.
- **Circular silhouette** sobre θ_pc (agrupado por pitch class dominante).
- **φ-retrieval probe** (Role B): query = puntos Roberts R-sequence sobre T²_pc×fifths, response = ∫ softmax(-‖z - P_{tφ}(q)‖² / τ) dt. Evalúa si el latente soporta recuperación sin locking.

### Brazos propuestos (en orden de costo/riesgo)
- **A — Product-Manifold diagnóstico**: `T¹_pc × ℝ_oct × ℝ^32`. Mínimo cambio sobre baseline d4a4. Test barato. Primera pregunta: ¿la geometría tórica respeta o degrada S?
- **B — Transposition-equivariant projection**: A + Toeplitz FC layer en la projection + augmentation pitch-shift.
- **C — Full product con fifths**: A + eje `T¹_fifths`. Solo si A no degrada.
- **D (opcional, alto riesgo)**: A + torus 12×12 à la Moyo-Chiurunge. Es el brazo más original del corpus pero más frágil (Davidson 2018).

**Precedencia**: E3 enseñó que toro puro perdió. Empezar por A, no por D.

---

## 5. φ-probe para E1 — propuesta acotada

Solo Role B (cobertura uniforme). Role A (residue test) y Role C (anti-resonancia) se difieren.

### φ-probe Role B (viable)
Evaluación **post-hoc sobre embeddings ya entrenados** (gratis en cómputo):

1. Congelar z_test (conjunto eval, pool=256).
2. Generar queries `q_t = q_0 + t · φ_d mod 1` sobre los bloques tóricos del latente, `t = 0..N-1`, `N=500`.
3. Medir discrepancia (∗): `D_N = sup_interval |empirical_frequency - Lebesgue|`.
4. Comparar contra: (i) `t · α mod 1` con α racional; (ii) `t · β mod 1` con β irracional cualquiera (√2); (iii) uniform random.

**Criterio GO**: `D_N^{φ} < D_N^{irrac-any}` significativamente. Si falla, φ no es matemáticamente especial en este contexto (y hay que reportarlo así).

Esto es independiente del training. Se puede correr **hoy** sobre el `d4a4` baseline congelado como sanity check antes de tocar nada.

### φ-probe Role A (opcional, residue test)
Usar embeddings congelados y entrenar un probe lineal sobre residuos `log₂(f/f_ref) mod 1 − nearest_rational_approximant`. Si el probe logra reconstruir mejor que chance, el latente preservó información fina más allá de la grilla 12-TET. Resultado negativo también es informativo.

### φ-probe Role C (diferido)
No se testa en E1. Se difiere a Escalón 4.

---

## 6. Riesgos identificados

1. **Latente tórico degrada S.** Davidson 2018 lo reporta. Mitigación: brazo A como diagnóstico previo.
2. **MAESTRO es piano solo.** Toda la geometría aprendida será específica de piano + Western tonal. No se generalizará a otros timbres ni a música microtonal. Hay que reportarlo al abrir E1-geom.
3. **LSBD-VAE se midió en synthetic.** Moyo-Chiurunge 2024 en corpus pequeño. Transfer a MAESTRO escala no está validado.
4. **Toeplitz FC agregando params.** Aunque <30k, hay que controlar que la mejora no venga solo de capacidad extra. Ablation requerido: Toeplitz FC con pesos sin constraint de circularidad.
5. **Interpretación de φ**: si Role B no diferencia φ de otros irracionales, hay que aceptar que la selección de φ fue un detalle del Lissajous setup, no un principio universal. Honesto reportarlo.
6. **No-transfer de Escalón 3**: E3 fue synthetic toy Lissajous. E1 es natural piano MAESTRO. La hipótesis "storage/retrieval duality" puede ser artefacto del régimen sintético.

---

## 7. Gaps identificados que podrían justificar 2ª ola de agentes

Flagged durante la síntesis, con grado de urgencia para decidir:

| # | Gap | ¿Urgente? |
|---|-----|-----------|
| G1 | Stability tricks para entrenar VAE/VICReg con latente tórico (Davidson fix) | Alta — antes de implementar |
| G2 | Literatura sobre Fisher-Lee circular correlation implementación eficiente en PyTorch | Media |
| G3 | Papers recientes (2024-2026) sobre equivariant MIR específicamente audio-MIDI | Media |
| G4 | ¿Alguien probó D_LSBD sobre audio? Corpus disponible fue imagen/synthetic | Baja — se puede estimar empírico |
| G5 | Relación formal entre Aubry-André + VICReg loss | Alta — si queremos justificar φ teóricamente |
| G6 | Microtonal / just intonation corpora para OOD test | Baja — E1 se queda en 12-TET por ahora |

Si el usuario aprueba, sugeriría 2ª ola focalizada solo en **G1 y G5** (alta prioridad) antes de plan mode.

---

## 8. Lo que NO corresponde decir

Parafraseando el estilo del documento E3-P5/P6:

- **no corresponde** decir que la geometría toroidal "va a resolver" Escalón 1 — E3 ya mostró que toro puro puede perder.
- **no corresponde** vender esto como replicación limpia de Escalón 3 — MAESTRO no es Lissajous, y eso puede romper el análogo.
- **no corresponde** asumir que +3 a +6pp es un piso — es una especulación de literatura LSBD que ni siquiera es en audio.
- **no corresponde** confundir "no hay paper que haga esto" con "esto va a funcionar". El hueco garantiza originalidad, no éxito.
- **no corresponde** ir directo a plan mode sin discutir antes con el usuario qué brazos priorizar y qué quedaría diferido.

---

## 9. Siguientes pasos (propuestos, no ejecutados)

En orden, **todos sujetos a aprobación explícita del usuario**:

1. **Discutir este documento** — el usuario decide qué descarta, qué enfatiza.
2. **Opcional**: 2ª ola de agentes sobre G1 + G5 si el usuario lo considera.
3. **φ-probe Role B sobre d4a4 congelado** (barato, informativo). Puede correrse antes de plan mode.
4. **Plan mode** para diseño experimental de Gate 11 (o como el usuario decida llamarlo).
5. **Implementación** post-plan firmado por el usuario.

---

## 10. Bibliografía — tabla maestra consolidada

Referencias estructuradas por dominio. Esto es un índice navegable; los 7 reportes individuales contienen citas completas.

### Geometrías toroidales en DL
- Rotman et al., "Semi-Supervised Learning of Partial Differential Operators and Dynamical Flows" ICLR 2022
- Tonnaer et al., "Quantifying and learning linear symmetry-based disentanglement" ICML 2022 — **LSBD-VAE**
- Davidson et al., "Hyperspherical Variational Auto-Encoders" UAI 2018 — advertencia torus inestabilidad
- Brehmer & Cranmer, "Flows for simultaneous manifold learning and density estimation" NeurIPS 2020 — **M-flows**
- Moyo & Chiurunge, "Structuring Concept Space with the Musical Circle of Fifths" arXiv:2403.00790 (2024)
- Riou et al., "PESTO: Pitch Estimation with Self-Supervised Transposition-Equivalence" ISMIR 2023
- Kong et al., "STONE: Self-Supervised Tonality Estimator" 2024
- Painblanc et al., "Equivariant Token Embeddings" 2025 (circle of fifths emergente)

### Geometrías no-euclídeas en ML (más allá del torus)
- Nickel & Kiela, "Poincaré Embeddings for Learning Hierarchical Representations" NeurIPS 2017
- Nickel & Kiela, "Learning Continuous Hierarchies in the Lorentz Model" ICML 2018
- Skopek, Ganea, Bécigneul, "Mixed-curvature Variational Autoencoders" ICLR 2020 — **product manifolds**
- Ganea et al., "Hyperbolic Neural Networks" NeurIPS 2018

### φ, golden angle, low-discrepancy
- Hurwitz, "Ueber die angenäherte Darstellung der Irrationalzahlen" (1891) — teorema Hurwitz
- Weyl, "Über die Gleichverteilung von Zahlen mod. Eins" Math. Annalen 1916
- Roberts, "The Unreasonable Effectiveness of Quasirandom Sequences" (2018) — R-sequence
- Winkelmann et al., "An optimal radial profile order based on the Golden Ratio for time-resolved MRI" IEEE TMI 2007
- Avila & Jitomirskaya, "The Ten Martini Problem" Annals of Math 2009
- Pletzer et al., "When frequencies never synchronize: The golden mean and the resting EEG" Brain Research 2010

### Storage/retrieval/activation (Hopfield moderno, EBMs)
- Ramsauer et al., "Hopfield Networks is All You Need" ICLR 2021 — modern Hopfield, 3 regímenes
- Hoover et al., "Energy Transformer" NeurIPS 2023
- Somepalli et al., "Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models" CVPR 2023
- LeCun, "A Path Towards Autonomous Machine Intelligence" (2022) — JEPA

### Representaciones armónicas audio
- Bittner et al., "Deep Salience Representations for F0 Estimation in Polyphonic Music" ISMIR 2017 — **HCQT**
- Lostanlen et al., "Learning the Helix Topology of Musical Pitch" ICASSP 2020
- Gfeller et al., "SPICE: Self-supervised Pitch Estimation" TASLP 2020
- Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions" NeurIPS 2020 — **SIREN**
- Li et al., "MERT: Acoustic Music Understanding Model" ICLR 2024

### Cross-modal contrastive
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" ICML 2021 — **CLIP**
- Wu et al., "Large-Scale Contrastive Language-Audio Pretraining" ICASSP 2023 — **CLAP**
- Bardes, Ponce, LeCun, "VICReg: Variance-Invariance-Covariance Regularization" ICLR 2022

### Geometría musical formal
- Mazzola, "The Topos of Music" Birkhäuser 2002
- Tymoczko, "A Geometry of Music" Oxford 2011
- Callender, Quinn, Tymoczko, "Generalized Voice-Leading Spaces" Science 2008
- Cohn, "Introduction to Neo-Riemannian Theory" Journal of Music Theory 1998
- Shepard, "Geometrical approximations to the structure of musical pitch" Psychological Review 1982
- Krumhansl & Kessler, "Tracing the dynamic changes in perceived tonal organization in a spatial representation of musical keys" Psychological Review 1982

*(Los 7 reportes individuales contienen ~200 citas adicionales a este índice.)*

---

## Cierre

Este documento es la síntesis. **No es propuesta de implementación**. La decisión de continuar, pivotar o cerrar queda en el usuario.

Mi recomendación operativa, ofrecida como tal y no como plan:

> Corrección honesta: antes de lanzar 2ª ola de agentes, correr **φ-probe Role B sobre d4a4 baseline congelado**. Es barato, es informativo, y nos dice si φ tiene algún privilegio matemático en el latente Audio↔MIDI actual. Si no lo tiene, buena parte del andamiaje conceptual heredado de E3 queda en entredicho para E1, y lo sabemos antes de invertir GPU-hours en T-VICReg. Si sí lo tiene, tenemos motivación independiente y empírica para seguir.

Lo que decida el usuario, lo respetamos. Pero esa corrida de sanity check parece el siguiente paso más limpio.
