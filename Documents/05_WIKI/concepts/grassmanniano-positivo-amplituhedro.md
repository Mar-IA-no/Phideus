---
schema_version: 1
id: grassmanniano-positivo-amplituhedro
kind: concept
page_status: current
front_status: transversal
architecture_status: candidate
experiment_status: not_started
evidence_status: fuentes matemáticas/físicas consultadas; transferencia a señales y química sin prueba local
decision_status: preserved
updated: 2026-09-30
verified_at: 2026-09-30
valid_at: 2026-09-30
recorded_at: 2026-09-30
evidence_commit: e917a8772560058f64045902c011d3045f2b5e1c
source_paths:
  - Biblioteca/Grassmanniano_Positivo_Amplituhedro/CROSS_REPORT.md
  - Biblioteca/Grassmanniano_Positivo_Amplituhedro/NARRATED_REPORT.md
  - Biblioteca/Grassmanniano_Positivo_Amplituhedro/BIBLIOGRAPHY.md
  - Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/32_cluster_algebras_positive_geometry.md
depends_on: [cajas-de-herramientas, lectura-y-transferencia, ground-truth-geometria-proporcional]
tangents: [algebra-geometrica-clifford, front-atencion-armonica, front-escalon-3]
---

# Grassmanniano positivo, amplituhedro y transferencia

## Pregunta y estado

¿Qué operación de consistencia, invariancia o composición podría conservar relaciones útiles del fenómeno al trasladarse a Phideus/CASMI? La [investigación de septiembre](../../../Biblioteca/Grassmanniano_Positivo_Amplituhedro/NARRATED_REPORT.md) amplía el antecedente R32 de agosto con física, avances de 2025, herramientas numéricas y lectura del modelo CASMI actual. SRC-GRASSMANN-POSITIVE-TRANSFER conserva la síntesis; los cuatro crudos permanecen separados.

La evidencia es documental y matemática dentro de los supuestos de sus fuentes. Las candidatas no fueron implementadas ni evaluadas. La investigación se hizo durante la pausa de CASMI-48. Mariano reanudó después el trabajo CPU el 30 de septiembre: CASMI-48 cerró desde su interfaz auditada, CASMI-49 completó la conversión observable y CASMI-50 prepara roles y procedencias. La GPU sigue suspendida y el primer brazo conserva prioridad; esta reanudación no activa las candidatas geométricas. El checkout CASMI consultado fue c534fe572269eb730bdf9b75737bb29e232624c2, distinto de la base de evidencia de esta wiki.

## Las separaciones que permiten reutilizar

| Objeto u operación | Qué establece | Qué no autoriza inferir |
|---|---|---|
| Gr(k,n) y Plücker | Subespacio módulo base; menores máximos ligados algebraicamente | Una lista positiva arbitraria no es un plano realizable |
| Positividad/no negatividad | Signos bajo orden y orientación; celdas de soporte | Intensidades positivas no satisfacen por eso positividad total |
| Redes/plabic y cluster | Parametrizaciones y cambios de carta bajo contratos precisos | Grafos moleculares no son redes plabic por semejanza |
| Amplituhedro/forma canónica | Relación con amplitudes/integrandos de teorías concretas | No da probabilidad molecular ni volumen euclídeo del dibujo |
| Proyectores/ángulos | Comparación de subespacios, sin depender de su base | Puede perder energía/dinámica y necesita ambiente común |
| DPP | Diversidad según kernel PSD y menores al cuadrado | Pierde signos y no crea propuestas ausentes del soporte |

La parte estrictamente positiva es la celda superior; las fronteras pertenecen a la clausura no negativa. Postnikov da construcciones de redes, Galashin–Karp–Lam prueban regularidad y cierres como bolas, y los trabajos BCFW de 2025 prueban tilings de árbol específicos. No se transportan automáticamente esos teoremas a imágenes, otros fenómenos o todo el régimen de loops. Los originales y ubicaciones están en la [bibliografía](../../../Biblioteca/Grassmanniano_Positivo_Amplituhedro/BIBLIOGRAPHY.md).

Tiling probado y geometría positiva con forma canónica no son una única garantía: el artículo de 2025 conserva como expectativa que los tiles sean geometrías positivas. Las formas racionales candidatas BCFW y el uso físico se distinguen del estatuto general de esa conjetura. La revisión independiente exige explicitar esta diferencia, además del límite de teoría/loops.

## Un falsador del traslado

Con masas distintas ordenadas t₁<…<tₙ, las columnas (1,t,t²,…) tienen menores Vandermonde positivos para cualquier conjunto de picos. Esa propiedad la suministra el encoding. Sus magnitudes podrían servir a un predictor; el signo positivo por sí solo no descubre química ni armonía. Además, permutar columnas puede mezclar signos aunque un cambio de etiquetas preserve el grafo químico.

La forma canónica tampoco es una posterior: dx/[x(1−x)] en el intervalo [0,1] tiene integral divergente. Traducirla a un modelo predictivo requeriría medida, observación, prior y normalización defendibles. Esa carencia no prueba imposibilidad futura; localiza el contrato todavía faltante.

## Candidatas recuperables

| Alternativa | A favor / en contra | Próximo discriminante, futuro |
|---|---|---|
| Subespacios ordinarios para señales | Elimina base arbitraria; puede eliminar información física necesaria | Proyección vs covarianza/representación completa, mismos inputs/unidades; controles de rango, energía y temporalidad |
| Witness fragmento–estructura conjunto | Coherencia entre explicaciones de productos; mapa químico no definido | Coherencia genérica vs variante positiva sólo si existe mapa; fórmula inferida y context/full comparables, identidad/cobertura |
| DPP para slots | Podría reducir redundancia; no diagnosticada aquí | Calidad vs deduplicación simple vs DPP sobre soporte/costo iguales, después de comprobar redundancia |

Productos MS/MS distintos pueden compartir átomos: no se suman sus fórmulas como una partición simultánea. Un witness requiere contexto, cargas, tolerancias y transformaciones admitidas, con límites ante rearrangements y ruido. Si opera sobre prefijos debe ser computable desde observación/prefijo y cambiar preferencias entre acciones; si requiere el grafo completo su alcance es evaluar/revisar esas propuestas.

La interfaz CASMI-48 ya permite influencia de productos sobre probabilidades relativas de acciones, bajo pesos aleatorios. Eso no prueba identificación ni recomienda reemplazarla. Un reranker conserva el soporte de propuestas; ampliar pool para seleccionar 25 altera costo y necesita control comparable. La evidencia relevante permanece en moléculas completas desde observaciones permitidas.

## Relación con el programa y criterio de horizonte

A13/P2p y A18/P2u preservan contratos formales previos, auditados y no ejecutados; no se reabre su campaña. La geometría sirve aquí para precisar objeto, gauge, relaciones y límites de inferencia. Se convertiría en un desvío si la siguiente tarea fuese enumerar celdas sin una pregunta del fenómeno o contraste identificable. No se exige utilidad inmediata a toda exploración ni se convierte la procedencia de una herramienta en garantía.

Ver [síntesis cruzada](../../../Biblioteca/Grassmanniano_Positivo_Amplituhedro/CROSS_REPORT.md), [fundamento de transferencia](../foundations/lectura-y-transferencia.md) y [base estratificada](ground-truth-geometria-proporcional.md). La pista MS/MS de 2026 en SEARCH_NOTES.md se conserva como lectura incompleta; no sostiene una decisión.

La [revisión independiente e integración](../../../Biblioteca/Grassmanniano_Positivo_Amplituhedro/REVIEW_RESOLUTION.md) corrigió el alcance de forma canónica/tiling y consideró alineado el horizonte. El cierre es documental y no abre trabajo experimental.
