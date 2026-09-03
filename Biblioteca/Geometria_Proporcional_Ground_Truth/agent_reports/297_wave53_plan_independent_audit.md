# R297 — auditoría independiente del plan de Ola 53

> **Rol:** auditor independiente con contexto mínimo
> **Fecha:** 2026-09-03
> **Resultado:** `REVISE`

## Informe verbatim

**Findings**
1. **Alta — `expected regret` no está definido operacionalmente.** El plan enumera conjuntos y minimiza regret esperado, pero no fija la pérdida \(L(a,S,u)\), la penalización por incompatibilidad, la escala cardinal de utilidad ni desempates. Distintas elecciones cambian la acción óptima y rompen comparabilidad con Ola 52. [Plan:39](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_53_UNCERTAINTY_AWARE_POLICY_PLAN.md#L39).
   **Corrección:** congelar explícitamente la pérdida de Ola 52: regret normalizado dentro del conjunto para acciones compatibles y penalización `1.25` para incompatibles; vincular niveles ordinales, condicionamiento a conjunto no vacío, ponderación uniforme de las 24 políticas y regla de desempate.

2. **Alta — la abstención no produce todavía un estimando reproducible.** Hay dos scores por decisión, pero no se define cuál gobierna cada corte; además, los scores varían por política mientras se exige mantener juntas las 24 políticas de cada token. Tampoco se fija qué riesgo integra el AURC ni cómo resolver empates. [Plan:58](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_53_UNCERTAINTY_AWARE_POLICY_PLAN.md#L58), [Plan:63](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_53_UNCERTAINTY_AWARE_POLICY_PLAN.md#L63), [Plan:78](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_53_UNCERTAINTY_AWARE_POLICY_PLAN.md#L78).
   **Corrección:** preregistrar un score primario y su orientación; agregar las 24 vistas a un único score por `pair_token` mediante una operación fija, como máximo regret o mínimo margen; aplicar una máscara común; definir AURC sobre regret restringido y fijar integración y desempate determinista independiente de labels.

3. **Alta — el ajuste Platt deja abierta la unidad muestral.** No se especifica si se ajusta sobre logits ensemble por `token×familia`, sobre logits por seed o sobre las 24 repeticiones contextuales. Las últimas dos alternativas alteran el calibrador y pueden introducir pseudorreplicación. [Plan:23](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_53_UNCERTAINTY_AWARE_POLICY_PLAN.md#L23), [Plan:39](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_53_UNCERTAINTY_AWARE_POLICY_PLAN.md#L39).
   **Corrección:** fijar un único `(a,b)` compartido, ajustado por NLL sobre una fila por `pair_token×familia` usando el logit ensemble y el target binario; no duplicar por política ni tratar seeds como observaciones independientes. Congelar además weighting, regularización y fallback numérico.

4. **Media — el bootstrap no queda completamente especificado ni preservado.** Se declara `pair_token` como unidad, pero no que los contrastes usen exactamente los mismos resamples pareados entre reglas, ni si el IC condiciona sobre calibrador/cortes ya ajustados. Tampoco aparecen draws o manifiesto de remuestreo entre los artefactos. [Plan:65](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_53_UNCERTAINTY_AWARE_POLICY_PLAN.md#L65), [Plan:73](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_53_UNCERTAINTY_AWARE_POLICY_PLAN.md#L73), [Plan:90](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_53_UNCERTAINTY_AWARE_POLICY_PLAN.md#L90).
   **Corrección:** usar bootstrap pareado de tokens con las 24 políticas intactas; tratar políticas y seeds como mediciones fijas, no réplicas independientes; declarar IC condicional al sistema congelado, método, número de draws y seed; preservar índices y resultados del remuestreo.

**Garantías Conformales**
PASS. No se importan garantías conformales. La nota las excluye expresamente y el plan sólo fija retenciones empíricas y reporta cobertura efectiva. [Nota:27](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_53_DECISION_UNDER_UNCERTAINTY_RESEARCH_NOTE.md#L27). Tampoco detecté leakage de selección hacia `val_monitor`; sigue siendo evidencia de desarrollo históricamente abierta, no confirmatoria. No se modificó ningún archivo.
