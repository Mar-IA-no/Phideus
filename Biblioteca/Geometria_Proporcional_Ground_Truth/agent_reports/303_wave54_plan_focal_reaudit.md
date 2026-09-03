# Reauditoría focal independiente — Ola 54

**Fecha:** 2026-09-03  
**Alcance:** verificación independiente, con contexto nuevo, del plan vigente contra la auditoría R302. Se revisaron exclusivamente `waves/WAVE_54_JOINT_SET_POSTERIOR_PLAN.md` y `agent_reports/302_wave54_plan_independent_audit.md`; no fue necesario consultar artefactos de la Ola 53 porque los cuatro puntos se resuelven de forma explícita y consistente en el plan revisado.

## Resultado

No quedan defectos sustantivos nuevos en los cuatro ejes solicitados. El estado `AUDIT-REVISED` no sustituye la futura verificación de implementación ni el replay, pero el diseño documentado ya es ejecutable y conserva el alcance inferencial restringido que corresponde.

### 1. Gauge de `J` y cardinalidad — resuelto

R302 identificaba una redundancia de una dimensión entre la componente uniforme de las seis interacciones y los sesgos por cardinalidad (`302_wave54_plan_independent_audit.md:7-12`). El plan vigente la elimina al imponer `sum J_tilde_ij=0`, parametrizar sólo cinco contrastes libres y absorber explícitamente la componente uniforme en `c_k` (`waves/WAVE_54_JOINT_SET_POSTERIOR_PLAN.md:65-74`). La contabilidad es coherente: cuatro slopes unary, cinco contrastes y tres sesgos de cardinalidad, para doce parámetros identificados (`waves/WAVE_54_JOINT_SET_POSTERIOR_PLAN.md:69-74`). Además, el criterio comparativo ya atribuye el efecto únicamente a interacciones heterogéneas frente a `joint_unary_cardinality`, no a una norma cruda de `J` (`waves/WAVE_54_JOINT_SET_POSTERIOR_PLAN.md:140-142`).

### 2. Barrera física anti-leakage y binding/versionado — resuelto

La objeción previa era que un único contenedor podía exponer outcomes de monitor antes del freeze (`302_wave54_plan_independent_audit.md:14-15`). El plan ahora exige dos bundles físicamente separados, un preparador ciego limitado a una lista cerrada y un manifest de hashes (`waves/WAVE_54_JOINT_SET_POSTERIOR_PLAN.md:21-26`). Su orden vinculante es suficiente: versionar los componentes y hashes upstream; ejecutar el preparador; bindear manifest y bundles en la config; volver a versionar; y recién entonces permitir la corrida, que falla cerrada ante Git sucio o cualquier hash divergente (`waves/WAVE_54_JOINT_SET_POSTERIOR_PLAN.md:30-34`). Durante el análisis, el ajuste recibe sólo `fit_select_bundle`, escribe `selection_freeze.json` con el hash esperado de monitor y abre después el bundle sellado, dejando recibo de acceso (`waves/WAVE_54_JOINT_SET_POSTERIOR_PLAN.md:47-51`).

### 3. Soporte escaso y clases no observadas — resuelto

R302 pedía transformar el reporte de masa sobre clases ausentes en una condición operativa y limitar el claim (`302_wave54_plan_independent_audit.md:17-18`). El plan lo vuelve vinculante: en `decision_select`, la masa media asignada por `joint_full` a conjuntos ausentes de `calibration_fit` no puede superar la de `independent_raw` por más de `0.02`; el mismo diagnóstico se informa en monitor sin presentar las ausencias como evidencia positiva (`waves/WAVE_54_JOINT_SET_POSTERIOR_PLAN.md:149-152`). También declara el soporte observado de `10/15` y `9/15`, las clases unitarias, y restringe expresamente toda lectura a ese soporte histórico, sin extenderla a los cinco conjuntos ausentes de fit (`waves/WAVE_54_JOINT_SET_POSTERIOR_PLAN.md:156-165`).

### 4. Selección de `lambda` y población primaria — resuelto

R302 observó que el selector global no respondía a la población ambigua primaria (`302_wave54_plan_independent_audit.md:20-21`). La versión actual selecciona cada `lambda` por NLL de conjunto exacto en `decision_select` restringido a `NEAR_RIVAL` y cardinalidad compatible `>=2`; la selección sobre los 192 tokens completos queda explícitamente como sensibilidad secundaria (`waves/WAVE_54_JOINT_SET_POSTERIOR_PLAN.md:36-45`). La regla se repite en la sección de selección, con empate hacia mayor regularización y una salvaguarda que marca el resultado como `selector-sensitive` si el selector global altera el signo de un contraste primario en monitor (`waves/WAVE_54_JOINT_SET_POSTERIOR_PLAN.md:108-115`).

## Veredicto

La revisión incorpora los cuatro findings materiales de R302 sin introducir una contradicción nueva de identificabilidad, leakage, soporte o selección. No se elevan mejoras opcionales ni se emite GO/NO-GO experimental.

PASS
