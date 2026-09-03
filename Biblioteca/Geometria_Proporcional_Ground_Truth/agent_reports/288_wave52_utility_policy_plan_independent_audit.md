# Auditoría independiente del plan de Ola 52

> **Auditor:** instancia independiente `Boyle`
> **Fecha:** 2026-09-03
> **Objeto:** `WAVE_52_UTILITY_CONDITIONED_POLICY_TRANSPORT_PLAN.md`, versión draft
> **Veredicto:** `REVISE`

## Findings

1. **Crítico: la comparación principal no aísla la factorización.**
   `direct_contextual_choice` debe aprender la regla de decisión, mientras
   `factored_set_then_utility` recibe explícitamente el algoritmo correcto
   `argmax` sobre el conjunto predicho. Una victoria puede deberse al
   conocimiento hard-coded de la política o a equivariancia perfecta, no a una
   representación set-valued superior. Corrección: definir el estimando como
   ventaja ingenieril de composición conocida, o agregar un baseline directo
   permutation-equivariant con igual conocimiento de la regla. Para aislar el
   reader, comparar ambos sobre la misma representación congelada y con
   idénticos datos de política.

2. **Crítico: el regret no está matemáticamente definido para acciones
   incompatibles.** Si el modelo elige una familia fuera del conjunto
   compatible con utilidad mayor que el máximo compatible, el regret
   convencional sería negativo y premiaría una acción inválida. Tampoco se
   define el denominador. Corrección: usar pérdida lexicográfica
   factibilidad-utilidad, o asignar a toda incompatibilidad una penalidad fija
   mayor que el peor regret compatible; congelar fórmula y rango antes del run.

3. **Alto: los controles pueden ser triviales aunque el modelo use utilidad.**
   Barajar vectores entre tokens no garantiza que cambie el vector ni la acción;
   los singletons nunca cambian. Corrección: definir una población primaria
   policy-sensitive con cardinalidad al menos dos y construir contrafactuales
   que garanticen cambio de ganador. Reportar también la fracción de
   oportunidades decisionales sobre toda `NEAR_RIVAL`.

4. **Alto: `utility_shuffled` no aísla uso contextual.** No se especifica si el
   shuffle ocurre durante entrenamiento o sólo en inferencia. Entrenar con
   labels incompatibles mide learnability bajo ruido, no dependencia causal.
   Corrección: intervenir la utilidad en evaluación sobre el mismo checkpoint y
   los mismos tokens; mantener separado cualquier control de entrenamiento.

5. **Alto: igualar parámetros y backprops no produce una comparación matched.**
   Los brazos reciben BCE, CE o BCE+CE; las capas inertes no igualan capacidad
   funcional ni gradientes. Repetir cada token bajo políticas también puede
   multiplicar artificialmente la BCE del set. Corrección: congelar
   exposiciones por token, normalización de pérdidas, optimizer steps,
   parámetros entrenables y FLOPs; tratar supervisiones distintas como
   comparaciones de sistemas, no atribución causal a la factorización.

6. **Alto: el umbral del set puede incorporar la política indirectamente.**
   Seleccionarlo con `policy_val` contamina la independencia si el objetivo usa
   acción o regret. Tampoco se explicita la disjunción por hash entre
   `val_threshold` y `val_monitor`. Corrección: calibrar threshold sólo con
   métricas set-valued y sin utilidades; reservar `policy_val` para readers.

7. **Alto: la unidad inferencial cubre tokens, no transporte entre políticas ni
   variabilidad de entrenamiento.** Bootstrap por token es correcto para un
   ensemble fijo, pero condiciona sobre pocas órdenes y oculta tres seeds.
   Corrección: limitar la inferencia, reportar por seed, repetir particiones de
   política o modelar políticas como segundo eje. Verificar si existe una unidad
   generativa superior a `pair_token`.

8. **Medio: el criterio permite selección post hoc entre dos readers.** El `OR`
   entre el reader hard y `score_composition` permite escoger el ganador con IC
   sin ajuste. Corrección: fijar un reader primario y dejar el otro exploratorio,
   o ajustar multiplicidad; un mismo reader debe cumplir todas las condiciones.

9. **Medio: el “sigmoid matched” no está definido.** No aparece como brazo
   formal. Si comparte checkpoint con los readers, conservar sus métricas es un
   chequeo de integridad; si es histórico, no está matched. Corrección: declarar
   checkpoint, loss, sampling y seeds exactos.

10. **Medio: el experimento prueba órdenes ordinales, no utilidad general.**
    Todas las políticas permutan cuatro niveles fijos. Además, grupos de seis
    no pueden balancear exactamente cuatro rangos por familia. Corrección:
    denominarlo transporte de ranking ordinal y usar una partición `8/8/8`, o
    variar gaps y escalas para claims cardinales.

## Cierre

El plan protege correctamente el lockbox y evita inflar `n` con réplicas
contextuales, pero los findings críticos y altos impiden atribuir un resultado
positivo al mecanismo propuesto. `REVISE`.
