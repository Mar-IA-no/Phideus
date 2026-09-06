# Ola 60 — recuperación del intento abortado durante preparación

> **Estado:** `DRAFT / PRE-IMPLEMENTATION / PRE-RECOVERY / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **Origen:** `wave60_frozen_policy_transport_attempt_v1`
> **Terminal firmado:** `PAIR_ABORTED_PRE_TRUTH`
> **Pregunta científica inalterada:** ¿la ley HGB/HGB congelada de Ola 59
> transporta a una realización independiente sin refit, recalibración ni
> selección?

## 1. Hecho que obliga a recuperar

El preflight final de Ola 60 pasó sobre el commit auditado
`2a10b6cb5a88fd2af4ca2f5f8230a296f59c8948`: recompuso exactamente las tres
seeds históricas sobre 384 pair tokens, validó ocho fuentes de ejecución y nueve
bindings upstream, y confirmó que el namespace del intento no existía. La
inicialización atómica creó después el intento v1 y la preparación primaria
generó el benchmark, el escrow, el freeze público y los logits ciegos.

La transacción falló al materializar los bundles preparados:

```text
KeyError: 'hard_set_tau'
```

El constructor heredado de Ola 59 usa `hard_set_tau` para derivar la referencia
hard y el dataset de evaluación. La config de Ola 60 fijó `penalty=1.25`, pero
omitió el threshold hard `0.5`; el validador aceptó esa omisión y las pruebas
integradas sustituyeron el materializador real por un doble. No hubo scoring de
la ley transportada, apertura del lockbox ni evaluación.

El intento fue cerrado mediante el propio lifecycle de Ola 60. La autoridad
física vigente es:

| Evidencia | Valor |
|---|---|
| pair terminal | `PAIR_ABORTED_PRE_TRUTH` |
| truth accedida | `false` |
| recovery permitida | `true` |
| `pair/FAILURE.json` | `05ede417b1f856488c1796210029aa74c74211f3f14858764d9705cbb0b3563d` |
| `pair/pair_status.json` | `922f227c9f74ec0b917533d58205a0ebdb7147bebdee111c37d5340973b3ee78` |
| escrow preservado | `930c9732b2d18a22ee8654b5478202abf51fa291579e459a34d244d46e0074a3` |
| pre-generation freeze | `88992f10ee270ff92f1ed286c1feb9418eb277fa2a11f886580d1ff52d9c4ebc` |
| benchmark manifest | `de43c7ebbfe0d3c3cc1d5f8d62bf04d094b68f835ece16e97fadd8708238b9f2` |
| preparation error | `7dee40f8c6ffbf9d6d317d4992c90514b2f3248a2274ce7f63a49e9ffa0d1547` |

El draw canónico vive bajo
`primary/failed_preparation/`. El árbol v1 completo, incluidas sus firmas e
inventarios, permanece inmutable.

## 2. Diagnóstico de alcance

Hay tres defectos relacionados, no uno:

1. **Contrato incompleto.** Ola 60 no declaró ni validó el `hard_set_tau=0.5`
   que exige el materializador utilizado por su propia preparación.
2. **Cobertura integrada insuficiente.** La suite probó el lifecycle con un
   materializador sustituido, pero no una preparación Wave 60 positiva con el
   materializador real y la config canónica.
3. **Recuperación inconsistente con su terminal.** `INVALID_PREPARATION` se
   publica como recuperable, pero la implementación vigente espera escrow,
   benchmark y ledgers de preparación en la raíz exitosa. En este terminal los
   artefactos están correctamente archivados dentro de `failed_preparation/` y
   todavía no existe un `preparation_receipt.json` firmado. Por tanto, el camino
   anunciado no puede consumir el mismo draw sin una corrección explícita.

No hay evidencia de defecto en el generador, los checkpoints, los re-forward,
la source law v2, los workers de score/evaluate ni la ley HGB congelada.

## 3. Invariantes de la recuperación

La recuperación v2 debe conservar simultáneamente:

1. el mismo escrow y las mismas tres claves;
2. el benchmark completo byte a byte, con inodos nuevos y sin hardlinks;
3. la misma source law v2, roster, estados, features y thresholds HGB;
4. la misma pregunta, estimandos, bootstrap, penalty, seeds, splits y límites;
5. cero refit, cero recalibración, cero selección post-draw y cero GPU;
6. un namespace nuevo:
   `wave60_frozen_policy_transport_attempt_v2`;
7. el v1 terminal intacto y revalidado antes, durante y después de copiar el
   draw;
8. una cadena Git lineal con auditorías independientes antes de ejecutar.

La única explicitación científica admitida es `hard_set_tau=0.5`. No es un
threshold elegido mirando el draw: es el valor congelado en la config fuente de
Ola 59 y el que define la referencia hard que Wave 60 ya se propuso transportar.

## 4. Presupuesto conservador del intento fallido

La corrida fallida terminó antes de publicar un ledger de preparación firmado.
El envoltorio externo observó 48,51 segundos, RSS máximo de 1.086.260 KiB y cero
swaps atribuibles al proceso, pero ese registro vive en el transcript y no debe
presentarse como artefacto autenticado del intento.

La recuperación debitará **60 segundos completos** del límite acumulado de 900
segundos. Es un redondeo conservador por encima del tiempo observado. El valor
se declarará en la amendment, se copiará a la autoridad de recuperación de la
config y será validado antes de extraer claves. No se reconstruirá una precisión
ficticia ni se asignará RSS firmado donde no existe.

## 5. Diseño de implementación

### 5.1 Config tipada y compatibilidad histórica

- La config v1 histórica seguirá validando exactamente como artefacto legado.
- Toda config recuperada `attempt.version > 1` deberá declarar
  `hard_set_tau: 0.5`.
- La autoridad `attempt.recovery` incorporará el débito conservador y los
  bindings del commit/auditoría de la corrección.
- El delta v1→v2 permitirá únicamente: namespace, autoridad de recuperación,
  auditoría final, self-binding de config, `hard_set_tau=0.5` y hashes de las
  fuentes de implementación expresamente auditadas. El resto del contrato
  científico deberá ser idéntico.

### 5.2 Origen físico polimórfico, no ambiguo

El validador distinguirá por terminal firmado:

- fallas posteriores a `PREPARED`: draw en la raíz primaria;
- `INVALID_PREPARATION`: draw en
  `primary/failed_preparation/`.

No se aceptará descubrimiento heurístico. La selección depende del
`pair_status`, del `FAILURE.json` de la raíz y de sus attestations. El mapa
`preserved_draw_sha256` seguirá siendo closed-world respecto del draw elegido:
escrow, freeze público, manifest y todos los archivos declarados por el
manifest.

### 5.3 Reuso y copia

La CLI de recuperación recibirá la raíz primaria v1 como autoridad. Sólo después
de validar amendment, terminal, firmas e inventario resolverá internamente el
draw archivado. Extraerá las claves desde ese escrow y copiará el benchmark con
bytes y modos idénticos, pero inodos distintos. La réplica v2 reutilizará como
siempre la primaria v2 ya preparada.

### 5.4 Ledger acumulado

Para `INVALID_PREPARATION`, el ledger heredado comenzará en 60 segundos. Para
terminales con `preparation_receipt` firmado seguirá usando la suma exacta ya
implementada. Se rechazará mezclar ambos regímenes, asignar débito cero al caso
sin ledger o reutilizar el débito conservador sobre una preparación firmada.

### 5.5 Superficie mínima de cambio

La corrección puede tocar solamente:

- `src/geometria_proporcional/wave60_frozen_policy_transport.py`;
- `experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py`;
- `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
- `tests/test_wave60_frozen_policy_transport.py`.

No se modifica `_wave60_phase_worker.py`, el materializador Wave 59, la source
law v2 ni los artefactos v1.

## 6. Pruebas obligatorias antes de una amendment ejecutable

1. regresión que demuestre que la config v1 histórica sigue validando;
2. rechazo de una recovery config sin `hard_set_tau`, con otro valor o con
   campos no declarados;
3. preparación positiva Wave 60 usando el materializador real, no un mock;
4. recuperación desde `failed_preparation/` que reproduzca escrow y benchmark
   byte a byte con inodos distintos;
5. rechazo si cambia un byte, modo, owner, symlink, hardlink, inventario, firma
   o hash del v1;
6. débito inicial exacto de 60 segundos y rechazo de cero, doble débito o mezcla
   con ledgers firmados;
7. preservación de las denegaciones de acceso y de `truth_accessed=false`
   durante preparación/source binding/score;
8. primaria y replay recuperadas exactas, sin redibujar claves ni benchmark;
9. regresión focal Wave 60 y regresión amplia Wave 56–60 en CPU, con RSS y swap
   registrados.

## 7. Cadena de autoridad prevista

1. commit exclusivo de este plan;
2. auditoría independiente del plan (`R478`);
3. commit exclusivo de la implementación corregida;
4. auditoría independiente de implementación (`R479`);
5. amendment JSON derivada del v1 terminal y del inventario físico;
6. auditoría independiente de amendment (`R480`);
7. config v2 exclusiva, con self-binding y fuentes actualizadas;
8. auditoría independiente de config (`R481`);
9. preflight final, inicialización v2, recuperación primaria, replay y ejecución
   del par;
10. auditoría independiente de resultados (`R482`) antes de integración
    documental.

Una auditoría `PASS` sólo autoriza el siguiente eslabón; no constituye decisión
científica ni `GO/NO-GO`.

## 8. Criterio de continuidad

Si la recuperación falla antes de truth, se conserva el nuevo terminal y se
evalúa otra recuperación versionada bajo el presupuesto restante. Si falla
después de truth, no se permite otro intento. Si v2 completa, el resultado se
interpreta como transporte sobre el draw original v1, no como un segundo draw.
