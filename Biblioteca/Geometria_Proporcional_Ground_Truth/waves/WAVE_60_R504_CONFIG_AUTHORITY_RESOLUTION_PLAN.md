# Ola 60 — plan de resolución de R504 antes de ejecutar el intento v4

> **Estado:** `PRE-IMPLEMENTATION / R504-REVISE / ATTEMPT-V4-ABSENT / NO-TRUTH / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **Finding de origen:** `R504-01`, HIGH
> **Config rechazada:** commit `201f9257c1e016b925719f6917dd1a9298ca493b`, SHA-256 `92bf02867579281f5df02b0cdd2c6980cbb8d1697f8df60a353457b86cea3960`
> **Auditoría R504:** commit `b70ee7e8f3d4f561a52017419aee0012914e2f54`, SHA-256 del informe `3dafae137041ddf183c5bd1d64aa6e7e728c1a89fcf59c623103f799fa01c6f6`

## 1. Estado físico y alcance

R504 ejecutó el preflight final después de congelar la primera config v4 y
publicar su auditoría como `HEAD`. El preflight falló antes de devolver un
contrato, antes del re-forward, antes de crear el namespace
`wave60_frozen_policy_transport_attempt_v4` y antes de abrir truth o scoring.
La config y el amendment no se reescriben: permanecen como artefactos exactos
de la candidata rechazada.

El finding no altera la pregunta científica, el draw, la source law, las
políticas HGB/HGB, los trece estados, las features, los thresholds, el
bootstrap, el penalty, los seeds ni el límite acumulativo de 900 segundos. La
corrección pertenece exclusivamente a la procedencia de sources durante el
preflight final.

## 2. Causa raíz

La recuperación estática v3 modificó legítimamente runner, preparer y test. La
config v3 auditada por R500 fijó el runner resultante con SHA-256:

```text
b35cd563f715bdff9b6e7489ac04712c728673563898d4a6aebf0144d4a50261
```

El recovery hard-set v4 sólo modifica preparer y test respecto de ese baseline;
por tanto declara runner, módulo científico y worker como invariantes v3→v4.
`validate_wave60_final_config_authority()` carga correctamente el mapa
`source_sha256` de la config v3, pero sólo lo usa para el `old_sha256` de los
sources que cambiaron. Para todo source fuera de la partición vuelve a
`implementation_binding.commit=9f1a229d...`, anterior a la recuperación
estática. Allí el runner tiene SHA-256 `1c778c3e...`, y el preflight rechaza el
blob v3 correcto.

El error consiste en mezclar dos autoridades temporales:

```text
implementación científica original  -> baseline de sources sin recovery
config v3 + R500                     -> baseline efectivo del recovery v4
```

Cuando existe un recovery tipado con config previa autenticada, los sources
invariantes deben validarse contra esa config previa. La implementación
científica original sigue gobernando las fuentes que no hayan sido sustituidas
en ningún eslabón, pero no puede borrar un delta intermedio ya auditado.

## 3. Corrección mínima

La implementación podrá modificar sólo:

```text
experiments/geometria_proporcional/prepare_wave56_fresh.py
tests/test_wave60_frozen_policy_transport.py
```

El preparador debe:

1. distinguir explícitamente entre `original implementation authority` y
   `authenticated prior-config authority`;
2. para recoveries estático y hard-set, validar cada source invariante contra
   `prior_config["source_sha256"]`;
3. conservar para los sources cambiados la comprobación acumulativa
   `old_sha256` desde la config previa y `new_sha256` desde el commit correctivo;
4. rechazar faltantes, extras o cruces entre las particiones changed/unchanged;
5. mantener la conducta histórica de los recoveries sin config previa tipada;
6. no relajar `require_sources_at_head()`, self-binding, parentage, auditorías ni
   comparación física de hashes.

La corrección no se formulará como una excepción exclusiva para el runner. La
regla válida es por procedencia: todo source declarado invariante hereda la
autoridad completa del baseline inmediatamente anterior.

## 4. Pruebas discriminantes

El test sintético de config final debe representar la historia que faltaba:

```text
implementación original
  -> recovery estático cambia runner
  -> config v3/R500 liga runner nuevo
  -> recovery hard-set cambia sólo preparer/test
  -> config v4 conserva runner v3
  -> R504 sintética
  -> validate_wave60_final_config_authority PASS
```

Debe agregarse al menos un ataque donde la config v4 intente volver al runner de
la implementación original; el validador debe rechazarlo aunque ese blob sea
históricamente válido. También debe conservarse un ataque donde un source
declarado invariante difiera del mapa v3.

El caso `test_hard_set_v4_source_baseline_is_v3_not_v1_escrow` depende hoy del
archivo canónico de `HEAD` para construir su supuesto prior v3. Como preparer y
test deberán adquirir hashes nuevos de todos modos, el costo incremental de
eliminar esa dependencia de fase es bajo: el fixture debe recuperar el blob v3
autenticado desde R500 o desde el snapshot sellado, nunca inferir su versión del
archivo canónico vigente. Esta corrección restaura una suite ejecutable después
del freeze sin cambiar el contrato productivo.

Pruebas obligatorias:

1. composición positiva del validador final con runner original ≠ runner v3;
2. rechazo de rollback del runner al blob pre-v3;
3. rechazo de deriva física o de config en cualquier source invariante;
4. delta acumulativo v3→implementación final limitado a preparer/test;
5. caso phase-bound reescrito con prior v3 explícito;
6. suite focal `hard_set_v4` completa después de congelar una config sintética;
7. suite Wave 60 completa y regresión Waves 56–59;
8. preflight real read-only contra una composición sintética completa;
9. medición de wall, RSS y swaps; GPU ausente y no consultada.

## 5. Autoridad suplementaria

La implementación ya aceptada por R502 y el amendment/R503 permanecen
inalterados. La corrección no puede injertarse en la config rechazada porque
cambiará los hashes ligados de preparer/test. Se publicará un amendment
suplementario, bajo el mismo recovery hard-set y el mismo intento v4, que
reconstruya toda la autoridad previa y agregue una resolución cerrada de R504.

El amendment suplementario debe ligar como mínimo:

- amendment inicial `3a21c039...` y R503 `2b98cfa8...`;
- config rechazada `201f9257...` y R504 REVISE `b70ee7e8...`;
- este plan y su auditoría independiente;
- implementación correctiva y su auditoría independiente;
- mapa acumulativo `prior_source_sha256` de v3;
- partición final donde cambian preparer/test y los otros cinco sources no
  autorreferentes conservan exactamente el baseline v3;
- el mismo terminal v3, draw, contrato hard-set, población y presupuesto.

La nueva config seguirá usando el namespace v4 porque ese namespace nunca fue
creado. Debe ser un nuevo commit exclusivo, no una enmienda del commit
`201f9257...`; su auditoría final debe ejecutar el preflight real después de
publicarse como `HEAD`.

## 6. Cadena prevista

```text
R504 REVISE
  -> este plan
  -> R505 auditoría independiente del plan
  -> implementación mínima preparer/test
  -> R506 auditoría independiente de implementación
  -> amendment suplementario v4
  -> R507 auditoría independiente del amendment
  -> nueva config v4
  -> R508 auditoría independiente + preflight final
  -> preparación y ejecución v4
  -> R509 auditoría independiente de resultado o terminal
```

Cada eslabón documental será un commit exclusivo y cada transición tendrá
parent directo. Un informe REVISE se conserva; no se reemplaza por prosa de
PASS. Ninguna auditoría decide promoción arquitectónica ni `GO/NO-GO`.

## 7. Criterio de continuidad

Sólo se implementará si R505 confirma que la regla de autoridad usa el baseline
v3 sin debilitar la trazabilidad de los blobs originales y que el amendment
suplementario puede cerrar el cambio sin reescribir artefactos previos. Sólo se
ejecutará si R508 obtiene PASS y el preflight final real retorna un contrato
contra el `HEAD` auditado.

Si el preflight vuelve a fallar, se preservará el nuevo terminal documental sin
crear el namespace. Si la preparación abre truth de scoring/evaluación y luego
falla, no habrá otra corrida automática. La decisión experimental seguirá
perteneciendo al usuario.
