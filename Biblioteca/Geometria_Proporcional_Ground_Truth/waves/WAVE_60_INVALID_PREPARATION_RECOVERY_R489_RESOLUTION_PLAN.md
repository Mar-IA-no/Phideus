# Ola 60 — resolución de la precondición blob/físico de R489

> **Estado:** `PRE-CORRECTION / PRE-AMENDMENT / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **Implementación auditada:**
> `fb0248f0430c640099f97e7e7c69eedc271fc9d3`
> **Auditoría R489:** commit
> `ae9f0a5fe0e16c7eb9346a5b6fc40513872ec6c0`, informe
> `489_wave60_invalid_preparation_recovery_implementation_acceptance_audit.md`,
> SHA-256
> `6bc09bfeeca1ddae7c6911ea25cf0fcd29ff54fd237782002c192b976a01f6c5`
> **Dictamen:** `REVISE / 0 HIGH + 1 MEDIUM + 0 LOW`

## 1. Finding y alcance

R489 confirmó la implementación productiva del lineage, la frontera científica,
la provenance firmada, las tres ramas negativas de sources, los deltas cruzados
y las matrices de parent y exclusividad. También reprodujo en CPU las suites
Wave 60 y Waves 56–59 sin fallos ni swaps del proceso.

El único finding es probatorio. Los 45 reportes alternativos de la matriz
semántica comprueban diferencia de una sola clave, path exclusivo, parent
directo y `sha256(archivo físico) == binding.sha256`, pero no afirman en cada
caso:

```text
git_blob_sha256(bad_commit, report_path)
  == sha256(archivo físico)
  == binding.sha256
```

Como el bloque termina con un `pytest.raises(RuntimeError)` genérico, una
divergencia accidental anterior al parser podría satisfacer la prueba. La
construcción vigente hace probable la igualdad, pero el contrato R486 exige
evidencia explícita. No se observó un bypass productivo ni se autoriza cambiar
la source law, el draw, el modelo, los thresholds, el roster o el estimando.

## 2. Corrección probatoria

Dentro del loop común a las auditorías semánticas, inmediatamente antes de
invocar el validador, se calcularán y compararán los tres valores:

```python
physical_sha256 = file_sha256(repo / relative)
blob_sha256 = preparer.git_blob_sha256(repo, bad_commit, relative)
assert blob_sha256 == physical_sha256 == case[binding][sha_field]
```

Permanecen además las precondiciones ya presentes:

- el commit malo cambia exclusivamente el reporte esperado;
- es hijo directo del parent correcto;
- sólo el campo semántico bajo prueba difiere de la autoridad positiva;
- el path externo del binding permanece canónico.

No se sustituye el negativo por un mock, una mutación sólo en memoria o el test
separado de divergencia blob/físico. La corrección debe ejecutarse para las
once auditorías de la cadena ampliada y sus cinco campos, es decir, 55 casos
semánticos materializados.

## 3. Historia que debe preservarse

El commit `fb0248f` queda como implementación probatoriamente incompleta y R489
permanece `REVISE 0/1/0`. No pueden ocupar el objeto positivo
`recovery_implementation`. A las 35 claves fijadas por el plan R487 se añaden
exactamente:

```text
r487_resolution_implementation
r487_resolution_implementation_audit
r489_resolution_plan
r489_resolution_plan_audit
```

El keyset top-level final de la amendment tendrá 39 claves. Los objetos se
definen así:

- `r487_resolution_implementation`: commit `fb0248f`, parent R488 y deltas de
  preparer/test calculados contra R475;
- `r487_resolution_implementation_audit`: commit, path, SHA, audit ID, scope,
  verdict y findings de R489 `REVISE 0/1/0`;
- `r489_resolution_plan`: commit, path y SHA de este documento;
- `r489_resolution_plan_audit`: commit, path, SHA, audit ID, verdict y findings
  de R490 `PASS 0/0/0`.

Los hashes de `fb0248f` son:

```text
preparer
  old = 7d7ead44f6d0e64802dafa585a59a20ae78f43f5e975e03c60e6bd8a1de33d66
  new = 0b52f06012c92feef02bf7a93892bb35c34c7196574d94b465c3796cbb6e16a4

test
  old = 328c934c63f2cb402633b72b52699e2d48433ff7e94966169a5b1428e6f63519
  new = f1d3d9668d5d09bd5fa87618fb7db42d7ac49550cf763f820d39d345cdf7f80a
```

## 4. Sufijo sintético ampliado

La positiva tendrá once auditorías consecutivas:

```text
R481 REVISE 0/1/0
R482 PASS   0/0/0
R483 REVISE 0/1/0
R484 PASS   0/0/0
R485 REVISE 0/1/0
R486 REVISE 0/2/0
R487 REVISE 0/1/0
R488 PASS   0/0/0
R489 REVISE 0/1/0
R490 PASS   0/0/0
R491 PASS   0/0/0
```

R490 audita este plan. R491 audita la implementación sucesora. Para R489 y
R490 se añaden los mismos negativos aislados que para las autoridades
anteriores: cinco campos semánticos internos, tres bindings externos, parent
saltado y un único path extra. La matriz de etapas exclusivas pasa de 18 a 22.
Las tres ramas científicas secundarias reconstruyen también esos cuatro
eslabones antes de alcanzar la implementación final.

Los negativos de deltas `old_sha256`, `new_sha256`, path y cruces incorporan
`fb0248f` como cuarta implementación rechazada. El sucesor aceptado será la
quinta implementación y seguirá cambiando sólo preparer/test.

## 5. Provenance, attestation y autoridades futuras

La positiva de transaction/provenance usará exclusivamente:

```text
implementation_audit.audit_id = R491
```

R489 no puede aparecer como autoridad positiva. La provenance completa seguirá
siendo idéntica en generation receipt, preparation freeze y preparation
receipt. La attestation Ed25519 continuará firmando transitivamente el receipt
por path, bytes y SHA-256 exactos, sin cambiar schema.

La secuencia futura queda:

```text
R489 REVISE
  -> este plan
  -> R490 PASS
  -> implementación sucesora
  -> R491 PASS
  -> amendment
  -> R492 PASS
  -> config v2
  -> R493 PASS / HEAD de ejecución
  -> R494 auditoría de resultados
```

Los paths futuros serán:

```text
Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/
  490_wave60_invalid_preparation_recovery_r489_resolution_plan_audit.md
  491_wave60_invalid_preparation_recovery_implementation_acceptance_audit.md
  492_wave60_invalid_preparation_recovery_amendment_audit.md
  493_wave60_frozen_policy_transport_v2_config_audit.md
  494_wave60_frozen_policy_transport_result_audit.md
```

El schema de amendment permanece
`wave60-invalid-preparation-recovery-amendment-v1`; cambia la historia, no la
semántica de recuperación. `attempt.recovery` conserva exactamente sus doce
claves. La config canónica no incorpora `hard_set_tau`; el valor `0.5` continúa
inyectándose sólo en la copia efímera del materializador.

## 6. Frontera científica y ejecución requerida

Módulo, runner y worker permanecen bajo R475 y deben conservar exactamente:

```text
46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65
1c778c3e60c1bbcebeb5c83430601a7c0b148e447528195f1dec4296322825aa
c6c5c832633d9ed8a99079dd4fc4f5677073f7e1883a204fa6bb41f6d646bac7
```

La implementación sucesora sólo puede modificar:

```text
experiments/geometria_proporcional/prepare_wave56_fresh.py
tests/test_wave60_frozen_policy_transport.py
```

R491 deberá leer el cambio completo y ejecutar en CPU la suite Wave 60 y la
regresión explícita Waves 56–59, registrando duración, RSS y swaps. Un nuevo
finding se preservará con su verdict real; no se promoverá por conveniencia.

## 7. Autorización

Este documento requiere auditoría independiente R490 con scope
`INVALID_PREPARATION_RECOVERY_R489_RESOLUTION_PLAN`, target exacto
`plan_commit + plan_sha256`, PASS y `0/0/0` antes de implementar.

No se publica amendment ni se ejecuta el draw hasta encadenar implementación,
R491 PASS, amendment, R492 PASS, config v2 y R493 PASS como HEAD limpio. Todo
continúa CPU-only. Esta cadena no promueve arquitectura ni decide `GO/NO-GO`
científico.
