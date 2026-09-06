# R502 — Auditoría de implementación del recovery de autoridad `hard_set_tau` de Ola 60

**Dictamen técnico: `REVISE` — 0 HIGH / 1 MEDIUM / 0 LOW.**

La implementación cierra correctamente el camino de CLI, autentica el intento
terminal v3, preserva el draw, resuelve `hard_set_tau=0.5` desde la autoridad
física y mantiene el ledger acumulativo. Sin embargo, la autoridad validada no
es transitiva a las dos entradas programáticas que efectivamente materializan y
firman la preparación. Un caller interno puede construir un `recovery_context`
con la forma esperada, omitir la autoridad de implementación y amendment, y aun
así producir una preparación estructuralmente válida. El propio E2E nuevo
ejercita y acepta ese recorrido.

## Target y alcance

El target auditado es exactamente:

- commit de implementación:
  `2b1359bcc8bf689122dc342032c1f69f31ff2b61`;
- parent directo:
  `2fe5f1f83dc266e8671ca5cc8dd1580ceef6b7a1`;
- paths modificados: sólo
  `experiments/geometria_proporcional/prepare_wave56_fresh.py` y
  `tests/test_wave60_frozen_policy_transport.py`;
- delta: `+1201/-177` en el preparer y `+913/-89` en el test;
- SHA-256 físicos respectivos:
  `ee84023b957618ae80cc24e9a5c8fe7a3379a57c9361ab71b6ea7b58a488cb3d`
  y `556b0efaae71db062f7cebae0945e05454900d4010f1d2a4b13e87f6cf60a462`.

El commit es hijo directo exclusivo de R501 y `git diff --check` queda limpio.
La auditoría leyó la implementación y sus tests, el plan aprobado, R498–R501,
la config v3, el amendment v3 y las superficies de validación y ejecución
pertinentes. No modificó código, tests, config ni datos.

## Finding R502-01 — MEDIUM: la autoridad validada puede eludirse por las entradas programáticas

El camino de CLI sí compone la autoridad completa: `main()` llama a
`validate_recovery_amendment()` antes de entrar en la transacción
(`prepare_wave56_fresh.py:9144-9206`). La garantía se pierde, no obstante, en
la frontera pública que recibe el resultado de ese preflight:

1. `run_preparation_transaction()` sólo llama a
   `resolve_wave60_materializer_config()` antes de mutar el output y después
   delega en `execute_preparation()` (`prepare_wave56_fresh.py:8636-8671`).
2. El resolver comprueba que el contexto contenga un amendment con schema v2 o
   v4 y vuelve a autenticar únicamente la cadena física del hard-set hasta
   `tau=0.5`; no demuestra que el contexto provenga de
   `_validate_wave60_hard_set_authority_recovery_amendment()`
   (`prepare_wave56_fresh.py:8004-8036`).
3. `execute_preparation()` llama a
   `revalidate_authorized_recovery_origin()`
   (`prepare_wave56_fresh.py:8340-8369`). En la rama v4, esa revalidación
   recarga la config previa, autentica el terminal v3 y el ledger v2, pero no
   revalida el amendment completo, R501/R502, el delta de sources ni la
   autoridad de implementación (`prepare_wave56_fresh.py:7497-7558`).

La brecha está demostrada por el test E2E agregado, no sólo inferida. Su helper
construye un amendment v4 con `recovery_implementation: {}`
(`test_wave60_frozen_policy_transport.py:211-317`). Luego fabrica directamente
un contexto con `implementation_commit` falso e `implementation_audit: {}`
(`test_wave60_frozen_policy_transport.py:4873-4891`), lo entrega a
`run_preparation_transaction()` y finaliza la autoridad firmada tanto para
primary como para replay (`test_wave60_frozen_policy_transport.py:4953-4977` y
`5007-5035`). Ese test pasa. La prueba separada de composición completa no
compensa esta cobertura porque sustituye
`validate_wave60_hard_set_recovery_authority` por un stub
(`test_wave60_frozen_policy_transport.py:2312-2328`).

### Impacto

No es una vulnerabilidad del comando operativo normal: la CLI obliga a pasar
por la validación completa. Sí es una ruptura fail-closed de la API interna y
del seam que los tests usan como camino E2E. Cualquier caller programático que
importe el preparer puede materializar, atestar y firmar un paquete de recovery
sin acreditar R502 ni la cadena completa de amendment/source delta. El runner
posterior comprueba presencia, hashes y firma de los records, pero no recompone
semánticamente la procedencia que originó el `recovery_context`
(`run_wave60_frozen_policy_transport.py:1220-1377`). Por alcance y necesidad de
un caller interno se clasifica MEDIUM, no HIGH.

### Corrección requerida

Hacer que la autoridad completa sea una precondición verificable de
`run_preparation_transaction()`/`execute_preparation()`, antes de cualquier
mutación. Dos soluciones compatibles con el diseño son:

- revalidar allí el amendment v4 completo a partir de paths y bindings
  canónicos; o
- reemplazar el diccionario libre por un capability/token opaco que sólo pueda
  emitir el validador completo y cuya integridad se compruebe en la frontera de
  ejecución.

Agregar tests negativos que llamen directamente a ambas entradas con el
contexto fabricado actual y exijan rechazo antes de `prepare_output()`. El E2E
positivo debe obtener su contexto ejecutando el validador completo, no
ensamblándolo a mano. El costo estimado es bajo a moderado y no requiere cambiar
la pregunta científica, el draw, el namespace ni el presupuesto.

## Controles que sí quedan satisfechos

Fuera de R502-01, la implementación observada concuerda con el plan:

- autentica la cadena manifest → request → alias único → snapshot de Ola 59 y
  obtiene exclusivamente `hard_set_tau=0.5`;
- conserva la config canónica sin `hard_set_tau` y lo incorpora sólo en una
  copia efímera para el materializador;
- valida firmas, terminales e inventarios v3, incluidos 44 records de primary,
  8 de replay, ausencia de preparados durables y el hash exacto del `KeyError`;
- reutiliza el escrow y copia el draw con igualdad de bytes e inodos nuevos;
- aplica una sola vez el débito conservador de 60 s sobre el acumulado v2 de
  `215.36700256168842 s`, dejando `275.3670025616884 s` como inicio primary;
- mantiene runner, módulo científico, worker, R475 y R476 como sources
  invariantes, y limita el delta de implementación a preparer y test;
- rechaza la ausencia ordinaria de autoridad hard-set antes de mutar el output,
  además de cubrir force, redraw, namespace, inventario y replay.

Estas observaciones no promueven una arquitectura ni constituyen un
`GO/NO-GO`. El finding impide otorgar PASS a la implementación hasta que la
misma cadena de autoridad gobierne también los entrypoints programáticos.

## Verificación CPU-only

La regresión prescrita se ejecutó con `CUDA_VISIBLE_DEVICES=''`,
`PYTHONDONTWRITEBYTECODE=1` y plugins externos de pytest deshabilitados, usando
un `--basetemp` bajo `/mnt/m2-1TB` con parent accesible para los casos que bajan
privilegios. Resultado:

```text
504 passed, 1 skipped in 789.03s (0:13:09)
elapsed /usr/bin/time: 13:10.29
maximum resident set size: 1,065,332 kB
swaps: 0
exit status: 0
```

La suite incluyó `test_wave60_frozen_policy_transport.py` y la regresión Waves
56–59 solicitada. El basetemp fue retirado después del cierre exitoso. No se
usó ni se consultó GPU.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R502",
  "scope": "HARD_SET_AUTHORITY_RECOVERY_IMPLEMENTATION",
  "target": {
    "implementation_commit": "2b1359bcc8bf689122dc342032c1f69f31ff2b61"
  },
  "technical_verdict": "REVISE",
  "findings": {
    "high": 0,
    "medium": 1,
    "low": 0
  },
  "files_modified": false,
  "gpu_used_or_queried": false
}
```
