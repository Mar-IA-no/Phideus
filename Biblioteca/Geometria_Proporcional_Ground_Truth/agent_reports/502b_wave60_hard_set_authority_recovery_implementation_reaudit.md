# R502 — Reauditoría final de implementación del recovery `hard_set_tau` de Ola 60

**Dictamen técnico: `PASS` — 0 HIGH / 0 MEDIUM / 0 LOW.**

La implementación congelada en
`7eba44b9622d7e6ecd90d0345feee32c00748833` resuelve R502-01. Tanto
`run_preparation_transaction()` como `execute_preparation()` reconstruyen la
autoridad canónica v4 antes de su primera mutación y comparan el contexto
recibido con el emitido por la validación completa. Los contextos fabricados se
rechazan sin crear la raíz transaccional o, para la entrada directa, sin cambiar
el árbol preexistente.

## Target y cadena Git

La cadena observada es exacta y lineal:

```text
R501                    2fe5f1f83dc266e8671ca5cc8dd1580ceef6b7a1
implementación inicial  2b1359bcc8bf689122dc342032c1f69f31ff2b61
R502 REVISE             00b433a688554d5f7d2c37caf87d32be02abc466
corrección 1            410a9189af5f0e87ce396aaa121ff593b7b5a322
corrección 2            e29069f68ea2fdf27e6a6fae2480a835144bc5ac
implementación final    7eba44b9622d7e6ecd90d0345feee32c00748833
```

Cada implementación/corrección modifica exclusivamente el preparador y el
test; R502 modifica exclusivamente su informe. El target final tiene parent
directo `e29069f68ea2fdf27e6a6fae2480a835144bc5ac` y `git diff --check`
queda limpio.

## Resolución de R502-01

`revalidate_wave60_hard_set_execution_context()` vuelve a localizar el
amendment y el source canónicos y llama a `validate_recovery_amendment()` antes
de aceptar el contexto. Esta precondición se ejecuta al inicio de ambos
entrypoints. La transacción repite además la resolución de la config efectiva
antes de `prepare_output()`; el executor directo lo hace antes de escribir
escrow, freeze, draw, inferencia o bundles. Las pruebas negativas alteran
`implementation_commit` y vacían `implementation_audit`; ambas entradas fallan
con el árbol byte-inmutable.

La resolución materializadora permanece separada de la config canónica:
manifest de source law → request → alias único → snapshot físico de Ola 59 →
SHA-256 `f6edfd2106fe87c8150562d096469e29b64a108a73de2dae0d371bd689a4a9b6`
→ valor finito exacto `0.5`. La copia efímera contiene `hard_set_tau=0.5`; la
config canónica no lo contiene. El seam real de `materialize_prepared_bundles()`
se ejecutó satisfactoriamente.

## Composición de autoridad futura

El mismo helper de keyset exige en amendment y preflight de config final las
nueve claves exactas del binding de implementación, incluida `resolution_of`
para v4. La cadena `resolution_of` liga el target inicial, el informe R502
`REVISE` y sus findings, las dos correcciones intermedias y el parent directo de
la implementación final. La auditoría futura debe ser un `PASS` R502 exclusivo,
con parent directo en el target presente.

La ruta de config final recompone esa misma cadena, toma los `old_sha256` del
mapa v3/R500, exige que preparer y test provengan del commit final y conserva
runner, módulo científico, worker, R475 y R476 contra el baseline v3. También
exige auditoría final R504 después de un commit exclusivo de config v4. El test
sintético construye físicamente el sufijo implementación-auditoría-amendment-
R503-config-R504 y confirma la aceptación.

Hay dos seams simulados declarados, ninguno oculta R502-01. El E2E de replay
v4 sustituye sólo la revalidación del namespace futuro y se rotula
`lineage-only`; primary conserva la revalidación canónica. El test sintético de
config R504 sustituye sólo la validación de la cadena Git R502 que no existe en
su repositorio temporal y comprueba que la composición la invoca con el binding
exacto. La cadena real se valida separadamente contra el repositorio físico y
los dos entrypoints fabricados se prueban sin sustituir su revalidación.

## Invariantes físicos y científicos

Los hashes físicos invariantes coinciden con la config v3:

```text
runner             b35cd563f715bdff9b6e7489ac04712c728673563898d4a6aebf0144d4a50261
módulo científico  46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65
worker             c6c5c832633d9ed8a99079dd4fc4f5677073f7e1883a204fa6bb41f6d646bac7
R475               e5c49ca16506469ac5099c2a3f1992819c1a54fed790f1638a82e93b9b1d9996
R476               497bb87f7e3677c4dae11a0281e0362fa4d1ef4d6ba60252f01cdc1f2c0a8a30
```

El delta v3→target sigue limitado a preparer y test. Sus hashes finales son,
respectivamente,
`19942cb81afda503f3ff975f6907183436cbff50f3d9ac419b5adc13a40b539a` y
`d249aba714c14584cdfae1ce242523def9dcc088256f6a3d68eb0d2c2f449f9e`.

La validación read-only recompuso el terminal v3, sus firmas y el inventario
closed-world: `44` records en primary y `8` en replay, ambos
`INVALID_PREPARATION`, pair `PAIR_ABORTED_PRE_TRUTH`, sin preparación durable ni
truth científico accedido. El access receipt conserva inferencia sin fit ni
labels y el probe denegado con `PermissionError`. El draw preservado permanece
byte-exacto; el E2E exige inodos nuevos. El ledger recompuesto es
`215.36700256168842 + 60.0 = 275.3670025616884 s`, con débito único y replay
encadenado al acumulado firmado de primary.

## Verificación CPU-only

Todas las ejecuciones usaron `CUDA_VISIBLE_DEVICES=''`,
`PYTHONDONTWRITEBYTECODE=1`, plugins externos de pytest deshabilitados y
`--basetemp` bajo `/mnt/m2-1TB` con parent modo `0755`. Los tres temporales
propios fueron retirados.

```text
focal R502/v4:       19 passed in 87.32s
                     max RSS 888,696 kB; swaps 0; exit 0
Wave 60 completo:   173 passed in 411.97s
                     max RSS 893,784 kB; swaps 0; exit 0
Waves 56–59:        336 passed, 1 skipped in 387.07s
                     max RSS 1,065,884 kB; swaps 0; exit 0
```

No se usó ni se consultó GPU. No se modificaron código, tests, config ni datos;
este informe es el único archivo creado. El dictamen no ejecuta v4, no promueve
arquitectura y no decide `GO/NO-GO`.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R502",
  "scope": "HARD_SET_AUTHORITY_RECOVERY_IMPLEMENTATION",
  "target": {
    "implementation_commit": "7eba44b9622d7e6ecd90d0345feee32c00748833"
  },
  "technical_verdict": "PASS",
  "findings": {
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "files_modified": false,
  "gpu_used_or_queried": false
}
```
