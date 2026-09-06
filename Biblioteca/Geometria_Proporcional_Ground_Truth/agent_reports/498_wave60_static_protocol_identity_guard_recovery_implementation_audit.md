# R498 — Auditoría de implementación del recovery del guard estático de Ola 60

**Dictamen técnico: `PASS` — 0 HIGH / 0 MEDIUM / 0 LOW.**

La implementación corrige de forma cerrada el falso positivo que abortó el
intento v2. La única excepción nueva permite que
`benchmark/protocol_config.json` coincida por bytes con un antecedente; siguen
siendo obligatorias la concordancia con el manifest, la igualdad byte-exacta
entre primaria y replay y la separación de inodos frente a replay y a todos los
antecedentes. Ningún otro archivo queda exceptuado.

## Target y superficie

El target auditado es exactamente:

- commit: `24fbb4a3dfd43e9c7eb9293921f068636c190900`;
- parent directo R497:
  `ff708b0d6d7a8dfe00fee7473b9238867325bbf5`;
- paths modificados:
  - `experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py`;
  - `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
  - `tests/test_wave60_frozen_policy_transport.py`.

No hay ningún cuarto path y `git diff --check` queda limpio. Los hashes del
blob target coinciden con los archivos físicos:

- runner:
  `b35cd563f715bdff9b6e7489ac04712c728673563898d4a6aebf0144d4a50261`;
- preparer:
  `85069cd9e889e8dd5124ed0107c9e8fba69ab79a5edf52f3e25fee9e34ea4999`;
- test:
  `ca344305754669d15a86456e148a22d45af8e9d0f79ada0e98fd7a37034e2bef`.

Los dos sources declarados invariantes permanecen byte-exactos respecto del
parent, del target y del filesystem:

- módulo científico:
  `46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65`;
- worker de fase:
  `c6c5c832633d9ed8a99079dd4fc4f5677073f7e1883a204fa6bb41f6d646bac7`.

Los objetos anteriores `af3323d…` y `8ecc792…` no forman parte del target ni
de su lineage autoritativo.

## Corrección del guard

El runner añade una allowlist inmutable de un solo path y la consulta
exclusivamente al clasificar colisiones de bytes con antecedentes. No altera
la comparación de commitments, el dominio de archivos, los SHA-256 entre
primaria y replay ni ninguna de las comprobaciones de dispositivo e inodo.
`opaque_draw_fingerprint()` continúa autenticando para cada archivo declarado
por el manifest su path regular, hash y tamaño.

Los tests positivos y adversariales confirman:

- aceptación de protocolo estático byte-idéntico cuando el resto del draw es
  nuevo;
- rechazo de cualquier colisión byte-exacta no estática;
- rechazo de hardlinks del protocolo contra antecedente o replay;
- rechazo de drift protocolo/manifest y de desigualdad primaria/replay.

## Recovery v2→v3 y autoridad

El preparer incorpora un schema propio para el recovery estático y autentica
la cadena plan inicial → R496 `REVISE` → plan corregido → R497 `PASS`. La
implementación futura queda limitada al runner, preparer y test, con R498 como
auditoría exclusiva y con módulo científico y worker fijados como invariantes.

La procedencia del delta distingue correctamente dos planos. El escrow puede
conservar el mapa de sources del intento v1, mientras `prior_source_sha256`
debe ser exactamente el mapa de la config v2 auditada. Los `old_sha256` de los
tres cambios se comparan contra esa config v2 y los `new_sha256` contra el
commit de implementación, la futura config y los archivos físicos. Así, el
lineage del draw permanece anclado al escrow sin atribuir erróneamente a v1 el
delta de código v2→v3.

El validador del amendment exige además parentage directo, auditorías R499 y
R500, inventario closed-world, hashes del draw preservado, terminal completo
v2 y presupuesto durable firmado. La revalidación previa a uso repite la
autenticación del origen y del terminal. La provenance materializada declara
la excepción cerrada, y el ledger de v3 hereda una sola vez el acumulado v2.

El test integral materializa en un repositorio efímero ambas recuperaciones,
incluida la rama estática completa: crea implementación y auditorías
sintéticas, valida amendment y config, prepara primaria y replay v3 con el
mismo draw por bytes e inodos nuevos, y verifica continuidad del ledger. Sus
ataques cubren terminal, presupuesto, mapa previo de sources, lineage y delta
de sources; la suite histórica conserva además los negativos de inventario,
firmas, amendment, config y doble débito.

## Reproducción física read-only

Ejecuté el guard corregido sobre
`data/geometria_proporcional/wave60_frozen_policy_transport_attempt_v2` sin
modificarlo. El resultado fue `PASS` con cinco antecedentes y 22 archivos
compartidos por bytes entre primaria y replay sin inodos compartidos. La
validación firmada confirmó:

- terminal de par: `PAIR_ABORTED_PRE_TRUTH`;
- `any_truth_accessed: false`;
- preparación durable: `215.36700256168842 s`;
- tiempo durable de fases: `0.0 s`.

La cadena documental real R496/R497 y el binding de los quince archivos del
terminal v2 también fueron aceptados directamente por los nuevos validadores.
Esta reproducción fue CPU-only; `/usr/bin/time -v` registró RSS máximo de
827.792 KiB y 0 swaps para la validación conjunta de chain y terminal.

## Pruebas ejecutadas

La corrida focal comprendió el E2E v2→v3 y los siete casos específicos de
partición, delta, provenance y guard: `8 passed in 20.18s`, RSS máximo
858.864 KiB y 0 swaps.

La regresión obligatoria ejecutó completos los diez archivos de tests de Waves
56–60: `490 passed, 1 skipped in 805.00s`, con tiempo de pared 13:26.29, RSS
máximo 1.065.520 KiB y 0 swaps. Todas las corridas usaron
`CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1` y basetemps bajo
`/mnt/m2-1TB`; los temporales fueron eliminados por su path exacto. No usé ni
consulté GPU y no modifiqué código, config ni datos.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R498",
  "scope": "STATIC_PROTOCOL_IDENTITY_GUARD_RECOVERY_IMPLEMENTATION",
  "target": {
    "implementation_commit": "24fbb4a3dfd43e9c7eb9293921f068636c190900"
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
