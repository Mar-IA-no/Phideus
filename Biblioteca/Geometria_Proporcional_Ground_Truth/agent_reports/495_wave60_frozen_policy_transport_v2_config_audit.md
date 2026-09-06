# R495 — Auditoría independiente de config Wave 60 v2

**Dictamen técnico: `PASS` — 0 HIGH / 0 MEDIUM / 0 LOW.**

La config v2 es un sucesor exclusivo de R494 y conserva íntegro el contrato
científico de v1. El único cambio funcional es la autorización cerrada del
recovery `INVALID_PREPARATION`: nuevo namespace v2, doce bindings exactos hacia
el failure v1, la amendment aprobada y R494, y la partición de source law R475
frente a preparer/test R493. Los validadores canónicos pre-draw, prospectivo y
de autoridad final aceptaron el archivo real. El preflight read-only también
cerró sin materializar el intento v2.

## Identidad, forma y commit

El target auditado es exactamente:

- commit: `ef4a620ae79f5eb4502eea9cca1547da86216f93`;
- parent directo: `d53ad522ac3e984ed8b1c5c658e75ce361940101` (R494);
- path único: `experiments/geometria_proporcional/configs/wave60_frozen_policy_transport.json`;
- SHA-256 del blob Git y del archivo físico:
  `b965ec2d414df0dc6ce3460dd3cb7cc741537971aac92000795dd3265a18f626`.

El commit modifica sólo ese path y `git diff --check` queda limpio. La config
tiene 11.598 bytes, modo `0600`, owner `0:0`, parsea como un único objeto JSON y
satisface el keyset cerrado de `wave60-frozen-policy-transport-v1`. Leí
completos la config, la amendment v2, R494 y los validadores afectados. No
modifiqué código, config, amendment ni datos; este informe es el único archivo
creado y no forma parte del target.

## Recovery y enlace al intento v1

`attempt` contiene exactamente `version`, `container`, `primary`, `replay`,
`pair` y `recovery`. La versión es 2 y los namespaces/output son coherentes:

```text
container              data/geometria_proporcional/wave60_frozen_policy_transport_attempt_v2
primary_output         .../attempt_v2/primary
replay_output          .../attempt_v2/replay
output_parent_relative .../attempt_v2
```

El objeto `attempt.recovery` tiene exactamente doce claves. Liga el contenedor
v1, su failure de par
`05ede417b1f856488c1796210029aa74c74211f3f14858764d9705cbb0b3563d`,
la auditoría de config R477 y la amendment canónica. Esta última coincide en
path y SHA-256
`0b470ed781a841dddda063a2ef40d4bbadea8896d052d778994184c8befaa6eb`;
su auditoría R494 coincide en commit
`d53ad522ac3e984ed8b1c5c658e75ce361940101`, path y SHA-256
`45e1e8f116c0399e621b2d6824a1e18052adee99b71463ab45624028920ec7e8`.
Amendment y R494 fueron introducidas cada una por un commit exclusivo y en
parent directo, antes del commit de config.

`preserved_draw_sha256` contiene exactamente 17 archivos y es idéntico al mapa
aprobado en la amendment. No aparece `hard_set_tau` en la config: el valor 0.5
continúa confinado al contrato duro de la amendment y a la vista efímera del
materializador.

## Comparación contra v1 y self-binding

La config v1 canónica proviene de
`ba4066879dcc8dbba4fd73d76b018c142e3e3eb5` y su SHA-256 es
`657a3912d2c799a396fe8eb5710c7a63ab7eea6dac3a8489b57fbaf5bfa282cc`.
Una comparación recursiva exacta muestra que, después de normalizar solamente
`attempt`, los tres paths de output, `final_audit` y los tres hashes autorizados
de config/preparer/test, el objeto v2 es idéntico a v1. Por tanto permanecen
iguales schema, status, device, threads, penalty, bootstrap, runtime budget,
políticas, features, splits, seeds, batch size, fresh benchmark, plan,
implementation binding R475, source binding y source-law authority.

El self-binding calculado por `config_self_binding_sha256()` —sustituyendo sólo
su propia entrada por 64 ceros y serializando canónicamente— coincide con la
entrada declarada:

```text
01c0bf9c408bc39eea62dfaea239c52b4bfb6db5e550547354238b1a661dd708
```

## Partición de fuentes y autoridad final

Los tres sources de la ley científica permanecen byte-exactos respecto de
R475, del commit de config y del filesystem:

- módulo: `46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65`;
- runner: `1c778c3e60c1bbcebeb5c83430601a7c0b148e447528195f1dec4296322825aa`;
- worker: `c6c5c832633d9ed8a99079dd4fc4f5677073f7e1883a204fa6bb41f6d646bac7`.

Los únicos sources ejecutables reasignados a la aceptación R493 son:

- preparer: `3d0532cd840461fa07ce1fccc81d5c1b085b8c00397f6162c88f435e0da23ac9`;
- test: `12611bd90e13a5e1ff7d601b4b654802e021f90b9f3c4be8fe9f38183e178e9e`.

Los informes R475/R476 también conservan sus hashes físicos declarados. El
manifest tiene exactamente ocho sources, sin duplicados, y su mapa
`source_sha256` tiene el mismo dominio. `final_audit` reserva de manera única
R495 y el path de este informe.

## Validación ejecutable y no materialización

`validate_pre_draw_config(config)` y `validate_prospective_config(config)`
aceptaron directamente el JSON físico. Para validar la autoridad circular sin
mover el estado real, añadí este informe al árbol del commit de config mediante
un índice temporal y construí con plumbing un commit object directo hijo de
`ef4a620…`. El objeto contiene un único delta: este informe. Un gitdir, índice y
worktree efímeros bajo `/mnt/m2-1TB` expusieron ese objeto como HEAD detached a
los subprocess Git de los validadores, mientras HEAD, branch y refs del
repositorio real permanecieron intactos.

Sobre esa vista,
`validate_wave60_final_config_authority(repo, config_path, config, HEAD)`
recorrió el commit exclusivo de config, R494, R493, la amendment y la partición
de blobs; aceptó R495 con target exacto `config_commit + config_sha256`.
`preparation_preflight()` se ejecutó después con los paths canónicos, fuente de
recovery v1 y amendment real. Fue estrictamente read-only: no creó
`wave60_frozen_policy_transport_attempt_v2`, no ejecutó draw, entrenamiento,
scoring ni evaluación.

Todas las corridas fueron CPU-only con `CUDA_VISIBLE_DEVICES=''`; no usé ni
consulté GPU. Los basetemps, gitdir e índice efímeros se inventariaron y
eliminaron por path exacto al terminar.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R495",
  "scope": "CONFIG",
  "target": {
    "config_commit": "ef4a620ae79f5eb4502eea9cca1547da86216f93",
    "config_sha256": "b965ec2d414df0dc6ce3460dd3cb7cc741537971aac92000795dd3265a18f626"
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
