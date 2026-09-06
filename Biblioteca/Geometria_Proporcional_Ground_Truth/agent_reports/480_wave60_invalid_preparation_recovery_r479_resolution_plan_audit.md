## Auditoría independiente R480

**Dictamen técnico: `PASS` — 0 HIGH / 0 MEDIUM / 0 LOW.**

### Identidad y alcance

Se releyeron completos:

- el plan focal R479;
- la auditoría R479;
- la resolución R478;
- la auditoría R478;
- el plan base rechazado.

La identidad del target es exacta:

- Commit: `305cdfcc47a0d5f537019a4a5fa8cf3ca271fd2c`.
- Parent directo: R479, `3f20c79db2311cfefebc90dd049b4862d2a04a11`.
- Commit exclusivo: añade únicamente `WAVE_60_INVALID_PREPARATION_RECOVERY_R479_RESOLUTION_PLAN.md`.
- SHA-256 físico y blob Git: `ec82136d826564effb50b848c61d674c950e9e83a3b6371496a24756efc92081`.
- R479 físico/blob: `4089ef7cd4812714fa731efae45719fc1685dafb767e1e92c26574abdddb9fa4`.
- Worktree limpio.

### Cierre del MEDIUM R479

El plan focal cierra completamente el finding. `hard_set_contract` pasa de cuatro a nueve claves exactas y congela todos los eslabones relevantes:

```text
manifest v2
  → source_law_request.json exacto
  → alias wave59_config_snapshot.json
  → path + SHA-256
  → snapshot físico
  → hard_set_tau = 0.5
```

El orden operativo también queda cerrado: toda la cadena debe validarse antes de construir `materializer_config` y, por tanto, antes de llamar al materializador real.

En particular, exige:

- resolver canónicamente el authority root;
- ligar su manifest mediante el hash fijado por la config;
- ejecutar `validate_source_authority()` real con `implementation_binding` R475;
- comprobar el registro físico y metadata del request dentro del manifest;
- exigir el hash exacto del request;
- comparar explícitamente:

```text
request.source_paths[request_alias] == hard_set_contract.source_path
request.source_sha256[request_alias] == hard_set_contract.source_sha256
```

- resolver después el snapshot físico, verificar su hash y exigir un `hard_set_tau` numérico, finito e igual a `0.5`;
- incorporar el valor únicamente a una copia efímera de la config.

No existe default, inferencia desde código ni modificación de la config canónica.

### Contraste con los artefactos reales

La cadena declarada coincide con el estado físico:

- Manifest v2: `9c69745a0661994049530e917e59e0a68b99d5f15a7c0d2bae3da66f1df43dc2`.
- Request v2: `983af4bb024f966b60b4e79fe753ebd38665747eab21b95db27ec3f0ab889a99`.
- El registro del request en el manifest coincide exactamente en bytes, owner, group, modo y SHA-256.
- El alias `wave59_config_snapshot.json` apunta al path fijado por el nuevo contrato.
- Su hash declarado es `f6edfd2106fe87c8150562d096469e29b64a108a73de2dae0d371bd689a4a9b6`.
- El snapshot físico coincide con ese hash y contiene `hard_set_tau=0.5`.
- `validate_source_authority()` acepta la autoridad v2 real con la config vigente.

### Negativos y prueba positiva

La cobertura focal es suficiente. Se exigen rechazos independientes para:

- alias ausente o duplicación por otro alias;
- path alternativo aunque conserve los mismos bytes;
- hash cruzado;
- request no ligado por el manifest;
- manifest alternativo o no ligado por la config;
- snapshot alterado, traversal, symlink;
- threshold ausente, distinto, no finito o no numérico.

La prueba positiva debe recorrer los artefactos v2 reales y ejecutar el materializador real sin mockear ninguna de esas fronteras. Esto evita repetir la insuficiencia que permitió el `KeyError` original.

### Realizabilidad con `preparer + test`

La corrección continúa siendo realizable modificando exclusivamente:

- `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
- `tests/test_wave60_frozen_policy_transport.py`.

El preparador puede reutilizar mediante import diferido el `validate_source_authority()` vigente. El runner no importa el preparador y no aparece una dependencia circular. Las verificaciones adicionales del contrato, request, snapshot y vista efímera pertenecen naturalmente al adaptador de recuperación.

Los blobs científicos permanecen byte-exactos respecto de R475:

- módulo: `46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65`;
- runner: `1c778c3e60c1bbcebeb5c83430601a7c0b148e447528195f1dec4296322825aa`;
- worker: `c6c5c832633d9ed8a99079dd4fc4f5677073f7e1883a204fa6bb41f6d646bac7`.

Por tanto, no se rompe la partición entre R475 y la futura autoridad de recuperación R481 ni se requiere regenerar source law v2.

### Schema y lineage

El nuevo keyset top-level de amendment incorpora separadamente las historias R478 y R479. `attempt.recovery` conserva sus doce claves y liga la amendment por hash, sin duplicar la autoridad ampliada.

La renumeración es coherente:

```text
R477
  → plan rechazado
  → R478
  → resolución R478
  → R479
  → resolución R479
  → R480
  → implementación preparer+test
  → R481
  → amendment
  → R482
  → config v2
  → R483 / HEAD de ejecución
```

Cada eslabón puede ser directo y exclusivo. La amendment no depende de su propia auditoría ni la config de su auditoría futura, por lo que no hay circularidad. Las reglas legacy permanecen separadas.

No se modificaron archivos ni artefactos. La auditoría fue CPU-only y no se usó ni consultó GPU.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R480",
  "scope": "INVALID_PREPARATION_RECOVERY_R479_RESOLUTION_PLAN",
  "target": {
    "plan_commit": "305cdfcc47a0d5f537019a4a5fa8cf3ca271fd2c",
    "plan_sha256": "ec82136d826564effb50b848c61d674c950e9e83a3b6371496a24756efc92081"
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
