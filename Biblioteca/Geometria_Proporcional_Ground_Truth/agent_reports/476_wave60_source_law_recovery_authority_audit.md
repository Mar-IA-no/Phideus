## Auditoría independiente R476 — PASS

Releí completos el plan base, el plan de recuperación, la resolución de R473 y R475. No encontré findings altos, medios ni bajos.

- Identidad: `HEAD 92de45191b60acfb82f2e9a6588751d9dc96983e`, hijo directo de `9f1a229d9c0ccb5e46b921e6c92281becc317139`; árbol `4bb222805b4a2ec16e19c7e440ba5f826c784c33`.
- R475 es el único archivo de su commit, su hash es `e5c49ca16506469ac5099c2a3f1992819c1a54fed790f1638a82e93b9b1d9996` y su autoridad PASS 0/0/0 quedó autenticada.
- v2 contiene exactamente 10 archivos y un directorio `journals/`. Directorios `0700`; archivos `0444`, `root:root`, `nlink=1`, sin symlinks ni nodos especiales.
- El inventario físico coincide exactamente con los nueve registros no autorreferenciales del manifest. Su autorreferencia, clases y terminal `SOURCE_LAW_VERIFIED` son correctos.
- Manifest canónico: `9c69745a0661994049530e917e59e0a68b99d5f15a7c0d2bae3da66f1df43dc2`.
- Request operacional y embebido son byte-exactos: `983af4bb024f966b60b4e79fe753ebd38665747eab21b95db27ec3f0ab889a99`. Declara 11 fuentes y nueve hashes; todos los hashes declarados coinciden.
- El receipt registra exactamente 12 inputs y 12 paths abiertos —request más 11 fuentes—, cinco outputs y tres probes denegados. Worker `65534:65534`, capabilities cero y `no_new_privs=1`.
- La firma Ed25519 y su payload de seis campos pasaron validación, incluyendo request, freeze, receipt, journal e implementation commit.
- Presupuesto: `4.078716456890106 s` actuales + `0.00401783362030983 s` durables de v1 = `4.082734290510416 s`; RSS `740962304`, ambos bajo los límites. `truth_accessed=false`.
- v1 conserva exactamente sus cinco hashes pactados y su terminal inválido durable; no sufrió mutación.
- Roster: 13 HGB usados y tres modelos excluidos explícitamente. Hay 1300 tree keys únicos y 3900 arrays exactos, sin arrays adicionales ni splits categóricos.
- Esquema: 17 features `float64` en el orden canónico.
- Política: 13 umbrales exactos, proposer `>`, guards `<`, `score_mask=disagreement` y `decision_mask=primary AND disagreement`.
- Reproducción: 16 scores retrospectivos completos, 13 scores de transporte, 26 arrays seleccionados y referencia `HARD-SET` exacta.
- Una reproducción aislada sólo CPU regeneró los cinco outputs científicos byte por byte: freeze, manifest, NPZ, spec y esquema.
- El camino auditado no contiene fit, recalibración, cuantiles, selección adaptativa, bootstrap, evaluación, draw ni referencia al bundle de verdad.
- Ocho pruebas dirigidas completaron con exit 0; el último worker se reconfirmó aisladamente: `1 passed`.
- No existen staging, initializing, attempt ni temporales R476. `git status` quedó limpio; HEAD y árbol permanecieron inmóviles. Un eventual commit de archivo R476 deberá ser hijo directo y exclusivo de este HEAD.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R476",
  "scope": "SOURCE_LAW",
  "target": {
    "source_authority_manifest_sha256": "9c69745a0661994049530e917e59e0a68b99d5f15a7c0d2bae3da66f1df43dc2"
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
