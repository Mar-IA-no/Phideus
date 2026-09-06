# R504 — Auditoría de config Wave 60 v4

## Dictamen: REVISE

La configuración congelada de Wave 60 v4 no puede autorizar ejecución. El
preflight final contra el `HEAD` real R504 autentica informe, config y cadena
R501–R503, pero falla al verificar el runner: usa como autoridad el commit de
implementación científica original en vez del baseline v3/R500 que el
amendment hard-set declara invariante. La config y el filesystem coinciden
entre sí y con v3, pero no con ese blob histórico; por tanto la composición
ejecutable queda bloqueada antes del re-forward y antes de crear cualquier
output.

El resto del target compone la autoridad de `hard_set_tau=0.5` sin incorporarla
a la configuración canónica, conserva el roster cerrado de ocho sources,
reserva un namespace nuevo y mantiene intactos la pregunta científica, las
políticas, el scoring y el presupuesto total. La auditoría no corrigió código o
config, no materializó ni ejecutó el intento prospectivo, no abrió truth y no
produjo resultados científicos.

El fallo posterior al freeze de un caso de pytest se reproduce y se explica en
este informe. Es un test histórico dependiente de fase que carga desde `HEAD`
el archivo que pretende usar como baseline v3; no es evidencia de una deriva de
la config ni del preflight productivo. Por eso no genera un finding sobre el
target `CONFIG`.

## Identidad, parentage y forma canónica

El target auditado es exactamente el commit
`201f9257c1e016b925719f6917dd1a9298ca493b`, hijo directo de R503
`2b98cfa8af9d207547ab81bc9eafc0e0cc77b399`. Modifica un único path:

`experiments/geometria_proporcional/configs/wave60_frozen_policy_transport.json`

El blob Git y el archivo físico son byte-exactos, tienen 11.599 bytes y SHA-256
`92bf02867579281f5df02b0cdd2c6980cbb8d1697f8df60a353457b86cea3960`.
La serialización física coincide con `json.dumps(..., indent=2,
sort_keys=True) + "\n"`; `validate_pre_draw_config()` y
`validate_prospective_config()` aceptaron el objeto con sus keysets exactos.
El self-binding, calculado sobre el JSON canónico compacto después de sustituir
sólo su propia entrada por 64 ceros, es
`94861c2afbc30753177249307f1199ab319443dd53ce7488f59d249af6e1dece`.
El mecanismo está definido en
`src/geometria_proporcional/wave60_frozen_policy_transport.py:245-262`.

## Delta v3 → v4

La comparación estructural usa como baseline el blob v3 congelado en
`0cc349c10b91cffe0eb53ae6669fcffbfaf8c756`, SHA-256
`8d8edeed1d4c7943655a2824bd906001f711581d079ec79da9b9704739f94ae2`.
Los veinte leaf changes pertenecen únicamente a cinco clases autorizadas:

- namespace y versión del intento, de v3 a v4;
- once bindings de recovery hacia el terminal v3, R500, el amendment v4 y R503;
- reserva de auditoría final R504;
- paths de primary, replay y parent bajo `attempt_v4`;
- self-binding y hashes nuevos de preparer/test ligados por R502.

No cambian schema, status, CPU/threads, penalty, bootstrap, límite de 900 s,
políticas mean/tail, features, seeds, batch size, splits físicos, benchmark,
source bindings, source law, implementación científica ni plan. El roster sigue
teniendo exactamente ocho sources distintos:

| Source | binding verificado |
|---|---|
| config (self-binding) | `94861c2a...dece` |
| módulo científico | `46e31fa1...c65` |
| runner | `b35cd563...0261` |
| worker | `c6c5c832...bac7` |
| preparer | `19942cb8...b539a` |
| test | `d249aba7...f449f9e` |
| R475 | `e5c49ca1...d9996` |
| R476 | `497bb87f...c0a8a30` |

Los siete hashes no autorreferentes coinciden con filesystem y con sus blobs de
autoridad; el octavo coincide con el algoritmo de self-binding. El validador
closed-world del roster está en
`src/geometria_proporcional/wave60_frozen_policy_transport.py:1198-1225`.

## Autoridad R501–R503 y `hard_set_tau`

La cadena revalidada es lineal y no borra la revisión intermedia:

```text
plan v4                    a9bb0cfd1062da5076f8cbdd8c2d64978270dce7
R501 PASS                  2fe5f1f83dc266e8671ca5cc8dd1580ceef6b7a1
implementación inicial     2b1359bcc8bf689122dc342032c1f69f31ff2b61
R502 REVISE                00b433a688554d5f7d2c37caf87d32be02abc466
correcciones               410a9189af5f0e87ce396aaa121ff593b7b5a322
                           e29069f68ea2fdf27e6a6fae2480a835144bc5ac
implementación final       7eba44b9622d7e6ecd90d0345feee32c00748833
R502 PASS                  7182d97533c801cc718360fa2f444a140d0e06dc
amendment v4               3a21c039118a4d9d95e84802d5b0e50b48be7cc9
R503 PASS                  2b98cfa8af9d207547ab81bc9eafc0e0cc77b399
config v4                  201f9257c1e016b925719f6917dd1a9298ca493b
```

R502 REVISE conserva `0/1/0`; `resolution_of` liga ese informe, ambas
correcciones y el target final. R502 PASS es exclusivo y autentica el commit
final con `0/0/0`. El amendment v4 es JSON canónico, exclusivo, hijo directo de
R502 PASS y tiene SHA-256
`1e4b097d7c9882e4037a608f65ac29ef10080addff8e4c24c2c922db2d84f4d8`;
R503 es su hijo directo exclusivo y lo acepta con `0/0/0`.

La config canónica no contiene `hard_set_tau`. La validación física recorrió
manifest → request → alias único → snapshot de Ola 59 y resolvió el único valor
finito autenticado `0.5`. Sólo una copia efímera de la config recibe ese valor
antes del materializador; la guarda correspondiente está en
`experiments/geometria_proporcional/prepare_wave56_fresh.py:8098-8115`.

## Origen, terminales, presupuesto y frontera pre-truth

El recovery recompuso el único origen permitido:
`data/geometria_proporcional/wave60_frozen_policy_transport_attempt_v3`.
Las firmas e inventarios físicos validan pair `PAIR_ABORTED_PRE_TRUTH`, primary
y replay `INVALID_PREPARATION`, `any_truth_accessed=false` y
`recovery_allowed=true`. Los inventarios contienen 44 records de primary y 8 de
replay, no tienen hardlinks ni estado durable `prepared`, labels, receipt o
attestation de preparación. El draw reutilizable es únicamente
`primary/failed_preparation`, ligado por los 17 hashes preservados.

La distinción semántica permanece explícita: durante la preparación fallida se
computaron transitoriamente labels de train antes del `KeyError`, pero no
persistieron; no hubo scoring/evaluación científica, acciones de lockbox ni
apertura de truth con valor decisional. Esta auditoría hizo sólo validación
read-only sobre esos artefactos.

El ledger v2 firmado conserva `215.36700256168842 s`; v3 agrega una única vez
el débito conservador de `60.0 s`, dejando
`275.3670025616884 s` como inicio acumulado de primary v4. El límite continúa
en 900 s y replay deberá heredar el acumulado firmado de primary. El path
`data/geometria_proporcional/wave60_frozen_policy_transport_attempt_v4` no
existe: el namespace está libre y no se materializó ningún output prospectivo.

## Finding R504-01 — HIGH: el preflight final usa la autoridad pre-v3 para el runner

El validador final reconoce correctamente que el recovery hard-set cambia sólo
preparer y test. También carga el `source_sha256` de la config previa v3/R500 en
`experiments/geometria_proporcional/prepare_wave56_fresh.py:1403-1414`. Sin
embargo, ese mapa previo se usa únicamente para comprobar los `old_sha256` de
los sources que sí cambiaron. En el loop final (`:1452-1487`), todo source fuera
de esa partición vuelve a `implementation_binding.commit`, es decir,
`9f1a229d9c0ccb5e46b921e6c92281becc317139`.

Eso es incorrecto para el runner. El runner físico, la config v3/R500, el
amendment v4/R503 y la config v4 coinciden en:

`b35cd563f715bdff9b6e7489ac04712c728673563898d4a6aebf0144d4a50261`

El blob del mismo path en la implementación científica original `9f1a229d...`
es, en cambio:

`1c778c3e60c1bbcebeb5c83430601a7c0b148e447528195f1dec4296322825aa`

La diferencia fue autorizada previamente por la recuperación estática R498 y
ya integra el baseline v3. R501, R502 PASS y R503 exigen expresamente que los
sources invariantes de v4 se comparen contra v3/R500, no contra el escrow v1 ni
contra una implementación anterior a R498. El preflight viola esa procedencia
y emite `Wave 60 executed blob differs from audited implementation` para el
runner.

### Impacto y corrección requerida

El impacto es bloqueante: `preparation_preflight()` no retorna contrato y el
intento v4 no puede comenzar. No hubo contaminación científica porque la falla
ocurre antes del re-forward, la materialización, truth o scoring. Se clasifica
HIGH por impedir la única ejecución autorizada, aunque el defecto lógico sea
acotado.

La corrección debe hacer que, para recovery hard-set, los sources declarados
invariantes se validen contra el mapa autenticado de la config v3/R500; sólo
preparer y test deben resolverse desde la implementación R502. Debe agregarse
una prueba de composición física final que ejecute
`validate_wave60_final_config_authority()` después de congelar la config y
publicar una auditoría R504 sintética. Corregir el preparer y su test cambia
sources ligados, de modo que no puede injertarse sobre esta config: requiere
una nueva cadena auditada y un nuevo freeze/namespace. R504 no implementa esa
corrección.

## Evidencia adversa de pytest: 18 PASS / 1 FAIL

La orden focal contra el `HEAD` ya congelado produjo exactamente `18 passed, 1
failed, 154 deselected`. Falló
`test_hard_set_v4_source_baseline_is_v3_not_v1_escrow`.

La causa es localizable y no contradice el contrato. En
`tests/test_wave60_frozen_policy_transport.py:5388-5392`, el test carga el path
canónico vigente y lo asigna a `prior_config`; no recupera el blob v3 ni fija
explícitamente `attempt.version=3`. Después construye `current_config` como v4
en `:5499-5533`. Antes del freeze, cuando el path físico todavía contenía v3,
esa composición era v3→v4 y R502 obtuvo `19/19`. Después del freeze el mismo
código construye v4→v4. El rechazo en
`src/geometria_proporcional/wave60_frozen_policy_transport.py:1101-1113` es el
comportamiento correcto: una recovery no puede declarar como prior una versión
igual o posterior a la actual.

La prueba discriminante fue ejecutar sólo ese caso con el mismo preparer/test
del commit final `7eba44b9622d7e6ecd90d0345feee32c00748833`, cuya config física aún
es el blob v3 auténtico: `1 passed in 1.27s`. Esto demuestra que el test está
ligado a la fase pre-freeze; no demuestra una falla del validador v4.

La resolución metodológica correcta de ese caso es correrlo contra el
blob v3 o, en una versión futura, hacer explícito el fixture v3. Editar ahora el
test ligado cambiaría su SHA-256, invalidaría el mapa de ocho sources y exigiría
otra ronda de implementación/config, sin modificar ninguna garantía productiva.
Ese costo aislado no repara R504-01 ni está justificado como condición adicional
para diagnosticarlo. La suite genérica `-k
hard_set_v4` contra HEAD deja de ser una evidencia autónoma válida para este
caso específico. El test adverso queda visible como limitación de replay de
fase, no como segundo finding `CONFIG`.

## Checks CPU-only

Todas las ejecuciones usaron `CUDA_VISIBLE_DEVICES=''`,
`PYTHONDONTWRITEBYTECODE=1` y, para pytest, plugins externos deshabilitados. No
se usó ni se consultó GPU.

| Check | Resultado | wall | max RSS | swaps |
|---|---:|---:|---:|---:|
| validación directa config/authorities/origin/budget | PASS | 3.96 s | 828.696 kB | 0 |
| suite `-k hard_set_v4` contra HEAD | 18 PASS / 1 FAIL esperado por fase | 85.52 s | 891.484 kB | 0 |
| caso adverso contra blob v3 | 1 PASS | 2.44 s | 862.080 kB | 0 |
| identidad/delta/roster físico | PASS | 1.57 s | 826.192 kB | 0 |
| preflight final contra `HEAD` R504 | FAIL en autoridad del runner | 2.00 s | 826.560 kB | 0 |

Los temporales y el worktree detached creados para aislar el blob v3 fueron
retirados y el inventario de worktrees volvió a su estado previo. El preflight
final se ejecutó después de commitear una primera versión de este informe,
porque exige que R504 sea el `HEAD` exclusivo hijo de la config. Falló antes de
`historical_preflight()` y antes de crear `attempt_v4`; el commit del informe se
enmendó sólo para registrar el dictamen REVISE, conservando el parent directo y
la exclusividad.

La configuración solicitada al launcher fue Codex `gpt-5.6-sol`, effort
`high`. Esta instancia no expone una interfaz confiable para verificar de manera
independiente el identificador efectivo del modelo o el esfuerzo; no simulo esa
confirmación.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R504",
  "scope": "CONFIG",
  "target": {
    "config_commit": "201f9257c1e016b925719f6917dd1a9298ca493b",
    "config_sha256": "92bf02867579281f5df02b0cdd2c6980cbb8d1697f8df60a353457b86cea3960"
  },
  "technical_verdict": "REVISE",
  "findings": {
    "high": 1,
    "medium": 0,
    "low": 0
  },
  "files_modified": false,
  "gpu_used_or_queried": false
}
```
