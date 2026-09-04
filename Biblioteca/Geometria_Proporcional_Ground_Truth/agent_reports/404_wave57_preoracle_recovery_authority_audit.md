# R404 — Auditoría final de autoridad para recovery pre-oracle de Ola 57

**Audited package commit:** `72eb98b958d89c484a03a4ecc40e899f734d5ceb`
**Amendment SHA-256:** `8cfcb0453bd1cd19b41db518ffca6f702cee03c31966f137cd8204fbad248fc5`
**Result:** `PASS`

## Dictamen

El paquete queda apto para conferir autoridad de recovery pre-oracle una vez que
este informe sea incorporado como único commit sucesor del amendment. No encontré
un desvío material entre el plan auditado, la implementación, sus auditorías, el
amendment canónico y el origen físico declarado. Esta conclusión no ejecuta el
recovery, no abre semánticamente el escrow o la truth sellada y no constituye una
decisión experimental sobre Ola 57.

## Provenance y cierre de la cadena

La cadena Git observada es lineal y de padre directo en cada transición:

- plan `d21f3c96e0313077f8e121dab731c9f8a7a93320`;
- auditoría de plan R401 `9e9a71493ea99c2dfe466bc7adc37db3a5df666b`;
- implementación `5b335cb47c00cb9770bfcacd815459c5109a1213`;
- reauditoría de implementación R403 `db3b1a3cbc82a1a74329d11d23d30bc8ae6b2f0e`;
- amendment `72eb98b958d89c484a03a4ecc40e899f734d5ceb`.

Cada commit cambia exclusivamente el path o los dos paths previstos para su
etapa. El HEAD auditado coincide exactamente con el amendment y el worktree estaba
limpio antes de crear este reporte. R402 fue leído como antecedente `REVISE`, pero
su archivo ignorado no integra el DAG activo ni fue usado como fuente de autoridad.

Los hashes del plan, R401 y R403 coinciden con el amendment. Los blobs del commit
de implementación coinciden con los hashes nuevos declarados: preparador
`a4de65cdccc33b7709ce79952848116ff6d0a13705612231adaa29710b821253` y
test Wave 57 `7ee02f622c61633509a58973966cdafd586fe497bd4c287793ac0775c2e84594`.
Los blobs en el commit de origen coinciden asimismo con los hashes viejos
`796b5e8f580c1e98f0f1061ed16c7116cd2db752cc38a44c3651250d56c3bea7` y
`5238778afb0008c200529b9883430d321793023006f8b57a5d9c1ec12022e52b`.
El amendment es JSON canónico y sus bloques de auditoría previos satisfacen el
parser estricto de atestaciones.

## Separación de autoridad y acceso semántico

La revisión completa del preparador confirma que la ruta Wave 57 exige primero
schema y estado aprobados, las cuatro aserciones pre-oracle, los hashes y paths
canónicos, el DAG cerrado, HEAD final exacto, worktree limpio, delta restringido a
preparador y test, y el preflight físico content-blind. Sólo después llama a la
validación semántica y, posteriormente, a la lectura del escrow de la fuente.

El inventario del origen tiene exactamente 24 entradas: seis directorios y
dieciocho archivos. Coincide byte por byte con el amendment en paths, tipos,
ownership, modos, tamaños y SHA-256; también están ausentes los prefijos y
artefactos prohibidos de inferencia, labels autorizados, oracle materializado,
bundles y fases. El contrato secret-free del freeze, el manifest público, los
visibles y la firma detached se validan sin interpretar los archivos sensibles.

Durante esta auditoría no se invocaron `read_escrow`, `keys_from_escrow`,
`validate_semantic_attestation`, `sealed_population_counts` ni `read_jsonl` sobre
truth sellada, y no se parseó ningún archivo bajo `benchmark/sealed/`. Escrow,
secret files y truth sellada sólo fueron recorridos como bytes opacos mediante
`O_NOFOLLOW`, `fstat` y streaming dirigido a SHA-256. No se extrajeron claves, no
se materializaron labels u oracle y no se inició inferencia.

## Verificación ejecutable

Con `CUDA_VISIBLE_DEVICES` vacío y un único thread para OpenMP, OpenBLAS y MKL,
la selección focal de `tests/test_wave57_prospective.py` produjo 28 pruebas
pasadas y 23 deseleccionadas en 6.42 segundos. Incluyó el dispatcher tipado, el
validator completo sobre un DAG sintético positivo, dieciséis fallos de autoridad,
el parser público, el probe instrumentado content-blind contra el origen real,
los rechazos de inventario, ownership, modo y hash, el delta de contrato, el orden
content-blind antes de semántica, provenance y parsing estricto de R401.

Como control adicional, confronté los campos consumidos por el preflight del
fixture instrumentado con el amendment canónico y resultaron idénticos. La llamada
content-blind usando directamente ese amendment contra el origen real pasó con 24
entradas y reprodujo el SHA-256 del contrato público. La compilación Python de los
dos archivos modificados pasó. No se ejecutaron tests que materializan recovery,
regeneración, inferencia u oracle, ni se usó GPU.

## Findings

No quedan findings materiales dentro del alcance de autoridad pre-oracle. La
condición operativa pendiente es deliberada: este reporte debe ser agregado y
commiteado solo, como hijo directo del amendment. Recién con ese commit como HEAD
y el worktree globalmente limpio el validator podrá atravesar la barrera final y
habilitar la etapa semántica posterior.

## Machine-verifiable decision

**Final decision:** `PASS`

