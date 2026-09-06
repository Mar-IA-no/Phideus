# R505 — Auditoría del plan de resolución de autoridad R504 de Ola 60

## Dictamen: PASS

El plan identifica correctamente la causa de R504-01 y propone la corrección
mínima: en un recovery tipado, la config previa autenticada gobierna todos los
sources que esa transición declara invariantes; el commit correctivo gobierna
sólo los sources de la partición cambiada. Aplicada sobre el roster cerrado,
esta regla acepta el runner v3 ligado por R500 y sigue rechazando tanto un
rollback al runner original como cualquier deriva física, de config o de Git.

No encontré findings materiales. La implementación deberá conservar como capas
distintas la autoridad hard-set inicial R502/R503 y la resolución R506/R507; esa
separación ya está exigida por el plan mediante enlaces inmutables y un delta
acumulativo v3→final. R505 no corrige el preparador, los tests, el amendment ni
la config rechazada, no crea el namespace v4 y no autoriza la ejecución.

## Identidad y estado físico

El target es exactamente el commit
`0e2768b665a5df329ca53de1a68ab6e14a86d70f`, hijo directo de R504 REVISE
`b70ee7e8f3d4f561a52017419aee0012914e2f54`. Modifica un único path, el plan
auditado. El blob Git y el archivo físico coinciden y tienen SHA-256
`49e6ba42b7f6802ead82b90f042a47022902dec8b1a9e53ae1c774b99adaef0d`.
El worktree estaba limpio al iniciar la auditoría.

El namespace físico
`data/geometria_proporcional/wave60_frozen_policy_transport_attempt_v4` no
existe. La config rechazada permanece congelada en `201f9257...`; R504 constató
que el preflight no retornó contrato y falló antes de re-forward,
materialización, scoring o apertura de truth. Por eso conservar el nombre v4 es
correcto: todavía no identifica una ejecución ni otro draw. Si un preflight
futuro falla nuevamente antes de crear la raíz, el plan conserva también ese
terminal documental sin fabricar un intento científico.

## Verificación de la corrección de autoridad

El defecto es visible en
`prepare_wave56_fresh.py:1403-1414,1452-1487`. El validador final recupera el
`source_sha256` de la config anterior, pero lo consulta sólo al formar el
`old_sha256` de un source cambiado. Para un source invariante deja
`authority_commit=implementation_binding.commit`. En el caso real, ese commit
original contiene runner `1c778c3e...`, mientras config v3/R500, config v4 y
filesystem contienen `b35cd563...`.

La regla de `plan:65-78` es correcta por tres razones:

1. la autoridad inmediata de una transición v3→v4 es la config v3 autenticada,
   no el escrow de draw v1 ni un commit anterior a R498;
2. changed e unchanged permanecen no compensables: preparer/test deben tener
   `old_sha256` de v3 y `new_sha256` del commit correctivo, mientras runner,
   módulo, worker, R475 y R476 deben coincidir con v3 en commit correctivo,
   config final y filesystem;
3. `require_sources_at_head()`, el self-binding de config, el parentage, las
   auditorías exclusivas y la comparación física siguen operando sin ninguna
   excepción por nombre de archivo.

La corrección no debe reducirse al runner. El universo hard-set no
autorreferente tiene siete sources: dos cambiados y cinco invariantes. La
igualdad exacta de las claves y paths de ambas particiones, su disjunción y su
unión completa con esos siete sources cierran faltantes, extras y cruces. La
config se valida aparte por self-binding. Para recoveries históricos sin una
config previa tipada se conserva el comportamiento anterior, como exige el
plan.

## Amendment suplementario y continuidad de lineage

Un amendment suplementario puede cerrar R504 sin reescribir ningún artefacto.
La cadena prevista es lineal: R502/R503 y la config rechazada/R504 permanecen
como ancestros y bindings explícitos; el nuevo amendment se introduce después
de R506, R507 lo audita, y una config nueva y exclusiva v4 queda como hija de
R507. El validador ya permite que la config de recovery tome como predecessor
su `amendment_audit_commit`, por lo que no existe conflicto de parentage.

Guard de aceptación para R506/R507: el schema suplementario debe tener keyset
exacto y conservar el bloque R502 como autoridad histórica, agregando una capa
correctiva R506 separada. No debe reinterpretar ni reemplazar R502. El mapa
final debe calcularse acumulativamente desde `prior_source_sha256` de v3 hasta
el commit R506: preparer/test cambian; runner, módulo, worker, R475 y R476
permanecen byte-exactos a v3. Los bindings del amendment inicial, R503, config
rechazada, R504, plan/R505, implementación/R506 y amendment/R507 deben incluir
commit, path, SHA-256, scope/veredicto y parentage verificables donde aplique.
Estos requisitos son una concreción de `plan:121-160`, no un finding adicional.

## Poder discriminante de los tests

Las pruebas de `plan:82-117` atacan el common mode que dejó pasar el bug. La
composición positiva obliga a que el runner original y el runner v3 sean
distintos; el ataque inverso exige rechazo del blob original aunque sea
históricamente auténtico. La deriva de cualquiera de los cinco invariantes y
el delta acumulativo v3→implementación final evitan que el arreglo se limite al
caso nominal. El preflight read-only sobre una cadena Git sintética completa
comprueba finalmente config exclusiva, auditoría en HEAD y hashes físicos, no
sólo un helper aislado.

El test actual
`test_hard_set_v4_source_baseline_is_v3_not_v1_escrow` sí debe corregirse
ahora. En `test_wave60_frozen_policy_transport.py:5388-5392` lee el archivo
canónico vigente como supuesto prior; después del freeze ese archivo ya es v4,
y en `:5499-5533` construye otra v4. Reproduje el efecto CPU-only: el caso
falló en `validate_pre_draw_config()` porque el supuesto prior no tiene versión
menor, mientras el test sintético de autoridad final pasó (`1 failed, 1
passed`). Recuperar el blob v3 desde el padre de R500 o usar el snapshot v3
sellado elimina esa dependencia de fase. El costo incremental es bajo porque
preparer y test cambiarán de hash de todos modos, y el beneficio es sustantivo:
la suite focal vuelve a ser ejecutable después del freeze y puede detectar una
regresión futura.

R506 deberá además demostrar que el test positivo incluye los cinco invariantes
hard-set, no sólo los cinco sources de implementación que hoy recorre el loop
genérico. R508 sólo podrá autorizar ejecución si el preflight real retorna el
contrato contra su propio HEAD auditado.

## Alcance metodológico y recursos

La corrección no cambia draw, source law, hard-set `0.5`, políticas HGB/HGB,
features, thresholds, bootstrap, penalty, seeds ni el límite acumulativo de
`900 s`. Tampoco produce nueva evidencia científica: el resultado sigue siendo
una sola realización originada en v1. El costo es acotado a preparer/test,
fixtures sintéticos y regresión CPU; el beneficio es bloqueante porque restaura
la única composición ejecutable autorizada sin debilitar sources.

La comprobación focal usó CPU con `CUDA_VISIBLE_DEVICES=''`, plugins externos
de pytest deshabilitados y temporales retirados al terminar. Registró RSS máximo
`845880 kB`, cero swaps y `2.50 s` wall. No se usó ni se consultó GPU. La
configuración solicitada al launcher fue Codex `gpt-5.6-sol`, effort `high`;
esta instancia no expone una interfaz confiable para verificar por sí misma el
modelo o esfuerzo efectivos, por lo que no afirmo haberlos comprobado.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R505",
  "scope": "HARD_SET_AUTHORITY_R504_RESOLUTION_PLAN",
  "target": {
    "plan_commit": "0e2768b665a5df329ca53de1a68ab6e14a86d70f",
    "plan_sha256": "49e6ba42b7f6802ead82b90f042a47022902dec8b1a9e53ae1c774b99adaef0d"
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
