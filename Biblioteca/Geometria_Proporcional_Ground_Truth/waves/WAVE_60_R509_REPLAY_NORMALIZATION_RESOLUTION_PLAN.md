# Ola 60 — plan de resolución de la normalización de replay observada por R509

> **Estado:** `PRE-IMPLEMENTATION / R509-REVISE / PAIR-COMPLETE-IMMUTABLE / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **Findings de origen:** `R509-01` MEDIUM, `R509-02` LOW
> **Intento:** `data/geometria_proporcional/wave60_frozen_policy_transport_attempt_v4`
> **Pair manifest:** `4a51993c420a96f9f8283a3ced5e9dee279097ce37453fbd2b7480580805e686`
> **R509:** commit `92305f4e54e72ee78924ca4b51ae5889369d805b`, informe SHA-256 `006a43e9257a340b162583bcf1190cf34a27085186643a1ea37d987b4fa45e28`

## 1. Estado que no se reabre

Wave 60 v4 terminó como `COMPLETE`, con primary y replay
`EVALUATED_IMMUTABLE`, firmas válidas, manifests cerrados y decisión científica
`null`. R509 recompuso exactamente 14 acciones, 56 arrays métricos, 301
pair-tokens y 5.000 réplicas bootstrap. La ejecución, los números y los ocho
artefactos científicos comparados entre roots son íntegros.

El intento es post-truth y no admite recovery ni repetición automática. No se
editará, añadirá, moverá ni volverá a sellar ningún archivo dentro de
`wave60_frozen_policy_transport_attempt_v4`. Tampoco se modificará la config
auditada por R508 ni los ocho sources ligados por ella. Esos bytes constituyen
la evidencia histórica de la corrida efectivamente realizada.

## 2. Defecto observado

`compare_evaluated_roots()` compara correctamente
`generation_receipt.json` mediante seis campos semánticos. Primary y replay
coinciden en todos ellos; sólo difieren, como corresponde, en
`execution_mode=recovery` y `execution_mode=replay`. Cada
`preparation_receipt.json` liga mediante SHA-256 su propio generation receipt.

La comparación de `preparation_receipt.json` vuelve a exigir igualdad
cross-root de esos dos SHA locales. Como los archivos role-specific son
legítimamente distintos, esa igualdad falla aunque ambos enlaces locales sean
correctos y la comparación semántica ya haya pasado. El único mismatch
publicado es entonces:

```text
operational:preparation_receipt.json
```

El finalizador transforma el status global en `replay_exact=false`. Además
copia a `final_analysis.json` la limitación root-level
`replay_exact_pending_pair_finalize`, que queda obsoleta una vez publicado el
terminal del par.

## 3. Alcance de la corrección

La resolución será una adjudicación derivada fuera del paquete sellado. Tendrá
dos componentes nuevos y auditables:

```text
experiments/geometria_proporcional/adjudicate_wave60_v4_result.py
tests/test_wave60_v4_result_adjudication.py
```

El checker no reemplazará al runner congelado ni fingirá que el artefacto
original decía otra cosa. Debe leer el terminal tal como existe, autenticar la
evidencia y publicar una corrección explícita con doble estado:

- observación original: `MISMATCH`, `replay_exact=false`;
- adjudicación normalizada: replay científico y operacional semántico exactos,
  con los enlaces locales validados por separado.

La salida canónica prevista será:

```text
Biblioteca/Geometria_Proporcional_Ground_Truth/waves/
WAVE_60_V4_REPLAY_NORMALIZATION_CORRECTION.json
```

Debe crearse de manera exclusiva después de que la implementación reciba una
auditoría independiente. No puede escribirse dentro del intento ni sobre un
path existente.

## 4. Regla de normalización

La normalización es admisible sólo si se prueban simultáneamente estas
condiciones:

1. el pair manifest, pair status, final analysis, replay comparison y ambos
   root manifests coinciden con los hashes auditados por R509;
2. el terminal es `COMPLETE`, ambos roots son `EVALUATED_IMMUTABLE` y las
   firmas/manifests continúan válidos;
3. `replay_comparison.json` se recompone exactamente desde los roots con el
   runner histórico y contiene un único mismatch, exactamente el declarado;
4. las 35 comprobaciones restantes son verdaderas, incluidos todos los hashes
   científicos, arrays tipados, estados funcionales y secretos opacos;
5. cada `preparation_receipt.json` apunta al SHA-256 del generation receipt de
   su propio root;
6. ambos preparation receipts comparten `preparation_freeze_sha256` y
   `next_state`, y los preparation freezes son byte-exactos;
7. los generation receipts coinciden en el allowlist semántico ya congelado y
   su única diferencia estructural es `execution_mode` con los valores
   `recovery`/`replay`;
8. los análisis primary/replay son byte-exactos y la recomposición numérica de
   R509 permanece ligada por hash.

No se aceptará una regla genérica de “ignorar diferencias”. Cualquier mismatch
adicional, enlace local roto, diferencia científica, diferencia secreta,
campo extra role-specific o cambio en los hashes target debe abortar sin
publicar corrección.

## 5. Adjudicación corregida

Una vez satisfechas las ocho guardas, el checker derivará
`normalized_replay_exact=true` y volverá a aplicar la función de agregación a
las condiciones root-level ya selladas. No recalculará logits, acciones,
métricas ni bootstrap.

La corrección debe dejar explícito que:

- todas las métricas y deltas permanecen byte-ligados al analysis original;
- los patrones `incompatibility` y `harm` continúan `false` aun con replay
  normalizado;
- incompatibility sigue fallando el contraste de regret contra controles
  máximamente desplazados, cuyo IC95 superior es `+0.001548...`;
- harm sigue fallando el contraste de worst regret contra esos controles, cuyo
  IC95 superior es `+0.009468...`;
- `scientific_decision` permanece `null` y `decision_authority=user`;
- la limitación pendiente se sustituye en la vista derivada por una nota de que
  replay fue adjudicado post-terminal mediante la resolución R509;
- las demás limitaciones —generador sintético, ausencia de identificación del
  efecto target e intervalos sin corrección de multiplicidad— permanecen.

Por lo tanto, resolver el falso negativo de replay corrige reproducibilidad y
estado de fase, pero no convierte ninguno de los dos patrones en positivo ni
autoriza promoción arquitectónica.

## 6. Autoridad y closed world de la salida

El JSON derivado deberá usar un schema propio y keyset exacto. Como mínimo
ligará:

- path y seis hashes target del intento auditado;
- config R508, su SHA físico y self-binding;
- R509 REVISE y sus conteos `0/1/1`;
- este plan y su auditoría independiente;
- implementación del checker y su auditoría independiente;
- observación original, evidencia normalizada y adjudicación corregida;
- hashes de los analysis primary/replay;
- condiciones y patrones antes/después;
- limitaciones corregidas;
- `scientific_decision=null` y autoridad del usuario.

El checker debe validar también su propia salida. La auditoría del artefacto
repetirá la derivación desde los estados crudos y comprobará serialización
canónica, introducción Git exclusiva, ausencia de mutaciones en el intento y
ausencia de cambios en los sources congelados.

## 7. Pruebas discriminantes

La implementación debe cubrir, al menos:

1. caso real positivo del terminal v4, sin escribir durante validación;
2. fixture con receipts role-specific que reproduzca el common mode de R509;
3. rechazo si se compara o acepta el SHA local del otro root;
4. rechazo por enlace local roto en cualquiera de los dos receipts;
5. rechazo ante cualquier mismatch científico, funcional, secreto u
   operacional adicional;
6. rechazo si los generation receipts difieren en algo más que
   `execution_mode` o si los roles no son `recovery`/`replay`;
7. recomposición exacta de condiciones/patrones con replay normalizado;
8. transición de la limitación pendiente sin borrar las tres limitaciones
   científicas vigentes;
9. publicación exclusiva, canónica, fuera del intento y sin overwrite;
10. prueba de que los seis hashes target y los ocho source hashes congelados no
    cambian;
11. medición CPU de wall, RSS y swaps, sin uso ni consulta de GPU.

## 8. Cadena prevista

```text
R509 REVISE
  -> este plan
  -> R511 auditoría independiente del plan
  -> implementación checker/tests
  -> R513 auditoría independiente de implementación
  -> JSON de adjudicación corregida
  -> R515 auditoría independiente del artefacto
  -> documentación y wiki
```

Los números pares R510/R512/R514 designan los commits de plan,
implementación y artefacto; los impares designan sus auditorías. Un finding
nuevo se conserva y se resuelve en otro eslabón, nunca mediante reescritura del
informe previo.

## 9. Criterio de continuidad

Sólo se implementará si R511 confirma que la normalización distingue igualdad
semántica cross-root de integridad referencial local y que la salida externa
puede resolver R509 sin alterar el terminal. Sólo se publicará el JSON si R513
acepta el checker y los ataques discriminantes. La documentación usará la vista
corregida únicamente después de R515 y mostrará a la vez el defecto original,
su alcance y la evidencia científica inalterada.

Ningún eslabón declara `GO/NO-GO`. La decisión científica continúa reservada a
Mariano.
