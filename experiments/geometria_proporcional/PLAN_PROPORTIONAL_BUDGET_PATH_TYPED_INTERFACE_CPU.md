# Plan CPU — interfaz tipada `BudgetPath`

**Fecha:** 2026-09-04
**Estado:** oficial y replay byte-exacto completados
**Régimen:** materialización checker-only sobre evidencia abierta R369–R372
**Arquitectura:** candidata separada; el baseline y sus resultados no cambian
**Autoridad:** valida estructura, lineage y recomputación; no elige política, utilidad, arquitectura ni GO/NO-GO

## Problema arquitectónico

R371 mostró que una salida set-valued representa mejor la tensión IID/grouped
que el firewall binario. R372 explicó su mecanismo: la ruta de acciones es
anidada, pero las relaciones pareadas transportan sólo parcialmente. Guardar
únicamente `front_policy_ids` pierde la diferencia entre:

- una expansión que mejora ambas coordenadas;
- una expansión que compra beneficio grouped con costo IID;
- una membresía de frente estable producida por relaciones que cambiaron.

La candidata `BudgetPath` conserva la ruta completa y deja la elección fuera de
su jurisdicción.

## Fuentes congeladas

- acciones R369, manifest
  `bc898c97ed3730e8719e344dc51d102a61637324eb7c3d79cc3c7e677941f994`;
- objetivos y frentes R371, manifest
  `88333aed65a78444a92b4236979981ccfedc1d695fe0c86fd8a26ab89625a34c`;
- relaciones pareadas R372, manifest
  `2e21aab7b3b7b5f1cea49bb91add2fc939033addb6432572cc637915e1d03b63`.

No se ajusta ningún modelo ni se crean vistas, solves, bootstraps o thresholds.

## Unidad tipada

Se materializan `608` artefactos independientes:

```text
2 cohortes × 2 regímenes × 4 brazos × 2 roles ×
(3 familias principales + 16 controles permutados)
```

Cada `BudgetPath` declara:

1. `lineage`: cohorte, role, propuesta, brazo, familia, réplica opcional y los
   tres manifests fuente;
2. `axes`: exactamente IID y grouped, estimador
   `mean_quotient_rmse_delta`, unidad `quotient_rmse`, dirección `minimize`;
3. `policies`: identity y presupuestos `1/2/5/10/20/40%` en orden, con fraction,
   cantidad de acciones, hash del vector de acción, coordenadas medias, hash
   del objetivo por master, frecuencia bootstrap y membresía al frente;
4. `pairs`: los 21 pares `base→expansion`, estado exclusivo, incremento medio,
   frecuencia de cinco estados y hash del incremento por master;
5. `reader`: salida `PARETO_SET` que devuelve todos los IDs no dominados;
6. estados de autoridad separados: artefacto chequeable, claim estructural,
   evidencia post hoc abierta, autoridad física no reclamada y decisión no
   resuelta;
7. `utility_boundary`: `ABSENT_EXTERNAL_REQUIRED`, sin peso, score ni política
   recomendada embebidos.

## Separación builder/checker

El builder ensambla R369, R371 y R372. El checker vive en un módulo distinto y
no importa el builder ni las funciones de Pareto/dominancia R371–R372. A partir
de acciones, objetivos e índices fuente reconstruye independientemente:

- schema y tipos exactos;
- orden de políticas y semántica de ejes;
- anidamiento de soporte y preservación de alpha;
- hashes, cantidades y medias;
- frente no dominado y frecuencias bootstrap;
- estados, incrementos y frecuencias de los 21 pares;
- consistencia del reader y separación de autoridades.

Un checker inválido no refuta la hipótesis empírica; invalida el artefacto.

## Suite adversarial congelada

Sobre copias de un artefacto representativo se ejecutan mutaciones
`PROTOCOL_INVALID`, todas con rechazo esperado:

1. `selected_policy_id` añadido;
2. peso de utilidad embebido;
3. semántica IID/grouped intercambiada;
4. hash de acción alterado;
5. media de objetivo alterada;
6. frente declarado alterado;
7. estado pareado alterado;
8. manifest fuente alterado.

Además, tests unitarios ejercen una ruta de acciones no anidada. La suite no
contiene un negativo semántico ambiguo: cada mutación viola una condición
mecánica declarada.

## Criterios de ejecución

- `608/608` artefactos aceptados por el checker independiente;
- todas las mutaciones rechazadas por el motivo esperado;
- ningún campo de utilidad o selección en la salida canónica;
- replay byte-exacto y arrays fuente sin modificación;
- `CUDA_VISIBLE_DEVICES=''`, un thread, máximo `5 min` y `4 GiB`.

Cumplir estos criterios acredita una interfaz estructuralmente consistente, no
su utilidad científica, seguridad, generalización, promoción ni GO/NO-GO. Toda
ejecución GPU permanece en cola.

## Artefactos

Output canónico:
`data/geometria_proporcional/proportional_budget_path_typed_interface_v1/`.

Contendrá `budget_paths.jsonl`, receipts del checker, resultados adversariales,
resumen, config resuelta, entorno, manifest y replay. Los arrays pesados no se
duplican: cada hash apunta a los NPZ preservados por R369–R372.

## Ejecución

Diseño `443760b`, implementación `239d363`. Oficial y replay terminaron en
`163,513/162,829 s`, ambos con `0,691 GiB`. Los `608/608` artefactos fueron
aceptados por reconstrucción independiente y las `8/8` mutaciones rechazadas.
Los siete archivos deterministas y el manifest fueron byte-idénticos:

```text
2e767e7e1a67afa97ac8295429e2a157452363f0fbd54a1ea42904e72b50886c
```

La regresión ampliada cerró `214/214`; `gpu_queried: false`.
