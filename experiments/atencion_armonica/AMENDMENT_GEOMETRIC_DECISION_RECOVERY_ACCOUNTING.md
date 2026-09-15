# Contabilidad de recuperación observable

2026-09-14. Propuesta operativa previa al primer test nuevo; requiere auditoría
e integración en el supervisor antes de ejecutarse. No cambia datos, modelos,
pérdidas, métricas, cantidad de recuperaciones ni límites numéricos.

## Motivo medido

El perfil CPU de cierre terminó en25,693755s. Su proyección original, antes
del tail final, es15857,815986s para fresh y8100,458485s para evaluación/replay.
Quedan7090,675836s en esta última etapa: la asignación original no admite la
campaña. No se congela ni se genera ningún test bajo esa asignación.

La recuperación observable post-sello consume5475,718419s proyectados, sin
redes, fitting ni lectura de respuestas. Su naturaleza es la misma que la
recuperación pre-sello ya asignada a fresh. Se propone contabilizar ambas en
fresh y reservar evaluación para reconstrucción de verdad, métricas, bootstrap
y replay de métricas. Es un cambio explícito respecto de la asignación de
replay del protocolo original; no se presenta como si éste ya lo dispusiera.

## Operaciones y límites

1. Fresh conserva producción/predicciones, recuperación pre-sello y sello.
2. Evaluación abre respuestas sólo tras COMPLETE y sello global válido.
3. Replay comienza con una operación CPU de recuperación observable post-sello,
   cargada a fresh: revalida el sello y recupera los cuatro batches completos.
4. Sólo tras COMPLETE de esa operación, el replay de métricas corre bajo
   evaluación. Ambos recibos quedan enlazados al mismo freeze y ledger.

Se transfiere a fresh exactamente la recuperación post-sello más una
verificación global del inventario. El perfil contabiliza cinco inventarios:
uno pasa con esa recuperación y cuatro permanecen en evaluación. No se
elimina trabajo ni se cobra dos veces el mismo tramo. Se conserva margen25%,
setup y tail en las proyecciones y se admite el total antes del freeze.

Los topes permanecen fresh21600s, evaluación7200s, entrenamiento14400s,
auditoría3600s y total49200s, con los demás límites intactos. No se edita el
módulo de presupuesto congelado ni se reinicia el ledger. Cada recuperación
es una operación distinta con reserva explícita; su tiempo no puede consumir
silenciosamente la reserva destinada a otra operación.

## Cálculo reproducible

Raíz: `data/atencion_armonica/geometric_decision_energy_v1/`.

- Perfil `profiles/closing-cpu-0/result.json`, SHA256
  `dc1e3b20723b91303c1af7916c20497e8a36ec3679aca2d9f731ba8bcbc968b9`.
- Finish `control/attempts/0011/finish.json`, SHA256
  `4f08e8fbd6252c087acac654928f2e7aefdf4075a6e132e3735ec1df5fb8e615`.
- Recuperación: `admission.json`, `observed_profile.observable_recovery_with_closing_seconds`.

Tail=finish.seconds−elapsed_to_forecast_seconds=1,107129735s. Cada etapa
conserva5×tail como en la admisión original. Inventario transferido=
1,25×128×timings.observable-inventory=61,066988632s. Transferencia total=
5536,785407672s. Resultan21400,137042s para el conjunto fresh y2569,208726s
para evaluación. Con los cargos al cierre0011, fresh suma21410,075762s sobre
21600s; evaluación2678,532890s sobre7200s. El nuevo setup todavía debe
cargarse y la admisión recalcularse desde su recibo real antes del freeze.

El margen de fresh por encima de esta proyección con25% es pequeño, unos190s
antes de ese setup. No es garantía OOD: si una reserva o guard se agota,
preservar artefactos y detenerse sin reducir roster, sustituir escenas ni
relanzar automáticamente. El forecast completo y las reservas por operación
deben quedar en el freeze. Esta enmienda no declara suficiencia científica.
