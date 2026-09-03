# Ola 54 — cierre del posterior conjunto regularizado

> **Estado:** `CLOSED / CPU-ONLY / OPENED-HISTORICAL / EXACT-REPLAY / NO-GO-NOGO`
> **Fecha:** 2026-09-03
> **Plan congelado:** `WAVE_54_JOINT_SET_POSTERIOR_PLAN.md`
> **Hash analítico:** `694bc4c050d617b1c271ddd5dac1e7f51aa09c5335ba957ac0fdac7b047f1e8b`

## Qué se puso a prueba

La Ola 53 había conservado la incertidumbre de cada familia por separado y
reconstruido los conjuntos compatibles como producto de cuatro pertenencias
Bernoulli. Esa representación mejoró compatibilidad y permitió abstenerse, pero
distorsionó cardinalidad y dependencias. La Ola 54 aisló ese problema: mantuvo
congelados encoder, logits, splits, utilidades, pérdida y población, y sustituyó
solamente la factorización independiente por un posterior regularizado sobre
los quince subconjuntos no vacíos.

El modelo conjunto aprendió cuatro pendientes unary, tres sesgos de cardinalidad
y cinco contrastes de interacción que inducen seis coeficientes de suma cero.
Esta parametrización separa la preferencia global por tamaño de conjunto de las
dependencias heterogéneas entre familias. La comparación incluyó variantes unary,
unary+cardinalidad y full, además de marginales raw/Platt, conjunto duro, prior
empírico, targets barajados y una referencia privilegiada de decisión.

## Protocolo e integridad

Un preparador ciego produjo bundles físicamente separados para ajuste/selección
(`192 + 192` tokens) y monitor sellado (`384` tokens), con cero `pair_token`
compartidos. El monitor no se abrió ni se recorrió para hashing antes de congelar
modelos, regularización y comparador independiente. Todas las fuentes de ejecución
debían pertenecer a `HEAD`; los bundles y manifiestos quedaron ligados por hash.

La corrida primaria y una réplica en proceso y directorio independientes
coincidieron exactamente en los cuatro estados NPZ y en el hash analítico. La
auditoría independiente `R306` verificó los once criterios, signos, intervalos,
selección, masa fuera de soporte y controles sin findings invalidantes. Este
resultado acredita cálculo y trazabilidad del smoke histórico; no revalida los
logits upstream ni abre el lockbox.

## Resultado primario

Población `NEAR_RIVAL`, cardinalidad mayor o igual que dos, `148` tokens de
monitor. `best_independent` quedó fijado como `independent_platt` en
`decision_select`.

| Contraste o propiedad | Resultado | Lectura |
|---|---:|---|
| NLL conjunta vs Platt independiente | `-0.0818`, IC95 `[-0.1323, -0.0369]` | mejora clara |
| L1 de distribución de cardinalidad | reducción `0.0871` | mejora clara |
| NLL vs unary+cardinalidad | `-0.0488`, IC95 `[-0.0922, -0.0118]` | las interacciones agregan señal |
| Compatibilidad vs conjunto duro | `+0.0363` | mejora |
| Regret vs conjunto duro | `-0.0130`, IC95 `[-0.0378, +0.0084]` | margen e IC insuficientes |
| Accuracy vs conjunto duro | `-0.0577` | falla la no inferioridad |
| Accuracy vs prior empírico | `+0.1405` | control superado |
| Accuracy vs target-shuffled | `+0.0386` | señal positiva, menor que `+0.05` requerido |
| Masa sobre clases no vistas, joint vs raw | `0.0252` vs `0.0667` | control satisfecho |

El patrón conjunto predeclarado quedó `false`. Ocho de once checks pasaron; los
tres negativos fueron regret frente al conjunto duro, no inferioridad de
accuracy y margen frente al control shuffled. Ninguno se explica por sensibilidad
del selector: cambiar la regularización primaria por la seleccionada globalmente
no invirtió el signo de los ocho contrastes vinculantes.

## Implicancia arquitectónica

El resultado separa dos problemas que una sola métrica habría confundido. Como
modelo probabilístico, el posterior conjunto recupera estructura que las
marginales independientes no expresan: predice mejor el conjunto exacto, corrige
su cardinalidad y obtiene una contribución adicional de interacciones
heterogéneas. Como política de acción, esa mejora no alcanza todavía el balance
exigido: eleva compatibilidad, pero sacrifica exactitud y no reduce regret con la
magnitud predeclarada.

La consecuencia no es volver al conjunto duro ni escalar silenciosamente el
encoder. El objeto conjunto queda como representación candidata recuperable,
mientras la deuda se desplaza a la interfaz entre posterior y decisión. Un
discriminante posterior deberá probar, sin usar monitor para seleccionar, si una
regla híbrida o un objetivo decisional explícito puede conservar la ganancia de
NLL/cardinalidad sin pagar la caída de accuracy. También deberá ampliar soporte:
`calibration_fit` observó sólo `10/15` conjuntos, por lo que esta ola no autoriza
inferencias sobre las cinco clases ausentes.

## Alcance

La evidencia corresponde a un smoke CPU sobre logits históricos congelados y a
la población observada. No establece utilidad natural, geometría física,
causalidad del encoder, transferencia externa ni promoción arquitectónica. La
decisión GO/NO-GO permanece en el usuario.

## Artefactos

- `data/geometria_proporcional/wave54_joint_set_inputs_v1/`
- `data/geometria_proporcional/wave54_joint_set_v1/`
- `data/geometria_proporcional/wave54_joint_set_v1_replay/`
- `agent_reports/304_wave54_implementation_independent_audit.md`
- `agent_reports/305_wave54_implementation_focal_reaudit.md`
- `agent_reports/306_wave54_result_independent_audit.md`
