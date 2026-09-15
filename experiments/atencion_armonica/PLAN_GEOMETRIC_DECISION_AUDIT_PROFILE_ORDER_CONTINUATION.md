# Continuación por corrección del orden del inventario

2026-09-15. Diseño previo revisado en R785; implementación y su revisión pendientes.

El intento0023 pasó la apertura OPEN y se detuvo en el inventario del perfil
observado. Productor y auditor comparaban los mismos2106 archivos con órdenes
distintos: objetos Path frente a strings relativos. El checker corregido
conserva el orden del productor, el roster exacto, el rechazo de symlinks y
artefactos desconocidos, y la autenticación de cada referencia. No cambia
ninguna fórmula, fuente congelada, dato experimental ni estimando.

## Autoridad y fronteras

1. No ejecutar de nuevo ninguno de los dos wrappers previos. La continuación
   será una entrada nueva y fija, no un mecanismo general de retries ni roots
   elegibles por CLI. Conservar los sources previos en Git y sus receipts como
   historia ligada a esas versiones.
2. Antes del start, bajo lock común, exigir ledger con tail0023 FAILED, sin
   intentos posteriores; autenticar finish, start, manifest y discrepancia.
   Finish SHA37515c1098a58d057ffff326330d4f47c4d621f0533266cbbc0d5cd7cfd52e90;
   discrepancia SHA0b9bab7e7744310ae5c30e69bca55fecbdf067f3683cc1c02816fa9993046133.
3. Autenticar binding/PRECOMMIT de `audit-final-adapter-verify`, cuyo PRECOMMIT
   tiene SHA7b973254c40cfe46800eb16eaac993cab1fd0c29a4fa5434849c35c68f917b19.
   Deben ligar el mismo manifest de0023 y distinguir checker7b1622f de la
   revisión nueva. Validar también sus referencias a0022 y el PRECOMMIT original,
   sin exigir que los sources viejos coincidan con el checker actual.
   Exigir topología exacta `{binding.json, precommit.json}`, sin symlinks ni
   entradas adicionales, en ambos roots históricos.
4. Usar `audit-final-profile-order-verify` y nombres nuevos fijos
   `final-technical-audit-profile-order.json` bajo manifests/outputs/discrepancies.
   Toda presencia de esos destinos o de un intento0024 bloquea otro lanzamiento.
5. Reutilizar funciones puras pertinentes de la continuación anterior, ligando
   su archivo por hash si se importa. No modificar sus constantes globales ni
   publicar outputs nuevos bajo nombres antiguos. El nuevo camino mantiene
   `operation=final-technical-audit`, como exige VERIFY.
   La allowlist de reutilización es `approved_source`, `current_checker` y
   `fixed_precommit`. Los helpers de destinos, start, ejecución y fallo son
   locales; el reloj de lanzamiento nuevo precede las importaciones pesadas.
6. Ligar por hash wrapper/plan nuevos, checker/core, plan científico/protocolo,
   helper importado, R784, R785 y registro de acceso posterior. Conservar en
   binding y manifest nuevos las referencias al root/binding/PRECOMMIT0023 y
   transitivamente al original0022. Revalidar esos sources antes y después.

## Presupuesto, corte y ejecución

El cargo audit tras0023 es1764.753717084881s y el saldo1835.246282915119s.
La nueva reserva no superará1500s ni saldos revalidados del ledger; caps
audit3600/global49200 intactos. No otra liability:0023 cobró117.82238421600778s
medidos y0021 sigue computado exactamente una vez por su reserva histórica.

Reloj desde lanzamiento, manifest/start antes del store nuevo o lecturas
pesadas, guards CPU1thread/CUDAhidden/RSS/disco y cierre medido de fallos.
Conservar el preflight y su proyección; no repetir campaña, perfiles,
trainings, selección, fitting, forward, evaluación, replay ni informe.
El VERIFY íntegro aún debe ejecutarse; no convertir las fases atravesadas de
un intento fallido en un certificado completo ni reducir el alcance.

El PRECOMMIT nuevo debe mantener igualdad canónica completa con el0023 y el
original0022: cuts, terminales científicos, freeze, seal, contrato y origen
del informe, además del preflight. Conservar el acceso posterior declarado
como true y ligar el registro histórico pinneado. Publicar antes de VERIFY.
Revalidar fuentes revisadas antes del start y al cierre.

## Condición previa a corrida

Auditar el camino nuevo concreto y sus fixtures de store/budget reales,
historia mutada, corte alterado con iguales conteos, no-retry y fallos cobrables.
Las regresiones del orden prueban también contenido modificado y symlink extra.
Publicar código revisado antes de lanzar una única vez y conservar su handle.
Un COMPLETE técnico no sustituye informe humano, auditoría de horizonte,
propagación documental ni cierre íntegro del goal.
