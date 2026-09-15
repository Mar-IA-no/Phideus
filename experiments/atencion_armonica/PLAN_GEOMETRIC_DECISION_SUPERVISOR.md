# Supervisor del contraste prospectivo

2026-09-14. Ejecución del protocolo vigente, sin cambios científicos. Tres
modos explícitos: observaciones nuevas, evaluación sellada y replay. El perfil
CPU completo es prerrequisito; ningún modo entrena, selecciona ni repite perfiles.

Cada lanzamiento abre sólo el binding autenticado del control antes de tomar
el lock. La admisión material de OPEN, entrenamiento, selección, archivo,
perfiles, cabezas y procedencia corre con alarmas y presupuesto vivo de120s,
cargado a fresh o evaluación según el modo. Conserva un recibo independiente
en el mismo ledger acumulativo; un intento parcial no se reejecuta en silencio.
Esta fase permite conocer la reserva final sin ejecutar datos nuevos. El
perfil de cierre ya reserva cuatro setups por etapa para estos accesos.

Después del recibo COMPLETE de admisión, congelar los valores autenticados,
los cuatro escenarios de512escenas y144estados, normalización, escala,
exclusiones, runtime y código. La admisión de recursos incluye bytes ya
ocupados más bytes nuevos proyectados, espacio libre, recuperación antes y
después del sello, métricas y auditoría final. La publicación del freeze y el
resto del lanzamiento se cargan desde el instante posterior al cierre de la
admisión. Una reapertura exige igualdad con el freeze; no cambia semillas.

El operador fresh usa guard de VRAM propia además de RSS/disco/tiempo,
fracción del asignador CUDA0,25, comprobación de ownership y runtime exacto.
Produce los cuatro batches, conserva todos los outputs, recupera sin cálculo
neuronal y sella el árbol completo. Evaluación y replay usan CUDA oculto;
el replay vuelve a verificar el árbol antes de recuperar observables y luego
recalcula métricas desde los mismos bytes. Nunca repara un resultado ausente.

La enmienda `AMENDMENT_GEOMETRIC_DECISION_RECOVERY_ACCOUNTING.md` asigna
recuperación observable post-sello a fresh, con reserva y finish propios;
sólo ese COMPLETE habilita replay de métricas en evaluación. El freeze fija
la enmienda y las reservas separadas. Al reabrirse, exige igualdad de todos
los inputs y procedencia con la admisión fresh original, no sólo del código;
los cargos históricos se validan contra aquel finish, no contra el saldo
actual. La fórmula de costes es un helper puro: importar el supervisor no
carga el parser de respuestas, reservado al puerto posterior al sello.

No repetir un modo que ya cerró ni continuar automáticamente una corrida
parcial: preservar sus artefactos y diagnosticar primero. Los fixtures deben
probar orden de admisión, interrupción, cargos acumulados, VRAM sin CUDA real,
rechazo de presupuesto insuficiente y bloqueo de reejecución. Auditar el
supervisor integrado antes de ejecutar el contraste con datos nuevos.
