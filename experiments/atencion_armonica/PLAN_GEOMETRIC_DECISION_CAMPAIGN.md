# Ejecución recuperable de las 72 celdas

2026-09-14. Implementación del protocolo ya aprobado; no modifica sus brazos,
épocas, fuentes, selección ni tests. Los perfiles reales están completos.

1. Autenticar los cierres COMPLETE de OPEN, inputs de perfil y ambas cabezas
   CPU/CUDA, sus bindings, tiempos y digests de recuperación. Proyectar todo
   el roster: arranque de cinco updates por celda, restantes updates, snapshots,
   calibración y tres cargas completas. No multiplicar el arranque CUDA cada
   25 updates; conservar los tiempos originales. Elegir menor costo proyectado
   y aplicar 25% de reserva antes de la admisión. El forecast no es cota probada.
2. Congelar protocolo, código importado, runtime/dispositivo, perfiles, fuente,
   escala, roster y presupuesto. Verificar ownership antes de CUDA. Reusar
   el ledger global y lock exclusivo; no iniciar tests ni seleccionar épocas.
3. Cargar un backbone completo por vez, reautenticando OPEN. Ejecutar sus ocho
   brazos × tres seeds con el runner auditado; repetir para los tres backbones.
   Reanudar por snapshots completos y conservar todas las calibraciones.
   No mutar los arrays después de validar. Las celdas ya completas se verifican
   sin volver a entrenar ni calcular outputs.
4. Registrar duraciones fuera del estado numérico: segmentos de tiempo entre
   eventos de época, con intento y frontera de reanudación. Incluyen trabajo
   de recuperación/calibración que caiga dentro del segmento; no se presentan
   como tiempo puro de optimización. Conservar segmentos parciales al pausar.
5. Publicar cierre sólo con las 72 celdas, 50 épocas y once calibraciones por
   celda verificadas. El finish COMPLETE del operador autoriza ese cierre;
   un archivo parcial, timeout o presupuesto agotado no lo sustituye.

La siguiente fase selecciona una época por brazo con las 720 calibraciones
elegibles y luego congela la evaluación prospectiva. El entrenamiento completo
no cierra por sí solo el goal: faltan 2048 tests, probes, replay y auditorías.

Auditoría independiente de controlador, admisión y forecast antes de ejecución.
Fixtures separan topología del roster y recuperación real de una celda; no
presentar simulaciones del controlador como 72 entrenamientos realizados.
