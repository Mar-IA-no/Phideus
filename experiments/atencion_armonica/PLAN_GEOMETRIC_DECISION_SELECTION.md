# Selección común de época, sin abrir tests

2026-09-14. Continuación del protocolo congelado. No cambia modelos, loss,
roster, criterio de selección ni presupuesto acumulado.

1. Autenticar el finish COMPLETE de entrenamiento, su manifiesto, binding y
   cierre de las 72 celdas. No basta encontrar checkpoints o un complete.json.
   El operador comparte el lock y ledger de la campaña y funciona sólo en CPU.
2. Releer las calibraciones conservadas, sin forward, fitting ni entrenamiento.
   Verificar identidad y orden de escenas/candidatos, precisión, offsets y
   referencias de estados; comprobar igualdad de targets entre backbones.
3. Elegir una época por brazo entre 5, 10, ..., 50 mediante el selector ya
   auditado: regret tD, media de las nueve celdas dentro de cada escena y
   después media entre escenas elegibles; empate exacto favorece la anterior.
   La época cero se verifica pero no compite. Conservar las 720 referencias,
   los regrets por escena/época y las épocas elegidas.
4. Perfilar únicamente el nuevo trabajo CPU de selección/lectura con arrays
   aritméticos declarados de 512 escenas y 82 candidatos, sin llamar samplers.
   Reusar la carga por backbone ya medida y añadir 25% a la proyección.
   Contar ambas lecturas de calibración (validación y selección), la carga
   CPU del estado final por celda y la autenticación de sus once blobs.
   No atribuir a los fixtures evidencia científica. El perfil consume el
   presupuesto de perfil existente; la selección, evaluación/replay.
5. Antes de ejecutar, auditoría independiente de admisión, implementación,
   recursos y recuperación. La recuperación revalida bytes y conserva
   artefactos idénticos; no sobrescribe una selección previa diferente.

Este cierre sólo autoriza preparar el sello prospectivo posterior. Los tests
y probes nuevos siguen sin abrirse hasta congelar selección, exclusiones y
roster de outputs. El goal completo permanece pendiente de esas etapas.
