# Compatibilidad entre parciales: perfil de recursos GPU

Fecha: 2026-09-07. GPU: NVIDIA GeForce RTX 3090.
Código congelado: `bfc71c8`. Estado de este corte: **perfil completo**.
La campaña posterior ya tiene [resultados completos](RESULTS_SHARED_PARTIAL_STUDY.md).

Después del [preflight CPU](RESULTS_SHARED_PARTIAL_PREFLIGHT.md), la prueba
autorizada ejercitó los cinco brazos con fixtures máximos, batch 128, FP32 y
TF32 deshabilitado. Por brazo hizo un warmup y tres pasos medidos de
entrenamiento N=24, seguidos por un warmup y tres forwards N=32. No abrió
splits de test ni entrenó el corpus del estudio.

| Medición | Resultado |
|---|---:|
| Tiempo total del perfil | 3,473 s |
| Máximo reservado por el allocator PyTorch | 1.235.222.528 bytes (1,15 GiB) |
| Máximo asignado a tensores | 975.554.560 bytes (0,91 GiB) |
| Proyección de quince trainings | 1.255,73 s |
| Proyección de sus seis evaluaciones finales | 13,85 s |

La proyección suma unos 21,16 minutos de cómputo. Usa el máximo de tres tiempos
por brazo, no una cota superior. Excluye generación/cache, lectura de disco,
checkpoints, lector/métricas CPU y espera del scheduler. La memoria informada
es la del allocator, no todo consumo externo del dispositivo. El runner debe
seguir controlando el presupuesto agregado de 24 horas GPU del
[protocolo](PLAN_SHARED_PARTIAL_COMPATIBILITY.md), sin cambiar batch o épocas
para presentar una receta reducida como completa.

Los estados iniciales, estados después de cuatro pasos mecánicos, optimizador,
RNG, logits y componentes quedan preservados por brazo. Son artefactos de
perfil, no checkpoints de la campaña ni evidencia de generalización. Se usó
un snapshot del código auditado, separado de la implementación posterior del
cache. Tras terminar no quedó un proceso propio ocupando GPU.

Fuentes: [manifest](../../data/atencion_armonica/shared_partial_gpu_profile_v1/manifest.json)
y [mediciones](../../data/atencion_armonica/shared_partial_gpu_profile_v1/profile.json).
El manifest liga siete fuentes runtime y trece artefactos de resultado.

La medición respalda la factibilidad de cómputo de la receta. El informe de
la campaña posterior distingue estos tiempos proyectados de los medidos
durante training y evaluación. No hay promoción arquitectónica ni GO/NO-GO.
