Informe Técnico Senior — Revisión de Gate 4 (Ratio Auxiliary)

Documento evaluado: plan_gate4.md (line 1)
Contexto integrado: estado de Gate 2/3 y scripts actuales de entrenamiento/evaluación.

Dictamen Ejecutivo
El plan es conceptualmente sólido y está bien orientado al aprendizaje del Gate 3: pasar de invariancia adversarial (DANN) a inyección de estructura (ratios) es una decisión técnicamente defendible para este proyecto.
Mi veredicto es GO condicional: lo aprobaría, pero con correcciones bloqueantes antes de ejecutar.

Hallazgos Críticos (bloqueantes)
Riesgo de evaluación inválida por incompatibilidad de checkpoints Gate 4 vs evaluador estructurado.
Evidencia: Gate 4 guarda self.model.state_dict() de MultiViewModel en gate4_ratio_auxiliary.py (line 480), mientras el evaluador carga un CrossModalModel con checkpoint['model_state_dict'] en evaluate_structured_pool.py (line 371) y evaluate_structured_pool.py (line 372).
Impacto: podés terminar evaluando pesos no alineados al modelo esperado (silenciosamente por strict=False).
Recomendación: exportar explícitamente un checkpoint “base-only” (compatible con CrossModalModel) para evaluación estructurada.

Inconsistencia de régimen de datos respecto al baseline (comparabilidad comprometida).
Evidencia: Gate 2 seleccionado fue con segment_len=4.0, hop=1.0, batch_size=16 en INFORME_GATE2_COMPLETO.md (line 76) y INFORME_GATE2_COMPLETO.md (line 77); además el ajuste 8→4 y 2→1 está documentado en INFORME_GATE2_COMPLETO.md (line 485).
Gate 4 hoy tiene defaults segment_len=8.0, hop=2.0 en gate4_ratio_auxiliary.py (line 491), gate4_ratio_auxiliary.py (line 492) y no expone esos flags en CLI (gate4_ratio_auxiliary.py (line 615)).
Impacto: comparación Gate 4 vs Gate 2/Gate 3 deja de ser limpia.
Recomendación: alinear Gate 4 a 4.0/1.0 y exponer flags CLI.

Comandos del plan con incompatibilidades de interfaz.
Evidencia: el plan usa --checkpoint para evaluación en plan_gate4.md (line 133), pero el script espera --model en evaluate_structured_pool.py (line 349).
Además, el plan pasa un directorio como --output (plan_gate4.md (line 136)) y el evaluador escribe archivo JSON con open(args.output, 'w') en evaluate_structured_pool.py (line 520).
Recomendación: corregir comandos antes de correr.

El problema principal detectado por el plan es real y crítico.
Evidencia: freeze total del base model en gate4_ratio_auxiliary.py (line 547) y optimizador sólo de ratio módulos en gate4_ratio_auxiliary.py (line 293); evaluación usa sólo embeddings audio/MIDI en gate4_ratio_auxiliary.py (line 370).
Conclusión: tal como está, el retrieval no puede mejorar.
Recomendación: correcto lo propuesto por el plan: descongelar MIDI + projections, mantener MERT congelado.

Hallazgos Importantes (metodología)
Criterio de selección de “best model” no está alineado al endpoint final.
Gate 4 selecciona por recall promedio de validación global en gate4_ratio_auxiliary.py (line 453), pero la decisión científica final del plan se basa en structured pool (plan_gate4.md (line 99)).
Sugerencia: evaluar structured pool sobre checkpoints por epoch, no solo best_model.pt.

Falta control causal para atribuir mejora al ratio auxiliary.
Si descongelás y reentrenás base, una mejora puede venir del fine-tuning y no del ratio.
Sugerencia: correr control ratio_weight=0.0 con misma configuración.

Riesgo estadístico de sobreinterpretación por una sola corrida.
Con 500 queries, diferencias chicas en R@10 pueden estar en ruido muestral.
Sugerencia: al menos 2-3 seeds para los mejores settings.

Vectorización de histogramas: cuidado con memoria.
La vectorización propuesta es buena, pero una implementación ingenua B x N x N x bins puede explotar VRAM/ram.
Sugerencia: vectorización por chunks y/o cap de eventos por segmento.

Fortalezas del plan
Excelente diagnóstico sobre freeze total + mismatch DANN/baseline (plan_gate4.md (line 13)-21).
Buena dirección conceptual: “agregar estructura” en vez de “forzar invariancia” (plan_gate4.md (line 28)).
Correcta insistencia en checkpoints por epoch y métricas separadas de ratio loss (plan_gate4.md (line 51)-66).
Recomendación Final
Aprobación condicional (alta confianza), con 4 bloqueantes previos a ejecución: compatibilidad de checkpoint para evaluación, alineación 4.0/1.0, corrección de comandos, y cierre de interfaz baseline/CLI.
Si eso se corrige, Gate 4 es el siguiente experimento correcto para BIAS_CONTROL dada la evidencia acumulada de Gate 3 cerrado.