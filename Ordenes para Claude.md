# Órdenes para Claude - Proyecto Phideus

## Instrucciones Generales de Organización

### 🗂️ **MANTENER ORDEN SIEMPRE**

**Cada vez que trabajes en el proyecto Phideus, debes:**

1. **Ubicar archivos en su lugar correcto**:
   - **Scripts principales del pipeline** → `src/`
   - **Scripts temporales/testing** → `src/temp/`
   - **JSONs de prueba** → `test-json/`
   - **Documentación** → `Documents/`
   - **Research papers** → `Biblioteca/`

2. **Actualizar documentación en paralelo**:
   - Cada cambio técnico → actualizar `Documents/bitacora_desarrollo.md`
   - Cambios significativos → actualizar `Documents/Proyecto_Estado_Actual.md`
   - Nuevas órdenes → añadir a este archivo

3. **Usar TodoWrite sistemáticamente**:
   - Crear todos para tareas multi-paso
   - Marcar progreso en tiempo real
   - Completar todos al finalizar tareas

4. **Nomenclatura consistente**:
   - Scripts finales: `nombre_v4.1_Optimized.py`
   - Scripts temporales: `test_`, `compare_`, `validate_`
   - JSONs: descriptivos con fecha si relevante

---

## Estructura de Directorios

```
Phideus/
├── src/                          # 🎯 PIPELINE PRINCIPAL
│   ├── analizador_4.1_Enriched.py
│   ├── auditor_v4.0.py
│   ├── generador_wavs_ratios_complejos_v3.0_Ninja.py
│   ├── train_ratio_model.py
│   └── temp/                     # 🧪 SCRIPTS TEMPORALES
│       ├── test_enriched_validation.py
│       ├── compare_bins.py
│       └── [otros scripts de testing]
├── Documents/                    # 📚 DOCUMENTACIÓN
│   ├── bitacora_desarrollo.md    # Log técnico detallado
│   └── Proyecto_Estado_Actual.md # Overview del proyecto
├── test-json/                    # 🧪 DATASETS DE PRUEBA
├── test_wavs/                    # 🎵 AUDIOS SINTÉTICOS
├── validation_plots/             # 📊 VISUALIZACIONES
└── Biblioteca/                   # 📖 RESEARCH PAPERS
```

---

## Flujo de Trabajo Estándar

### Para Cambios Técnicos:
1. **Implementar** cambio en archivo correspondiente
2. **Probar** funcionalidad si aplica
3. **Actualizar bitácora** con detalles técnicos
4. **Actualizar proyecto overview** si es cambio significativo
5. **Organizar** archivos nuevos en estructura correcta

### Para Nuevas Features:
1. **Crear todos** para planificación
2. **Documentar** en bitácora antes de empezar
3. **Implementar** de forma ordenada
4. **Validar** con tests si aplica
5. **Actualizar** documentación final

### Para Research/Análisis:
1. **Documentar** hallazgos en bitácora
2. **Mover** scripts de análisis a `src/temp/`
3. **Conservar** resultados importantes en `test-json/`
4. **Integrar** conclusiones al proyecto overview

---

## Reglas de Commit y Documentación

### Bitácora (Log Técnico):
- **Formato**: Fecha | Título | Descripción técnica
- **Incluir**: Cambios de código, resultados de tests, decisiones técnicas
- **Actualizar**: Cada sesión de trabajo significativa

### Proyecto Overview:
- **Actualizar**: Cambios en arquitectura, nuevas fases completadas
- **Mantener**: Estado actual, próximos pasos, especificaciones
- **Revisar**: Consistencia con código actual

---

## 📝 Órdenes Específicas

*[Aquí se irán añadiendo órdenes específicas que me vayas dando]*

### Última actualización: 2025-08-06
- Organización inicial del repositorio
- Estructura de directorios establecida
- Pipeline Fase 0 completado y documentado