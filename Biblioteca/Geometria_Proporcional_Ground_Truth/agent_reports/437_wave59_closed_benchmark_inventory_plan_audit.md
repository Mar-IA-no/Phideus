# Wave 59 closed benchmark inventory plan audit R437

**Plan commit:** `44e62eadd01f42ec67975816a8f48d7ffda1a95c`  
**Plan SHA:** `dbd5deef037193445b41d062c0364955fc0f48fc6c02d1eb4fd38679a11d96f3`  
**R436 report SHA-256:** `44ea7c0c416bb6808375934ce2222c19243d34fa6fbd8c3aeb311cb4c69a6366`  
**Result:** `REVISE`

## Findings

### P1 — ninguno en el inventario físico

La revisión cierra exactamente el finding de R436: exige rechazar todo nodo no regular —FIFO, socket, device u otro— y comparar los directorios físicos contra el cierre exacto de padres de `manifest.files`, incluida la raíz `benchmark/`. También exige probes separados para FIFO y directorio vacío no declarado. El control puede implementarse exclusivamente con recorrido físico, `lstat`, paths y hashing opaco; no requiere parsear truth, commitments, secrets ni escrow.

### P2 — La cronología de la cadena quedó desactualizada

El plan afirma que los seis commits comienzan después de R434 y que son “posteriores a R432”. La cadena real ya contiene `R434 → revisión → R435 → c716e68 → R436 → 44e62ea`. Los seis pasos ejecutables restantes deben describirse como iniciados en esta revisión `44e62ea` y posteriores a R436.

El commit auditado es hijo directo de R436, modifica exclusivamente el plan y pasa `git diff --check`. La matriz de pruebas requerida es suficiente; los probes nuevos todavía no existen porque corresponden al siguiente commit de implementación.

**Final decision:** `REVISE`
