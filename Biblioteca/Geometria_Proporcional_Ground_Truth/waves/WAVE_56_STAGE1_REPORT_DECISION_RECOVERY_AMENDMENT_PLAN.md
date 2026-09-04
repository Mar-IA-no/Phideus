# Wave 56 Stage 1 — Plan de cierre de decisión documental del recovery

**Estado:** `FROZEN_IMPLEMENTATION_PLAN`
**Alcance:** cerrar el bypass R384 y conservar la corrección de entrada a fase
**Decisión científica:** reservada a Mariano; no declara `GO/NO-GO`

## 1. Hecho que obliga a revisar la cadena

I3 (`7b37b5381b0c7540e86de2d53001903475d321ab`) corrigió el inventario
visible y dejó 34/34 pruebas focales verdes. R384 confirmó esa corrección, pero
cerró `REVISE`: `_require_report_fields` aceptó un informe con cabecera
`**Result:** `PASS`` y decisión terminal `REVISE`.

R378 y R380 ya habían mostrado accidentalmente esta clase de contradicción.
La lectura humana completa evitó usarlos como autorización, pero el preflight
ejecutable sigue fail-open. R384 queda preservado como evidencia y no puede ser
el `A` aprobatorio de ningún amendment.

## 2. Gramática terminal obligatoria

Todo nuevo informe aprobatorio de implementación o paquete debe conservar la
cabecera canónica vigente y terminar exactamente con:

```markdown
## Machine-verifiable decision

**Final decision:** `PASS`
```

El parser deriva el resultado esperado de la línea única `**Result:** ...` de
la cabecera y exige que la última sección use el mismo valor. Debe rechazar:

- cabecera `PASS` y decisión terminal `REVISE`;
- decisión terminal ausente, duplicada, cercada o seguida por contenido;
- comentarios HTML y separadores no canónicos ya prohibidos;
- múltiples líneas de resultado o de decisión final.

La sección terminal no pretende automatizar el juicio semántico del cuerpo. La
auditoría independiente y la lectura completa siguen siendo obligatorias; esta
gramática cierra la contradicción explícita que el preflight sí puede decidir.

## 3. Implementación acumulativa real

El runner correcto ya fue introducido por I3 y R384 lo auditó positivamente en
ese punto. El nuevo commit `I4` debe modificar exclusivamente:

1. `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
2. `tests/test_wave56_preoracle_recovery.py`.

No se fabrica un cambio adicional del runner. El schema v4 debe representar la
lineage real con:

- `runner_commit = I3`, cuyo diff contiene exactamente preparador, runner y
  test bajo P3;
- `implementation.commit = I4`, cuyo diff contiene exactamente preparador y
  test para cerrar R384;
- el blob final del runner en I4 idéntico al blob auditado de I3;
- los blobs finales de preparador y test tomados de I4;
- I3 como ancestro de I4 y este plan congelado antes de I4.

El contrato dual mantiene exactamente dos source deltas frente al origen:
preparador y runner. El hecho de que el test cambie no lo convierte en fuente
del contrato prospectivo.

## 4. Nueva DAG y paths

La cadena anterior se conserva completa como evidencia:

```text
P3 → R383 → I3 → R384(REVISE)
```

La nueva cadena continúa:

```text
P4 → R385(plan audit) → I4 → A4(R386) → J4 → F4(R387)
```

`A4`, `J4` y `F4` deben ser hijos directos sucesivos; cada uno introduce un
solo path. `I4` cambia exactamente preparador y test. El preflight exige
`HEAD == F4`, worktree globalmente limpio e introducciones únicas.

Paths nuevos:

```text
experiments/geometria_proporcional/configs/wave56_stage1_report_decision_recovery_amendment_v4.json
Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/386_wave56_stage1_report_decision_recovery_implementation_audit.md
Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/387_wave56_stage1_report_decision_recovery_final_package_audit.md
```

La cabecera de R386 liga `I4`, `runner_commit=I3` y los hashes finales de
preparador, runner y test. R387 liga `J4` y el hash del amendment v4. Ambos
terminan con la sección verificable exacta.

## 5. Tests mínimos

La focal debe agregar negativos end-to-end para implementación y auditoría
final con cabecera `PASS` y decisión terminal `REVISE`. También debe cubrir
ausencia, duplicación y contenido posterior a la decisión. Los positivos
sintéticos deben usar la nueva sección terminal.

Se conservan las 34 pruebas existentes: inventario visible oficial-like,
calibración intacta/faltante/mutada, visible extra, conflicto manifest/freeze,
dos source deltas, no-redraw, origen físico, DAG, blobs y replay manifest.

R386 ejecuta la focal completa. R387 repite focal y suite amplia exacta Wave
49–56. El contenido entero de cada informe se lee antes de aceptarlo; el parser
terminal no sustituye esa auditoría cruzada.

## 6. Reejecución

Sólo con `HEAD == F4`:

1. recuperar otra vez desde el intento fallido original y el amendment v4;
2. usar `--force` para archivar recuperablemente el primario v2 `PREPARED`;
3. ejecutar `fit`, `select`, `adjudicate` del nuevo primary;
4. preparar y ejecutar replay exacto desde ese primary;
5. preservar todos los estados crudos y receipts.

No se usa el primario v2 como fuente de claves, no se borra ningún artefacto y
no se modifica el protocolo científico. Todo sigue CPU-only por contrato.

## 7. Criterio de cierre

El cambio queda listo sólo si un auditor independiente confirma:

- rechazo del probe exacto de R384;
- identidad del runner final con I3 y lineage acumulativa verificable;
- dos y sólo dos source deltas frente al escrow de origen;
- focal y suite amplia verdes;
- DAG, hashes, decisiones terminales y limpieza exactos;
- recovery, fases y replay oficiales completos sin bypass ni redraw.
