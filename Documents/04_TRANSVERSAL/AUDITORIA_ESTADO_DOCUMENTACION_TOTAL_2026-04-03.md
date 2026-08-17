# Auditoria del Estado de la Documentacion Total del Repositorio

> **Auditoría histórica.** Este documento conserva el diagnóstico ejecutado el 2026-04-03 y las correcciones que motivó. Para el estado documental vigente y la revisión posterior de todas las capas usar `AUDITORIA_ESTADO_DOCUMENTACION_TOTAL_2026-08-17.md`.

Fecha: 2026-04-03  
Repo: `Phideus`

---

## 1. Objetivo y criterio de auditoria

Esta auditoria no parte de un criterio ingenuo de "todo documento viejo que no refleje el presente esta mal".  
El repositorio de Phideus ya tiene suficiente densidad historica como para exigir una distincion mas fina entre capas documentales.

El criterio usado fue este:

1. **Documentacion canonica viva**: debe reflejar el estado actual del proyecto con la mayor precision posible.
2. **Documentacion operativa/interna**: puede contener trazas locales, handoffs, auditorias, memos de trabajo y rutas machine-specific, siempre que no se la confunda con la capa publica canonica.
3. **Documentacion historica de frente**: puede y debe preservar hipotesis, planes, lecturas provisionales y decisiones del momento, incluso si despues el proyecto fue por otro lado.
4. **Documentacion pausada/cerrada/archivistica**: no debe reescribirse para simular un presente que ya no le corresponde, salvo que tenga errores formales graves.

La pregunta rectora no fue "que cosas estan viejas", sino:

> **Que documentos deben sincronizarse porque hoy funcionan como capa viva del repo, y que documentos conviene preservar justamente como registro historico del proceso?**

---

## 2. Alcance y metodologia

### 2.1 Alcance auditado

Se relevo la documentacion textual del repo con este recorte practico:

- `README.md`
- `MARCO_EPISTEMOLOGICO_PHIDEUS.md`
- `Documents/**`
- `Para_GPT/**`
- `Para_Sai/**`
- `Para_Share/**`

### 2.2 Volumen relevado

Conteo actual de documentos textuales (`.md`, `.txt`, `.rst`):

| Zona | Cantidad |
|---|---:|
| `README.md` | `1` |
| `MARCO_EPISTEMOLOGICO_PHIDEUS.md` | `1` |
| `Documents/` | `229` |
| `Para_GPT/` | `26` |
| `Para_Sai/` | `17` |
| `Para_Share/` | `2` |
| **Total** | **`276`** |

Desglose util dentro de `Documents/`:

| Zona | Cantidad |
|---|---:|
| `Documents/00_TRONCAL/` | `10` |
| `Documents/01_FRENTES_ACTIVOS/` | `120` |
| `Documents/02_FRENTES_PAUSADOS/` | `5` |
| `Documents/03_FRENTES_CERRADOS/` | `37` |
| `Documents/04_TRANSVERSAL/` | `13` |
| `Documents/90_ARCHIVO_GLOBAL/` | `37` |
| `Documents/Skills/` | `5` |
| `Documents/BITACORA_UNC.md` | `1` |
| `Documents/NOTAS_CLAUDE-CODEX.md` | `1` |

### 2.3 Tipologia usada para clasificar hallazgos

| Tipo | Regla |
|---|---|
| **Canonico vivo** | Debe quedar sincronizado con el estado actual |
| **Operativo interno** | Puede conservar trazas locales y memoria de trabajo |
| **Historico de frente** | Puede conservar planes, hipotesis y lecturas del momento |
| **Archivistico** | Debe preservarse como pasado del proyecto |

---

## 3. Diagnostico ejecutivo

La documentacion del repo **no esta globalmente rota**.  
De hecho, despues de las auditorias recientes y de las correcciones forenses sobre `d4a4`, `Gate 10`, `Gate 8`, `Gate 6` y `Escalon 2`, la mayor parte de la capa viva ya quedo razonablemente sincronizada.

El problema actual esta concentrado en **pocos nodos de alta visibilidad**:

1. `README.md`
2. `Documents/00_TRONCAL/INDICE_DOCUMENTACION.md`
3. `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/INDEX_BIAS_CONTROL.md`
4. `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/PHIDEUS_MASTER_BRIEFING.md`
5. Un cluster acotado de links rotos en `CURADURIA_VISUAL/`
6. Un documento mixto (`Rosetta_triplescaloneta.md`) que tiene el cuerpo historico bien, pero arrastra un addendum/final operativo parcialmente viejo

La conclusion fuerte de esta auditoria es esta:

> **El repo ya no necesita una reescritura documental masiva. Necesita una correccion fina de la capa canonica viva y, al mismo tiempo, una defensa explicita de la documentacion historica para no borrar el proceso real.**

---

## 4. Hallazgos obligatorios en documentacion canonica viva

### 4.1 `README.md` quedo un estado atras en dos puntos centrales

Archivo: `README.md`

#### Problema A: `d4a4` sigue narrado como si fuera training multi-seed homogeneo

Evidencia:

- `README.md:24`
- `README.md:51`
- `README.md:56`
- `README.md:259`
- `README.md:264`

Que dice hoy, en sustancia:

- `d4a4=84.1% +/-2.3pp`, `+9.4pp sobre baseline`, `5 seeds`
- encabezado de tabla: `S (multi-seed)`

Por que esta mal:

- hoy la lectura canonica del repo ya distingue con claridad:
  - `D0`, `a4r`, `d4-a4r` = **training-seed replication** en UNC
  - `d4a4=84.1%±2.3pp` = **eval-seed reference** sobre un unico checkpoint `e30`

Fuentes canonicas correctas:

- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md:14`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` (bloque Gate 5B actualizado)
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/README.md:56-60`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/INFORME_COMPLETO_GATE5B.md:199-230`

Correccion necesaria:

- dejar de usar `multi-seed` como etiqueta homogenea en el `README`
- recalificar `d4a4` como **referencia eval-seed**
- reformular la lectura causal para no vender como training variance algo que hoy es evaluator variance

#### Problema B: `Escalon 2` sigue contado como si `S2-P3` fuera el siguiente paso, no una primera pasada ya completada

Evidencia:

- `README.md:24`
- `README.md:37`
- `README.md:41`
- `README.md:81`
- `README.md:83`
- `README.md:192`
- `README.md:203`

Que dice hoy, en sustancia:

- `el siguiente paso real ... es S2-P3`
- `siguiente fase S2-P3`
- `proximo contraste (S2-P3)`

Por que esta mal:

- la capa canonica actual ya absorbio que `P3` **se corrio y se leyo en una primera pasada**
- la pregunta viva ya no es "abrir P3", sino **cerrar `P2 vs P3`**

Fuentes canonicas correctas:

- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md:18`
- `Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md:12-15`
- `Documents/01_FRENTES_ACTIVOS/ESCALON_2/ROADMAP_ESCALON_2.md` (bloque P3 actualizado)

Correccion necesaria:

- reescribir el framing de Escalon 2 en el `README`
- pasar de `S2-P3 decidido/siguiente fase` a `S2-P3 primera pasada completada; siguiente tarea = diagnostico comparativo P2 vs P3`

---

### 4.2 `INDICE_DOCUMENTACION.md` ya no coincide con la capa canonica que el mismo indice enlaza

Archivo: `Documents/00_TRONCAL/INDICE_DOCUMENTACION.md`

Evidencia:

- `INDICE_DOCUMENTACION.md:149`
- `INDICE_DOCUMENTACION.md:155`
- `INDICE_DOCUMENTACION.md:221`
- `INDICE_DOCUMENTACION.md:246`

Problemas detectados:

1. Todavia describe `S2-P3` como `implementado/en ejecución` o `P3-D0 en ejecución`, cuando la capa canonica del frente ya lo trata como **primera pasada completada**.
2. Sigue resumiendo `Gate 10` como `concat > pca >> attn_bias`, cuando la lectura ya fue endurecida a `concat > FiLM/pca >> attn_bias`.

Por que importa:

- `INDICE_DOCUMENTACION.md` no es un memo historico: funciona como mapa canónico de entrada al repo.
- Si el indice queda atras, arrastra una contradiccion interna entre el mapa y los documentos que el propio mapa recomienda.

Correccion necesaria:

- sincronizar la descripcion de Escalon 2 con `P3` ya corrido
- sincronizar la formula breve de `Gate 10` con la lectura vigente

---

### 4.3 `INDEX_BIAS_CONTROL.md` arrastra una version abreviada vieja de `Gate 10`

Archivo: `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/INDEX_BIAS_CONTROL.md`

Evidencia:

- `INDEX_BIAS_CONTROL.md:12`
- `INDEX_BIAS_CONTROL.md:85`

Problema:

- el índice del frente sigue diciendo `concat > pca >> attn_bias`

Por que esta mal:

- el cierre canónico actual del gate ya no opone solo `concat` a `pca`, sino `concat` a **`FiLM/pca`**, precisamente porque la lectura final es sobre **mecanismo** y no solo sobre una rama reducida.

Fuentes correctas:

- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md:19`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/17_GATE_10_MECHANISM_SWEEP/README.md:76`
- `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md:47`

Correccion necesaria:

- actualizar el corte operativo y la descripcion corta del Gate 10 dentro del índice

---

### 4.4 `PHIDEUS_MASTER_BRIEFING.md` tiene una inconsistencia interna en su tabla de roadmap

Archivo: `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/PHIDEUS_MASTER_BRIEFING.md`

Evidencia:

- la introduccion ya esta actualizada:
  - `PHIDEUS_MASTER_BRIEFING.md:18`
  - `PHIDEUS_MASTER_BRIEFING.md:20`
- pero la tabla de roadmap queda atras:
  - `PHIDEUS_MASTER_BRIEFING.md:264`
  - `PHIDEUS_MASTER_BRIEFING.md:266`

Problemas:

1. `Escalon 2` aparece como `S2-P3 decidido`
2. `Gate 10` aparece como `concat > pca >> attn_bias`

Por que importa:

- `PHIDEUS_MASTER_BRIEFING.md` funciona como una de las sintesis transversales mas fuertes del repo.
- No conviene que en el mismo documento convivan una introduccion corregida y una tabla-resumen atrasada.

Correccion necesaria:

- poner la tabla en el mismo estado que la cabecera del documento:
  - `P3` primera pasada completada
  - `Gate 10 = concat > FiLM/pca >> attn_bias`

---

### 4.5 El cluster `CURADURIA_VISUAL` tiene links rotos reales

Archivos:

- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/04_DIAGNOSTICO_GATE_6_Y_GATE_4_2/CURADURIA_VISUAL/SNAPSHOT_DEC005.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/04_DIAGNOSTICO_GATE_6_Y_GATE_4_2/CURADURIA_VISUAL/INDEX_VISUAL.md`

Hallazgo:

- chequeo automatico de links internos: `16` links rotos
- todos concentrados en este cluster

Ejemplos:

- `../../../data/bias_control_medium/evaluations/gate42/h426_prered_results.json`
- `../../../data/bias_control_medium/evaluations/gate6/layer_drift.json`

Que pasa:

- las rutas relativas quedaron mal calculadas
- hoy resuelven hacia `Documents/01_FRENTES_ACTIVOS/data/...`, que no existe

Por que esto si es un error y no solo "historia":

- aunque el contenido sea historico/diagnostico, un índice visual con links rotos deja de cumplir su función documental mínima

Correccion necesaria:

- arreglar la profundidad relativa o reemplazar por rutas repo-root consistentes

---

## 5. Hallazgos importantes, pero no equivalen a "todo ese documento esta mal"

### 5.1 `Rosetta_triplescaloneta.md` es un documento mixto: cuerpo historico valido + addendum/final operativo parcialmente viejo

Archivo: `Documents/00_TRONCAL/ROADMAP_GENERAL/Rosetta_triplescaloneta.md`

Evidencia:

- `Rosetta_triplescaloneta.md:16` todavia habla de apertura de `S2-P3`
- `Rosetta_triplescaloneta.md:172` sigue diciendo que el proximo paso correcto es `S2-P3`
- en cambio el addendum central ya esta bastante actualizado en `Rosetta_triplescaloneta.md:19`

Diagnostico:

- **no** conviene tocar el cuerpo original del texto, porque justamente es una pieza historica/argumental sobre la escalera metodologica
- **si** conviene corregir el addendum operativo y el bloque final de "estado actual" cuando estos se presentan como corte vigente

Lectura recomendada:

- preservar el documento como texto historico vivo
- actualizar solo los tramos que se autoidentifican como `addendum operativo` o `estado actual`

---

### 5.2 `Documents/04_TRANSVERSAL/` mezcla dos capas diferentes

Observacion estructural:

`Documents/04_TRANSVERSAL/` hoy contiene:

1. **Transversal canonico** (`TEORIA_Y_FUNDAMENTOS/`)
2. **Memos operativos/auditorias internas** (`AUDIT_REPORT_TRAZABILIDAD.md`, `INFORMES_CORRECCIONES_LIBRO_HIT.md`, `INFORME_CAMBIOS_LIBRO_HIT_DESDE_AUDITORIA_PARA_CLAUDE.md`, `INFORME_D4A4_MULTISEED_PARA_CODEX.md`)

Consecuencia:

- no es un error en si mismo
- pero en futuras auditorias conviene no tratar toda `04_TRANSVERSAL/` como una sola capa documental

Importante:

- los memos operativos pueden contener rutas absolutas, referencias al repo del libro y otras trazas locales
- eso **no** invalida el repo mientras no se los haga pasar por documentacion publica canonica

---

## 6. Documentacion historica que conviene preservar y no "actualizar por reflejo"

### 6.1 Principio general

No todo documento con lenguaje desactualizado es un error.

Hay documentos cuyo valor esta precisamente en mostrar:

- que hipotesis se estaba considerando en ese momento
- que decision aun no estaba tomada
- que experimento se imaginaba antes de ser corrido
- que interpretacion se tenia antes de la evidencia posterior

Reescribir eso en clave presente destruiria trazabilidad historica.

### 6.2 Conservar como historico salvo error formal

En general, **no conviene reescribir**:

- `Documents/02_FRENTES_PAUSADOS/**`
- `Documents/03_FRENTES_CERRADOS/**`
- `Documents/90_ARCHIVO_GLOBAL/**`
- entradas fechadas de `Documents/00_TRONCAL/bitacora_desarrollo.md`
- planes e informes ya superseded dentro de frentes activos, cuando el propio documento funciona como memoria de proceso
- briefings de UNC ya cumplidos, si estan claramente marcados como historicos/operativos
- auditorias y memos internos de sincronizacion libro/repo

### 6.3 Casos concretos donde **no** hay que sobreactuar

- Que un documento viejo diga que `S2-P3` era el paso siguiente **no es un problema** si ese documento es un plan viejo o una nota de transicion.
- Que aparezca `80.4%` en docs de `Gate 2` **no es inconsistencia**: ahi sigue siendo el valor correcto.
- Que existan planes iniciales de Escalon 3 o viejas explicaciones de Gate 4.2/4.3 **no obliga** a reescribirlos a la luz de lo que despues paso.

---

## 7. Documentacion viva que hoy se ve sana

Despues de la auditoria, estos nodos parecen bien alineados con el estado actual y no muestran desfasajes fuertes:

- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md`
- `Documents/01_FRENTES_ACTIVOS/ESCALON_3/README.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/17_GATE_10_MECHANISM_SWEEP/README.md`
- `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md`
- `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md`

Esto importa porque evita una conclusion exagerada: **la capa viva ya no esta globalmente desalineada**.

---

## 8. Clasificacion final de acciones

### 8.1 Correcciones obligatorias en capa viva

1. `README.md`
2. `Documents/00_TRONCAL/INDICE_DOCUMENTACION.md`
3. `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/INDEX_BIAS_CONTROL.md`
4. `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/PHIDEUS_MASTER_BRIEFING.md`
5. Links rotos en `CURADURIA_VISUAL/`

### 8.2 Correcciones recomendables pero quirurgicas

6. `Documents/00_TRONCAL/ROADMAP_GENERAL/Rosetta_triplescaloneta.md`
   - solo en addendum operativo / estado actual
   - no en el cuerpo historico principal

### 8.3 Documentos que deben preservarse como archivo de proceso

- `Documents/00_TRONCAL/bitacora_desarrollo.md` (entradas fechadas)
- planes historicos dentro de frentes activos
- briefings de UNC ya cumplidos
- auditorias internas y memos de sincronizacion
- carpetas `02_FRENTES_PAUSADOS`, `03_FRENTES_CERRADOS`, `90_ARCHIVO_GLOBAL`

---

## 9. Regla propuesta para futuras auditorias

Para no repetir el mismo problema en cada sync, conviene institucionalizar esta regla:

### Si el documento es...

#### A. Canonico vivo

Ejemplos:

- `README.md`
- `Proyecto_Estado_Actual.md`
- `INDICE_DOCUMENTACION.md`
- READMEs/roadmaps vivos de frente
- `TEORIA_Y_FUNDAMENTOS/**`

Entonces:

- **debe** reflejar el estado actual
- y cualquier desfasaje semantico o numerico **si** cuenta como error

#### B. Operativo interno

Ejemplos:

- handoffs
- auditorias
- memos para Claude/Codex
- `Para_GPT`, `Para_Sai`, `Para_Share`

Entonces:

- puede contener lenguaje local, rutas absolutas y trazas de trabajo
- no debe usarse como fuente publica canonica sin mediacion

#### C. Historico de frente

Ejemplos:

- planes viejos
- briefings ya ejecutados
- informes de decisiones del momento

Entonces:

- debe preservarse salvo error formal grave
- si hace falta, agregar una marca de `historico` o `superseded`, no reescribirlo entero

---

## 10. Conclusion

La documentacion total del repositorio **no necesita una normalizacion brutal**.  
Necesita una politica madura de capas.

El hallazgo central de esta auditoria es doble:

1. **La capa canonica viva quedo mucho mejor que antes y ya no tiene decenas de problemas; tiene pocos, concretos y localizados.**
2. **La documentacion historica no debe tratarse como "basura desactualizada", sino como parte del activo epistemologico del proyecto.**

Por eso la recomendacion final no es "actualizar todo".

La recomendacion final es:

- **corregir la capa viva donde todavia quedo un estado atras**, y
- **preservar explicitamente la documentacion historica como registro del proceso**, evitando reescribir el pasado con lenguaje del presente.
