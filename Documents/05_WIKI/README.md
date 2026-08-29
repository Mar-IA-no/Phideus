# Wiki viva de Phideus

Esta wiki es la capa de conocimiento compilado del programa Phideus. No reemplaza
los README, roadmaps, informes, resultados ni artefactos experimentales: los
conecta, explicita sus relaciones y mantiene una imagen recuperable del estado
del programa.

El diseño adapta el patrón **LLM Wiki** propuesto por Andrej Karpathy:

- fuente primaria: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
- las fuentes se preservan;
- una capa intermedia de Markdown acumula síntesis y relaciones;
- un esquema gobierna cómo los agentes actualizan esa síntesis.

## Dos entradas

| Lectura | Documento | Para qué sirve |
|---|---|---|
| Humana | [MAPA_VISUAL_DEL_PROGRAMA.md](MAPA_VISUAL_DEL_PROGRAMA.md) | Ver frentes, estados, dependencias y bifurcaciones en tablas y diagramas |
| Agentes / LLM | [LLM_CONTEXT.md](LLM_CONTEXT.md) | Recuperar en una sola lectura el mapa denso, las autoridades documentales y las preguntas abiertas |

El catálogo completo de páginas está en [index.md](index.md).

## Por qué no es sólo RAG

Un sistema RAG recupera fragmentos de fuentes para cada consulta. Esta wiki
conserva una síntesis persistente entre consultas: integra una vez relaciones,
supersesiones, tensiones y estados, y permite que esa integración se acumule en
Markdown versionado. La consulta sigue volviendo a las fuentes para verificar
evidencia; la wiki evita tener que reconstruir el mapa completo desde cero.

## Las tres capas

1. **Fuentes canónicas.** Permanecen en sus ubicaciones actuales: documentación
   troncal, README y roadmaps de frente, informes, código y resultados. La wiki
   no los copia ni los reescribe para hacerlos coincidir.
2. **Wiki.** Este directorio compila el estado vigente, las dependencias, las
   contradicciones y las alternativas preservadas.
3. **Esquema.** [SCHEMA.md](SCHEMA.md) define estados, procedencia, operaciones y
   límites de autoridad.

## Adaptación específica a Phideus

El patrón original deja la estructura exacta abierta al dominio. Phideus agrega
cuatro defensas necesarias para investigación multifrente:

- estados ortogonales de arquitectura, experimento, evidencia y decisión;
- temporalidad separada entre el corte descrito y su fecha de registro;
- IDs estables para páginas y fuentes, con catálogo machine-readable;
- lint ejecutable para esquema, relaciones, procedencia y enlaces.

El lint es una defensa estructural, no un juez semántico. Los cambios de estado
siguen requiriendo releer las fuentes y auditar en conjunto todas las vistas que
repiten ese estado.

## Qué problema resuelve

La estructura física de `Documents/` conserva la historia del proyecto, pero ya
no coincide exactamente con el estado científico. Hay frentes cerrados dentro
de `01_FRENTES_ACTIVOS/`, un antecedente superseded que conserva su propio
roadmap y un roadmap maestro cuyo orden original fue superado por la ejecución.
La wiki hace explícitas esas diferencias sin borrar ni falsificar la historia.

## Relación con los documentos existentes

| Artefacto | Pregunta que responde | Autoridad |
|---|---|---|
| `README.md` | ¿Qué es Phideus y cuál es su estado público? | Entrada pública canónica |
| `Proyecto_Estado_Actual.md` | ¿Cuál es el corte ejecutivo consolidado? | Estado global |
| README/roadmap de cada frente | ¿Qué se ejecutó y qué pregunta sigue? | Estado local del frente |
| Informes y artefactos | ¿Qué se observó exactamente? | Evidencia |
| Bitácora | ¿Qué ocurrió y cuándo? | Cronología |
| Wiki | ¿Cómo se conecta todo y qué alternativas existen hoy? | Síntesis mantenida |

La wiki no declara GO/NO-GO, no convierte una hipótesis en observación y no
funciona como una segunda bitácora ni como una segunda cola de tareas.
