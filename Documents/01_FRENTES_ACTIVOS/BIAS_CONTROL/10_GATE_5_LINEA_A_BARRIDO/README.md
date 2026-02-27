# Gate 5 Linea A — Replanteo Estrategico y Exploracion Oportunista

**Estado**: REPLANTEADO / OPORTUNISTA  
**Fecha de actualizacion**: 2026-02-27  
**Rol en roadmap**: Gate 5A ya no es paso bloqueante antes de Escalon 2. Corre en paralelo cuando hay recursos libres, mientras Gate 5B completa el cierre cientifico de Escalon 1-C.

---

## Resumen

Gate 5A nacio como una linea de barrido amplio: descriptores no probados x mecanismos x combinaciones cruzadas. Ese framing ya no describe bien la prioridad real del programa.

Los resultados de Gate 4.3, Gate 4.4, Gate 4.5 y el hallazgo del Pre-Proj A/B test cambiaron el criterio de valor:

- ya no importa completar casilleros del barrido original por completitud;
- importa atacar el bottleneck real detectado;
- importa combinar mecanismos que ya mostraron valor;
- importa sostener Gate 5A como exploracion paralela, sin bloquear la transicion a Escalon 2.

En este nuevo encuadre, Gate 5A queda organizado en tres cajas: lo ya explorado, lo de alta prioridad y el backlog legacy de baja prioridad.

---

## Caja 1 — Ya explorado / parcialmente cerrado

Estos items ya tienen lectura suficiente para no ocupar la ruta critica actual.

| Item | Estado | Lectura actual |
|---|---|---|
| Concat same-modality (`d4`, `a4`, `a7`, `d4a4`, `d4a7`) | probado | `d4a4` emerge como uno de los grandes ganadores |
| Cross-attention regular (`a4x`, `a7x`, `d4x`) | probado | no desplazo a los mejores mecanismos |
| Reverse cross-attention (`a4r`, `d4a4r`, `d4-a4r`) | probado | `a4r` y `d4-a4r` quedaron entre los mejores brazos |
| FiLM per-layer (`film-a4`, `film-d4`, `film-dual`) | probado | negativo, por debajo de `D0` |
| MoE (`moe-a4`, `moe-dual`, v2-v4) | probado | lateral / sin desplazamiento claro |
| Cross-modal bidireccional (`d4a4cm`) | probado | negativo: ya fue ejecutado y quedo muy por debajo del dual same-modality |
| Third Tower (`t3-tri`, `t3-anc`, `t3-wt`) | probado | `t3-wt` mostro valor real como mecanismo complementario |

Nota importante:
- `d4a4cm` implica que **cross-modal injection no esta “pendiente” en abstracto**. Lo que queda pendiente son variantes unidireccionales especificas (`CM-a`, `CM-m`), no la hipotesis bidireccional completa.

---

## Caja 2 — Alta prioridad operativa

Esta es la parte activa de Gate 5A. No forma parte de la ruta critica inmediata, pero si vale la pena correrla cuando LOCAL o UNC tengan ventana libre.

### C1. Descriptor-Conditioned Projection Heads

**Estado**: implementado y verificado (`8/8` tests pass).  
**Codigo**:
- `src/bias_control/encoders/projection.py`
- `experiments/bias_control/gate5a_proj_cond.py`

**Motivacion**:
- El Pre-Proj A/B test mostro que la projection head, especialmente en MIDI (`512 -> 256`), destruye gran parte de la informacion condicionante.
- Esta linea ataca ese bottleneck sin tocar el encoder.

**Brazos activos**:

| Arm | Audio proj | MIDI proj | Condicion | Rol |
|---|---|---|---|---|
| `a4r-ctrl` | standard | standard | — | control reproducido |
| `a4r-pca` | conditioned | standard | A4 -> audio | aislar lado audio |
| `a4r-pcm` | standard | conditioned | D4 -> midi | aislar lado MIDI |
| `a4r-pcd` | conditioned | conditioned | A4 + D4 | brazo principal |
| `a4r-pcd-zero` | conditioned | conditioned | zeros fijos | control de overhead |

**Lectura esperada**:
- si `a4r-pcd > a4r-pcd-zero`, la mejora viene del conditioning y no solo de parametros extra;
- si mejora probing generativo sin mover `S`, la proyeccion preserva mas informacion aunque VICReg no la este explotando del todo.

### C2. Combinatorios `t3-wt`

**Estado**: diseno listo, no implementado.

Brazos con prioridad real:
- `t3-wt-vanilla`: tower weighted sin inyeccion en encoders; control barato y de bajo riesgo.
- `t3-wt-a4r`: combinacion entre Third Tower y `d4-a4r`; hipotesis fuerte porque junta dos mecanismos con valor demostrado.

### C3. TBD del usuario
### C4. TBD del usuario

Se reservan dos lugares para hipotesis nuevas que todavia no quedaron cerradas en diseño. No se abren hasta que exista pregunta clara y costo acotado.

---

## Caja 3 — Backlog legacy de baja prioridad

Estos items siguen existiendo como historial del roadmap, pero hoy no justifican la inversion principal.

### Barrido amplio de descriptores no probados

Descriptores heredados del plan original:
- MIDI: `D3`, `D8`, `D9`, `D10`, `D2`, `D5`, `D6`, `D7`
- Audio: `A1`, `A2`, `A3`, `A5`, `A6`

Motivo de depriorizacion:
- Gate 4.3/4.4/4.5 ya seleccionaron mecanismos fuertes (`d4a4`, `a4r`, `d4-a4r`, `t3-wt`).
- El cuello nuevo esta en proyeccion y combinatoria, no en abrir otro barrido amplio por default.

### Cross-modal unidireccional (`CM-a`, `CM-m`)

Quedan como hipotesis abiertas solo si aparece una justificacion nueva.

Lectura actual:
- el bidireccional (`d4a4cm`) ya dio senal negativa fuerte;
- por eso las variantes unidireccionales no entran hoy en la ruta critica.

### Deep injection por capas (AdaLN / variantes afines)

No se descarta, pero queda pospuesto.

Motivo:
- FiLM per-layer ya fue negativo en Gate 4.4;
- AdaLN no es identico, pero pertenece a la misma familia de “modular capas internas”; 
- antes de subir complejidad, conviene observar si conditioned projections mueve algo real.

---

## Modo de ejecucion

Gate 5A queda definido como **exploracion oportunista**:

1. **Ruta critica principal**: cerrar Gate 5B y habilitar la transicion a Escalon 2.
2. **Ruta paralela**: correr Gate 5A cuando haya GPU o slots UNC libres.
3. **Regla de prioridad**: Gate 5A no bloquea Escalon 2.

Lectura operativa actual:
- Gate 5B sigue siendo el cierre cientifico de Escalon 1-C.
- Escalon 2 (Speech <-> EGG) pasa a foco principal una vez cerrado Gate 5B.
- Gate 5A puede seguir corriendo en paralelo como linea de exploracion si los recursos lo permiten.

---

## Orden sugerido de ejecucion

### Local first

1. smoke/control corto de `a4r-ctrl`
2. `a4r-pcd`
3. `a4r-pcd-zero`
4. `a4r-ctrl` completo
5. `a4r-pca`
6. `a4r-pcm`

### Luego, si hay senal

7. `t3-wt-vanilla`
8. `t3-wt-a4r`

### Criterio de corte

- si `a4r-pcd` pierde retrieval fuerte frente a `a4r-ctrl`, pausar y re-evaluar;
- si `a4r-pcd-zero ~= a4r-pcd`, la mejora aparente no es causal del conditioning;
- si `a4r-pcd` mejora probing pero no retrieval, documentar el insight y decidir si vale una nueva loss.

---

## Apéndice histórico — Plan original de Gate 5A

El plan original incluia:
- barrido descriptor x mecanismo,
- variantes cross-modal adicionales,
- combinatorios mas amplios.

Ese plan no se borra: queda como contexto historico del frente. Pero desde este corte deja de ser la descripcion canonica del estado actual de Gate 5A.

Documentos relacionados:
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_UNC.md`
- `Documents/NOTAS_CLAUDE-CODEX.md`
