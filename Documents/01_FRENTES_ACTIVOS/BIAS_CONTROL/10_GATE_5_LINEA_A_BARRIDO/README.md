# Gate 5 Linea A — Replanteo Estrategico y Exploracion Oportunista

**Estado**: REPLANTEADO / OPORTUNISTA  
**Fecha de actualizacion**: 2026-03-11  
**Rol en roadmap**: Gate 5A ya no es paso bloqueante antes de Escalon 2. Corre en paralelo cuando hay recursos libres, con Gate 5B ya cerrado como lectura científica estable y Gate 8 ya absorbido como línea positiva cerrada.

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

**Estado**: implementado, ejecutado y ya leído como Gate 8 cerrado `5/5`.  
**Codigo**:
- `src/bias_control/encoders/projection.py`
- `experiments/bias_control/gate5a_proj_cond.py`

**Motivacion**:
- El Pre-Proj A/B test mostro que la projection head, especialmente en MIDI (`512 -> 256`), destruye gran parte de la informacion condicionante.
- Esta linea ataca ese bottleneck sin tocar el encoder.

**Brazos de C1 (ya cerrados como Gate 8)**:

| Arm | Audio proj | MIDI proj | Condicion | Best S | Lectura |
|---|---|---|---|---:|---|
| `a4r-ctrl` | standard | standard | — | `79.2%` | control reproducido |
| `a4r-pca` | conditioned | standard | A4 -> audio | `82.6%` | el audio-side sí gana con conditioning aislado |
| `a4r-pcm` | standard | conditioned | D4 -> midi | `80.0%` | mejora marginal del lado MIDI |
| `a4r-pcd` | conditioned | conditioned | A4 + D4 | `84.2%` | brazo principal; mejor resultado de la línea |
| `a4r-pcd-zero` | conditioned | conditioned | zeros fijos | `81.8%` | control de overhead |

**Lectura actual**:
- `a4r-pcd > a4r-pcd-zero`: la mejora no viene solo de parametros extra;
- `a4r-pcd-zero > a4r-ctrl`: la arquitectura conditioned agrega expresividad por si misma;
- `a4r-pca > a4r-pcm`: el audio-side responde mas que el MIDI-side al conditioning aislado;
- Gate 5A/C1 ya no es una hipotesis abierta sino una linea positiva cerrada, aunque siga sin desplazar a Escalon 2 como foco principal.

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
- Gate 5B ya quedó cerrado como cierre cientifico de Escalon 1-C.
- Escalon 2 (Speech <-> EGG) pasa a foco principal una vez cerrado Gate 5B.
- Gate 5A puede seguir corriendo en paralelo como linea de exploracion si los recursos lo permiten.
- Gate 9 / revision `A10` no cuentan como continuidad automatica de Gate 5A: son una rama retrospectiva separada sobre armonia natural.

---

## Orden residual sugerido de ejecucion

La linea conditioned projections ya no necesita orden activo: Gate 8 quedó cerrada `5/5`.

Lo que queda vivo dentro de Gate 5A es el remanente combinatorio:

1. `t3-wt-vanilla`
2. `t3-wt-a4r`
3. cualquier brazo nuevo del usuario solo si entra con hipotesis clara y costo acotado

### Criterio de corte

- si `t3-wt-vanilla` no sostiene una señal mínimamente comparable a `t3-wt`, pausar la rama combinatoria;
- si `t3-wt-a4r` no mejora o no iguala razonablemente a sus padres, documentar el no y no escalar complejidad;
- no reabrir conditioned projections salvo nueva hipótesis puntual del usuario.

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
