# Nota: Cimatica, topologia nodal y control armonico en tiempo real

> Fecha: 2026-03-13
> Estado: nota interna de orientacion para Escalon 4

## 1. Contexto

Esta nota registra una linea de ideas surgida a partir de conversaciones sobre un setup fisico de cimatica con control en tiempo real de:

- ganancia por armonico;
- fase por armonico;
- paneo por armonico;
- feedback visual inmediato sobre el patron de interferencia.

La pregunta de fondo no es solo si el setup es "lindo" o sugestivo, sino si agrega variables experimentales y descriptores nuevos para `ESCALON_4`.

## 2. Que ya estaba en el radar

Buena parte del encuadre general ya era compatible con el frente:

- forward determinista `parametros -> patron`;
- problema inverso `patron -> parametros`;
- relevancia de ratios racionales vs no racionales;
- posibilidad de retrieval y generacion condicionada;
- lectura de Lissajous/cimatica como banco de ground truth fisico o semi-fisico.

Esto ya aparece, con distintos niveles de madurez, en:

- `README.md`
- `ROADMAP_ESCALON_4.md`
- `Plan_Claude.md`
- `Plan_inaugural_construccion_dataset_Codex.md`
- `Documents/90_ARCHIVO_GLOBAL/Legacy/Rosetta/PROPUESTA_ROSETA_2_AUDIO_CINEMATICA.md`

## 3. Lo nuevo que si agrega valor

### 3.1 Topologia nodal como objeto primario

La idea mas fuerte que aparece aca no es "la figura" en general, sino la **topologia del patron nodal**:

- conectividad de regiones;
- aparicion o desaparicion de loops;
- merges y splits;
- numero de regiones encerradas;
- cruces y singularidades locales.

Esto es mas fuerte que tratar la salida solo como imagen raster o como trayectoria XY. Sugiere una familia de descriptores topologicos para Escalon 4.

### 3.2 Barridos racional vs irracional

La observacion sobre la estabilidad de patrones armonicos y el drift inmediato cuando entra una razon irracional sugiere un eje experimental claro:

- ratios armonicos simples;
- ratios racionales no armonicos;
- irracionales cercanos a racionales;
- irracionales "duros" como `phi`.

No solo para clasificacion visual, sino para estudiar:

- estabilidad temporal;
- cierre/no cierre del patron;
- velocidad de drift;
- transiciones entre familias de patron.

### 3.3 Descomposicion local/global por armonico

La distincion empirica entre:

- `f1` afectando la organizacion global del patron;
- armonicos altos perturbando subestructuras mas locales,

es valiosa porque abre una lectura jerarquica de fase y amplitud por armonico. Esto puede traducirse en:

- descriptores por banda armonica;
- perturbaciones controladas por armonico;
- lectura de sensibilidad local/global del patron.

### 3.4 Espacio de parametros con fronteras topologicas

La idea de que el espacio `ganancia/fase/pan por armonico` pueda estar particionado en regiones estables separadas por fronteras donde la topologia cambia discretamente es experimentalmente muy fuerte.

Esto empuja a pensar el frente no solo como dataset de pares `audio-imagen`, sino como:

- mapa de regiones de estabilidad;
- deteccion de boundaries;
- estudio de transiciones topologicas inducidas por parametros.

### 3.5 Framing forward fisico / inverse learned

Como framing experimental, la idea de:

- forward = fisica determinista;
- inverse = aprendizaje,

es muy buena. No hace falta usar todavia la palabra "codec" en documentos publicos, pero si sirve para ordenar el problema de aprendizaje.

### 3.6 Intuicion impulso-respuesta y medio resolvente

La conversacion tambien abre otra capa valiosa: pensar el setup no solo como generador de escenas, sino como **medio que resuelve una excitacion**.

La intuicion tipo `Jpsh!` o impulso inicial no deberia entrar todavia como tesis fuerte del programa, pero si puede traducirse a una pregunta experimental sobria:

- que parte del patron queda determinada por la estructura del medio;
- que parte depende del impulso o de la parametrizacion instantanea;
- y hasta que punto el sistema puede leerse como `query -> respuesta del medio` mas que como simple render de senales.

Eso conversa bien con la idea de respuesta impulsional y con una lectura mas fisica del setup, sin necesidad de importar por ahora afirmaciones fuertes sobre cognicion, subjetividad o "perspectiva armonica".

### 3.7 Linea derivada: computacion resonante

La parte mas promisoria de la deriva hacia ML no es la metafora total de "todo sistema piensa armonicamente", sino una linea mas acotada:

- medios resonantes con estructura fija;
- excitacion o query de entrada;
- respuesta organizada por modos propios;
- aprendizaje concentrado en inversion, lectura o ajuste de parametros del medio.

Esto hace visible una familia de conexiones que hoy no es el nucleo de `ESCALON_4`, pero si merece quedar en radar:

- `reservoir computing` / `echo state`;
- redes con parametrizacion sinusoidal o de sesgo espectral;
- interpretaciones de `attention` como busqueda de minimos de energia o memoria asociativa;
- espacios de fase toroidales o manifolds periodicos como sustrato parametrico.

Como linea derivada, esto es interesante. Como claim actual del frente, todavia seria prematuro.

## 4. Lo que conviene bajar a hipotesis o poner en cuarentena

Varias afirmaciones surgidas en la conversacion son inspiradoras, pero no deberian entrar al programa como si fueran conocimiento ya asentado:

- `ratio racional = condicion necesaria de standing wave`
  - para Lissajous o superposiciones periodicas cerradas, si;
  - como afirmacion general sobre standing waves, no.

- `la serie armonica forma un grupo bajo multiplicacion`
  - asi dicho, no.

- `phase shift del armonico n produce transformaciones rigidas generales`
  - puede pasar en casos limpios y muy simetricos;
  - no debe elevarse a ley general del setup fisico.

- `de una snapshot 2D nodal se reconstruye el campo 3D completo`
  - demasiado fuerte tal como esta formulado.

- `hay infinitos estados estables practicos`
  - matematicamente puede haber continuidad parametricamente rica;
  - experimentalmente siempre hay resolucion finita, ruido, hysteresis y drift.

- `todo sistema es su perspectiva armonica del universo`
  - como intuicion filosofica puede ser fertil;
  - como afirmacion cientifica directa, todavia no.

- `la inferencia es instantanea porque el medio ya computo todo`
  - como intuicion sobre medios resonantes, interesante;
  - como afirmacion ingenieril o teorica fuerte, demasiado prematura.

- `una red armonica o fasica tendria mas capacidad por definicion`
  - plausible como hipotesis de investigacion;
  - no como conclusion ya establecida.

## 5. Implicancias operativas para Escalon 4

### 5.1 Variables nuevas a registrar

Si el frente toca una capa cimatico-fisica o semi-fisica, conviene agregar campos como:

- `ratio_class`: harmonic / rational_nonharmonic / irrational_near / irrational_far
- `drift_rate`
- `closure_score`
- `nodal_components`
- `nodal_crossings`
- `connected_regions`
- `topology_state_id`
- `harmonic_index_perturbed`
- `phase_role`: global / meso / local (si resulta operacionalizable)

### 5.2 Nuevas tareas experimentales

1. **Boundary mapping**
   - mapear fronteras topologicas en el espacio de fase y ganancia.

2. **Rational vs irrational discrimination**
   - medir estabilidad, drift y recuperabilidad de parametros.

3. **Local/global phase sensitivity**
   - cuantificar cuanto cambia la geometria global vs local cuando se perturba cada armonico.

4. **Inverse problem**
   - recuperar parametros desde patron nodal o trayectoria.

5. **Topology-aware retrieval**
   - retrieval no solo por scene exacta, sino por clase topologica o familia de ratio.

6. **Impulse-response probing**
   - estudiar que parte del patron depende del medio y que parte depende de la excitacion, usando protocolos de perturbacion minima y respuesta estabilizada.

### 5.3 Impacto sobre descriptores

Ademas de `R1 natural`, `R2 perceptual-control`, `R3 geometrico` y `R4 dinamico`, aparece una posible subfamilia:

- `R3b topologico`
  - conectividad;
  - numero de componentes;
  - merges/splits;
  - invariantes de contorno;
  - clase nodal.

### 5.4 Linea secundaria de exploracion computacional

Sin mover el foco del frente, esta conversacion tambien sugiere una rama secundaria y separada:

- toy models donde el espacio parametrico no sea un vector euclidiano plano sino una estructura periodica;
- comparaciones pequenas entre representaciones estandar y representaciones armonico-fasicas;
- pruebas de estabilidad bajo perturbacion y capacidad por parametro;
- lectura del medio fisico como inspiracion para arquitectura, no como prueba directa de una arquitectura nueva.

Esto no deberia contaminar la `v0.1` del dataset, pero si puede quedar como desprendimiento futuro de alto interes.

## 6. Relacion con HIT

Esta nota no prueba la tesis fuerte de HIT, pero si fortalece una intuicion importante:

- ciertas relaciones armonicas no solo organizan senales;
- tambien organizan geometria visible y estabilidad de interferencia de modo no trivial.

Eso vuelve mas interesante a Escalon 4 porque lo desplaza desde "audio XY e imagen bonita" hacia:

- banco de estudio de estabilidad;
- banco de transiciones topologicas;
- banco de descriptores de ratio visibles.

## 7. Relacion con Beacon

Por ahora el valor para Beacon es indirecto.

Lo mas util no es traducir de inmediato estas ideas a experiencia subjetiva o a tesis fuertes sobre perspectiva, sino conservarlas como:

- intuicion sobre sistemas resonantes situados;
- material para pensar coordinacion, acople y legibilidad de patrones;
- posible insumo futuro para la capa de bodily resonance.

No parece, por ahora, una linea primaria para Beacon.

## 8. Decision provisional

La conclusion mas razonable es esta:

- **si** hay informacion relevante nueva para Escalon 4;
- esa informacion esta en la capa experimental y descriptorial;
- **no** conviene importar sin filtro las afirmaciones metafisicas o las sobreafirmaciones fisicas;
- lo prioritario es traducir estas observaciones a:
  - variables,
  - metadatos,
  - tareas,
  - y criterios de comparacion.

## 9. Watchlist bibliografica derivada

Para esta nota conviene dejar registradas algunas familias bibliograficas que no pertenecen todavia al nucleo del frente, pero si a su periferia prometedora:

- `Lauterwasser` y `CymaScope`
  - como archivo visual e instrumental de patrones armonicos en medios fisicos;
  - con cautela, porque su estatus academico es desigual.

- `Green's functions` / respuesta impulsional
  - como marco fisico para pensar el medio como estructura que resuelve una excitacion.

- `reservoir computing` / `echo state networks`
  - como analogia computacional seria para medios fijos + dinamica rica + readout entrenable.

- `modern Hopfield networks`, `attention as associative memory`, `energy minima`
  - como puente posible entre excitacion, estabilizacion y lectura.

- `spectral bias`, `Fourier-feature networks`, `sinusoidal parameterizations`
  - como lenguaje mas tecnico para discutir por que una parametrizacion armonica podria cambiar capacidad o estabilidad.

- `Pribram`
  - solo como antecedente historico a tratar con mucha cautela si alguna vez se abre una rama cognitiva o neurocomputacional.
