# Gate 10 - Mechanism Sweep

**Fecha de apertura documental**: 2026-03-12  
**Estado**: completo en UNC. Los `9/9` arms ya cerraron `30ep` y dejaron una lectura final comparable.

Gate 10 nace como respuesta directa al cierre de Gate 9 y revision `A10`. Bajo `reverse cross-attention`, `a7r`, `a9r` y `a10a-d` convergieron todos a una banda muy estrecha (`69-72%`). Esa observacion no autoriza por si sola una conclusion fuerte sobre los descriptores. Puede querer decir que todos portan una senal parecida, pero tambien puede querer decir algo mas simple: que el mecanismo de inyeccion ya estaba dominando la lectura y comprimiendo sus diferencias.

Por eso Gate 10 no es "mas A10". Es un barrido causal para separar dos preguntas que Gate 9 y `A10` todavia mezclaban:

1. que parte del resultado depende del **contenido** del descriptor;
2. que parte depende del **mecanismo** con el que ese descriptor entra al encoder.

## Pregunta central

**¿Importa el mecanismo de inyeccion para estos descriptores audio-side o la banda estrecha de Gate 9/A10 ya refleja un techo descriptorial real?**

## Diseno del piloto

El piloto no abre la grilla completa `7 x 3`. Arranca con `3` descriptores representativos y `3` mecanismos:

| Descriptor | Dim | concat | pca | attn_bias | rev_xattn previo |
|------------|-----|--------|-----|-----------|------------------|
| `a7` | `12` | `a7` (re-run comparable) | `a7-pca` | `a7-ab` | `70.4%` |
| `a10a` | `12` | `a10a` | `a10a-pca` | `a10a-ab` | `70.6%` |
| `a10d` | `32` | `a10d` | `a10d-pca` | `a10d-ab` | `70.2%` |

Eso da `9` runs comparables bajo el mismo protocolo `30ep / from-scratch / run-d / seed=42`.

## Baselines correctos

Gate 10 trabaja en **audio-only**. Por eso sus referencias no son equivalentes a `d4a4`.

| Arm | Tipo | `S` | Uso comparativo |
|-----|------|-----|-----------------|
| `ctrl` | sin descriptor | `79.2%` | baseline comun |
| `a4r-pca` | FiLM audio-only | `82.6%` | referencia para columna `pca` |
| `a7r` / `a10ar` / `a10dr` | reverse x-attn | `70.4-70.6%` | referencia retrospectiva por descriptor |
| `d4a4` | dual concat | `84.1%` | no comparable directo |

## Lectura esperada

Gate 10 puede cerrar tres tipos de lectura:

1. **Si concat / pca / attn_bias se separan con claridad**, entonces el cuello de Gate 9/A10 era mecanistico y no meramente descriptorial.
2. **Si los tres mecanismos convergen a la misma banda**, entonces la compresion observada bajo `rev_xattn` probablemente refleje una limitacion del contenido o del problema.
3. **Si algun descriptor responde distinto segun mecanismo**, aparece una interaccion descriptor x mecanismo que Gate 9/A10 no podia leer.

## Estado operativo

- codigo en `main`;
- `--gate 10` agregado para trazabilidad;
- `slurm/gate10_pilot.sh` ya quedó ejecutado por completo en UNC;
- `BRIEFING_UNC_GATE10.md` sigue siendo el handoff canónico del frente;
- los jobs de resume ya quedaron absorbidos en el cierre final `9/9`;
- bug de reanudación corregido: `ls -t` podía elegir `_archive_base_not_for_eval.pt`; el fix fue filtrar `grep -v '_archive_'`.

## Lectura final del corte

Los números ya no son intermedios. El barrido completo deja una jerarquía nítida:

| Arm | concat | FiLM/pca | attn_bias |
|-----|--------|----------|-----------|
| `a7` | **`76.4%`** | `71.8%` | `55.8%` |
| `a10a` | `75.6%` | `74.0%` | `59.6%` |
| `a10d` | `75.4%` | `73.2%` | `57.4%` |

La lectura correcta de este corte ya no es provisoria:

- `concat` gana en los tres descriptores visibles;
- `FiLM/pca` queda en una segunda banda consistente, pero por debajo;
- `attn_bias` queda descartado como mecanismo competitivo en esta rama;
- los tres brazos `concat` convergen en una franja muy estrecha (`75.4-76.4%`), lo que refuerza que el spread grande está entre mecanismos y no entre descriptores.

## Conclusiones

1. **`concat > FiLM/pca >> attn_bias`**. La diferencia entre mecanismos es mucho mayor que la diferencia entre descriptores.
2. **El mecanismo domina sobre el descriptor**. El spread intra-mecanismo es de ~`1pp` en `concat`, mientras el spread inter-mecanismo llega a ~`16-20pp`.
3. **`a7-concat=76.4% @ e29`** es el mejor arm del gate y muestra que la rama puede levantar mucho más tarde de lo que sugerían los cortes tempranos.
4. **Gate 10 no rescata la línea retrospectiva por encima de los mejores brazos ya conocidos**:
   - sigue por debajo de `ctrl=79.2%`;
   - y muy por debajo de `d4a4=84.1%`.
5. **La deuda retrospectiva queda mejor acotada**. La compresión observada en `A7r/A9r/A10` no era solo “falta de descriptor”; en esta rama el mecanismo explicaba buena parte del fenómeno.

## Documentos del gate

- `README.md` - estado canonico y lectura del frente
- `BRIEFING_UNC_GATE10.md` - handoff operativo para corrida en UNC

## Rol dentro del programa

Gate 10 no compite con Escalon 2 por prioridad epistemologica. Su rol es mas acotado: limpiar retrospectivamente una deuda interna de Escalon 1. Si Gate 9 y `A10` querian decir algo sobre armonia natural en musica, primero hacia falta saber si estabamos comparando descriptores o comparando mecanismos. Ese barrido ya dejó su respuesta comparable: en esta rama, el mecanismo pesa más que el descriptor, y aun el mejor `concat` no desplaza a los mejores brazos canónicos del programa.
