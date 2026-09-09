# Protocolo: fuentes rivales observables

2026-09-08. Candidato para auditoría independiente; todavía no autorizado
para búsqueda experimental. Desarrolla [el diseño inicial](PLAN_OBSERVABLE_SOURCE_RIVALS.md).
Diagnóstico retrospectivo, no nuevo test de generalización ni arquitectura.

## 1. Datos, unidad y autoridad

Se reutiliza exclusivamente el roster del lector aprendido
`data/atencion_armonica/learned_partition_reader_v1/evaluation_release_recovery_v2/tests_01.json`,
SHA256 `7f4f368cf22ea60a2d02faa92099933cdbec2dcb13a69dc1df2a603f45c66153`.
Sus cuatro tests contienen 512 escenas cada uno; ya están abiertos.
No se generan escenas, entrenamientos, features ni forwards nuevos.

La unidad es una escena, no checkpoint × inicialización. Se eligen 24 IDs por
split con `default_rng(SeedSequence([2026090901, split_index]))`, choice sin
reemplazo entre 0..511, ordenados; índices de split 0..3 en el orden siguiente:

- IID: 13,30,66,131,163,190,228,231,277,297,308,341,348,367,391,398,424,444,445,453,472,482,492,507.
- Beta: 6,9,13,19,49,89,107,113,127,147,167,183,211,231,236,265,271,309,335,337,381,391,399,450.
- Polifonía: 1,18,87,92,139,181,242,257,272,293,304,307,354,367,372,376,391,408,431,451,491,492,494,499.
- Deformada: 6,9,35,50,65,81,86,90,111,149,162,169,250,257,278,279,304,350,440,466,471,472,487,500.

No reemplazar IDs por errores, empates o ausencia de candidatos. Se conserva
el vector observado q32, su permutación hacia rango ascendente estable y
conteo de empates. Las firmas de partición usan esos rangos, no source_ids.
Los empates se reportan y no acreditan invariancia de una elección indexada
entre eventos de frecuencia exactamente igual.

El buscador recibe sólo q32 y pools observables preservados. El nombre del
split sirve para localizar datos, nunca para escoger una familia de ajuste.
Supervisión, fuentes plantadas y predicciones/métricas previas se consumen
sólo después de sellar la búsqueda observable completa. La partición plantada
se ajusta después como referencia privilegiada con el mismo procedimiento;
no se añade retrospectivamente al universo del buscador.

## 2. Candidatos de partición

Se toma la unión exacta de los pools de los tres checkpoints2026090721/22/23.
Se conservan todas sus firmas y procedencia. La búsqueda generativa sólo
ajusta particiones con k en {2,3,4} y cada grupo con 4..8 miembros. Las demás
se registran OUTSIDE_GENERATIVE_CARDINALITY, no como fits fallidos.

De cada partición válida de esa unión se generan todos los intercambios de
un evento entre dos grupos y todos los traslados de un evento a otro grupo
que conserven 4..8 miembros. No se itera sobre los vecinos resultantes.
Se deduplican, se retiran firmas presentes en la unión y se ordenan por
SHA256 de su JSON compacto ASCII, con la firma como desempate. Se retienen
los primeros 64 vecinos por escena, o todos si hay menos; no hay selección
por residual ni verdad. Un resultado sin particiones admisibles se preserva
como NO_OBSERVABLE_CANDIDATE, sin usar la partición plantada para repararlo.

El número de grupos distintos es finito y se cachea por escena/firma/familia.
Se informa cobertura separada para pool original, vecinos y referencia
plantada. El oracle histórico conserva su universo original y no se compara
como si hubiera usado los vecinos nuevos.

## 3. Familias y soporte conjunto

Para cada grupo, asignar índices distintos de 1..8 en orden creciente de q.
La plantilla logarítmica es `t(n;b,g)=log(n)+0.5*log1p(b*n²+g*n⁴)`.
Se enumeran todas las combinaciones de índices del tamaño del grupo. Bajo
ruido intercambiable y plantillas monótonas, el emparejamiento ordenado
minimiza error cuadrático para esa asignación de grupo.

Se usan las mismas ramas en todas las escenas:

| Rama | beta por fuente | gamma por fuente | k admisible |
|---|---|---|---|
| base-low | [1e-4,1e-3] | 0 | 2,3,4 |
| base-high | [3e-3,1e-2] | 0 | 2,3 |
| deformed-low | [1e-4,1e-3] | [5e-6,5e-5] | 2,3 |

Una rama rige la escena completa: no mezclar low/high o gamma cero/positivo
por grupos. Esto conserva la unión de soportes del generador, no su prior
de mezcla. La familia Base reúne las dos primeras ramas; la familia Extendida
las tres. Ambas se evalúan siempre, sin consultar el régimen verdadero.
La extensión contiene Base: menor costo no es evidencia automática de mejor
identidad. No se evalúa un posterior ni una probabilidad de pertenencia.

Todas las fuentes deben admitir f0 en [100,500] Hz bajo una misma escala
global. Los offsets relativos a_j sólo pueden realizarse conjuntamente si
su rango es como máximo log(5). No basta comprobar cada grupo aisladamente.
El witness devuelve f0 de todas las fuentes y reconstruye la escena centrada;
los índices son witnesses, no índices identificados.

## 4. Objetivo, ruido y cuantización

Convertir q32 a float64 y aplicar `P=I-11ᵀ/N`: y=Pq. La predicción es Pμ,
con μ_i=log(f0_grupo)+t_i. El objetivo es
`J = ||y-Pμ||²/(2*sigma²)`, sigma=2*log(2)/1200. Se reporta también RMS en
cents. Antes de cuantización, Pε tiene covarianza sigma²P y rango N−1.
J es el término variable de la log-densidad gaussiana en ese subespacio para
parámetros fijos, no la likelihood integrada de las celdas float32, ni
evidencia bayesiana después de buscar parámetros/índices/particiones.

La cuantización no se ignora: conservar q, y y epsilon_i igual a media
separación máxima entre q_i y sus dos vecinos float32. Para un witness fijo,
con r=y-Pμ y E=||epsilon||, reportar
`quantization_objective_bound=(2*||r||*E+E²)/(2*sigma²)`.
Es una cota de variación por redondeo de observación, no incluye error de
grilla ni convierte el score en probabilidad cuantizada exacta. Los tests
separan tolerancia de aritmética float64 de esa cota de cuantización.

No usar chi-cuadrado con grados de libertad ajustados a mano después de
selección, ni comparar familias con Bayes factors a partir de estos mínimos.
Se conservan N, k y dimensión continua nominal (2k−1 en Base, 3k−1 en rama
deformada), además de búsqueda discreta; no se esconde la flexibilidad extra.

## 5. Búsqueda finita y cotas de la grilla

No hay optimización continua de beta/gamma ni inicializaciones aleatorias.
Cada intervalo beta tiene una grilla geométrica fina de 257 puntos; gamma
positivo tiene 65. La grilla gruesa toma cada cuarto punto de ambas (65 y17).
Gamma cero es un punto. Las grillas gruesas son subconjuntos exactos de las
finas almacenadas. No se extienden mirando resultados.

Para cada grupo y combinación de índices se calcula el mínimo de SSE
centrado dentro del grupo sobre toda la grilla. Desempate: primer índice
lexicográfico y primer punto beta/gamma en orden ascendente. Se guardan ese
mínimo y su plantilla, media y parámetros para todas las asignaciones de
índices; se retienen las cuatro mejores asignaciones distintas para el paso
conjunto, o todas si hay menos. La selección fina y gruesa se guarda separada.
Calcular SSE con diferencias centradas y suma de cuadrados float64, no con
la identidad de normas que resta cantidades grandes casi iguales.

Para cada producto cartesiano de witnesses retenidos de los grupos (≤4^4
por resolución y rama), sea a_j=mean(y_G−t_G) y m_j el tamaño. Encontrar
offsets b_j que minimicen sum_j m_j(b_j−a_j)² sujetos a rango(b)≤log5:
`b_j=clip(a_j,L,L+log5)`, con L raíz de
`sum_j m_j(clip(a_j,L,L+log5)−a_j)=0`.
Usar 64 pasos de bisección en [min(a)−log5,max(a)]. En el caso ya realizable,
conservar a sin bisección. Al convertir offsets a f0 usar
`log(f0_j)=b_j−min(b)+log100`. Se verifica rango y reconstrucción, y se
recalcula J directamente desde los eventos; no confiar sólo en sumas de
costos guardadas. No se etiqueta infeasible un fit malo: este paso construye
un witness válido, aunque su costo pueda ser grande.

La cota inferior por partición/rama es la suma de mínimos de grupo de la
grilla fina, dividida por 2sigma², relajando la escala conjunta. La cota
superior fina es `min(J(mejor producto fino), J(mejor witness grueso completo))`.
Se conserva y recalcula ese witness grueso aunque sus asignaciones ya no
estén entre las cuatro mejores finas; sus índices de grilla pasan a la fina
multiplicados por cuatro. Guardar procedencia y parámetros completos; para
costos exactamente iguales, desempatar por asignaciones y puntos de grilla
en orden lexicográfico. No se requiere mezclar grupos gruesos y finos.
El mínimo separado de LB y UB entre ramas permitidas define las cotas por
familia. Verificar LB≤UB con tolerancia aritmética de J igual a1e-7;
son cotas numéricas, no certificados de aritmética de intervalos.
Estas cotas son **del universo de la grilla**, nunca del óptimo continuo.
Una brecha grande indica una limitación de esta búsqueda/relajación, no
incertidumbre física cuantificada. Guardar ambas cotas, parámetros y costos
gruesos/finos permite separar error de lector y resolución insuficiente.

## 6. Evaluación retrospectiva sellada después de búsqueda

Por escena/familia: mejor partición observable encontrada y segunda distinta,
ARI frente a la plantada, J y RMS, tamaños, rama elegida, margen segunda−mejor,
cobertura de candidatos y brechas superior−inferior de grilla. Después de
ajustar la referencia plantada, definir rival como firma distinta de ella;
reportar `UB_rival−UB_plantada` y el intervalo de optimización en grilla
`[LB_rival−UB_plantada, UB_rival−LB_plantada]`. Aquí LB_rival y UB_rival
son mínimos **separados sobre todas las firmas rivales observables**, no
las dos cotas de la firma elegida por UB. Guardar ambos argmins, que pueden
diferir, además del mejor witness concreto. Si el extremo superior es
menor que −1e-7, registrar RIVAL_BETTER_ON_GRID; si el extremo inferior es
mayor que1e-7, PLANTED_BETTER_ON_GRID; en los demás casos,
UNRESOLVED_GRID_COMPARISON. Esos estados describen separación numérica en
la grilla, no decisiones científicas ni equivalencia estructural. Los deltas
puntuales negativos se reportan aparte. Si no hay rival, usar estado
explícito y null, nunca infinito serializado como una certeza.

Conservar separada la evaluación del vector ideal y parámetros verdaderos
del sidecar: referencia privilegiada fuera de grilla, no candidato observable
ni límite alcanzable por un lector. Reconstruir q32 desde ideal+ruido+centrado
y permutación preservados antes de usar esas referencias.

Reportar por cada split las 24 filas, conteos de soporte/errores, medianas,
cuantiles10/90 y fracción de márgenes encontrados negativos (punto cero como
igualdad de costo, no umbral de promoción). Relacionar márgenes con 1−ARI
de cada uno de los siete lectores previos, promediando primero sus nueve
celdas por escena; Spearman descriptivo con n efectivo y null si constante.
No agregar los cuatro splits en una victoria global ni crear un primario
confirmatorio. Las ramas/candidatos no disponibles mantienen su denominador.

No se busca una prueba general de inyectividad ni se implementa un
certificado de colisión entre medias ideales centradas de parametrizaciones
inequivalentes: `noiseless_collision_status=NOT_TESTED`. Reconstruir la
muestra observada mediante `float32(Pμ_witness)==q32` sólo se registra como
EXACT_SAMPLE_FIT; puede absorber el ruido realizado y no acredita una
colisión noiseless. Tampoco un fit cercano constituye ese certificado. La
búsqueda no prueba ni refuta identificabilidad estadística de distribuciones
completas, ni interpreta la ausencia de rivales encontrados como inyectividad.

## 7. Implementación, recursos y reproducción

Archivos nuevos `observable_source_rivals.py` (matemática observable),
`observable_rival_campaign.py` (carga/roster/búsqueda),
`observable_rival_evaluation.py` (supervisión posterior) y tests/CLI separados.
No modificar fuentes congeladas. Un manifiesto pequeño fija hashes de estos
archivos, protocolo, roster96, fuentes upstream usadas, versiones y dispositivo.
Auditoría de protocolo antes de implementar; revisión de código/fixtures antes
de ejecutar la campaña. No encadenar wrappers de la campaña anterior.

Perfil mecánico en grupos deterministas de tamaños4,6,8 con plantillas
conocidas y perturbaciones deterministas de hasta2cents, sin escenas de test.
Comparar grillas idénticas CPU/GPU float64. GPU usa tensores por lotes y la
misma aritmética definida, TF32 deshabilitado; la producción usa un solo
backend elegido por costo medido, no por mejores resultados. Comparar costos
a tolerancia1e-7 de J y reconstrucciones a1e-10 en logfrecuencia; empates
numéricos se reportan, no se transforman en afirmaciones semánticas.
Si estas tolerancias fallan, reparar la implementación antes de la búsqueda.

Presupuesto inicial: CPU con un hilo, ≤4GiB RSS; GPU local≤6GiB VRAM si
acelera materialmente el barrido; lote ajustable sólo por memoria. Perfil≤120s
por backend, campaña≤2h de cómputo acumulado y≤5GiB nuevos de artefactos.
Los límites son operativos, no criterios científicos. Si la proyección los
excede, revisar el plan antes de comenzar, no podar escenas o controles.
Mariano ya habilitó la GPU sin ventanas por corrida; verificar disponibilidad
y detener trabajo propio si lo solicita. No Mendieta por rutina.

Persistir por escena los candidatos y su selección, fits de grupo/cotas,
todos los witnesses conjuntos evaluados o sus factores reconstruibles,
salida por partición, estados de exclusión/fallo y tiempo/memoria. Archivos
nuevos, recuperables por escena y sin sobrescribir resultados. Sellar el
conjunto observable completo antes de evaluación privilegiada. Replay desde
factores y witnesses guardados, sin repetir barridos ni redes; reproducir
scores/elecciones y métricas, con recursos separados de payloads científicos.

Tests mínimos: selección exacta96IDs; no truth en API observable; permutación
sin ties; traslación/gauge; ruido centrado y cuantización; proyección conjunta
comparada con enumeración pequeña; rechazo de soporte incorrecto; nesting de
grillas/familias; bounds que encierran una enumeración exhaustiva pequeña;
separación de candidata y plantada; replay; recuperación sin sobrescritura.
Al final: auditoría de evidencia y alineación, balance por escenarios,
documentación/wiki y commit/push. Ninguna reparación de infraestructura ni
preflight aislado constituye cierre del goal.
