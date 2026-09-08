# Lector aprendido de particiones

Estado: siguiente goal formulado desde el contraste estructurado del
2026-09-08. Diseño inicial; requiere especificación ejecutable y auditoría
antes de producir datos. No implementado ni promovido.

## Pregunta finita

¿Puede un lector aprendido reconocer mejores particiones dentro del mismo
pool sin ocultar sobrefragmentación, y aporta la coherencia conjunta de
fuente información incremental frente a pares/estructura, compatibilidad
local y costos desacoplados?

El [contraste precedente](RESULTS_SOURCE_STRUCTURED_READER.md) no mostró
ventaja clara del factor fijo en mayor polifonía y perdió ARI medio en
inarmonicidad y familia deformada. El oracle del pool conserva mejores
particiones que los lectores deployables; es una razón para probar si esa
decisión puede aprenderse, no una prueba de que sea reconocible sin etiquetas.

Este goal termina con un contraste entrenado, evaluado sobre tests nuevos,
reproducido y auditado, incluso si el resultado es nulo o adverso. No termina
por dejar preparado otro preflight. Un impedimento material se registra
como tal; no se redefine la pregunta para simular su resolución.

## Hipótesis geométrica y cómputo

La unidad de decisión es una partición de eventos, invariante a renombrar
fuentes y permutar eventos fuera de los empates declarados. Las operaciones
de separar y fusionar grupos cambian esa estructura; su costo de aprendizaje
no tiene por qué coincidir con sumar errores independientes de aristas.

El sistema conserva las tres redes descriptor-guided congeladas y el
generador de candidatos anterior. Aprende únicamente una función de costo
de particiones, compartida entre candidatos y construida a partir de
agregaciones invariantes de grupos. No recibe índices verdaderos, k
verdadero, identidad de fuente ni nombre del régimen durante inferencia.
Una salida softmax sobre candidatos será una distribución del lector,
no un posterior físico calibrado por llamarse probabilística.

La arquitectura aprendida no debe llamarse «anti-fragmentación» por diseño:
esa propiedad es un resultado por medir. La loss y la evaluación deben
registrar tanto separar una fuente como fusionar fuentes distintas. No se
añaden penalties elegidos mirando los tests abiertos del experimento anterior.

## Contraste arquitectónico mínimo

| Brazo aprendido | Información adicional al canal común |
|---|---|
| Pares y estructura | Ninguna relación física de grupo adicional |
| Compatibilidad local | Costos de triples locales por grupo |
| Fuente compartida | Costos conjuntos por grupo |
| Fuente desacoplada | Los mismos costos conjuntos rotados dentro de tamaño |

Todos comparten pool, prior, logits, agregaciones estructurales, capacidad,
inicialización emparejada, batches, optimizador y presupuesto. Los brazos
local/global/desacoplado tienen un canal del mismo tamaño; el baseline sin
ese canal debe conservar capacidad efectiva comparable, no sólo parámetros
nominales inutilizados. La resolución concreta se fija y audita antes de datos.
No dar costos locales y globales simultáneamente sólo al brazo favorito.

El canal común incluye evidencia de pares y tamaños observables, con lista
exacta de features y normalizaciones train-only. Desacoplar conserva el
multiset por tamaño, no automáticamente el multiset ponderado por aparición.
Se registra soporte del control y dependencia efectiva del modelo del canal;
si aprende a ignorarlo, no hay contraste semántico activo por decreto.

Se conservan como referencias sin training Pares sobre el pool común, el
lector de factor fijo anterior y el lector histórico. Una mejora sólo frente
al histórico no acredita un aporte geométrico incremental.

## Arquitectura, loss y atribución

Se especificará una cabeza pequeña de costo sobre grupos/particiones con
agregación invariante y una pérdida de decisión estructurada. La comparación
de los cuatro brazos a loss común aísla el acceso al factor dentro de esa
receta. No aísla por sí sola el efecto de cambiar simultáneamente cabeza y
loss respecto del lector fijo: ese contraste pertenece al sistema completo.

La elección de loss requiere un contrato de errores de partición, no sólo
optimizar el nombre de la métrica favorable. Una consulta dirigida a la
[geometría de comparación de particiones](https://icml.cc/Conferences/2005/proceedings/papers/073_ComparingClustering_Meila.pdf)
recupera VI = H(C)+H(C′)−2I(C,C′) como candidata; no está seleccionada ni
se presume superior a ARI o error de pares. Si se normaliza por tamaño,
se declarará qué propiedades de composición se modifican.

Antes de implementar se fijan por escrito: ecuaciones, dimensiones, lista
de entradas, máscaras, normalización, target, entrenamiento, selección,
desempates y controles de soluciones triviales. No se abrirá un barrido
abierto de pérdidas para rescatar el factor anterior.

## Datos, selección y evaluación

Train y calibración nuevos; tests nuevos con semillas y rosters disjuntos
de todo lo ya observado. Conservar la ley frequency-only y los cuatro
regímenes para que el nuevo contraste no mezcle información adicional con
decisión aprendida. La cantidad de escenas y semillas, receta y estimando
primario quedan fijados en el protocolo ejecutable antes de generar datos.

Mantener al menos tres inicializaciones de lector y los tres checkpoints
históricos. Declarar si el diseño cruza ambos factores o los empareja;
no contar varias predicciones de una escena como muestras independientes.
El presupuesto elegirá el diseño completo, no una poda posterior por resultados.

El primario deberá responder al aporte de Compartida frente a los tres
controles aprendidos en mayor polifonía. ARI, partición exacta, errores de
pares, k, error de k, masa sub-3, geometría del error elegido y cobertura
oracle permanecen visibles en todos los slices. La selección usa únicamente
train/calibración y se sella antes de test, con incertidumbre pareada por
escena y tratamiento explícito de múltiples contrastes.

Conservar todos los modelos y checkpoints recuperables, predicciones de costo
por candidato, particiones, pools, features, logits, targets separados,
configuraciones, semillas, índices de remuestreo, manifests y recursos.
Replay desde esos estados, sin reentrenar ni repetir forwards innecesarios.

## Secuencia y recursos

1. Rebasar el diseño sobre fuentes actuales; fijar el protocolo ejecutable y
   auditarlo independientemente. Sólo consultas bibliográficas dirigidas por
   una dependencia concreta; no otra ola.
2. Implementar en archivos nuevos y verificar fixtures de permutación,
   invariancia de escala pertinente, máscaras, pérdida, selección, integridad
   y ausencia de etiquetas en inferencia. Auditar antes de datos.
3. Perfilar producción de pools y training. Usar CPU para preparación
   proporcionada y la 3090 para forwards/training cuando sea más eficiente,
   informando duración y VRAM y verificando disponibilidad vigente.
4. Ejecutar train/calibración, sellar y auditar freeze, abrir todos los tests
   previstos y completar análisis/replay. No tuning posterior ni recorte de
   controles para acomodar un resultado.
5. Auditar evidencia y alineación geométrica, documentar, commit/push y
   formular el siguiente goal desde lo aprendido.

## Bifurcaciones de salida

Si aprende a decidir mejor pero los factores físicos no añaden valor, conservar
la solución más simple y atribuir la mejora a lectura aprendida, no a armonía.
Si Compartida añade valor sobre todos los controles sin esconder fragmentación,
proponer desarrollo de ese mecanismo sin promoverlo unilateralmente.
Si no mejora pese al gap de cobertura, delimitar esa receta de aprendibilidad;
no declarar insuficiencia informacional a partir de un modelo fallido.

Tiempo, fase, medición/CQT y fuentes latentes permanecen alternativas. Pueden
convertirse en el siguiente contraste por una hipótesis propia verificable,
no porque este null pruebe que son necesarias. La extensión set-valued
continúa pausada e incompleta. GO/NO-GO y promoción corresponden al usuario.
