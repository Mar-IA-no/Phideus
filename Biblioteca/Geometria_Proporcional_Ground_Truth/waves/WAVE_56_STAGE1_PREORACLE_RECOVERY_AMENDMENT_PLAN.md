# Wave 56 Stage 1: amendment de recuperacion pre-oraculo

**Estado:** plan prospectivo de recuperacion, previo a cualquier cambio de codigo
y a cualquier apertura de inferencia, oraculo o labels.

**Fecha:** 2026-09-03.

## 1. Motivo y limite epistemico

La preparacion oficial de Wave 56 Stage 1 creo y verifico el escrow, genero el
benchmark fresco y aborto antes de inferencia por este error:

```text
RuntimeError: fresh benchmark pair-token count differs from prospective freeze
```

El intento quedo archivado en:

```text
data/geometria_proporcional/
  wave56_contextual_gate_fresh_v1.failed_20260903T171827485015Z/
```

El fallo no abrio resultados: no existen `inference/`, `authorized_labels/`,
`bundles/`, `preparation_freeze.json` ni `preparation_receipt.json`. Existen
solamente el escrow y su freeze publico, el benchmark sellado/visible y
`FAILURE.json`. Por lo tanto, este amendment no corrige una adjudicacion a la
vista de sus resultados: corrige un guard de integridad antes de cualquier
inferencia u oracle.

No se habilita un nuevo sorteo. Las tres claves ya extraidas son la unica
realizacion admisible. El escrow original permanece inmutable y el intento
fallido se conserva como evidencia.

## 2. Diagnostico demostrado

El contrato prospectivo fija `768` **pair tokens elegibles** por split. El
benchmark generado contiene, en cada uno de `train`, `val` y `lockbox`:

- 4992 fixtures;
- 1152 `pair_token` distintos en toda la verdad sellada;
- 768 `pair_token` distintos con
  `is_out_of_catalog == false` y
  `calibration_population == "canonical_preserving"`;
- 384 tokens out-of-catalog; 192 tokens pertenecen a la poblacion de
  calibracion no canonica.

El preparador conto los 1152 tokens sellados totales y los comparo con 768. La
cantidad congelada es correcta: coincide con la poblacion que posteriormente
consume `load_labeled_records`. El bug esta exclusivamente en el predicado del
guard pre-oraculo.

Evidencia congelada del intento:

- commit del contrato de escrow:
  `51aae0715dfe8318f5333c568429c8e9af59f866`;
- SHA-256 canonico del contrato:
  `6fb76aef60722fca85c586e985cf13258d3eb515611bffa448d6e8ce1c17581f`;
- SHA-256 de `benchmark/manifest.json`:
  `7582efe3fdcd40125929cbe2c6783a37b1ba3f8ffb2fb6cce6b5578979d29ef8`;
- SHA-256 del preparador original:
  `7ff5919d2b0bdd607ca179180c4f94de3ff5be6e23e6024b21e748d22c61fb44`;
- SHA-256 del escrow publicado:
  `f86fb936651552a757b46acd56e1c17674635eb37cfc6d0cd2a8a02e2f06e978`;
- SHA-256 del freeze publico:
  `c65d581a755d611f9f86264402bfea89503a8599d7967d94882d2db91f5083e8`.

Los compromisos publicos de las claves se preservan en el escrow y en el
manifest; las claves no se copian a este documento.

## 3. Unico delta cientifico permitido

El guard pasa de contar todos los tokens sellados a contar solamente los tokens
elegibles:

```python
not row["is_out_of_catalog"]
and row["calibration_population"] == "canonical_preserving"
```

No cambian generador, claves, protocolo, benchmark, splits, features,
checkpoints, modelos, seeds, criterios, estimando, selector, bordes ni workers.
No se cambia el valor esperado `768`.

El recibo de generacion debe distinguir `sealed_pair_token_counts_total` de
`sealed_eligible_pair_token_counts`; el guard usa solo el segundo. Esto evita
que la correccion vuelva ambiguo que poblacion fue validada.

## 4. Amendment ejecutable y doble procedencia

La excepcion no se codifica como una comparacion laxa de contratos. Se agrega
un argumento explicito `--recovery-amendment` y un JSON canonico versionado que
autoriza una sola cadena de recuperacion. Ese JSON debe ligar:

1. este plan final y su hash;
2. el commit y hash canonico del contrato del escrow original;
3. el hash original y el hash corregido de
   `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
4. el commit de implementacion corregida, que debe ser ancestro de `HEAD`;
5. el informe independiente que aprueba la implementacion y su hash;
6. el basename exacto del intento fallido y el hash de su manifest;
7. el inventario permitido del intento pre-oraculo;
8. el predicado de elegibilidad y los conteos esperados/observados;
9. la declaracion `no_redraw`, `no_inference`, `no_oracle`, `no_labels`.

El JSON final se publica recien despues de que la implementacion tenga commit y
auditoria independiente. No forma parte del escrow retroactivamente: su hash y
contenido se incorporan a `generation_receipt.json`,
`preparation_freeze.json`, `preparation_receipt.json` y al replay. El escrow y
`pre_generation_freeze.json` conservan bytes y semantica originales.

La preparacion distingue entonces dos contratos:

- **contrato de origen del escrow:** inmutable, anterior al sorteo;
- **contrato de ejecucion recuperada:** `HEAD` limpio y fuentes actuales, con
  el unico delta autorizado por el amendment.

La procedencia dual debe quedar explicita; no se presenta el codigo corregido
como si hubiera sido el codigo que sorteo las claves.

## 5. Validacion antes de crear output

Antes de crear o archivar el output canonico, la recuperacion debe abortar si no
se cumplen todas estas condiciones:

1. El amendment es un archivo regular, versionado, identico a `HEAD` y con
   schema/estado exactos.
2. El commit de implementacion declarado es ancestro de `HEAD`.
3. El escrow fuente es root-owned `0600`, verifica sus tres compromisos y su
   freeze publico es la proyeccion exacta ya definida.
4. El contrato del escrow y su hash coinciden con los fijados en el amendment.
5. El contrato de ejecucion actual coincide con el contrato original en config,
   config prospectiva, upstream, preflight historico y bindings.
6. El mapa de fuentes difiere exclusivamente en el preparador y exactamente
   entre los hashes old/new autorizados. Agregar archivos auxiliares no permite
   cambiar ninguna fuente declarada por el contrato original.
7. El intento fallido tiene el basename fijado, el `FAILURE.json` esperado y el
   manifest fijado. `validate_manifest`, `validate_visible_package` y la
   atestacion semantica vuelven a pasar.
8. El inventario no contiene inferencia, oracle materializado, labels,
   bundles, fases analiticas ni freezes/receipts posteriores a generacion.
9. Los tres splits del benchmark fallido vuelven a demostrar 4992 filas, 1152
   tokens totales y 768 tokens elegibles.

La recuperacion ordinaria sin amendment conserva la igualdad estricta de
contrato existente. El amendment no puede usarse para un primary nuevo ni para
otro archivo fallido.

## 6. Ejecucion de la recuperacion

La recuperacion usa `--recovery-secrets-from` apuntando al intento fallido y
`--recovery-amendment` apuntando al JSON aprobado. El preparador:

1. extrae las claves solamente del escrow durable;
2. vuelve a publicar el mismo escrow y el mismo freeze publico;
3. invoca el mismo generador, sin ningun sorteo, con esas claves;
4. exige que el nuevo `benchmark/manifest.json` sea byte-identico al hash del
   intento fallido antes de continuar;
5. valida por split los conteos totales y elegibles por separado;
6. recien entonces ejecuta la inferencia ciega y publica el freeze de
   preparacion con la doble procedencia.

Si cualquier verificacion falla, el nuevo estado se archiva y no se abre
oraculo. Nunca se modifica ni elimina el intento fallido original.

## 7. Replay exacto

El replay usa el primary recuperado como fuente de claves y referencia, y exige
el mismo `--recovery-amendment`. Debe verificar, ademas de la matriz de replay
ya congelada:

- igualdad del hash del amendment copiado al primary;
- igualdad del contrato de origen del escrow;
- igualdad del contrato de ejecucion y del delta autorizado;
- igualdad del manifest con el intento fallido y el primary;
- mismos conteos total/elegible por split.

El replay no convierte el amendment en permiso general para futuros cambios.
Si cambia cualquier fuente requerida, hace falta un nuevo plan y una nueva
auditoria; nunca se reutiliza silenciosamente esta excepcion.

## 8. Implementacion y archivos

Se permiten estos cambios:

- `experiments/geometria_proporcional/prepare_wave56_fresh.py`: predicado
  correcto, validacion del amendment, procedencia dual y binding de replay;
- `tests/test_wave56_preoracle_recovery.py`: pruebas nuevas, separadas para no
  alterar retroactivamente el hash del test original ligado al escrow;
- un JSON canonico de amendment bajo
  `experiments/geometria_proporcional/configs/`, creado despues del commit de
  implementacion y ligado a la auditoria independiente;
- informes de auditoria y cierre en `Biblioteca/`.

No se modifica el config prospectivo original ni ninguna otra fuente declarada
por su `required_execution_sources`.

## 9. Verificacion obligatoria

1. Test unitario del predicado: 1152 totales, 768 elegibles; el guard acepta
   768 y rechaza cualquier drift de ambas poblaciones.
2. Amendment ausente, no versionado, sucio, mal hasheado, no aprobado o ligado
   a otro intento: aborta antes de crear output.
3. Cualquier delta adicional en el contrato/fuentes: aborta.
4. Presencia de inferencia, oracle, labels o bundles en el intento fuente:
   aborta.
5. Escrow, freeze, manifest o benchmark alterados: aborta.
6. Recuperacion fisica sintetica: reutiliza claves, no llama a
   `secrets.token_bytes`, reproduce el benchmark y llega a `PREPARED`.
7. Crash matrix de recuperacion: todo estado fallido se archiva y conserva
   escrow; nunca habilita redraw.
8. Replay fisico sintetico exacto con el mismo amendment.
9. Suite focal y suite amplia Wave 49--56 completas.
10. Auditoria independiente de este plan antes de implementar y de codigo,
    tests y JSON final antes de ejecutar la recuperacion oficial.

## 10. Criterio de salida

Este amendment queda tecnicamente habilitado solamente con auditoria
independiente `PASS`, `HEAD` limpio y todos los tests verdes. No declara GO
cientifico ni interpreta resultados. Su unica funcion es restaurar la validez
transaccional del draw ya realizado y permitir que el protocolo prospectivo
continue desde el limite pre-oraculo sin cambiar la pregunta experimental.
