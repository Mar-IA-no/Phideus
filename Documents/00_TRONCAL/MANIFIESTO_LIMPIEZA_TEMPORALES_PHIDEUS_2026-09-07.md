# Manifiesto de limpieza de temporales externos de Phideus — 2026-09-07

## Alcance y autorización

Mariano autorizó explícitamente el 7 de septiembre de 2026 limpiar las
carpetas de Phideus dispersas en `/mnt/m2-1TB` y adoptar la política del
mensaje recursivo 022. Esta autorización nueva satisface la restricción del
mensaje 010 y se limita a residuos atribuibles a Phideus. No alcanza otros
proyectos, sesiones Codex, terminales Orca, el checkout canónico ni worktrees
Git.

## Inventario previo

El inventario del filesystem encontró **21 directorios** temporales, no 22,
con **24.688.349.094 bytes aparentes (22,993 GiB)**. El SHA-256 de la lista
ordenada de nombres es
`f4d3913a2846e5550ebb30288d70cc81106557a3a15860289355057d5dece9dd`.
Todos contienen salidas derivadas de pytest, fixtures copiadas, repos de
prueba o árboles de recovery; ninguno contiene marcador `.git` en su raíz.

| Directorio relativo a `/mnt/m2-1TB` | Bytes aparentes | Clasificación |
|---|---:|---|
| `.phideus-pytest-9-quarantine-20260906` | 1.760.945.136 | pytest/quarantine regenerable |
| `.phideus-pytest-r504-sQfxgM` | 1.983.161.848 | pytest regenerable |
| `.phideus-pytest-r516-jRpnSM` | 16.816 | pytest regenerable |
| `.phideus-pytest-r520-4q2qjj` | 19.270 | pytest regenerable |
| `.phideus-pytest-r520-regression-Ude1Y5` | 13.264 | pytest regenerable |
| `phideus-r520-debug.Gbjkcf` | 0 | temporal vacío |
| `phideus-r520-pytest.AE7lLh` | 4.866.129.050 | pytest regenerable |
| `phideus-r520-pytest.rQKX6K` | 4.866.129.128 | pytest regenerable |
| `phideus-r528-focal` | 100.138 | pytest regenerable |
| `phideus-r528-focal.Kb0Kfr` | 100.448 | pytest regenerable |
| `phideus-r528-full.L7mTqx` | 4.866.210.317 | pytest regenerable |
| `phideus-r528-refocal.7vOav3` | 100.574 | pytest regenerable |
| `phideus-wave60-r498-corrected` | 111.162 | pytest regenerable |
| `phideus-wave60-r498-delta` | 2.253 | pytest regenerable |
| `phideus-wave60-r498-full` | 4.251.271.288 | pytest regenerable |
| `phideus-wave60-r498-guard` | 8.522 | pytest regenerable |
| `phideus-wave60-r498-regressions` | 1.983.161.598 | pytest/replay regenerable |
| `phideus-wave60-r498-static` | 57.939 | pytest regenerable |
| `phideus-wave60-r498-static-e2e` | 110.705.071 | pytest regenerable |
| `phideus-wave60-r498-static2` | 46.666 | pytest regenerable |
| `phideus-wave60-r498-static3` | 58.606 | pytest regenerable |

Los tamaños de la tabla provienen de `du -sb --one-file-system`; por eso no
incluyen el tamaño del contenido apuntado por symlinks. Se verificó por `/proc`
que ningún proceso tenía `cwd` ni descriptor abierto dentro de los candidatos,
y se contrastó que no aparecieran como worktrees en Git u Orca.

## Objetos protegidos o fuera del alcance

- `/mnt/m2-1TB/Phideus`: checkout canónico activo, protegido.
- `/mnt/m2-1TB/Phideus-piddock`: se protegió durante la primera fase por ser un
  worktree Git, no un temporal. Una auditoría posterior confirmó que estaba
  limpio, ocupaba 593 MiB según `du -sh`, apuntaba a `480c7ef` —el mismo commit
  que `altermundi/main`—, no tenía commits propios y era ancestro de `main`.
  Con esa evidencia se retiró luego mediante Orca; la rama local `piddock` se
  restauró en `480c7ef` después de que la operación de Orca la eliminara junto
  con el checkout.
- `/mnt/m2-1TB/phideus-r528-check-attempt.json`: archivo regular de 6.207
  bytes; no es una carpeta y queda conservado porque la autorización pidió
  limpiar carpetas.
- cambios sin commit del goal físico CPU: permanecen en el checkout canónico y
  no se mezclan con la operación destructiva.

## Procedimiento aprobado

1. Intentar tomar `/tmp/backup_m2_to_drive.lock` para serializar la fase
   destructiva. El intento quedó esperando porque el backup iniciado el 2 de
   septiembre conserva el lock mientras su único hijo ejecuta `rclone copy`
   desde RAID hacia Drive. Esa etapa ya no lee `/mnt/m2-1TB`. Se retiró el
   waiter propio sin tocar el backup y se adoptó, para estos temporales que no
   se copiarán a RAID, `/tmp/phideus_external_temp_cleanup.lock`. Antes de
   eliminar se vuelve a comprobar que el backup no regresó a una etapa que lea
   el origen local.
2. Revalidar nombre, tipo, device/inode, ausencia de `.git`, liveness y lista
   completa inmediatamente antes de retirar.
3. No copiar estos árboles a RAID: son scratch regenerable que, bajo la nueva
   política, debe estar excluido del backup. Los resultados canónicos que
   justificaron las pruebas permanecen en Git, `Biblioteca/` y `data/`.
4. Eliminar sólo los 21 paths explícitos, sin globs dentro de la operación
   destructiva y sin seguir symlinks ni cruzar filesystems.
5. Recontar candidatos, verificar Git y registrar bytes liberados.

La restauración de estos árboles no es necesaria para retomar el proyecto. Si
se requiere reproducirlos, se vuelven a ejecutar las suites pytest de las
campañas R498, R504, R516, R520 y R528 usando una raíz local bajo
`Phideus/.agent-work/<request_id>/pytest`.

## Política desde este cierre

Todo nuevo temporal del proyecto vive en
`/mnt/m2-1TB/Phideus/.agent-work/<request_id>/`, con subdirectorios `tmp`,
`pytest`, `staging` o `quarantine` según corresponda. `.agent-work/` queda
excluido explícitamente de Git y del backup ordinario. Cada ejecución conserva
owner, propósito y condición de cierre, y retira su zona exacta apenas deja de
ser necesaria.

## Resultado posterior

La primera fase terminó con `21/21` directorios retirados y
`24.688.349.094` bytes aparentes liberados. La reconsulta top-level devolvió
cero directorios `.phideus-*` o `phideus-*`. El checkout canónico,
`Phideus-piddock` y el JSON fuera de alcance permanecían presentes en ese
corte. La auditoría específica posterior habilitó retirar también el checkout
`Phideus-piddock`; su rama local y commit permanecen recuperables, sin carpeta
hermana residual.

Git y Orca registran ahora únicamente el worktree `Phideus` (`main`); la rama
local `piddock` sigue apuntando a `480c7ef`. Orca conserva el workspace canónico
de Phideus sin haber cerrado ni modificado terminales. Como prueba mínima
posterior, la suite `tests.test_proportional_set_valued_physical` pasó `14/14`
por CPU. No se usó ni consultó GPU/CUDA.
