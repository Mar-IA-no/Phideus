## Tercera revisión Wave 58

**Veredicto: `REVISE`.**

Los cinco findings anteriores fueron materialmente corregidos, pero el ledger histórico conserva una inconsistencia numérica y un desempate incompleto que impiden considerarlo literalmente reproducible.

### Findings

1. **MEDIO — El threshold fijo de `P1` no es exacto.**

   El plan fija `P1` en `0.3719541471` (`WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md:121`), pero el valor `float64` congelado en `selection_arrays.npz` es:

   ```text
   0.37195414707872076
   ```

   La diferencia es aproximadamente `2.13e-11`, mayor que el `atol=1e-12`, y la propuesta usa desigualdad estricta. Por tanto, el decimal publicado no puede sustentar igualdad exacta de threshold y potencialmente puede alterar una máscara.

   **Corrección:** usar el valor completo `0.37195414707872076` o definir `P1` como recomputación de `q=0.8, method="higher"` sobre el score Wave 57, seguida de aserción contra el escalar congelado.

2. **MEDIO — El tie-break y los tests no cubren todos los guards del ledger.**

   El orden total sólo enumera `harm, incompatibility` (`:275-285`), mientras el ledger contiene `compatibility_loss`, `accuracy_loss`, `tail_breach` y la conjunción triple `P1-HCT` (`:121-144`). La gramática general describe como máximo dos guards (`:244-251`) y la matriz de tests no exige específicamente la conjunción triple ni un orden total para esos targets históricos (`:374-375`).

   **Corrección:** congelar un orden general de targets —o el orden izquierda-a-derecha de cada ID— y generalizar la máscara a `proposal AND ∧ᵢ(scoreᵢ<thresholdᵢ)`. Añadir fixture de `P1-HCT`, incluyendo producto `Qg8³` y desempate.

### Verificación solicitada

| Punto | Estado |
|---|---|
| Ledger de 24 IDs únicos | **Parcial:** el conteo `7 + 6 + 6 + 5 = 24` es correcto y las variantes están expandidas; persisten los dos defectos anteriores |
| Binding plan + auditoría y replay completo | **Resuelto:** binding post-aceptación en `:350-353`; replay cubre todos los artefactos científicos FIT/SELECT en `:333-342` |
| `LEGACY-W57` | **Resuelto:** refit → reselect → re-evaluate ocurre antes de abrir referencias (`:227-235`) |
| Transformaciones y autoridad HGB | **Resuelto:** scaler lineal, HGB crudo y scores HGB como autoridad (`:210-218`) |
| Bootstrap | **Resuelto:** PCG64, orden, percentiles, rango y pairing compartido (`:385-393`) |

No encontré inconsistencias nuevas en el factorial canónico de `36` candidatos, selectores canónicos, prioridad de `HARD_ONLY`, separación FIT/SELECT/MONITOR, nominación adaptativa ni artefactos. Los hashes de fuentes listados continúan coincidiendo.

Plan auditado: SHA-256 `04383ad462384122daeb7bf0bc3a702436766c8ad8c43d5e529bc5876dc4efbd`.

El segundo dictamen quedó preservado verbatim en [409_wave58_open_model_class_diagnostic_plan_reaudit.md](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/409_wave58_open_model_class_diagnostic_plan_reaudit.md).

**SHA-256 R409:** `533c90ba4da10de4e2b3e2fa2c719d32618c2b0acf8ef7304dfbca47fe9ad1b0`

No modifiqué ningún otro archivo ni ejecuté suites, GPU o web.
