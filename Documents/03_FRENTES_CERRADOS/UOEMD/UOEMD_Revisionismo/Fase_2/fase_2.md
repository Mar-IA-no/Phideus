 Plan de Ejecución: Fase 2 - Re-entrenar Rosetta con Extractor v2.2                                                                        
                                                                                                                                           
 Fecha: 2026-01-30                                                                                                                         
 Objetivo: Validar H3 (cross-modality) usando histogramas discriminativos del Extractor v2.2                                               
 Criterio de éxito: Gap aligned-shuffled del modelo > 0.15                                                                                 
                                                                                                                                           
 ---                                                                                                                                       
 Contexto                                                                                                                                  
                                                                                                                                           
 Fase 1 Completada                                                                                                                         
                                                                                                                                           
 - Extractor v2.2 implementado con filtrado temporal                                                                                       
 - Sweep de 36 configuraciones ejecutado                                                                                                   
 - Mejor config: config_002 (K=8, prom=0.1, stab=0.7, warped=No)                                                                           
 - Gap pre-red: 0.691 (vs 0.004 de Rosetta1 2.0 = 172× mejor)                                                                              
                                                                                                                                           
 Problema Original (Rosetta1 2.0)                                                                                                          
                                                                                                                                           
 El modelo aprendía "histograma promedio" porque los histogramas eran uniformes:                                                           
 - aligned ≈ shuffled (gap = 0.002)                                                                                                        
 - Retrieval Top-1 = 0.78% (= random)                                                                                                      
                                                                                                                                           
 Hipótesis Fase 2                                                                                                                          
                                                                                                                                           
 Con histogramas discriminativos (gap pre-red = 0.69), el modelo debería poder aprender correspondencia real audio ↔ vibración.            
                                                                                                                                           
 ---                                                                                                                                       
 Plan de Ejecución                                                                                                                         
                                                                                                                                           
 Tarea 1: Regenerar Dataset con Config Óptima                                                                                              
                                                                                                                                           
 Archivo a generar: data/datasets/roseta_v22_full.npz                                                                                      
                                                                                                                                           
 Comando:                                                                                                                                  
 python src/analizador/analizador_roseta.py \                                                                                              
     --input-dir data/datasets/UOEMD/raw/2_CSV_Data_Files \                                                                                
     --output data/datasets/roseta_v22_full.npz \                                                                                          
     --top-k-peaks 8 \                                                                                                                     
     --min-prominence 0.1 \                                                                                                                
     --temporal-stability-threshold 0.7 \                                                                                                  
     --use-warped-bins false \                                                                                                             
     --workers 12                                                                                                                          
                                                                                                                                           
 Verificación:                                                                                                                             
 - Confirmar 128 archivos procesados                                                                                                       
 - Verificar shape correcto [n_files, T, 256, 3] para audio y vib                                                                          
                                                                                                                                           
 ---                                                                                                                                       
 Tarea 2: Entrenar RosetaVAE                                                                                                               
                                                                                                                                           
 Script: experiments/run_roseta_experiment.py                                                                                              
                                                                                                                                           
 Configuración recomendada:                                                                                                                
 python experiments/run_roseta_experiment.py \                                                                                             
     --data data/datasets/roseta_v22_full.npz \                                                                                            
     --output data/training_outputs/roseta_v22 \                                                                                           
     --all-data \                                                                                                                          
     --epochs 100 \                                                                                                                        
     --batch-size 8 \                                                                                                                      
     --beta-kl-private 0.01 \                                                                                                              
     --dropout-shared 0.5 \                                                                                                                
     --lambda-diff 0.1 \                                                                                                                   
     --lambda-infonce 1.0                                                                                                                  
                                                                                                                                           
 Parámetros clave (fix de z_private collapse):                                                                                             
 - --beta-kl-private 0.01: KL más bajo para z_private → permite varianza                                                                   
 - --dropout-shared 0.5: Fuerza uso de z_private en decoding                                                                               
 - --lambda-diff 0.1: Separa z_private entre audio y vib                                                                                   
                                                                                                                                           
 Output esperado:                                                                                                                          
 - data/training_outputs/roseta_v22/best_model.pt                                                                                          
 - data/training_outputs/roseta_v22/training_log.json                                                                                      
                                                                                                                                           
 ---                                                                                                                                       
 Tarea 3: Evaluación con Protocolo P0                                                                                                      
                                                                                                                                           
 3.1 Cross-Reconstruction con Controles Negativos                                                                                          
                                                                                                                                           
 python experiments/evaluate_cross_reconstruction.py \                                                                                     
     --model data/training_outputs/roseta_v22/best_model.pt \                                                                              
     --data data/datasets/roseta_v22_full.npz \                                                                                            
     --run-all-controls                                                                                                                    
                                                                                                                                           
 Controles obligatorios:                                                                                                                   
 ┌──────────┬───────────────────┬───────────────┐                                                                                          
 │ Control  │    Descripción    │   Criterio    │                                                                                          
 ├──────────┼───────────────────┼───────────────┤                                                                                          
 │ Aligned  │ Pares correctos   │ Baseline      │                                                                                          
 ├──────────┼───────────────────┼───────────────┤                                                                                          
 │ Shuffled │ Pares aleatorios  │ Debe degradar │                                                                                          
 ├──────────┼───────────────────┼───────────────┤                                                                                          
 │ Random z │ z_shared ~ N(0,1) │ Debe degradar │                                                                                          
 └──────────┴───────────────────┴───────────────┘                                                                                          
 Criterio GO:                                                                                                                              
 - Δcorr (aligned - shuffled) > 0.15                                                                                                       
 - shuffled_retrieval < 5%                                                                                                                 
                                                                                                                                           
 3.2 Retrieval Global                                                                                                                      
                                                                                                                                           
 python experiments/evaluate_retrieval.py \                                                                                                
     --model data/training_outputs/roseta_v22/best_model.pt \                                                                              
     --data data/datasets/roseta_v22_full.npz                                                                                              
                                                                                                                                           
 Métricas:                                                                                                                                 
 - Top-1, Top-5, Top-10 accuracy                                                                                                           
 - MRR (Mean Reciprocal Rank)                                                                                                              
 - Variantes: global, intra-condición, intra-archivo                                                                                       
                                                                                                                                           
 Criterio GO: Top-1 > 10× random chance                                                                                                    
                                                                                                                                           
 3.3 Regime Separation                                                                                                                     
                                                                                                                                           
 python experiments/evaluate_regime_separation.py \                                                                                        
     --model data/training_outputs/roseta_v22/best_model.pt \                                                                              
     --data data/datasets/roseta_v22_full.npz                                                                                              
                                                                                                                                           
 Métricas:                                                                                                                                 
 - Silhouette score (en embedding real, no UMAP)                                                                                           
 - Linear probe AUC (Healthy vs Fault)                                                                                                     
 - Fisher ratio                                                                                                                            
                                                                                                                                           
 Criterio GO: Silhouette > 0.3                                                                                                             
                                                                                                                                           
 ---                                                                                                                                       
 Tarea 4: Análisis de Resultados y Documentación                                                                                           
                                                                                                                                           
 Si GO (3 de 4 criterios pasan):                                                                                                           
                                                                                                                                           
 1. Documentar en Documents/Roseta/ROSETTA_V22_RESULTS.md:                                                                                 
   - Configuración exacta del extractor                                                                                                    
   - Métricas con 5 seeds (o 3 mínimo)                                                                                                     
   - Comparación con Rosetta1 2.0                                                                                                          
   - Claim: "H3 supported under Protocol P0 + Extractor v2.2"                                                                              
 2. Actualizar Documents/00_TRONCAL/Proyecto_Estado_Actual.md                                                                                         
 3. Commit y merge a main                                                                                                                  
                                                                                                                                           
 Si NO-GO:                                                                                                                                 
                                                                                                                                           
 1. Documentar qué falló y por qué                                                                                                         
 2. Opciones:                                                                                                                              
   - A) Ajustar hiperparámetros del modelo                                                                                                 
   - B) Probar config_012 (máximo gap: 0.702)                                                                                              
   - C) Ir a Grupo 2 (log-spectrogram, JEPA, etc.)                                                                                         
                                                                                                                                           
 ---                                                                                                                                       
 Archivos Críticos                                                                                                                         
 ┌──────────────────────────────────────────────┬────────┬────────────────────────┐                                                        
 │                   Archivo                    │ Acción │       Propósito        │                                                        
 ├──────────────────────────────────────────────┼────────┼────────────────────────┤                                                        
 │ src/analizador/analizador_roseta.py          │ Usar   │ Generar dataset v2.2   │                                                        
 ├──────────────────────────────────────────────┼────────┼────────────────────────┤                                                        
 │ experiments/run_roseta_experiment.py         │ Usar   │ Entrenar RosetaVAE     │                                                        
 ├──────────────────────────────────────────────┼────────┼────────────────────────┤                                                        
 │ experiments/evaluate_cross_reconstruction.py │ Usar   │ Controles negativos    │                                                        
 ├──────────────────────────────────────────────┼────────┼────────────────────────┤                                                        
 │ experiments/evaluate_retrieval.py            │ Usar   │ Métricas retrieval     │                                                        
 ├──────────────────────────────────────────────┼────────┼────────────────────────┤                                                        
 │ experiments/evaluate_regime_separation.py    │ Usar   │ Regime probing         │                                                        
 ├──────────────────────────────────────────────┼────────┼────────────────────────┤                                                        
 │ data/datasets/roseta_v22_full.npz            │ Crear  │ Dataset discriminativo │                                                        
 └──────────────────────────────────────────────┴────────┴────────────────────────┘                                                        
 ---                                                                                                                                       
 Criterios GO/NO-GO Fase 2                                                                                                                 
 ┌──────────────────────┬──────────────┬─────────┐                                                                                         
 │       Criterio       │    Umbral    │  Peso   │                                                                                         
 ├──────────────────────┼──────────────┼─────────┤                                                                                         
 │ Gap aligned-shuffled │ > 0.15       │ CRÍTICO │                                                                                         
 ├──────────────────────┼──────────────┼─────────┤                                                                                         
 │ Retrieval Top-1      │ > 10× random │ Alto    │                                                                                         
 ├──────────────────────┼──────────────┼─────────┤                                                                                         
 │ Silhouette score     │ > 0.3        │ Medio   │                                                                                         
 ├──────────────────────┼──────────────┼─────────┤                                                                                         
 │ var(z_private)       │ > 0.1        │ Medio   │                                                                                         
 └──────────────────────┴──────────────┴─────────┘                                                                                         
 Decisión:                                                                                                                                 
 - GO si 3 de 4 pasan (incluyendo gap obligatorio)                                                                                         
 - NO-GO si gap < 0.15 (automático)                                                                                                        
                                                                                                                                           
 ---                                                                                                                                       
 Estimación de Tiempo                                                                                                                      
 ┌────────────────────────────┬───────────────────┐                                                                                        
 │           Tarea            │ Duración Estimada │                                                                                        
 ├────────────────────────────┼───────────────────┤                                                                                        
 │ Regenerar dataset          │ ~30 min           │                                                                                        
 ├────────────────────────────┼───────────────────┤                                                                                        
 │ Entrenamiento (100 epochs) │ ~2-3 horas        │                                                                                        
 ├────────────────────────────┼───────────────────┤                                                                                        
 │ Evaluación completa        │ ~30 min           │                                                                                        
 ├────────────────────────────┼───────────────────┤                                                                                        
 │ Documentación              │ ~1 hora           │                                                                                        
 ├────────────────────────────┼───────────────────┤                                                                                        
 │ Total                      │ ~4-5 horas        │                                                                                        
 └────────────────────────────┴───────────────────┘                                                                                        
 ---                                                                                                                                       
 Verificación End-to-End                                                                                                                   
                                                                                                                                           
 # 1. Verificar dataset generado                                                                                                           
 python -c "import numpy as np; d=np.load('data/datasets/roseta_v22_full.npz', allow_pickle=True); print(f'Files: {len(d.files)}')"        
                                                                                                                                           
 # 2. Verificar modelo entrenado                                                                                                           
 ls -la data/training_outputs/roseta_v22/best_model.pt                                                                                     
                                                                                                                                           
 # 3. Verificar evaluación                                                                                                                 
 cat data/training_outputs/roseta_v22/evaluation_report.md | head -50                                                                      
                                                                                                                                           
 ---                                                                                                                                       
 Plan preparado para Fase 2 del Revisionismo de Extracción de Ratios   