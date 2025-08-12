#!/usr/bin/env python3
"""
Script para actualizar documentación completa - Phideus v4.1
Automatiza la actualización de todos los documentos markdown cuando se solicita
"""

import os
import sys
import glob
from datetime import datetime
from pathlib import Path

def actualizar_documentos():
    """
    Actualiza todos los documentos de Phideus según estructura dual:
    
    OBLIGATORIO - NUNCA saltar ninguno:
    1. README.md - Descripción proyecto dual
    2. ARCHITECTURE.md - Documentación arquitectura dual  
    3. Documents/bitacora_desarrollo.md - Log compartido
    4. Documents/vae/* - Documentos línea VAE
    5. Documents/hrm/* - Documentos línea HRM
    """
    
    print("📚 Actualizando Documentación Phideus v4.1")
    print("=" * 50)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Lista de documentos OBLIGATORIOS - NUNCA omitir
    documentos_root = [
        "readme.md",
        "ARCHITECTURE.md"
    ]
    
    documentos_shared = [
        "Documents/bitacora_desarrollo.md"
    ]
    
    documentos_vae = [
        "Documents/vae/Hoja_de_Ruta_VAE.md",
        "Documents/vae/Proyecto_Estado_VAE.md", 
        "Documents/vae/RNA_Arqu_VAE.md",
        "Documents/vae/Scripts_src_VAE.md",
        "Documents/vae/Arquitectura_Neural_VAE_2025-08-12.md"
    ]
    
    documentos_hrm = [
        "Documents/hrm/Hoja_de_Ruta_HRM.md",
        "Documents/hrm/Proyecto_Estado_HRM.md",
        "Documents/hrm/RNA_Arqu_HRM.md", 
        "Documents/hrm/Scripts_src_HRM.md",
        "Documents/hrm/Arquitectura_Neural_HRM_2025-08-12.md"
    ]
    
    print("🎯 Documentos a actualizar:")
    print(f"  Root level: {len(documentos_root)} archivos")
    print(f"  Shared: {len(documentos_shared)} archivos")  
    print(f"  VAE line: {len(documentos_vae)} archivos")
    print(f"  HRM line: {len(documentos_hrm)} archivos")
    print(f"  TOTAL: {len(documentos_root + documentos_shared + documentos_vae + documentos_hrm)} archivos")
    
    print("\n⚠️  CRÍTICO: Si algún documento se omite, esto es un ERROR MAYOR")
    print("    Claude debe actualizar TODOS los documentos listados")
    
    # Verificar existencia
    todos_documentos = documentos_root + documentos_shared + documentos_vae + documentos_hrm
    faltantes = []
    
    for doc in todos_documentos:
        if not os.path.exists(doc):
            faltantes.append(doc)
    
    if faltantes:
        print(f"\n❌ DOCUMENTOS FALTANTES ({len(faltantes)}):")
        for doc in faltantes:
            print(f"    - {doc}")
        print("\n🔧 ACCIÓN REQUERIDA: Crear documentos faltantes")
    else:
        print(f"\n✅ TODOS LOS DOCUMENTOS ENCONTRADOS ({len(todos_documentos)})")
    
    print(f"\n⏰ Timestamp actualización: {timestamp}")
    print("\n📋 PROCESO DE ACTUALIZACIÓN:")
    print("1. Revisar estado actual del proyecto (ambas líneas)")
    print("2. Actualizar hoja de ruta con progreso reciente") 
    print("3. Sincronizar estado de proyecto con código actual")
    print("4. Verificar arquitectura neural matches implementación")
    print("5. Actualizar scripts documentation")
    print("6. Añadir entrada nueva en bitácora")
    print("7. Reportar qué cambios se hicieron")
    
    return todos_documentos

if __name__ == "__main__":
    documentos = actualizar_documentos()
    
    print(f"\n🚀 READY: {len(documentos)} documentos identificados para actualización")
    print("👤 ACCIÓN HUMANA REQUERIDA: Solicitar a Claude actualización completa")
    print('   Comando: "actualizar documentos" o "actualizar documentación"')