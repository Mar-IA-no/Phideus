#!/bin/bash
# Script de activación del entorno virtual para Temporal VAE

echo "🚀 Activating Temporal VAE Environment..."
cd /root/Phideus/src/vae/temporal
source temporal_vae_env/bin/activate

echo "✅ Environment activated!"
echo "📍 Working directory: $(pwd)"
echo "🐍 Python: $(which python)"
echo ""
echo "Available commands:"
echo "  python setup_temporal_vae.py       # Setup inicial"  
echo "  python train_temporal_vae.py       # Entrenar modelo"
echo "  python run_temporal_analysis.py    # Análisis temporal"
echo "  python test_temporal_vae.py        # Testing completo"
echo ""