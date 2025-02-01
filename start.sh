#!/bin/bash


echo "🔍 Instalando dependencias de Python..."
pip install -r requirements.txt
echo "✅ Dependencias de Python instaladas correctamente"

echo "🔍 Cargando configuración..."
if [ -f config.env ]; then
    export $(cat config.env | xargs)
    echo "API_KEY cargada desde config.env"
else
    echo "⚠️ No se encontró el archivo config.env"
fi

exec gunicorn -w 4 -b 0.0.0.0:$PORT main:app