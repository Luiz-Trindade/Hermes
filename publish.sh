#!/bin/bash
set -e

# Carrega variáveis do arquivo .env
export $(grep -v '^#' .env | xargs)

if [ -z "$PYPI_TOKEN" ]; then
    echo "Erro: variável PYPI_TOKEN não encontrada no .env"
    exit 1
fi

echo "==> Limpando diretórios antigos"
rm -rf dist build *.egg-info

echo "==> Gerando nova build"
uv build
uv publish
