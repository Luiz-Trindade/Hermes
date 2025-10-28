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

echo "==> Publicando no PyPI usando token"
uv config set pypi-token.pypi "$PYPI_TOKEN"
uv publish --no-cache --username __token__ --password "$PYPI_TOKEN" -y

echo "==> Commit e push automáticos"
git add .
git commit -m "chore: release $(date +'%Y-%m-%d %H:%M:%S')"
git push

echo "==> Publicação concluída com sucesso."
