#!/usr/bin/env bash
set -euo pipefail

COMPOSE="docker compose -f docker-compose-prod.yml"
WORKFLOW=/workflows/canario.json
ENV_FILE=.env

KEY=$(grep '^CANARIO_SCRIBE_API_KEY=' "$ENV_FILE" | cut -d= -f2-)

if [ -z "$KEY" ]; then
    echo "Falta CANARIO_SCRIBE_API_KEY en $ENV_FILE (créala con app.manage_keys, dueño canario-local)." >&2
    exit 1
fi

CREDENTIALS=$(cat <<JSON
[
  {
    "id": "canario-api-key",
    "name": "API key canario-local",
    "type": "httpHeaderAuth",
    "data": {"name": "X-API-Key", "value": "$KEY"}
  },
  {
    "id": "canario-postgres",
    "name": "Postgres scribe",
    "type": "postgres",
    "data": {"host": "postgres", "port": 5432, "database": "scribe", "user": "scribe", "password": "scribe", "ssl": "disable"}
  }
]
JSON
)

printf '%s' "$CREDENTIALS" | $COMPOSE exec -T n8n sh -c 'umask 077; cat > /tmp/credenciales.json'
$COMPOSE exec -T n8n sh -c 'n8n import:credentials --input=/tmp/credenciales.json; rm -f /tmp/credenciales.json'
$COMPOSE exec -T n8n n8n import:workflow --input="$WORKFLOW"
$COMPOSE exec -T n8n n8n publish:workflow --id=canario-diario
$COMPOSE restart n8n
