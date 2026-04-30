#!/bin/sh
set -e

apk add --no-cache curl > /dev/null 2>&1

G="/garage-bin/garage"
S="${GARAGE_RPC_SECRET}"
ADMIN="http://127.0.0.1:3903"
TOKEN="${GARAGE_ADMIN_TOKEN}"

echo "==> Checking binary..."
$G --version

echo "==> Waiting for admin API..."
until curl -sf -H "Authorization: Bearer $TOKEN" "$ADMIN/v2/GetClusterStatus" > /dev/null; do
  sleep 2
done

echo "==> Getting full node ID..."
FULL_ID=$(curl -s -H "Authorization: Bearer $TOKEN" "$ADMIN/v2/GetClusterStatus" \
  | grep '"id"' | head -1 | grep -o '"[0-9a-f]*"' | tr -d '"')
echo "    Full ID: $FULL_ID"

RPC_HOST="${FULL_ID}@127.0.0.1:3901"
echo "    RPC host: $RPC_HOST"

echo "==> Checking layout..."
LAYOUT_VERSION=$(curl -s -H "Authorization: Bearer $TOKEN" "$ADMIN/v2/GetClusterLayout" \
  | grep -o '"version":[0-9]*' | grep -o '[0-9]*$')
echo "    Current layout version: ${LAYOUT_VERSION:-0}"

if [ "${LAYOUT_VERSION:-0}" -eq 0 ]; then
  echo "==> Assigning layout..."
  $G -h "$RPC_HOST" -s "$S" layout assign -z dc1 -c 10G "$FULL_ID"

  echo "==> Applying layout..."
  $G -h "$RPC_HOST" -s "$S" layout apply --version 1
else
  echo "==> Layout already applied (version $LAYOUT_VERSION), skipping."
fi

echo "==> Importing key..."
$G -h "$RPC_HOST" -s "$S" key import --yes \
  -n mlflow-key "$GARAGE_ACCESS_KEY" "$GARAGE_SECRET_KEY"

echo "==> Creating bucket..."
$G -h "$RPC_HOST" -s "$S" bucket create "$GARAGE_BUCKET" || true

echo "==> Allowing key on bucket..."
$G -h "$RPC_HOST" -s "$S" bucket allow \
  --read --write --owner "$GARAGE_BUCKET" \
  --key "$GARAGE_ACCESS_KEY"

echo "==> Done!"
$G -h "$RPC_HOST" -s "$S" bucket list