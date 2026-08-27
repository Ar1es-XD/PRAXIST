#!/usr/bin/env bash
set -Eeuo pipefail

: "${COLLECTOR_HOST:?COLLECTOR_HOST is required}"
: "${COLLECTOR_IMAGE:?COLLECTOR_IMAGE is required}"

if [[ ! "$COLLECTOR_HOST" =~ ^[A-Za-z0-9.-]+$ ]]; then
  echo "COLLECTOR_HOST contains unsupported characters" >&2
  exit 1
fi
if [[ ! "$COLLECTOR_IMAGE" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "COLLECTOR_IMAGE must use an immutable sha256 digest" >&2
  exit 1
fi

app_dir=/opt/praxist-collector
asset_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
compose_file="$app_dir/compose.yml"
current_image_file="$app_dir/.current-image"
nginx_rate_file=/etc/nginx/conf.d/praxist-collector-rate.conf
nginx_site_file=/etc/nginx/sites-available/praxist-collector
nginx_enabled_file=/etc/nginx/sites-enabled/praxist-collector
previous_image=""
previous_revision=""
migration_started=false
maintenance_active=false
backup_dir=""
had_compose=false
had_nginx_rate=false
had_nginx_site=false
had_nginx_enabled=false
had_current_image=false
compose_cmd=()
if docker compose version >/dev/null 2>&1; then
  compose_cmd=(docker compose)
elif command -v docker-compose >/dev/null 2>&1 \
  && docker-compose version >/dev/null 2>&1; then
  compose_cmd=(docker-compose)
else
  echo "Docker Compose is unavailable" >&2
  exit 1
fi
if [[ -f "$current_image_file" ]]; then
  previous_image="$(<"$current_image_file")"
fi

restore_runtime_assets() {
  if [[ -z "$backup_dir" ]]; then
    return
  fi
  if [[ "$had_compose" == true ]]; then
    install -m 0644 "$backup_dir/compose.yml" "$compose_file"
  else
    rm -f "$compose_file"
  fi
  if [[ "$had_current_image" == true ]]; then
    install -m 0644 "$backup_dir/current-image" "$current_image_file"
  else
    rm -f "$current_image_file"
  fi
}

restore_nginx_assets() {
  if [[ "$had_nginx_rate" == true ]]; then
    install -m 0644 "$backup_dir/nginx-rate.conf" "$nginx_rate_file"
  else
    rm -f "$nginx_rate_file"
  fi
  if [[ "$had_nginx_site" == true ]]; then
    install -m 0644 "$backup_dir/nginx-site.conf" "$nginx_site_file"
  else
    rm -f "$nginx_site_file"
  fi
  rm -f "$nginx_enabled_file"
  if [[ "$had_nginx_enabled" == true ]]; then
    cp -a "$backup_dir/nginx-enabled" "$nginx_enabled_file"
  fi
}

restore_assets() {
  restore_runtime_assets
  restore_nginx_assets
}

activate_maintenance_proxy() {
  printf '%s\n' \
    'server {' \
    '    listen 80;' \
    "    server_name $COLLECTOR_HOST;" \
    '    location / { return 503; }' \
    '}' > "$nginx_site_file"
  ln -sfn "$nginx_site_file" "$nginx_enabled_file"
  nginx -t
  nginx -s reload
}

service_healthy() {
  service_name="$1"
  container_id="$("${compose_cmd[@]}" -f "$compose_file" ps -q "$service_name")"
  [[ -n "$container_id" ]] \
    && [[ "$(docker inspect --format '{{.State.Health.Status}}' "$container_id")" == healthy ]]
}

rollback() {
  status="${1:-$?}"
  trap - ERR
  rm -f "$current_image_file.tmp" || true
  if [[ "$maintenance_active" != true ]]; then
    restore_assets
    [[ -z "$backup_dir" ]] || rm -rf "$backup_dir"
    exit "$status"
  fi
  if [[ -f "$compose_file" ]]; then
    "${compose_cmd[@]}" -f "$compose_file" stop collector retention || true
  fi
  if [[ "$migration_started" == true ]]; then
    rollback_revision="${previous_revision:-base}"
    if ! "${compose_cmd[@]}" -f "$compose_file" run --rm --no-deps collector \
      alembic -c services/product_usage/alembic.ini downgrade "$rollback_revision"; then
      echo "Database rollback failed; refusing to restore public ingress." >&2
      exit "$status"
    fi
  fi
  restore_runtime_assets
  if [[ -n "$previous_image" ]]; then
    echo "Deployment failed; restoring previous Collector image." >&2
    if ! COLLECTOR_IMAGE="$previous_image" "${compose_cmd[@]}" -f "$compose_file" \
      up -d collector retention; then
      echo "Previous Collector failed to start; public ingress remains unavailable." >&2
      exit "$status"
    fi
    for _ in {1..30}; do
      if curl -fsS --max-time 3 http://127.0.0.1:8110/healthz >/dev/null \
        && service_healthy collector \
        && service_healthy retention; then
        break
      fi
      sleep 2
    done
    if ! curl -fsS --max-time 3 http://127.0.0.1:8110/healthz >/dev/null \
      || ! service_healthy collector \
      || ! service_healthy retention; then
      echo "Previous services are unhealthy; public ingress remains unavailable." >&2
      exit "$status"
    fi
    restore_nginx_assets
    if ! nginx -t || ! nginx -s reload; then
      echo "Previous proxy configuration failed; maintenance response remains active." >&2
      exit "$status"
    fi
  elif ! activate_maintenance_proxy; then
    echo "Maintenance proxy could not be persisted after first-deployment failure." >&2
    exit "$status"
  fi
  [[ -z "$backup_dir" ]] || rm -rf "$backup_dir"
  exit "$status"
}
trap rollback ERR

install -d -m 0755 "$app_dir"
test -s "$app_dir/.env" || {
  echo "Missing $app_dir/.env" >&2
  exit 1
}
env_owner="$(stat -c '%u' "$app_dir/.env")"
env_mode="$(stat -c '%a' "$app_dir/.env")"
if [[ "$env_owner" != "$(id -u)" ]] || (( (8#$env_mode & 077) != 0 )); then
  echo "$app_dir/.env must be owned by the deploy account with no group/other permissions" >&2
  exit 1
fi
grep -Eq '^COLLECTOR_INGESTION_ENABLED=(true|false)$' "$app_dir/.env" || {
  echo "COLLECTOR_INGESTION_ENABLED=true|false must be explicit in $app_dir/.env" >&2
  exit 1
}
backup_dir="$(mktemp -d "$app_dir/.deploy-backup.XXXXXX")"
if [[ -f "$compose_file" ]]; then
  cp -a "$compose_file" "$backup_dir/compose.yml"
  had_compose=true
fi
if [[ -f "$nginx_rate_file" ]]; then
  cp -a "$nginx_rate_file" "$backup_dir/nginx-rate.conf"
  had_nginx_rate=true
fi
if [[ -f "$nginx_site_file" ]]; then
  cp -a "$nginx_site_file" "$backup_dir/nginx-site.conf"
  had_nginx_site=true
fi
if [[ -e "$nginx_enabled_file" || -L "$nginx_enabled_file" ]]; then
  cp -a "$nginx_enabled_file" "$backup_dir/nginx-enabled"
  had_nginx_enabled=true
fi
if [[ -f "$current_image_file" ]]; then
  cp -a "$current_image_file" "$backup_dir/current-image"
  had_current_image=true
fi
install -m 0644 "$asset_dir/compose.yml" "$compose_file"

export COLLECTOR_IMAGE
"${compose_cmd[@]}" -f "$compose_file" pull collector retention
previous_revision="$(
  "${compose_cmd[@]}" -f "$compose_file" run --rm --no-deps collector \
    alembic -c services/product_usage/alembic.ini current \
    | awk 'NF {print $1; exit}'
)"
if [[ -n "$previous_revision" && ! "$previous_revision" =~ ^[0-9a-f]+$ ]]; then
  echo "Unable to determine the current database revision" >&2
  rollback 1
fi
if [[ -n "$previous_image" && -z "$previous_revision" ]]; then
  echo "A previous image exists but its database revision is unknown" >&2
  rollback 1
fi

# Keep public clients on a deterministic 503 boundary until schema migration,
# service health, and the final proxy configuration have all committed.
activate_maintenance_proxy
maintenance_active=true
"${compose_cmd[@]}" -f "$compose_file" stop collector retention || true

migration_started=true
"${compose_cmd[@]}" -f "$compose_file" run --rm --no-deps collector \
  alembic -c services/product_usage/alembic.ini upgrade head
"${compose_cmd[@]}" -f "$compose_file" up -d collector retention

for _ in {1..30}; do
  if curl -fsS --max-time 3 http://127.0.0.1:8110/healthz >/dev/null \
    && service_healthy collector \
    && service_healthy retention; then
    break
  fi
  sleep 2
done
curl -fsS --max-time 3 http://127.0.0.1:8110/healthz >/dev/null
service_healthy collector
service_healthy retention

install -m 0644 "$asset_dir/nginx/collector-rate.conf" "$nginx_rate_file"
sed "s/__COLLECTOR_HOST__/$COLLECTOR_HOST/g" "$asset_dir/nginx/collector.conf.template" \
  > "$nginx_site_file"
ln -sfn "$nginx_site_file" "$nginx_enabled_file"
nginx -t
printf '%s\n' "$COLLECTOR_IMAGE" > "$current_image_file.tmp"
mv -f "$current_image_file.tmp" "$current_image_file"
nginx -s reload
trap - ERR
rm -rf "$backup_dir" || true
echo "Collector and retention deployment succeeded: $COLLECTOR_IMAGE"
