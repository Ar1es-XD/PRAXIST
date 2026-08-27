#!/usr/bin/env bash
set -Eeuo pipefail

read -r action first second extra <<< "${SSH_ORIGINAL_COMMAND:-}"

case "$action" in
  ghcr-login)
    if [[ -n "${second:-}" || -n "${extra:-}" || ! "${first:-}" =~ ^[A-Za-z0-9-]+$ ]]; then
      exit 64
    fi
    exec docker login ghcr.io -u "$first" --password-stdin
    ;;
  deploy)
    if [[ -n "${extra:-}" ]]; then
      exit 64
    fi
    if [[ ! "${first:-}" =~ ^[A-Za-z0-9.-]+$ ]]; then
      exit 64
    fi
    if [[ ! "${second:-}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
      exit 64
    fi
    image="ghcr.io/sapientinc/praxist-collector@$second"
    asset_dir="$(mktemp -d "${TMPDIR:-/tmp}/praxist-collector-assets.XXXXXX")"
    container_id=""
    cleanup() {
      if [[ -n "$container_id" ]]; then
        docker rm -f "$container_id" >/dev/null 2>&1 || true
      fi
      rm -rf "$asset_dir"
    }
    trap cleanup EXIT

    # Compose, proxy, migration, and rollback behavior must come from the same
    # immutable image as the service being deployed, never a stale host copy.
    docker pull "$image" >/dev/null
    container_id="$(docker create "$image")"
    docker cp \
      "$container_id:/app/services/product_usage/deploy/." \
      "$asset_dir"
    docker rm "$container_id" >/dev/null
    container_id=""
    chmod 0700 "$asset_dir/deploy.sh"
    env \
      COLLECTOR_HOST="$first" \
      COLLECTOR_IMAGE="$image" \
      "$asset_dir/deploy.sh"
    ;;
  *)
    echo "unsupported deploy command" >&2
    exit 126
    ;;
esac
