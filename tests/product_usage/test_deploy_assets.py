from __future__ import annotations

import stat
from pathlib import Path


def test_deploy_starts_and_rolls_back_collector_and_retention_together() -> None:
    root = Path(__file__).resolve().parents[2]
    script_path = root / "services" / "product_usage" / "deploy" / "deploy.sh"
    script = script_path.read_text(encoding="utf-8")

    assert "pull collector retention" in script
    assert "up -d collector retention" in script
    assert "stop collector retention" in script
    assert "State.Health.Status" in script
    assert "COLLECTOR_INGESTION_ENABLED=(true|false)" in script
    assert "stat -c '%u'" in script
    assert "stat -c '%a'" in script
    assert "8#$env_mode & 077" in script
    assert "return 503" in script
    assert "restore_assets" in script
    assert "maintenance_active=true" in script
    assert "Previous services are unhealthy" in script
    assert "service_healthy collector" in script
    assert "service_healthy retention" in script
    assert 'rollback_revision="${previous_revision:-base}"' in script
    assert "alembic -c services/product_usage/alembic.ini current" in script
    assert 'downgrade "$rollback_revision"' in script
    assert "refusing to restore public ingress" in script
    stop_position = script.index("stop collector retention")
    downgrade_position = script.index('downgrade "$rollback_revision"')
    restore_position = script.index('COLLECTOR_IMAGE="$previous_image"')
    assert stop_position < downgrade_position < restore_position
    final_reload = script.rindex("nginx -s reload")
    commit_position = script.rindex("trap - ERR")
    assert final_reload < commit_position
    assert "curl " not in script[final_reload:]
    maintenance_position = script.index("maintenance_active=true")
    active_stop_position = script.index(
        '"${compose_cmd[@]}" -f "$compose_file" stop collector retention || true',
        maintenance_position,
    )
    assert maintenance_position < active_stop_position
    assert script_path.stat().st_mode & stat.S_IXUSR
    dispatch_path = root / "services/product_usage/deploy/ssh-dispatch.sh"
    assert dispatch_path.stat().st_mode & stat.S_IXUSR
    dispatch = dispatch_path.read_text(encoding="utf-8")
    assert 'docker pull "$image"' in dispatch
    assert 'docker create "$image"' in dispatch
    assert "container_id:/app/services/product_usage/deploy/." in dispatch
    assert '"$asset_dir/deploy.sh"' in dispatch
    assert "/usr/local/lib/praxist-collector/deploy.sh" not in dispatch

    compose = (root / "services/product_usage/deploy/compose.yml").read_text(encoding="utf-8")
    assert "--health-file" in compose
    assert "praxist-retention-health" in compose


def test_current_http_deployment_workflow_is_development_only() -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github" / "workflows" / "deploy-product-usage.yml").read_text(
        encoding="utf-8"
    )

    assert "environment: development" in workflow
    assert "environment: production" not in workflow
    assert "product-usage-collector-development" in workflow


def test_development_proxy_has_per_client_and_global_admission_limits() -> None:
    root = Path(__file__).resolve().parents[2]
    rate_config = (root / "services/product_usage/deploy/nginx/collector-rate.conf").read_text(
        encoding="utf-8"
    )
    virtual_host = (root / "services/product_usage/deploy/nginx/collector.conf.template").read_text(
        encoding="utf-8"
    )

    assert "zone=praxist_collector:" in rate_config
    assert "zone=praxist_collector_global:" in rate_config
    assert "limit_req zone=praxist_collector " in virtual_host
    assert "limit_req zone=praxist_collector_global " in virtual_host
    assert 'proxy_set_header User-Agent ""' in virtual_host


def test_collector_build_context_retains_packaged_plugin_fixtures() -> None:
    root = Path(__file__).resolve().parents[2]
    dockerignore = (root / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert "tests/**" in dockerignore
    assert "!tests/fixtures/" in dockerignore
    assert "!tests/fixtures/plugins/" in dockerignore
    assert "!tests/fixtures/plugins/**" in dockerignore


def test_v2_migration_preserves_a_rollback_compatible_v1_archive() -> None:
    root = Path(__file__).resolve().parents[2]
    migration = (
        root / "praxist/product_usage/migrations/versions/0002_create_v2_environment_events.py"
    ).read_text(encoding="utf-8")

    assert 'op.rename_table("raw_events", "raw_events_v1_archive")' in migration
    assert 'op.rename_table("raw_events_v1_archive", "raw_events")' in migration
    assert 'op.drop_table("raw_events_v1_archive")' not in migration
