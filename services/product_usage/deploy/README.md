# Product-usage collector deployment

The topology uses Nginx in front of one stateless Collector container and a
separate retention-worker container using the same image.
The container binds only to `127.0.0.1:8110` and reaches managed PostgreSQL over
its private endpoint. Development hostnames, database addresses, credentials,
SSH material, and cloud resource IDs stay outside this repository.

`deploy.sh` accepts only an image pinned by `sha256` digest. It temporarily
quiesces public ingestion, runs the rollback-compatible V2 migration while
preserving the V1 table as an archive, waits for Collector database-backed
health and a successful retention pass, then commits the Nginx route. It backs
up and restores the prior Compose and Nginx assets together with the previous
image if a deployment step fails.

The final Nginx reload is the deployment commit point. Every fallible schema,
service, health, proxy-validation, and image-record step completes while public
ingress still returns `503`; no schema-dropping rollback is possible after the
new route becomes public.

Before that commit point, rollback keeps the maintenance response live until
the previous Collector and retention worker have restarted and both container
health checks pass. A failed first deployment has no prior service to expose,
so it deliberately leaves the maintenance response active.

The server must already contain `/opt/praxist-collector/.env`, owned by the
deploy account with mode `0600`:

```dotenv
DATABASE_URL=postgresql+psycopg://collector:secret@private-host/database
COLLECTOR_INGESTION_ENABLED=true
COLLECTOR_MAX_TABLE_BYTES=2147483648
```

The GitHub `development` environment holds only the public host and SSH target,
bastion, host-key, and deploy-key material. Database credentials remain on the
server. Development deployment is a separate, manually approved workflow that
accepts an immutable image digest.

The CI deploy key does not provide a general root shell. Its bastion entry is
restricted to forwarding only to the Collector SSH endpoint, while its target
entry is forced through `ssh-dispatch.sh`. The dispatcher accepts only GHCR
authentication on standard input and deployment of this repository's image by
an immutable digest. It extracts the Compose, migration, proxy, and rollback
assets from that exact image before deployment, so a stale host checkout cannot
control a newer service image.

The retention worker runs immediately at startup and at least every 24 hours.
Every pass must succeed. A failed pass exits the worker so the container restart
policy retries promptly; successful passes refresh a bounded-age health marker.
It places raw events into the deletion window after 179 days, leaving one day
of scheduling safety margin so normally running deployments do not exceed the
declared 180-day maximum. Intervals longer than 24 hours are rejected. There is
no delete-by-Environment-ID endpoint; withdrawing consent stops future capture
and removes only unsent local events.

Nginx applies both per-client and deployment-wide request-rate limits. The
database adapter also refuses new events when `COLLECTOR_MAX_TABLE_BYTES` is
reached, so an exposed development endpoint cannot grow storage without bound.

This workflow deploys only the disclosed development endpoint; it does not
deploy a production collector. The operator guide at
`docs/operations/product-usage.md` owns the repository-level workflow.
