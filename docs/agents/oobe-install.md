# Agent OOBE Runbook

This runbook is the machine-facing contract for a Codex- or Claude Code-managed
Praxist installation. It keeps the agent-managed lane separate from the local
terminal wizard while reusing the same setup profiles, configuration files,
and validation.

## Boundary

Use this runbook when the operator asks Codex or Claude Code to install and
configure Praxist from PyPI, or explicitly points the agent at a source checkout
for development.
Do not use it for an ordinary runtime repair after OOBE.
An installation request does not authorize project selection or research
launch. Stop after setup and readiness; takeover is a separate action after the
operator reads [Your First Task](../getting-started/first-task.md).

Set one conversation-local skill host, `codex` or `claude`, from the interface
running this workflow. [Agent Skills](../user-guide/skills.md#install-or-refresh)
owns the installation locations and refresh commands.

The skill host is only the human interaction surface. It does not choose or
change the Praxist peer runtime selected by the setup profile.

Keep these choices in the current conversation until setup completes:

- selected Praxist executable;
- selected setup profile ID.

Do not create an OOBE state file. Legal-terms acceptance, installed configuration
(including the explicitly selected profile ID), product-usage consent,
`praxist doctor`, and task artifacts are the durable sources from which
interrupted setup is resumed.

## Interaction Contract

1. Prefer the current agent's native structured choice UI. If it is unavailable, ask the
   operator to run the matching local TTY selector. Do not ask for typed
   `yes`/`no` answers.
2. Use Up/Down and Enter for local choices. Treat Esc as back/cancel without
   undoing a completed package installation.
3. Never ask the operator to paste an API key into chat. API providers must use
   the local masked input opened by `praxist setup`; it displays one `*` per
   character and keeps the key out of chat, argv, logs, and shell history.
4. Ask for a path only when bounded discovery cannot identify the intended
   project. Never scan the whole home directory, a storage mount, or a dataset
   tree.
5. Do not use `--force-unmanaged` automatically. If bundled skill names collide
   with operator-owned paths, show the complete conflict list and offer: keep
   the existing skills, back them up and replace them, or cancel. Continue only
   with the operator's selected action.
6. Never accept the Praxist Fair Source License or User Agreement on the
   operator's behalf or infer acceptance from an installation request. Keep
   legal acceptance separate
   from optional product-usage consent.
7. A usable API provider default, saved login, exported key, or successful doctor
   result is not a profile choice. Only the operator may choose the profile.

## Workflow

1. **Preflight and install.** Verify Python 3.11+ and respect the operator's
   active or explicitly selected Python environment. Do not replace a task
   environment or modify system Python. Install the package and maintained
   runtime integrations without selecting an API provider on the operator's
   behalf:

   ```bash
   python3 -m pip install --index-url https://pypi.org/simple "praxist[agents,codex]"
   ```

   If no suitable writable environment is active, create a dedicated virtual
   environment first and keep using its Python and `praxist` entrypoint for the
   rest of OOBE. Do not silently install into an externally managed interpreter.
   A successful pip command proves only that the distribution is installed.
   Legal terms, privacy, runtime, skills, writable examples, and readiness
   remain pending. Continue with the steps below; do not report
   OOBE completion at this boundary. Never launch a read-only
   source/package-resource example copy.
   Immediately run `praxist setup --agent-managed`. This read-only command is
   the machine-owned decision checkpoint. Follow its `next_required_action`
   and rerun it after each completed decision.

2. **Legal terms.** Run `praxist user-agreement status --json`. If the current
   legal bundle is not accepted, present three native choices with no
   acceptance preselected: review the complete terms, agree and continue, or
   cancel setup. For review, surface the canonical root `LICENSE.md`, the two
   packaged Markdown files under `docs/legal/`, and the `license_url` and
   `review_url` returned by the status command as scrollable links. Use
   `praxist user-agreement review --print` only when neither interface is
   available, because printing the full text into chat is the least usable
   fallback. Repeat the choices after review. Explain that the Fair Source
   License includes eligibility, revenue-threshold, attribution, distribution,
   and use restrictions. Only after the operator explicitly agrees to the
   complete bundle may the Agent run:

   ```bash
   praxist user-agreement accept --agent-reply Agree
   ```

   Verify that status now reports `accepted: true`. A cancellation ends OOBE
   without changing runtime configuration.
3. **Privacy.** Run `praxist product-usage status --json`. Legal acceptance is
   not optional telemetry consent. If collection is available and consent is
   unset, offer review, share, and skip choices with neither sharing nor
   skipping preselected. Review uses the packaged
   `docs/legal/product-usage-data-notice.md` or its hosted page rather than
   dumping the notice into chat. Record only the explicit selection with
   `praxist product-usage consent --agent-reply <Yes-or-No>`. If no choice is
   made, leave consent unset. When `collection_available` is false, explicitly
   tell the operator that this build collects no product-usage data and no
   privacy authorization is required.
4. **Runtime.** Read the machine-owned options from
   `praxist setup --agent-managed` (or `--list-profiles`). Present those
   complete setup profiles as selectable API provider, agent runtime, concrete
   model, and authentication combinations. Apply a profile only after the
   operator chooses it. Before applying it, state the
   profile's `authorization_detail`: Codex-native uses the saved ChatGPT/Codex
   login without an API provider key or new authorization code; API-backed
   setup profiles require the matching key through local masked input. Then run
   `praxist setup --profile <id> --install-skills <agent-host>`, where
   `<agent-host>` is `codex` or `claude` from the current interface. Run the
   command in a local interactive terminal when the profile needs either an API
   key or a Codex-native ChatGPT login. The latter uses and verifies the
   SDK-pinned Codex binary; no other profile may require that login. If secure
   local interaction is unavailable, print the exact setup command for the
   operator and wait; never route credentials through the agent conversation.
   `praxist setup --interactive` handles any same-name skill conflict locally
   with keep, backup-and-replace, and cancel choices. Non-interactive setup
   refuses to overwrite an operator-owned path. Rerun
   `praxist setup --agent-managed` and require `profile.selected: true`; a
   configured API provider default without that confirmation remains incomplete.
5. **Readiness and stop.** Require `setup_decisions_complete: true`, run the
   matching host diagnostics, and report the selected profile, installed skill
   host, writable example locations, and any concrete blocker. The final
   `next_required_action` is `run_doctor_then_finish_setup`. Do not discover or
   select a research project, invoke a takeover skill, or launch a run. Link the operator to
   [Your First Task](../getting-started/first-task.md) for the later, separate
   takeover workflow.

Finish when installation, explicit setup decisions, skill registration, and
readiness checks are complete, or when one concrete blocker remains.
