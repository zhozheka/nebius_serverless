"""Optional: run a ``RepeatableJob`` loop in a CPU job on Nebius.

Use it when nothing else stays up for the whole run. When an orchestrator
such as Prefect already runs your code, await ``run_async()`` there instead
and skip this module.

``launch(repeatable, account)`` starts the supervisor: an on-demand CPU job
that installs this SDK with pip (``sdk``), rebuilds the job from JSON,
and runs ``RepeatableJob.run_async()``. Its logs show every attempt.
Cancelling it cancels the current attempt.

A job's container gets no Nebius credentials of its own, so the supervisor
signs in as a service account whose private key is read from MysteryBox.

This file is also the supervisor's entry point:
``python supervisor.py spec.json``.
"""

import asyncio
import json
import logging
import os
import shlex
import signal
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from repeatable import RepeatableJob

import nebius_serverless as ns

HERE = Path(__file__).resolve().parent
DIR = "/sv"  # where the supervisor's files are uploaded
# pip requirement for this SDK. Injected files are limited to 64 KiB
# after encoding, so a wheel does not fit; install from GitHub instead.
SDK = (
    "https://github.com/zhozheka/nebius_serverless/archive/"
    "refs/heads/feat/repeatable_job.tar.gz"
)

log = logging.getLogger("repeatable")


@dataclass(frozen=True)
class ServiceAccount:
    """Credentials the supervisor signs in with.

    The account needs permission to create, get, and cancel jobs in the
    project, for example membership in the ``editors`` group.
    """

    id: str  # serviceaccount-...
    public_key_id: str  # publickey-... of an authorized key
    private_key_secret: str  # mbsec-... holding that key's PEM


async def launch(
    repeatable: RepeatableJob,
    account: ServiceAccount,
    *,
    timeout: str = "24h",
    name: str | None = None,
    sdk: str = SDK,
) -> ns.Run:
    """Start the supervisor and return its ``Run``.

    ``timeout`` bounds the whole loop, all attempts included. ``sdk`` is
    the pip requirement the supervisor installs this SDK from.
    """
    with tempfile.TemporaryDirectory() as tmp:
        supervisor = supervisor_job(
            repeatable,
            account,
            Path(tmp),
            timeout=timeout,
            name=name,
            sdk=sdk,
        )
        run = await supervisor.create_async(quiet=True)  # reads the files
    repeatable.log.info("supervisor %s started", run.id)
    return run


def supervisor_job(
    repeatable: RepeatableJob,
    account: ServiceAccount,
    tmp: Path,
    *,
    timeout: str,
    name: str | None,
    sdk: str = SDK,
) -> ns.Job:
    """Build the CPU job. Generated files are written to ``tmp``."""
    uploads = [
        f"{HERE / 'repeatable.py'}:{DIR}/repeatable.py",
        f"{HERE / 'supervisor.py'}:{DIR}/supervisor.py",
    ]
    recipe = job_kwargs(repeatable.job)
    # The job's own files move into the supervisor, which uploads them
    # again with every attempt.
    moved = []
    for n, item in enumerate(recipe["inject_files"]):
        local, container = item.split(":", 1)
        copy = f"{DIR}/files/{n}-{Path(local).name}"
        uploads.append(f"{local}:{copy}")
        moved.append(f"{copy}:{container}")
    recipe["inject_files"] = moved
    spec = tmp / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "job": recipe,
                "name": repeatable.name,
                "max_attempts": repeatable.max_attempts,
                "reschedule": list(repeatable.reschedule),
                "poll": repeatable.poll,
                "drain_timeout": repeatable.drain_timeout,
            }
        )
    )
    uploads.append(f"{spec}:{DIR}/spec.json")
    command = (
        f"pip install -q --root-user-action=ignore {shlex.quote(sdk)}"
        f" && exec python -u {DIR}/supervisor.py {DIR}/spec.json"
    )
    return ns.Job(
        ["sh", "-c", command],
        image="python:3.12-slim",
        timeout=timeout,
        inject_files=uploads,
        env={
            "NEBIUS_SA_ID": account.id,
            "NEBIUS_SA_PUBLIC_KEY_ID": account.public_key_id,
            "NEBIUS_PROJECT_ID": recipe["project"],
        },
        secrets={"NEBIUS_SA_PRIVATE_KEY": account.private_key_secret},
        name=name or f"{repeatable.name}-supervisor",
        profile=repeatable.job.profile,
        project=recipe["project"],
    )


def job_kwargs(job: ns.Job) -> dict[str, object]:
    """Return the arguments that build ``job`` again, as JSON values."""
    from nebius_serverless.profile import load_profile

    return {
        "command": list(job.command),
        "image": job.image,
        "gpu": job.gpu,
        "platform": job.platform,
        "preset": job.preset,
        "timeout": job.timeout_seconds,
        "pricing": job.pricing,
        "subnet_id": job.subnet_id,
        "volumes": list(job.volumes),
        "inject_files": list(job.inject_files),
        "env": dict(job.env),
        "name": job.name,
        "disk": job.disk,
        "project": job.project or load_profile(job.profile)[0],
        "retries": job.retries,
        "working_dir": job.working_dir,
        "secrets": dict(job.secrets),
        "registry_secret": job.registry_secret,
        "spot_policy": job.spot_policy,
    }


def write_profile() -> None:
    """Write a CLI profile that signs in as the service account."""
    config = Path.home() / ".nebius"
    config.mkdir(exist_ok=True)
    key = config / "sa.pem"
    key.write_text(os.environ["NEBIUS_SA_PRIVATE_KEY"])
    key.chmod(0o600)
    profile = {
        "endpoint": "api.nebius.cloud",
        "auth-type": "service account",
        "service-account-id": os.environ["NEBIUS_SA_ID"],
        "public-key-id": os.environ["NEBIUS_SA_PUBLIC_KEY_ID"],
        "private-key-file-path": str(key),
        "parent-id": os.environ["NEBIUS_PROJECT_ID"],
    }
    body = {"default": "supervisor", "profiles": {"supervisor": profile}}
    (config / "config.yaml").write_text(json.dumps(body))  # JSON is YAML


async def supervise(spec_path: str) -> int:
    """Run the loop from ``spec.json``; return the process exit code."""
    write_profile()
    spec = json.loads(Path(spec_path).read_text())
    repeatable = RepeatableJob(
        ns.Job(**spec["job"]),
        name=spec["name"],
        max_attempts=spec["max_attempts"],
        reschedule=tuple(spec["reschedule"]),
        poll=spec["poll"],
        drain_timeout=spec["drain_timeout"],
    )
    stopped = False

    def on_sigterm() -> None:
        """Cancel the current attempt; the loop then stops."""
        nonlocal stopped
        stopped = True
        log.warning("supervisor got SIGTERM")
        asyncio.ensure_future(repeatable.cancel_async())

    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, on_sigterm)
    try:
        run = await repeatable.run_async()
    except RuntimeError:
        return 1
    finally:
        log.info("attempts:\n%s", repeatable.summary())
    if stopped:
        return 143
    return 0 if run.status == "COMPLETED" else 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    sys.exit(asyncio.run(supervise(sys.argv[1])))
