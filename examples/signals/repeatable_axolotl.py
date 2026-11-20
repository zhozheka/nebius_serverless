"""Run the repeatable Axolotl fine-tune from the command line.

The script version of ``repeatable_axolotl.ipynb``: the cookbook's Axolotl
QLoRA fine-tune (``Qwen/Qwen2.5-0.5B``, spot H100) under ``RepeatableJob``.
When the spot VM is taken away, a new attempt continues from the last
checkpoint in the bucket. See the notebook for how each piece works.

Two places to run the loop:

``supervisor``
    A CPU job on Nebius runs the loop (``supervisor.py``). It keeps
    relaunching after this script exits. Ctrl-C stops watching; the
    supervisor keeps going. Cancel it with ``nebius ai job cancel <id>``.
``local``
    This process runs the loop. Ctrl-C cancels the current attempt and
    stops.

Examples::

    uv run python examples/signals/repeatable_axolotl.py supervisor
    uv run python examples/signals/repeatable_axolotl.py supervisor --detach
    uv run python examples/signals/repeatable_axolotl.py local
    # Print the create requests only
    uv run python examples/signals/repeatable_axolotl.py supervisor --dry-run
    # Pay at most $2.50 per GPU-hour; preempted when spot goes above it
    uv run python examples/signals/repeatable_axolotl.py supervisor \\
        --max-price 2.5

Spot pricing:

- By default an attempt pays the current spot price, whatever it is.
- ``--max-price USD`` caps it per GPU-hour. The cap is a Nebius pricing
  policy (``nebius billing pricing-policy``) for the job's platform; the
  script reuses one with the same platform and price or creates it. When
  the spot price rises above the cap, the VM is preempted (the attempt
  ends ``platform`` and is rescheduled); while the price stays above it,
  new attempts wait in ``PROVISIONING``.
- ``--spot-policy pricingpolicy-...`` uses an existing policy.
- ``--pricing on-demand`` turns spot off.

Every mode starts billable spot H100 attempts except ``--dry-run``.
``supervisor`` also runs an on-demand CPU job for as long as the loop lasts.
Uses the active Nebius profile.
"""

import argparse
import asyncio
import json
import logging
import re
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from repeatable import RepeatableJob, end_reason
from supervisor import ServiceAccount, supervisor_job

import nebius_serverless as ns

HERE = Path(__file__).resolve().parent
FILES = HERE / "axolotl"  # uploaded into each attempt
IMAGE = "docker.io/axolotlai/axolotl:main-20260309-py3.11-cu128-2.9.1"

# Bucket id, not name: nebius storage bucket get-by-name --name <bucket>
BUCKET_ID = "storagebucket-e008847641561172892054"  # ez-bucket-1

# MLflow server (nebius msp mlflow v1alpha1 cluster list) and its login
MLFLOW_URI = (
    "https://public-tracking-e00-th0rycpzb30p26v-ebn5j97mr4nqat7"
    "-mlflow.gw.msp.eu-north1.nebius.cloud"
)  # mlflow1
MLFLOW_USER = "admin"
# Local file with MLFLOW_TRACKING_PASSWORD=..., readable only by you
SECRETS_ENV = Path.home() / ".nebius" / "mlflow1.env"

# Service account the supervisor signs in as (see repeatable_axolotl.ipynb,
# section 6, for creating the key and the MysteryBox secret).
ACCOUNT = ServiceAccount(
    id="serviceaccount-e00pe5phdy0w0wxd23",  # perfect-pipeline
    public_key_id="publickey-e00rr9c847fm7w20dr",  # authorized key
    private_key_secret="mbsec-e00nv8q43vwjefc8mm",  # its private PEM
)

# `nebius ai job logs` returns the last 100 lines by default; 1000 is the max.
FULL_LOG = ("--since", "24h", "--tail", "1000")
# Lines from resumable_axolotl.py that show how each attempt went.
PATTERN = re.compile(r"starting fresh|resuming|SIGTERM|copied|training done")

log = logging.getLogger("repeatable")


def build_job(
    args: argparse.Namespace, spot_policy: str | None = None
) -> ns.Job:
    """The Axolotl job every attempt runs."""
    ckpt_dir = f"/ckpt/repeatable-axolotl/{args.run_name}"
    inject = [
        f"{FILES / name}:/app/{name}"
        for name in (
            "resumable_axolotl.py",
            "checkpoint_plugin.py",
            "config.yaml",
        )
    ]
    env = {"CKPT_DIR": ckpt_dir, "PYTHONPATH": "/app"}
    if not args.no_mlflow:
        if not SECRETS_ENV.exists():
            sys.exit(
                f"{SECRETS_ENV} not found. Create it with "
                "MLFLOW_TRACKING_PASSWORD=..., or pass --no-mlflow."
            )
        inject.append(f"{SECRETS_ENV}:/app/secrets.env")
        env["MLFLOW_TRACKING_URI"] = MLFLOW_URI
        env["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USER
    return ns.Job(
        ["python", "-u", "/app/resumable_axolotl.py"],
        image=IMAGE,
        gpu="h100",
        pricing=args.pricing,
        spot_policy=spot_policy,
        disk="500Gi",
        timeout=args.attempt_timeout,
        volumes=[f"{args.bucket_id}:/ckpt:rw"],
        inject_files=inject,
        env=env,
        name="repeatable-axolotl",
    )


def cli_json(*args: str) -> dict:
    """Run the nebius CLI with JSON output and parse it."""
    proc = subprocess.run(
        ["nebius", *args, "--format", "json"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"nebius {' '.join(args)} failed:\n{proc.stderr.strip()}")
    return json.loads(proc.stdout or "{}")


def field(obj: object, *path: str) -> object:
    """Read a nested field by its snake_case path (camelCase also works)."""
    for key in path:
        if not isinstance(obj, dict):
            return None
        camel = re.sub(r"_([a-z0-9])", lambda m: m.group(1).upper(), key)
        obj = obj.get(key, obj.get(camel))
    return obj


def find_policy(project: str, platform: str, max_price: str) -> dict | None:
    """A live pricing policy in the project for this platform and price."""
    listed = cli_json(
        "billing", "pricing-policy", "list", "--parent-id", project
    )
    for policy in listed.get("items") or []:
        price = field(policy, "spec", "pricing", "max_price_v1", "max_price")
        if (
            field(policy, "spec", "compute_instance_spec", "v1", "platform")
            == platform
            and price
            and Decimal(str(price)) == Decimal(max_price)
            and field(policy, "status", "state") != "STATE_DELETING"
        ):
            return policy
    return None


def resolve_spot_policy(args: argparse.Namespace) -> str | None:
    """The pricing policy id for ``--max-price``, creating one if needed.

    A policy caps one platform in one project, so both come from the job's
    dry-run request. A policy is reused rather than updated: its price can
    only change while no VM runs under it.
    """
    if args.spot_policy:
        return args.spot_policy
    if args.max_price is None:
        return None
    request = build_job(args).create(dry_run=True)
    project, platform = str(request["project"]), str(request["platform"])
    policy = find_policy(project, platform, args.max_price)
    if policy is None and args.dry_run:
        log.info(
            "would create a pricing policy: %s, max $%s per GPU-hour",
            platform,
            args.max_price,
        )
        return "pricingpolicy-(created-on-run)"
    if policy is None:
        name = f"{platform}-max-{args.max_price.replace('.', '-')}"
        log.info(
            "creating pricing policy %s: max $%s per GPU-hour",
            name,
            args.max_price,
        )
        policy = cli_json(
            "billing",
            "pricing-policy",
            "create",
            "--parent-id",
            project,
            "--name",
            name,
            "--compute-instance-spec-v1-platform",
            platform,
            "--pricing-max-price-v1-max-price",
            args.max_price,
        )
    policy_id = str(field(policy, "metadata", "id"))
    log.info(
        "spot price cap: %s (%s, max $%s per GPU-hour)",
        policy_id,
        platform,
        args.max_price,
    )
    if (
        field(policy, "status", "scheduling_state")
        == "SCHEDULING_STATE_BLOCKED"
    ):
        log.warning(
            "the spot price is above $%s now: attempts wait in PROVISIONING "
            "until it falls to the cap",
            args.max_price,
        )
    return policy_id


async def nebius(*args: str) -> str:
    """Run the nebius CLI and return its stdout."""
    proc = await asyncio.create_subprocess_exec(
        "nebius", *args, stdout=asyncio.subprocess.PIPE
    )
    out, _ = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"nebius {' '.join(args)} failed")
    return out.decode()


async def logged_attempts(supervisor_id: str) -> list[str]:
    """Attempt ids from the supervisor's `attempt N/M: created <id>` lines."""
    logs = await nebius("ai", "job", "logs", supervisor_id, *FULL_LOG)
    return re.findall(r"attempt \d+/\d+: created (\S+)", logs)


async def check_attempts(attempt_ids: list[str]) -> None:
    """Print how each attempt ended and its resume/checkpoint log lines."""
    print("\nAttempts:")
    for n, job_id in enumerate(attempt_ids, start=1):
        run = await ns.jobs.get_async(job_id, quiet=True)
        reason, code = end_reason(run)
        state = f"{run.status} {code}".rstrip()
        print(f"attempt {n}: {run.id} {state} ({reason})")
        try:
            logs = await nebius("ai", "job", "logs", run.id, *FULL_LOG)
        except RuntimeError as exc:
            print(f"    could not read logs: {exc}")
            continue
        for line in logs.splitlines():
            if PATTERN.search(line):
                print(f"    {line}")


async def run_local(repeatable: RepeatableJob) -> int:
    """Run the loop in this process. Ctrl-C cancels the current attempt."""

    def on_interrupt() -> None:
        log.warning("Ctrl-C: cancelling the current attempt")
        asyncio.ensure_future(repeatable.cancel_async())

    asyncio.get_running_loop().add_signal_handler(signal.SIGINT, on_interrupt)
    try:
        run = await repeatable.run_async()
    except RuntimeError:  # no attempt completed in max_attempts tries
        run = None
    print("\n" + repeatable.summary())
    await check_attempts([a.job_id for a in repeatable.attempts])
    return 0 if run is not None and run.status == "COMPLETED" else 1


async def run_supervisor(
    args: argparse.Namespace, repeatable: RepeatableJob
) -> int:
    """Run the loop in a CPU job. Ctrl-C stops watching, not the job."""
    supervisor = await repeatable.launch_async(ACCOUNT, timeout=args.timeout)
    cancel = f"nebius ai job cancel {supervisor.id}"
    logs = f"nebius ai job logs {supervisor.id} --since 24h --tail 1000"
    print(f"Supervisor {supervisor.id}\n  logs:   {logs}\n  cancel: {cancel}")
    if args.detach:
        return 0

    detached = asyncio.Event()

    def on_interrupt() -> None:
        log.warning("Ctrl-C: stopped watching; the supervisor keeps running")
        detached.set()

    asyncio.get_running_loop().add_signal_handler(signal.SIGINT, on_interrupt)
    waiting = asyncio.ensure_future(supervisor.wait_async(quiet=True))
    stop = asyncio.ensure_future(detached.wait())
    await asyncio.wait({waiting, stop}, return_when=asyncio.FIRST_COMPLETED)
    for task in (waiting, stop):
        task.cancel()
    if detached.is_set():
        print(
            f"Supervisor {supervisor.id} is still running.\n  cancel: {cancel}"
        )
        return 0
    print(f"\nSupervisor {supervisor.id} ended {supervisor.status}")
    await check_attempts(await logged_attempts(supervisor.id))
    return 0 if supervisor.status == "COMPLETED" else 1


def dry_run(args: argparse.Namespace, repeatable: RepeatableJob) -> int:
    """Print the create requests without creating anything."""
    print("Attempt 1:")
    print(repeatable.attempt_job(1).create(dry_run=True))
    if args.where == "supervisor":
        with tempfile.TemporaryDirectory() as tmp:
            job = supervisor_job(
                repeatable,
                ACCOUNT,
                Path(tmp),
                timeout=args.timeout,
                name=None,
            )
            print("\nSupervisor:")
            print(job.create(dry_run=True))
    return 0


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--run-name",
        default=f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}z-qwen05b-qlora",
        help="Names the attempts, the supervisor, the MLflow run, and the "
        "checkpoint folder. Reuse one to continue its checkpoints.",
    )
    common.add_argument("--bucket-id", default=BUCKET_ID)
    common.add_argument("--max-attempts", type=int, default=5)
    common.add_argument(
        "--attempt-timeout", default="2h", help="Timeout of each attempt."
    )
    common.add_argument(
        "--no-mlflow", action="store_true", help="Do not log to MLflow."
    )
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the create requests and exit.",
    )
    common.add_argument(
        "--pricing",
        choices=("spot", "on-demand"),
        default="spot",
        help="Pricing of each attempt (default spot).",
    )
    cap = common.add_mutually_exclusive_group()
    cap.add_argument(
        "--max-price",
        type=usd,
        metavar="USD",
        help="Spot price cap per GPU-hour, e.g. 2.5. Uses or creates a "
        "pricing policy. Default: no cap, pay the current spot price.",
    )
    cap.add_argument(
        "--spot-policy",
        metavar="ID",
        help="Existing pricing policy id (pricingpolicy-...) to cap with.",
    )

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    where = parser.add_subparsers(dest="where", required=True)
    sv = where.add_parser(
        "supervisor", parents=[common], help="Run the loop in a CPU job."
    )
    sv.add_argument(
        "--timeout", default="12h", help="Supervisor timeout (whole loop)."
    )
    sv.add_argument(
        "--detach",
        action="store_true",
        help="Exit right after the supervisor starts.",
    )
    where.add_parser("local", parents=[common], help="Run the loop here.")
    args = parser.parse_args()
    if args.pricing != "spot" and (args.max_price or args.spot_policy):
        parser.error("--max-price and --spot-policy need --pricing spot")
    return args


def usd(value: str) -> str:
    """Parse a price in USD into the API's 3-decimal string."""
    try:
        price = Decimal(value)
    except InvalidOperation:
        raise argparse.ArgumentTypeError(f"not a price: {value}") from None
    if not price.is_finite() or price <= 0:
        raise argparse.ArgumentTypeError(f"price must be positive: {value}")
    return f"{price:.3f}"


def main() -> int:
    """Build the job and run the loop where asked."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    job = build_job(args, resolve_spot_policy(args))
    repeatable = RepeatableJob(
        job, name=args.run_name, max_attempts=args.max_attempts
    )
    if args.dry_run:
        return dry_run(args, repeatable)
    print(f"Run {args.run_name}")
    if args.where == "local":
        return asyncio.run(run_local(repeatable))
    return asyncio.run(run_supervisor(args, repeatable))


if __name__ == "__main__":
    sys.exit(main())
