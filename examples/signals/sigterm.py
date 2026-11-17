"""Graceful shutdown on SIGTERM.

One CPU job runs ``sigterm_worker.py`` under coreutils ``timeout``.
The worker file is uploaded with ``inject_files``: it is read on
``create`` and written into the container at ``WORKER_IN_CONTAINER``.

After ``SIGTERM_AFTER`` seconds ``timeout`` sends SIGTERM to the worker,
which logs the signal, runs its cleanup, and exits 0, so the job ends
``COMPLETED``. If the worker is still alive ``GRACE`` seconds later,
``timeout`` sends SIGKILL and the job ends ``FAILED``.

The worker logs to stdout. The script reads the logs back with
``nebius ai job logs`` and checks for the shutdown lines.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

import nebius_serverless as ns

IMAGE = "python:3.12-slim"
SIGTERM_AFTER = 60  # seconds of work before SIGTERM
GRACE = 30  # seconds between SIGTERM and SIGKILL
TIMEOUT = "1h"  # remote deadline for the whole job; 1h is the minimum
LOG_ATTEMPTS = 6
LOG_PAUSE = 5

# Resolved from this file, so the script works from any directory.
WORKER = Path(__file__).resolve().parent / "sigterm_worker.py"
WORKER_IN_CONTAINER = "/app/sigterm_worker.py"


def _fetch_logs(job_id: str) -> str | None:
    """Return the job logs once they contain the shutdown line."""
    if shutil.which("nebius") is None:
        print("nebius CLI not found; skipping the log check.")
        return None
    output = ""
    for _ in range(LOG_ATTEMPTS):
        completed = subprocess.run(
            ["nebius", "ai", "job", "logs", job_id],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        output = completed.stdout
        if "graceful shutdown complete" in output:
            return output
        time.sleep(LOG_PAUSE)
    return output


def main() -> None:
    """Create the job, wait for it, and check that it shut down cleanly."""
    job = ns.Job(
        [
            "timeout",
            "--preserve-status",  # job exit code = worker exit code
            f"--kill-after={GRACE}",
            "--signal=TERM",
            str(SIGTERM_AFTER),
            "python",
            "-u",
            WORKER_IN_CONTAINER,
        ],
        image=IMAGE,
        timeout=TIMEOUT,
        inject_files=[f"{WORKER}:{WORKER_IN_CONTAINER}"],
        name="sigterm-example",
    )

    run = job.create()
    run.wait()
    run.display()

    logs = _fetch_logs(run.id)
    if logs is not None:
        print(logs)
        if "received SIGTERM" not in logs:
            sys.exit(f"job {run.id}: no SIGTERM line in the logs")

    if run.status != "COMPLETED":
        sys.exit(f"job {run.id} ended {run.status}")


if __name__ == "__main__":
    main()
