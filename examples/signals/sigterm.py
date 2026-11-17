"""Graceful shutdown on SIGTERM.

One CPU job runs a Python worker under coreutils ``timeout``. After
``SIGTERM_AFTER`` seconds ``timeout`` sends SIGTERM to the worker,
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

import nebius_serverless as ns

IMAGE = "python:3.12-slim"
SIGTERM_AFTER = 60  # seconds of work before SIGTERM
GRACE = 30  # seconds between SIGTERM and SIGKILL
TIMEOUT = "1h"  # remote deadline for the whole job; 1h is the minimum
LOG_ATTEMPTS = 6
LOG_PAUSE = 5

WORKER = """
import logging
import os
import signal
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("worker")

# Exit with an error if SIGTERM never comes, so a broken setup ends
# FAILED instead of running until the job timeout.
GIVE_UP_AFTER = 300

received = None


def on_sigterm(signum, frame):
    # Only record the signal here. Logging and cleanup run in the main
    # loop, outside the handler.
    global received
    received = signal.Signals(signum).name


def cleanup(step):
    log.info("cleanup: saving progress at step %d", step)
    time.sleep(2)  # stands in for writing a checkpoint
    log.info("cleanup: done")


def main():
    signal.signal(signal.SIGTERM, on_sigterm)
    started = time.monotonic()
    log.info("worker started, pid %d", os.getpid())
    step = 0
    while received is None:
        elapsed = time.monotonic() - started
        if elapsed > GIVE_UP_AFTER:
            log.error("no SIGTERM after %.0fs, giving up", elapsed)
            sys.exit(1)
        step += 1
        if step % 10 == 0:
            log.info("working: step %d, %.0fs elapsed", step, elapsed)
        time.sleep(1)
    elapsed = time.monotonic() - started
    log.warning("received %s after %.1fs at step %d", received, elapsed, step)
    cleanup(step)
    log.info("graceful shutdown complete")
    sys.exit(0)


if __name__ == "__main__":
    main()
"""


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
            "-c",
            WORKER,
        ],
        image=IMAGE,
        timeout=TIMEOUT,
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
