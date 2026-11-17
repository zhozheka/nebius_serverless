"""Simulated training that resumes from a checkpoint.

Used by ``repeatable_job.ipynb``. Each step takes one second. The step
number is saved to ``$CKPT_DIR/checkpoint.json`` every ``SAVE_EVERY``
steps and on SIGTERM. A new attempt starts from that checkpoint.

Exit codes: 0 when all ``TOTAL_STEPS`` are done, 143 after SIGTERM
(checkpoint saved), so an interrupted attempt ends ``FAILED``.
"""

import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", "150"))
SAVE_EVERY = 10
CHECKPOINT = Path(os.environ["CKPT_DIR"]) / "checkpoint.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("worker")

stop_requested = False


def on_sigterm(signum: int, frame: object) -> None:
    """Only set a flag. The loop saves at the next step boundary."""
    global stop_requested
    stop_requested = True


def load_step() -> int:
    """Return the saved step, or 0 when there is no checkpoint."""
    if not CHECKPOINT.exists():
        return 0
    return json.loads(CHECKPOINT.read_text())["step"]


def save_step(step: int) -> None:
    """Write the checkpoint through a temp file, then rename it."""
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT.with_suffix(".tmp")
    tmp.write_text(json.dumps({"step": step}))
    os.replace(tmp, CHECKPOINT)


def main() -> None:
    """Train from the last checkpoint until done or SIGTERM."""
    signal.signal(signal.SIGTERM, on_sigterm)
    step = load_step()
    log.info("starting at step %d of %d", step, TOTAL_STEPS)
    while step < TOTAL_STEPS:
        time.sleep(1)  # one training step
        step += 1
        if step % SAVE_EVERY == 0:
            save_step(step)
            log.info("checkpoint at step %d", step)
        if stop_requested:
            save_step(step)
            log.warning("SIGTERM: checkpoint at step %d, exiting", step)
            sys.exit(143)
    save_step(step)
    log.info("training done at step %d", step)


if __name__ == "__main__":
    main()
