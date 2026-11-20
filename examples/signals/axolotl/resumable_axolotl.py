"""Run Axolotl so that a new attempt continues from the last checkpoint.

Used by ``repeatable_axolotl.ipynb`` as the job's main process. Follows the
three rules of a resumable worker:

1. On start, copy the newest ``$CKPT_DIR/checkpoint-N`` that has a
   ``COMPLETE`` marker from the bucket into Axolotl's local
   ``output_dir``. ``config.yaml`` sets ``auto_resume_from_checkpoints``,
   so Axolotl continues from it.
2. ``checkpoint_plugin.py`` copies every checkpoint to the bucket, and
   saves one more on SIGTERM. This process forwards SIGTERM to Axolotl.
3. Exit 0 only when training is done, after copying the final adapter to
   ``$CKPT_DIR/final``.

MLflow: every attempt logs to one run. The first attempt starts it and
``checkpoint_plugin.py`` saves its id; later attempts set ``MLFLOW_RUN_ID``
so the Trainer reattaches, and ``MLFLOW_MAX_LOG_PARAMS=0`` because MLflow
rejects a parameter whose value changes within a run. Metrics are logged
at the global step, so the curves continue where the checkpoint left off.

Exit codes: 0 when done, 143 after SIGTERM, otherwise Axolotl's code.
"""

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

CKPT_DIR = Path(os.environ["CKPT_DIR"])
CONFIG = os.environ.get("AXOLOTL_CONFIG", "/app/config.yaml")
OUTPUT_DIR = Path("/workspace/output")  # output_dir in config.yaml
MARKER = "COMPLETE"  # checkpoint_plugin writes it last into each copy
RUN_ID_FILE = CKPT_DIR / "mlflow_run_id"  # written by checkpoint_plugin
MLFLOW_CLIENT = "mlflow-skinny==2.21.2"  # same version as the server
SECRETS_FILE = Path("/app/secrets.env")  # optional KEY=value lines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("resumable_axolotl")


def copy_dir(src: Path, dst: Path) -> None:
    """Copy file contents only. The bucket mount rejects chmod and utime,
    so ``shutil.copytree`` and ``copy2`` fail there."""
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.rglob("*")):
        target = dst / path.relative_to(src)
        if path.is_dir():
            target.mkdir(exist_ok=True)
        else:
            shutil.copyfile(path, target)


def latest_checkpoint() -> Path | None:
    """Return the newest complete checkpoint in the bucket, or None."""
    complete = [
        path
        for path in CKPT_DIR.glob("checkpoint-*")
        if path.name.split("-")[-1].isdigit() and (path / MARKER).exists()
    ]
    return max(
        complete, key=lambda p: int(p.name.split("-")[-1]), default=None
    )


def restore() -> None:
    """Copy the latest complete checkpoint from the bucket, if any."""
    latest = latest_checkpoint() if CKPT_DIR.exists() else None
    if latest is None:
        log.info("no checkpoint in %s, starting fresh", CKPT_DIR)
        return
    copy_dir(latest, OUTPUT_DIR / latest.name)
    (OUTPUT_DIR / latest.name / MARKER).unlink(missing_ok=True)
    log.info("resuming from %s", latest.name)


def load_secrets() -> None:
    """Export ``KEY=value`` lines from the injected secrets file, if any.

    Injected files are stored encrypted and their contents do not show in
    the job's record, unlike ``env`` values, so passwords such as
    ``MLFLOW_TRACKING_PASSWORD`` come this way.
    """
    if not SECRETS_FILE.exists():
        return
    names = []
    for line in SECRETS_FILE.read_text().splitlines():
        key, sep, value = line.partition("=")
        key = key.strip()
        if sep and key and not key.startswith("#"):
            os.environ[key] = value.strip()
            names.append(key)
    log.info("loaded %s from %s", ", ".join(names), SECRETS_FILE)


def ensure_mlflow() -> None:
    """Install the MLflow client if the image does not have it."""
    try:
        import mlflow  # noqa: F401
    except ImportError:
        log.info("installing %s", MLFLOW_CLIENT)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", MLFLOW_CLIENT],
            check=True,
        )


def start_attempt() -> int:
    """Record this attempt in the bucket and return its number.

    ``RepeatableJob`` sets ``REPEATABLE_ATTEMPT``; without it, attempts
    are counted from the bucket.
    """
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    number = int(
        os.environ.get("REPEATABLE_ATTEMPT")
        or len(list(CKPT_DIR.glob("attempt-*"))) + 1
    )
    (CKPT_DIR / f"attempt-{number}").write_text("started")
    return number


def training_env(attempt: int) -> dict[str, str]:
    """Environment for Axolotl: tags, and the MLflow run to continue."""
    env = dict(os.environ)
    tags = {"repeatable.ckpt_dir": str(CKPT_DIR)}
    if os.environ.get("REPEATABLE_NAME"):
        tags["repeatable.name"] = os.environ["REPEATABLE_NAME"]
    tags["repeatable.attempt"] = str(attempt)
    env["MLFLOW_TAGS"] = json.dumps(tags)
    if RUN_ID_FILE.exists():
        run_id = RUN_ID_FILE.read_text().strip()
        env["MLFLOW_RUN_ID"] = run_id
        env["MLFLOW_MAX_LOG_PARAMS"] = "0"  # logged by the first attempt
        log.info("attempt %d, continuing MLflow run %s", attempt, run_id)
    else:
        log.info("attempt %d, new MLflow run", attempt)
    return env


def log_adapter() -> None:
    """Log the trained adapter to the MLflow run. Best effort."""
    if not RUN_ID_FILE.exists() or not os.environ.get("MLFLOW_TRACKING_URI"):
        return
    run_id = RUN_ID_FILE.read_text().strip()
    try:
        from mlflow import MlflowClient

        client = MlflowClient()
        for path in OUTPUT_DIR.iterdir():
            if path.is_file():
                client.log_artifact(run_id, str(path), "adapter")
        log.info("adapter logged to MLflow run %s", run_id)
    except Exception as exc:  # training is done; do not fail the job
        log.warning("could not log the adapter to MLflow: %s", exc)


def save_final() -> None:
    """Copy the trained adapter to the bucket, then mark the run done."""
    final = CKPT_DIR / "final"
    final.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_DIR.iterdir():
        if path.is_file():  # adapter, tokenizer, logs; not checkpoint-N/
            shutil.copyfile(path, final / path.name)
    (CKPT_DIR / "done.json").write_text(json.dumps({"final": "final"}))
    log.info("training done, adapter in %s", final)
    log_adapter()


def main() -> None:
    """Restore, train, and save the result or forward the stop."""
    if (CKPT_DIR / "done.json").exists():
        log.info("training done in an earlier attempt, nothing to do")
        return
    load_secrets()
    restore()
    ensure_mlflow()
    child = subprocess.Popen(
        ["axolotl", "train", CONFIG, "--launcher", "python"],
        env=training_env(start_attempt()),
    )
    signal.signal(
        signal.SIGTERM, lambda signum, frame: child.send_signal(signum)
    )
    code = child.wait()
    if code == 0:
        save_final()
        return
    if code in (143, -signal.SIGTERM):
        log.warning("SIGTERM: Axolotl stopped, exiting")
        sys.exit(143)
    log.error("Axolotl exited with %d", code)
    sys.exit(code if code > 0 else 1)


if __name__ == "__main__":
    main()
