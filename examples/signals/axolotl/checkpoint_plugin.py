"""Axolotl plugin: save a checkpoint on SIGTERM, copy each one to a bucket.

Used by ``resumable_axolotl.py`` through ``plugins:`` in ``config.yaml``.
Runs inside the training process, so ``PYTHONPATH`` must include this
directory.

- Every checkpoint the Trainer saves (each ``save_steps``) is copied to
  ``$CKPT_DIR/checkpoint-N``, then a ``COMPLETE`` file is written into it.
  The marker is written last, so a folder without it is a partial copy.
  (A pointer file renamed into place is not safe: ``os.replace`` on the
  bucket mount can leave the target empty.)
- On SIGTERM the Trainer saves one more checkpoint at the end of the
  current step. After it is copied, the process exits 143.
- Restores the token counters on resume (works around an Axolotl bug).
- When the Trainer logs to MLflow, the run id is written to
  ``$CKPT_DIR/mlflow_run_id``, so later attempts continue the same run,
  and the run is named after ``REPEATABLE_NAME`` when it is set.
"""

import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

from axolotl.integrations.base import BasePlugin
from transformers import TrainerCallback

CKPT_DIR = Path(os.environ["CKPT_DIR"])
MARKER = "COMPLETE"  # written last into each copied checkpoint
RUN_ID_FILE = CKPT_DIR / "mlflow_run_id"  # read by resumable_axolotl.py


def say(message: str) -> None:
    """Print one line, flushed, so it shows in the job logs right away."""
    print(f"checkpoint_plugin: {message}", flush=True)


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


def publish(local: Path) -> None:
    """Copy a finished checkpoint to the bucket, then mark it complete."""
    start = time.monotonic()
    target = CKPT_DIR / local.name
    copy_dir(local, target)
    (target / MARKER).write_text("ok")
    seconds = time.monotonic() - start
    say(f"copied {local.name} to the bucket in {seconds:.1f}s")
    for old in CKPT_DIR.glob("checkpoint-*"):
        if old.name != local.name:  # best effort; marker first
            (old / MARKER).unlink(missing_ok=True)
            shutil.rmtree(old, ignore_errors=True)


class SigtermCheckpoint(TrainerCallback):
    """Turn SIGTERM into one last checkpoint, then exit."""

    def __init__(self) -> None:
        self.stop_requested = False
        signal.signal(signal.SIGTERM, self._on_sigterm)

    def _on_sigterm(self, signum: int, frame: object) -> None:
        """Only set a flag. The Trainer saves at the next step boundary."""
        self.stop_requested = True

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_requested and not control.should_training_stop:
            say(f"SIGTERM: saving checkpoint-{state.global_step}")
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_save(self, args, state, control, **kwargs):
        """Called after the checkpoint folder is completely written."""
        publish(Path(args.output_dir) / f"checkpoint-{state.global_step}")
        if self.stop_requested:
            say(f"SIGTERM: checkpoint-{state.global_step} saved, exiting")
            sys.exit(143)
        return control


class MlflowRunId(TrainerCallback):
    """Save the MLflow run id, so later attempts log to the same run.

    Runs after the Trainer's MLflow callback, which starts the run in its
    own ``on_train_begin``.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        if "mlflow" not in (args.report_to or []):
            return control
        import mlflow

        run = mlflow.active_run()
        if run is None:
            return control
        name = os.environ.get("REPEATABLE_NAME")
        if name:
            mlflow.set_tag("mlflow.runName", name)  # the name the UI shows
        if not RUN_ID_FILE.exists():
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            RUN_ID_FILE.write_text(run.info.run_id)
            say(f"MLflow run {run.info.run_id} ({name or 'unnamed'})")
        return control


class RestoreTokenCounts(TrainerCallback):
    """Restore the ``tokens/total`` and ``tokens/trainable`` counters.

    Axolotl saves them into each checkpoint, but in the image's version
    (main-20260309) the trainer writes ``tokens_state.`` while the callback
    that restores them reads ``tokens_state.json``, so they restart from 0
    on every resume. By ``on_train_begin`` the Trainer has loaded the
    checkpoint's state, so ``global_step`` names the checkpoint. Runs after
    Axolotl's own callbacks, so it also works once Axolotl is fixed.
    """

    NAMES = ("tokens_state.", "tokens_state.json")

    def on_train_begin(self, args, state, control, **kwargs):
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if state.global_step == 0 or not checkpoint.is_dir():
            return control
        for name in self.NAMES:
            path = checkpoint / name
            if path.is_file():
                import torch

                saved = json.loads(path.read_text())
                state.tokens = {
                    "total": torch.tensor(saved.get("total", 0)),
                    "trainable": torch.tensor(saved.get("trainable", 0)),
                }
                say(
                    f"restored token counts from {checkpoint.name}/{name}: "
                    f"total {saved.get('total', 0)}, "
                    f"trainable {saved.get('trainable', 0)}"
                )
                break
        return control


class CheckpointPlugin(BasePlugin):
    """Register the callbacks with the Axolotl trainer."""

    def add_callbacks_post_trainer(self, cfg, trainer):
        return [SigtermCheckpoint(), MlflowRunId(), RestoreTokenCounts()]
