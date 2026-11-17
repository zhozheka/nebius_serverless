# Examples

Scripts below use the active Nebius profile and call `create()`. `create(dry_run=True)` returns the create request and does not create a job. Run a script only when the project, region, and quota are authorized.

The alias for `gpu` must exist in that region. See the [alias table](reference.md#gpu-and-cpu-aliases). Logs are `nebius ai job logs <id>`.

Examples are grouped by use case. Settings other than a YAML path live in the Python file. The YAML runner takes `--config`. Decorator recipes are the proposal in [decorators.md](decorators.md). The scripts on this page call `Job.create`.

| Path | Use case |
| --- | --- |
| [`examples/job_quickstart.py`](../examples/job_quickstart.py) | One `nvidia-smi` job, the [README](../README.md) recipe |
| [`examples/from_yaml/`](../examples/from_yaml/) | That same recipe loaded from YAML |
| [`examples/endpoint_quickstart.py`](../examples/endpoint_quickstart.py) | vLLM endpoint on an L40S: create, wait, call `/v1/chat/completions`, delete. YAML: [`vllm_endpoint.yaml`](../examples/from_yaml/vllm_endpoint.yaml) |
| [`examples/notebook/endpoint_quickstart.ipynb`](../examples/notebook/endpoint_quickstart.ipynb) | The same endpoint in Jupyter, with the awaited twins and a separate delete cell |
| [`examples/train/`](../examples/train/) | Axolotl fine-tune |
| [`examples/finetuning/`](../examples/finetuning/) | Axolotl QLoRA fine-tune (`config.yaml` + S3 bucket mount) |
| [`examples/parallel_jobs/`](../examples/parallel_jobs/) | Parallel Axolotl fine-tuning sweep across two models, saving outputs to S3 |
| [`examples/batch/matrix_multiply.py`](../examples/batch/matrix_multiply.py) | Three GPU matrix multiplications, one size per job |
| [`examples/signals/sigterm.py`](../examples/signals/sigterm.py) | CPU job that catches SIGTERM after 60 seconds, logs it, and shuts down cleanly |
| [`examples/signals/repeatable_job.ipynb`](../examples/signals/repeatable_job.ipynb) | `RepeatableJob` prototype: submit an interrupted job again until it completes, resuming from a checkpoint in a bucket |
| [`examples/benchmark/`](../examples/benchmark/) | Side-by-side [SDK comparison](../examples/benchmark/sdk-comparison.md) (Official Nebius, Designed Nebius, Modal, RunPod) |

## Quickstart

One on-demand H100 job. The Python recipe and the commands that run it are in the [README](../README.md): [`examples/job_quickstart.py`](../examples/job_quickstart.py) and [`examples/from_yaml/nvidia_smi_yaml.py`](../examples/from_yaml/nvidia_smi_yaml.py) with [`examples/from_yaml/nvidia_smi.yaml`](../examples/from_yaml/nvidia_smi.yaml).

In `eu-north1`, `gpu="h100"` selects `gpu-h100-sxm` and `1gpu-16vcpu-200gb`. Omit `gpu` for a CPU machine (`cpu-d3`, `2vcpu-8gb`). `pricing="spot"` on that machine raises `UnsupportedFeature`.

The YAML runner calls `create()`, `wait()`, and `display()`, then exits non-zero when `run.status` is not `COMPLETED`.

## Train

One spot GPU job. The full cookbook command is [`examples/train/axolotl_lora.py`](../examples/train/axolotl_lora.py). In `eu-north1`, `gpu="h100"` selects `gpu-h100-sxm` and `1gpu-16vcpu-200gb`. Shared memory stays the `16Gi` default.

```python
job = ns.Job(
    ["bash", "-lc", "axolotl train /workspace/data/config.yaml"],
    image="docker.io/axolotlai/axolotl:main-20260309-py3.11-cu128-2.9.1",
    gpu="h100",
    pricing="spot",
    disk="500Gi",
    timeout="24h",
    volumes=[
        "s3://axolotl-data:/workspace/data:ro",
        "s3://axolotl-artifacts:/workspace/artifacts:rw",
    ],
)
run = job.create()
run.wait()
run.display()
```

```bash
uv run python examples/train/axolotl_lora.py
```

[`examples/finetuning/axolotl-finetuning.py`](../examples/finetuning/axolotl-finetuning.py) pairs that recipe with [`examples/finetuning/config.yaml`](../examples/finetuning/config.yaml) (`Qwen/Qwen2.5-0.5B` QLoRA on `Salesforce/wikitext`) and mounts `s3://fine-tuning-axolotl:/workspace/data:rw`. Training writes to local `/workspace/output` first and copies checkpoints to `/workspace/data/output/$RUN_ID` on completion:

```python
job = ns.Job(
    [
        "bash",
        "-lc",
        "RUN_ID=run-$(date +%Y%m%d-%H%M%S)"
        " && axolotl train /workspace/data/config.yaml"
        " && mkdir -p /workspace/data/output/$RUN_ID"
        " && cp -r /workspace/output/. /workspace/data/output/$RUN_ID",
    ],
    image="docker.io/axolotlai/axolotl:main-20260309-py3.11-cu128-2.9.1",
    gpu="h100",
    pricing="spot",
    disk="500Gi",
    timeout="24h",
    volumes=["s3://fine-tuning-axolotl:/workspace/data:rw"],
)
run = job.create()
run.wait()
run.display()
```

```bash
aws s3 cp examples/finetuning/config.yaml s3://fine-tuning-axolotl/config.yaml \
  --endpoint-url https://storage.eu-north1.nebius.cloud
uv run python examples/finetuning/axolotl-finetuning.py
```

## Parallel Fine-Tuning Sweep

[`examples/parallel_jobs/axolotl_sweep.py`](../examples/parallel_jobs/axolotl_sweep.py) runs two Axolotl jobs in parallel (`Qwen/Qwen2.5-0.5B` and `Qwen/Qwen2.5-1.5B`). Upload both configs to Object Storage first. Both jobs mount `s3://fine-tuning-axolotl:/workspace/data:rw`, train from `/workspace/data/configs/…`, and copy checkpoints into separate folders under `/workspace/data/output/`.

```python
jobs = (
    ns.Job(
        [
            "bash",
            "-lc",
            "axolotl train /workspace/data/configs/qwen_0_5b.yaml"
            " && mkdir -p /workspace/data/output/qwen_0_5b"
            " && cp -r /workspace/output/. /workspace/data/output/qwen_0_5b",
        ],
        image="docker.io/axolotlai/axolotl:main-20260309-py3.11-cu128-2.9.1",
        gpu="h100",
        pricing="spot",
        disk="500Gi",
        timeout="24h",
        volumes=["s3://fine-tuning-axolotl:/workspace/data:rw"],
        name="axolotl-qwen-0.5b",
    ),
    ns.Job(
        [
            "bash",
            "-lc",
            "axolotl train /workspace/data/configs/qwen_1_5b.yaml"
            " && mkdir -p /workspace/data/output/qwen_1_5b"
            " && cp -r /workspace/output/. /workspace/data/output/qwen_1_5b",
        ],
        image="docker.io/axolotlai/axolotl:main-20260309-py3.11-cu128-2.9.1",
        gpu="h100",
        pricing="spot",
        disk="500Gi",
        timeout="24h",
        volumes=["s3://fine-tuning-axolotl:/workspace/data:rw"],
        name="axolotl-qwen-1.5b",
    ),
)

runs = [job.create() for job in jobs]
for run in runs:
    run.wait()
    run.display()
```

```bash
aws s3 cp examples/parallel_jobs/configs/ s3://fine-tuning-axolotl/configs/ \
  --recursive \
  --endpoint-url https://storage.eu-north1.nebius.cloud
uv run python examples/parallel_jobs/axolotl_sweep.py
```

## Batch

Three independent GPU jobs, one matrix size each: `1024`, `2048`, and `4096`. There is no `.map()`. [`examples/batch/matrix_multiply.py`](../examples/batch/matrix_multiply.py) submits every job, prints each id, then waits and reads a JSON report per size. Matrices are created on the GPU. Object Storage is used only to bring back the small reports. The jobs may run at the same time. The script does not promise that they start together, or that the batch finishes sooner than the same jobs one after another.

The script sets `BUCKET` to `acceptance-bucket` and `ENDPOINT` to `https://storage.eu-north1.nebius.cloud`. Change both when that bucket is not the one you can mount and read. The active Nebius profile must be allowed to create jobs in a region where `gpu="h100"` exists. In `eu-north1` that alias is `gpu-h100-sxm` / `1gpu-16vcpu-200gb`. Quota has to cover up to three one-GPU H100 jobs, because overlap is allowed. The AWS CLI (v2) needs credentials that can read the report objects at `ENDPOINT`, and delete them when you clean up. The job mount `s3://<bucket>:/data:rw` has to be able to write those objects. No input files are uploaded. The controller does not import PyTorch.

The image is pinned by digest, checked against the `linux/amd64` manifest of `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime` (`pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime@sha256:f894dae26e1ee8557c544f9cfdb9dc011b1552bf3c1e656b422f2e221d380e40`):

| | |
| --- | --- |
| digest | `pytorch/pytorch@sha256:f894dae26e1ee8557c544f9cfdb9dc011b1552bf3c1e656b422f2e221d380e40` |
| PyTorch | 2.6.0 (`2.6.0+cu126`) |
| Python | 3.11.11 |
| CUDA | 12.6.3 |
| cuDNN | 9, as named by that tag |
| base | Ubuntu 22.04 |

The worker sets `torch.backends.cuda.matmul.allow_tf32 = False` and `torch.set_float32_matmul_precision("highest")` so the float32 multiply uses full FP32 rather than a default that can change. It requires CUDA, checks `[[1, 2], [3, 4]] @ [[5, 6], [7, 8]]`, then checks the shape, finiteness, and nine sampled entries of the large product. A failed check does not write a success report. Each report is `/data/<prefix>/<size>.json` with `schema_version` 1, `size`, `gpu`, `elapsed_ms`, `mean`, and `verified: true`. See [SDK comparison](../examples/benchmark/sdk-comparison.md) for the same workload beside the official Nebius SDK, Modal, and RunPod.

`timeout="1h"` is the remote Jobs deadline. `run.wait(timeout=5400)` only stops the local poll; it does not cancel the job. After `COMPLETED`, the script polls the report for at most 60 seconds, every 2 seconds, and only when the object is missing. An authentication failure, malformed JSON, or an invalid report stops immediately. Pricing stays the on-demand default.

```bash
uv run python examples/batch/matrix_multiply.py
```

```python
run = ns.Job(
    ["python", "-c", WORKER, prefix, str(size)],
    image=IMAGE,
    gpu="h100",
    timeout="1h",
    volumes=[f"s3://{BUCKET}:/data:rw"],
).create()
```

The script prints the object prefix and each job id as soon as create is accepted, then a table. The numbers below are an illustration. Timings, means, and the GPU name are not fixed results. On a real H100 the `gpu` column should name an H100; a different device means the allocation did not match `gpu="h100"`.

```text
prefix matrix-multiply/0124abcd0124abcd0124abcd0124abcd
size 1024: bt1example1024
size 2048: bt1example2048
size 4096: bt1example4096
size    gpu                     elapsed_ms    mean      verified
1024    NVIDIA H100 80GB HBM3   12.5          0.0011    True
2048    NVIDIA H100 80GB HBM3   48.2          -0.0003   True
4096    NVIDIA H100 80GB HBM3   190.4         0.0004    True
```

The script’s own table is tab-separated with the columns `size`, `gpu`, `elapsed_ms`, `mean`, and `verified`.

An error stops the script and includes the size and, once create has returned, the job id. Jobs already accepted keep running. Nothing is resubmitted, cancelled, or deleted. A local wait timeout does not stop the remote container. Cleanup is manual, and only for this invocation’s ids and prefix:

```bash
nebius ai job delete <id>
aws s3 rm --recursive "s3://<bucket>/matrix-multiply/<uuid>/" \
  --endpoint-url https://storage.eu-north1.nebius.cloud
```

## Graceful shutdown on SIGTERM

[`examples/signals/sigterm.py`](../examples/signals/sigterm.py) runs [`examples/signals/sigterm_worker.py`](../examples/signals/sigterm_worker.py) on a CPU machine (`python:3.12-slim`) under coreutils `timeout`. The worker is a plain Python file uploaded with `inject_files`: it is read on `create` (at most 64 KiB) and written to `/app/sigterm_worker.py` in the container. The local path is resolved from the script's own location, so it runs from any directory. After 60 seconds `timeout` sends SIGTERM. The worker logs the signal, runs a cleanup step, and exits 0, so the job ends `COMPLETED`. If the worker is still running 30 seconds after SIGTERM, `timeout` sends SIGKILL and the job ends `FAILED`. If SIGTERM never arrives, the worker exits 1 after 300 seconds.

```python
WORKER = Path(__file__).resolve().parent / "sigterm_worker.py"

job = ns.Job(
    [
        "timeout",
        "--preserve-status",
        "--kill-after=30",
        "--signal=TERM",
        "60",
        "python",
        "-u",
        "/app/sigterm_worker.py",
    ],
    image="python:3.12-slim",
    timeout="1h",
    inject_files=[f"{WORKER}:/app/sigterm_worker.py"],
    name="sigterm-example",
)
```

The worker also runs locally: `python examples/signals/sigterm_worker.py`, then `kill -TERM <pid>`.

`timeout` is PID 1 and forwards the signal to the worker. If the worker runs under `bash -lc`, use `exec python ...`: bash as PID 1 does not forward SIGTERM to its child. `--preserve-status` makes the job's exit code the worker's exit code instead of `124`. After the job finishes, the script reads `nebius ai job logs <id>` and exits non-zero when the `received SIGTERM` line is missing.

```bash
uv run python examples/signals/sigterm.py
```

The worker log looks like this. Times and the pid are an illustration.

```text
2026-11-17 13:29:22,144 INFO worker started, pid 7
2026-11-17 13:30:12,168 INFO working: step 50, 49s elapsed
2026-11-17 13:30:22,177 WARNING received SIGTERM after 60.0s at step 60
2026-11-17 13:30:22,178 INFO cleanup: saving progress at step 60
2026-11-17 13:30:24,184 INFO cleanup: done
2026-11-17 13:30:24,184 INFO graceful shutdown complete
```

## Secrets, private images, retries, and spot caps

Keep tokens in MysteryBox instead of `env`. `secrets` maps an env name to a secret id (`mbsec-…`, latest version) or a version id (`mbsecver-…`). The container gets the payload entry whose key matches the name. `registry_secret` is a secret version with `REGISTRY_USERNAME` and `REGISTRY_PASSWORD` for a private image. `retries` restarts a failed or preempted job. `spot_policy` caps the spot price instead of following it. The ids below are placeholders.

```python
job = ns.Job(
    ["python", "train.py", "--config", "config.yaml"],
    image="registry.example.com/team/trainer:1.4",
    gpu="h100",
    timeout="24h",
    pricing="spot",
    spot_policy="pricingpolicy-...",
    retries=3,
    working_dir="/workspace",
    env={"WANDB_MODE": "offline"},
    secrets={"HF_TOKEN": "mbsec-...", "WANDB_API_KEY": "mbsecver-..."},
    registry_secret="mbsecver-...",
    volumes=["s3://my-data:/workspace/data:ro"],
)
print(job.create(dry_run=True))   # shows secret ids, never values
```

An endpoint takes the same `working_dir`, `secrets`, `registry_secret`, and `spot_policy`. `token_secret` reads the bearer token from a secret whose payload has `AUTH_TOKEN`, so the token stays stable across creates. `Service.token` is then `None`; clients read the token from MysteryBox.

```python
endpoint = ns.Endpoint(
    image="vllm/vllm-openai:v0.19.1",
    port=8000,
    gpu="l40s",
    auth="token",
    token_secret="mbsec-...",
    secrets={"HF_TOKEN": "mbsec-..."},
)
```

The same keys work in YAML documents.

```yaml
schema_version: 1
kind: job
image: registry.example.com/team/trainer:1.4
command: ["python", "train.py"]
gpu: h100
timeout: 24h
pricing: spot
retries: 3
working_dir: /workspace
secrets:
  HF_TOKEN: mbsec-...
registry_secret: mbsecver-...
```

## Reload a saved id

`run.id` is the handle. `jobs.get` performs one get and does not create again.

```python
saved_id = run.id
loaded = ns.jobs.get(saved_id)
print(loaded.status)
```

## Cancel

`cancel` returns when the request is accepted, with the job usually `CANCELLING`. Stopping the container can take minutes. `wait()` returns once it is `CANCELLED`.

```python
payload = run.cancel()
print(payload)
print(run.get()["status"]["state"])   # usually CANCELLING
record = run.wait()
print(record["status"]["state"])      # CANCELLED
```

A local `wait` timeout leaves the job running:

```python
try:
    run.wait(timeout=0)
except ns.WaitTimeout as exc:
    print(exc.run_id)
```

## Quiet progress

Pass `quiet=True` on `create` and `wait` to suppress progress on stderr. Stdout still prints the id and the final state. Other calls on that run show progress again. `create(dry_run=True)` stays silent.

```python
run = job.create(quiet=True)
print(run.id)
record = run.wait(quiet=True)
print(run.id, record["status"]["state"])
```

## Inside an event loop

```python
import asyncio

import nebius_serverless as ns

async def main() -> None:
    job = ns.Job(
        ["nvidia-smi"],
        image="nvidia/cuda:13.1.1-runtime-ubuntu24.04",
        gpu="h100",
        timeout="1h",
    )
    run = await job.create_async()
    record = await run.wait_async()
    print(run.id, record["status"]["state"])

asyncio.run(main())
```

Calling `job.create()` from inside `main` raises `ValidationError` and tells you to use `create_async`.
