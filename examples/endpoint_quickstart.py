"""Serve a small model with vLLM on an endpoint, call it, then delete it.

``gpu="l40s"`` with token auth. Uses the active Nebius profile. The same
recipe as YAML is ``examples/from_yaml/vllm_endpoint.yaml``.

The endpoint bills from create until ``delete``. This script deletes it
in ``finally``, including when a step fails or you press Ctrl-C.
"""

import json
import time
import urllib.error
import urllib.request

import nebius_serverless as ns

MODEL = "Qwen/Qwen3-0.6B"
READY_TIMEOUT = 15 * 60


def main() -> None:
    """Create the endpoint, wait for the model, send one request, delete."""

    # Describe what should serve. Omit command to keep the image entrypoint.
    endpoint = ns.Endpoint(
        [
            "bash",
            "-c",
            "python3 -m vllm.entrypoints.openai.api_server "
            f"--model {MODEL} --host 0.0.0.0 --port 8000",
        ],
        image="vllm/vllm-openai:v0.19.1",
        gpu="l40s",
        port=8000,
        auth="token",
        disk="500Gi",
        name="qwen-tiny",
    )

    # Start one endpoint. The token exists only on this Service.
    service = endpoint.create()
    try:
        # RUNNING with a URL. The model may still be loading.
        service.wait()
        if service.url is None:
            raise SystemExit(f"{service.id} ended {service.status}")

        # Poll the application's own route before sending traffic.
        _wait_for_model(service)
        print(_chat(service, "Say hello in five words."))
    finally:
        service.delete()


def _request(
    service: ns.Service,
    path: str,
    body: dict[str, object] | None = None,
) -> urllib.request.Request:
    assert service.url is not None
    headers = {"Authorization": f"Bearer {service.token}"}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    return urllib.request.Request(
        service.url.rstrip("/") + path,
        data=data,
        headers=headers,
    )


def _wait_for_model(service: ns.Service) -> None:
    deadline = time.monotonic() + READY_TIMEOUT
    while True:
        try:
            with urllib.request.urlopen(
                _request(service, "/v1/models"), timeout=10
            ) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        if time.monotonic() > deadline:
            raise SystemExit(f"{service.url} did not answer /v1/models")
        time.sleep(10)


def _chat(service: ns.Service, prompt: str) -> str:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
    }
    with urllib.request.urlopen(
        _request(service, "/v1/chat/completions", body), timeout=60
    ) as response:
        payload = json.load(response)
    return payload["choices"][0]["message"]["content"]


if __name__ == "__main__":
    main()
