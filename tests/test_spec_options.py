"""Retries, working_dir, secrets, registry_secret, spot_policy, and
token_secret on jobs and endpoints."""

from pathlib import Path

import pytest
from fakes import FakeEndpointTransport, FakeTransport

import nebius_serverless as ns
from nebius_serverless.sdk_transport import (
    build_create_request,
    build_endpoint_request,
)

CUDA = "nvidia/cuda:13.1.1-runtime-ubuntu24.04"
VLLM = "vllm/vllm-openai:v0.19.1"
SECRET = "super-secret-value"
HF_SECRET = "mbsec-hf"
WANDB_VERSION = "mbsecver-wandb"
REGISTRY = "mbsecver-registry"
TOKEN_SECRET = "mbsec-token"
POLICY = "pricingpolicy-cap"


def _job(**kwargs: object) -> ns.Job:
    fields: dict[str, object] = {
        "image": CUDA,
        "gpu": "h100",
        "timeout": "1h",
    }
    fields.update(kwargs)
    return ns.Job(["nvidia-smi"], **fields)  # type: ignore[arg-type]


def _endpoint(**kwargs: object) -> ns.Endpoint:
    fields: dict[str, object] = {
        "image": VLLM,
        "port": 8000,
        "gpu": "l40s",
    }
    fields.update(kwargs)
    return ns.Endpoint(**fields)  # type: ignore[arg-type]


def _full_job() -> ns.Job:
    return _job(
        pricing="spot",
        spot_policy=POLICY,
        retries=3,
        working_dir="/workspace",
        env={"PLAIN": SECRET},
        secrets={"HF_TOKEN": HF_SECRET, "WANDB_API_KEY": WANDB_VERSION},
        registry_secret=REGISTRY,
    )


def _full_endpoint() -> ns.Endpoint:
    return _endpoint(
        auth="token",
        token_secret=TOKEN_SECRET,
        pricing="spot",
        spot_policy=POLICY,
        working_dir="/srv",
        secrets={"HF_TOKEN": HF_SECRET},
        registry_secret=REGISTRY,
    )


def _spec(body: dict[str, object]) -> dict[str, object]:
    spec = body["spec"]
    assert isinstance(spec, dict)
    return spec


# Validation


@pytest.mark.parametrize("retries", [-1, True, 1.5, "3"])
def test_retries_must_be_a_non_negative_integer(retries: object) -> None:
    with pytest.raises(ns.ValidationError, match="retries"):
        _job(retries=retries)


@pytest.mark.parametrize("factory", [_job, _endpoint])
def test_shared_option_rules(factory: object) -> None:
    make = factory  # type: ignore[assignment]
    with pytest.raises(ns.ValidationError, match="128"):
        make(working_dir="/" + "a" * 128)  # type: ignore[operator]
    with pytest.raises(ns.ValidationError, match="working_dir"):
        make(working_dir="")  # type: ignore[operator]
    with pytest.raises(ns.ValidationError, match="env names"):
        make(secrets={"BAD-NAME": HF_SECRET})  # type: ignore[operator]
    with pytest.raises(ns.ValidationError, match="mbsec"):
        make(secrets={"HF_TOKEN": "hf_plain_value"})  # type: ignore[operator]
    with pytest.raises(ns.ValidationError, match="both env and secrets"):
        make(  # type: ignore[operator]
            env={"HF_TOKEN": SECRET},
            secrets={"HF_TOKEN": HF_SECRET},
        )
    with pytest.raises(ns.ValidationError, match="mbsecver"):
        make(registry_secret="mbsec-not-a-version")  # type: ignore[operator]
    with pytest.raises(ns.ValidationError, match="pricing='spot'"):
        make(spot_policy=POLICY)  # type: ignore[operator]
    with pytest.raises(ns.UnsupportedFeature, match="profile"):
        make(volumes=["s3://data:/data:ro:aws"])  # type: ignore[operator]


def test_plain_value_errors_do_not_echo_the_value() -> None:
    with pytest.raises(ns.ValidationError) as caught:
        _job(secrets={"HF_TOKEN": SECRET})
    assert SECRET not in str(caught.value)


def test_token_secret_rules() -> None:
    with pytest.raises(ns.ValidationError, match="auth='token'"):
        _endpoint(token_secret=TOKEN_SECRET)
    with pytest.raises(ns.ValidationError, match="not both"):
        _endpoint(auth="token", token="t", token_secret=TOKEN_SECRET)
    with pytest.raises(ns.ValidationError, match="mbsec"):
        _endpoint(auth="token", token_secret="plain-token")


# Job mapping


def test_job_body_carries_the_new_fields() -> None:
    fake = FakeTransport()
    _full_job().create(region="eu-north1", project="p", transport=fake)
    spec = _spec(fake.bodies[0])
    assert spec["restart_attempts"] == 3
    assert spec["working_dir"] == "/workspace"
    assert spec["environment_variables"] == [
        {"name": "PLAIN", "value": SECRET},
        {"name": "HF_TOKEN", "mysterybox_secret": {"secret_id": HF_SECRET}},
        {
            "name": "WANDB_API_KEY",
            "mysterybox_secret": {"version_id": WANDB_VERSION},
        },
    ]
    assert spec["registry_credentials"] == {
        "mysterybox_secret_version": REGISTRY
    }
    assert spec["preemptible"] is True
    assert spec["spot_pricing_policy"] == POLICY
    assert "follows_spot_price" not in spec


def test_job_defaults_leave_the_new_fields_out() -> None:
    fake = FakeTransport()
    job = _job()
    job.create(region="eu-north1", project="p", transport=fake)
    spec = _spec(fake.bodies[0])
    for key in (
        "restart_attempts",
        "working_dir",
        "registry_credentials",
        "environment_variables",
        "spot_pricing_policy",
    ):
        assert key not in spec
    described = job.create(dry_run=True, region="eu-north1", project="p")
    assert isinstance(described, dict)
    for key in (
        "retries",
        "working_dir",
        "secrets",
        "registry_secret",
        "spot_policy",
    ):
        assert key not in described


def test_job_dry_run_shows_secret_ids_not_values() -> None:
    described = _full_job().create(
        dry_run=True, region="eu-north1", project="p"
    )
    assert isinstance(described, dict)
    assert described["retries"] == 3
    assert described["working_dir"] == "/workspace"
    assert described["env"] == ["PLAIN"]
    assert described["secrets"] == {
        "HF_TOKEN": HF_SECRET,
        "WANDB_API_KEY": WANDB_VERSION,
    }
    assert described["registry_secret"] == REGISTRY
    assert described["spot_policy"] == POLICY
    assert SECRET not in repr(described)


def test_build_create_request_maps_the_new_fields() -> None:
    fake = FakeTransport()
    _full_job().create(region="eu-north1", project="p", transport=fake)
    spec = build_create_request(fake.bodies[0]).spec
    assert spec.restart_attempts == 3
    assert spec.working_dir == "/workspace"
    plain, hf, wandb = spec.environment_variables
    assert (plain.name, plain.value) == ("PLAIN", SECRET)
    assert hf.name == "HF_TOKEN"
    assert hf.mysterybox_secret.secret_id == HF_SECRET
    assert hf.value == ""
    assert wandb.mysterybox_secret.version_id == WANDB_VERSION
    assert spec.registry_credentials.mysterybox_secret_version == REGISTRY
    assert spec.registry_credentials.password == ""
    assert spec.preemptible is True
    assert spec.pricing_model.spot_pricing_policy.id == POLICY
    payload = build_create_request(fake.bodies[0]).to_json(
        preserving_proto_field_name=True
    )
    assert "follows_spot_price" not in payload
    assert "followsSpotPrice" not in payload


# Endpoint mapping


def test_endpoint_body_uses_the_token_secret() -> None:
    fake = FakeEndpointTransport()
    service = _full_endpoint().create(
        region="eu-north1", project="p", transport=fake
    )
    assert isinstance(service, ns.Service)
    assert service.token is None
    spec = _spec(fake.bodies[0])
    assert "auth_token" not in spec
    assert spec["auth_token_mysterybox_secret"] == {"secret_id": TOKEN_SECRET}
    assert spec["working_dir"] == "/srv"
    assert spec["environment_variables"] == [
        {"name": "HF_TOKEN", "mysterybox_secret": {"secret_id": HF_SECRET}},
    ]
    assert spec["registry_credentials"] == {
        "mysterybox_secret_version": REGISTRY
    }
    assert spec["spot_pricing_policy"] == POLICY
    assert "follows_spot_price" not in spec
    assert "restart_attempts" not in spec


def test_endpoint_dry_run_names_the_token_secret() -> None:
    described = _full_endpoint().create(
        dry_run=True, region="eu-north1", project="p"
    )
    assert isinstance(described, dict)
    assert described["token"] == "secret"
    assert described["token_secret"] == TOKEN_SECRET
    assert described["working_dir"] == "/srv"
    assert described["secrets"] == {"HF_TOKEN": HF_SECRET}
    assert described["registry_secret"] == REGISTRY
    assert described["spot_policy"] == POLICY


def test_build_endpoint_request_maps_the_new_fields() -> None:
    fake = FakeEndpointTransport()
    _full_endpoint().create(region="eu-north1", project="p", transport=fake)
    spec = build_endpoint_request(fake.bodies[0]).spec
    assert spec.auth_token == ""
    assert spec.auth_token_mysterybox_secret.secret_id == TOKEN_SECRET
    assert spec.working_dir == "/srv"
    assert spec.environment_variables[0].mysterybox_secret.secret_id == (
        HF_SECRET
    )
    assert spec.registry_credentials.mysterybox_secret_version == REGISTRY
    assert spec.pricing_model.spot_pricing_policy.id == POLICY


def test_plain_spot_still_follows_the_spot_price() -> None:
    fake = FakeEndpointTransport()
    _endpoint(pricing="spot").create(
        region="eu-north1", project="p", transport=fake
    )
    spec = build_endpoint_request(fake.bodies[0]).spec
    assert spec.pricing_model.spot_pricing_policy is None
    payload = build_endpoint_request(fake.bodies[0]).to_json(
        preserving_proto_field_name=True
    )
    assert "follows_spot_price" in payload or "followsSpotPrice" in payload


# YAML


def test_job_yaml_accepts_the_new_keys(tmp_path: Path) -> None:
    path = tmp_path / "job.yaml"
    path.write_text(
        "schema_version: 1\n"
        "kind: job\n"
        f"image: {CUDA}\n"
        "command: [nvidia-smi]\n"
        "gpu: h100\n"
        "timeout: 1h\n"
        "pricing: spot\n"
        f"spot_policy: {POLICY}\n"
        "retries: 3\n"
        "working_dir: /workspace\n"
        "env: {PLAIN: value}\n"
        f"secrets: {{HF_TOKEN: {HF_SECRET}, WANDB_API_KEY: {WANDB_VERSION}}}\n"
        f"registry_secret: {REGISTRY}\n",
        encoding="utf-8",
    )
    job = ns.Job.from_yaml(path)
    assert job.retries == 3
    assert job.working_dir == "/workspace"
    assert job.secrets == (
        ("HF_TOKEN", HF_SECRET),
        ("WANDB_API_KEY", WANDB_VERSION),
    )
    assert job.registry_secret == REGISTRY
    assert job.spot_policy == POLICY


def test_endpoint_yaml_accepts_the_new_keys(tmp_path: Path) -> None:
    path = tmp_path / "endpoint.yaml"
    path.write_text(
        "schema_version: 1\n"
        "kind: endpoint\n"
        f"image: {VLLM}\n"
        "port: 8000\n"
        "gpu: l40s\n"
        "auth: token\n"
        f"token_secret: {TOKEN_SECRET}\n"
        "pricing: spot\n"
        f"spot_policy: {POLICY}\n"
        "working_dir: /srv\n"
        f"secrets: {{HF_TOKEN: {HF_SECRET}}}\n"
        f"registry_secret: {REGISTRY}\n",
        encoding="utf-8",
    )
    endpoint = ns.Endpoint.from_yaml(path)
    assert endpoint.token_secret == TOKEN_SECRET
    assert endpoint.working_dir == "/srv"
    assert endpoint.secrets == (("HF_TOKEN", HF_SECRET),)
    assert endpoint.registry_secret == REGISTRY
    assert endpoint.spot_policy == POLICY


def test_job_yaml_rejects_endpoint_only_keys(tmp_path: Path) -> None:
    path = tmp_path / "job.yaml"
    path.write_text(
        "schema_version: 1\n"
        "kind: job\n"
        f"image: {CUDA}\n"
        "command: [nvidia-smi]\n"
        "timeout: 1h\n"
        f"token_secret: {TOKEN_SECRET}\n",
        encoding="utf-8",
    )
    with pytest.raises(ns.ValidationError, match="token_secret"):
        ns.Job.from_yaml(path)
