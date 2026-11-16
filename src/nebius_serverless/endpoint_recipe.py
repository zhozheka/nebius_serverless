"""Validate an endpoint recipe and map it to a create request."""

import secrets
from collections.abc import Sequence

from .errors import ValidationError
from .recipe import (
    DISK_DEFAULT,
    SHM_BYTES,
    SHM_LABEL,
    describe_extras,
    parse_volume,
    quoted_args,
    secret_ref,
    spec_extras,
    validate_command,
    validate_secret_ref,
)

AUTH_MODES = ("none", "token")
TOKEN_GENERATED = "generated"
TOKEN_PROVIDED = "provided"
TOKEN_SECRET = "secret"
_OPERATION = "Endpoint"


def validate_optional_command(command: object) -> tuple[str, ...] | None:
    """``None`` keeps the image entrypoint. A list overrides it."""
    if command is None:
        return None
    return validate_command(command, operation=_OPERATION)


def validate_port(port: object) -> int:
    """One HTTP container port, an integer from 1 to 65535."""
    if (
        isinstance(port, bool)
        or not isinstance(port, int)
        or port < 1
        or port > 65535
    ):
        raise ValidationError(
            "port must be an integer from 1 to 65535.",
            operation=_OPERATION,
        )
    return port


def validate_auth(auth: object) -> str:
    """Accept ``none`` or ``token``."""
    if auth not in AUTH_MODES:
        raise ValidationError(
            "auth must be 'none' or 'token'.",
            operation=_OPERATION,
        )
    return auth


def validate_token(token: object, auth: str) -> str | None:
    """A bearer token only makes sense with ``auth='token'``.

    The value is never echoed in the error text.
    """
    if token is None:
        return None
    if not isinstance(token, str) or token == "":
        raise ValidationError(
            "token must be a nonempty string.",
            operation=_OPERATION,
        )
    if auth != "token":
        raise ValidationError(
            "token requires auth='token'.",
            operation=_OPERATION,
        )
    return token


def validate_token_secret(
    token_secret: object,
    auth: str,
    token: str | None,
) -> str | None:
    """A MysteryBox secret holding ``AUTH_TOKEN``. Excludes ``token``."""
    if token_secret is None:
        return None
    value = validate_secret_ref(
        token_secret, "token_secret", operation=_OPERATION
    )
    if auth != "token":
        raise ValidationError(
            "token_secret requires auth='token'.",
            operation=_OPERATION,
        )
    if token is not None:
        raise ValidationError(
            "Set token or token_secret, not both.",
            operation=_OPERATION,
        )
    return value


def generate_token() -> str:
    """A random bearer token, as the CLI makes for ``--auth token``."""
    return secrets.token_urlsafe(32)


def describe_endpoint(
    *,
    command: Sequence[str] | None,
    image: str,
    port: int,
    project: str,
    region: str,
    platform: str,
    preset: str,
    pricing: str,
    auth: str,
    token_supplied: bool,
    subnet_id: str | None,
    volumes: Sequence[str],
    inject_files: Sequence[str],
    env_keys: Sequence[str],
    name: str,
    disk: str | None,
    token_secret: str | None = None,
    working_dir: str | None = None,
    secrets: Sequence[tuple[str, str]] = (),
    registry_secret: str | None = None,
    spot_policy: str | None = None,
) -> dict[str, object]:
    """Build the dry-run description. Env values and tokens are left out."""
    description: dict[str, object] = {"image": image}
    if command is not None:
        description["command"] = list(command)
        args = quoted_args(command)
        if args is not None:
            description["args"] = args
    description["port"] = port
    description["project"] = project
    description["region"] = region
    description["platform"] = platform
    description["preset"] = preset
    description["pricing"] = pricing
    description["auth"] = auth
    if token_secret is not None:
        description["token"] = TOKEN_SECRET
        description["token_secret"] = token_secret
    elif auth == "token":
        description["token"] = (
            TOKEN_PROVIDED if token_supplied else TOKEN_GENERATED
        )
    if subnet_id is not None:
        description["subnet_id"] = subnet_id
    if volumes:
        description["volumes"] = list(volumes)
    if inject_files:
        description["inject_files"] = list(inject_files)
    description["env"] = list(env_keys)
    description["name"] = name
    description["disk"] = disk if disk is not None else DISK_DEFAULT
    description["shm"] = SHM_LABEL
    describe_extras(
        description,
        working_dir=working_dir,
        secrets=secrets,
        registry_secret=registry_secret,
        spot_policy=spot_policy,
    )
    return description


def endpoint_request_body(
    *,
    command: Sequence[str] | None,
    image: str,
    port: int,
    project: str,
    platform: str,
    preset: str,
    pricing: str,
    token: str | None,
    subnet_id: str | None,
    volumes: Sequence[str],
    inject_files: Sequence[dict[str, object]],
    env: Sequence[tuple[str, str]],
    name: str,
    disk: str | None,
    token_secret: str | None = None,
    working_dir: str | None = None,
    secrets: Sequence[tuple[str, str]] = (),
    registry_secret: str | None = None,
    spot_policy: str | None = None,
) -> dict[str, object]:
    """Map a recipe to the create body the endpoint transport sends."""
    spec: dict[str, object] = {"image": image}
    if command is not None:
        spec["container_command"] = command[0]
        args = quoted_args(command)
        if args is not None:
            spec["args"] = args
    spec["ports"] = [{"container_port": port, "protocol": "HTTP"}]
    spec["platform"] = platform
    spec["preset"] = preset
    spec["disk"] = disk if disk is not None else DISK_DEFAULT
    spec["shm_size_bytes"] = SHM_BYTES
    if subnet_id is not None:
        spec["subnet_id"] = subnet_id
    if volumes:
        spec["volumes"] = [parse_volume(item) for item in volumes]
    if inject_files:
        spec["injected_files"] = list(inject_files)
    spec_extras(
        spec,
        pricing=pricing,
        env=env,
        working_dir=working_dir,
        secrets=secrets,
        registry_secret=registry_secret,
        spot_policy=spot_policy,
    )
    if token is not None:
        spec["auth_token"] = token
    if token_secret is not None:
        spec["auth_token_mysterybox_secret"] = secret_ref(token_secret)
    return {
        "metadata": {"parent_id": project, "name": name},
        "spec": spec,
    }
