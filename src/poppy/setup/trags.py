"""One-command device-code onboarding for Trags cloud sync.

Flow:
1. Generate an ephemeral RSA key pair for this setup attempt.
2. POST /api/cli-setup/start (anonymous) with the PUBLIC key — get a one-time
   code + setup URL.
3. Open the browser to the setup URL.
4. Poll /api/cli-setup/poll?code=... until the *encrypted* API key arrives.
5. Decrypt it with the private key (which never leaves this process) and write
   it into Poppy's config.json via the standard config-set path.

The key is encrypted end-to-end (RSA-OAEP-SHA256). The server never
stores it plaintext and the unauthenticated poll endpoint only ever returns
ciphertext, so a leaked setup code or compromised DB yields nothing usable.
"""

from __future__ import annotations

import base64
import time
import webbrowser

import click
import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from poppy.config import load_config, save_config
from poppy.runtime import get_poppy_dir

_DEFAULT_POLL_INTERVAL_S = 2.0
_DEFAULT_TIMEOUT_S = 5 * 60
# OAEP-SHA256 padding for a 2048-bit key can wrap up to 190 bytes — far more
# than a usr_* key — and matches the Node `publicEncrypt` params on the server
# (web/app/api/cli-setup/authorize/route.ts).
_OAEP = padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)


def _generate_device_keypair() -> tuple[rsa.RSAPrivateKey, str]:
    """Return (private_key, public_key_pem) for one device-code attempt."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("ascii")
    )
    return private_key, public_pem


def _decrypt_api_key(private_key: rsa.RSAPrivateKey, api_key_encrypted: str) -> str:
    """Decrypt the base64 RSA-OAEP-SHA256 ciphertext returned by /poll."""
    ciphertext = base64.b64decode(api_key_encrypted)
    return private_key.decrypt(ciphertext, _OAEP).decode("utf-8")


def run_device_code_flow(api_url_override: str | None = None) -> None:
    """Walk the device-code flow end-to-end. Prints progress to stdout."""
    poppy_dir = get_poppy_dir()
    cfg = load_config(poppy_dir)
    base_url = (api_url_override or cfg.trags_api_url).rstrip("/")

    click.echo(f"Connecting to {base_url}…")

    # Ephemeral key pair for this attempt. The private key stays in memory and
    # is discarded when the flow ends; only the public key is sent to the server.
    private_key, public_pem = _generate_device_keypair()

    try:
        start = httpx.post(
            f"{base_url}/api/cli-setup/start",
            json={"device_pubkey": public_pem},
            timeout=15.0,
        )
    except httpx.HTTPError as e:
        raise click.ClickException(f"Could not reach Trags at {base_url}: {e}") from e
    if start.status_code != 201:
        raise click.ClickException(f"Failed to start setup (HTTP {start.status_code}): {start.text[:200]}")

    body = start.json()
    code = body["code"]
    setup_url = body["setup_url"]
    poll_interval = float(body.get("poll_interval_seconds") or _DEFAULT_POLL_INTERVAL_S)

    click.echo("\nOpening your browser to authorize this machine.")
    click.echo(f"  URL:  {setup_url}")
    click.echo(f"  Code: {code}")
    click.echo("\nIf the browser doesn't open, paste the URL into it yourself.")
    try:
        webbrowser.open(setup_url)
    except Exception:
        pass  # Best-effort; user has the URL.

    from poppy import telemetry

    device_id = telemetry.get_device_id(poppy_dir)
    poll_params: dict[str, str] = {"code": code}
    if device_id:
        poll_params["device_id"] = device_id

    click.echo("\nWaiting for authorization…")
    deadline = time.monotonic() + _DEFAULT_TIMEOUT_S
    while time.monotonic() < deadline:
        time.sleep(poll_interval)
        try:
            poll = httpx.get(
                f"{base_url}/api/cli-setup/poll",
                params=poll_params,
                timeout=15.0,
            )
        except httpx.HTTPError:
            continue  # Transient network blip; keep polling.

        if poll.status_code == 200:
            api_key_encrypted = poll.json().get("api_key_encrypted")
            if not api_key_encrypted:
                raise click.ClickException("Server returned an empty API key.")
            try:
                api_key = _decrypt_api_key(private_key, api_key_encrypted)
            except Exception as e:  # noqa: BLE001 — surface any decode/decrypt failure cleanly
                raise click.ClickException(
                    "Could not decrypt the API key the server returned. Run `poppy setup trags` again."
                ) from e
            cfg.set("trags-api-key", api_key)
            if api_url_override:
                cfg.set("trags-api-url", base_url)
            save_config(cfg)
            click.echo("\n✓ Connected. Your API key is saved in ~/.poppy/config.json.")
            click.echo("  Try: poppy sync push")
            return

        if poll.status_code == 202:
            continue

        if poll.status_code in (404, 410):
            raise click.ClickException("Setup code expired or already used. Run `poppy setup trags` again.")

        raise click.ClickException(f"Unexpected response while polling (HTTP {poll.status_code}): {poll.text[:200]}")

    raise click.ClickException("Timed out waiting for authorization. Run `poppy setup trags` again.")
