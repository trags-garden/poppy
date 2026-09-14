"""The device-code key pair — keygen + decrypt of the API key.

The PRODUCTION encryptor is the Next.js server (Node `publicEncrypt` with
RSA_PKCS1_OAEP_PADDING + oaepHash 'sha256'), proven to interoperate with this
decrypt path. These unit tests encrypt with the *same* OAEP params in Python so
they run without Node, and guard the Poppy side against parameter drift.
"""

import base64

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from poppy.setup.trags import _OAEP, _decrypt_api_key, _generate_device_keypair


def test_generate_keypair_returns_spki_pem():
    private_key, public_pem = _generate_device_keypair()
    assert public_pem.startswith("-----BEGIN PUBLIC KEY-----")
    assert public_pem.strip().endswith("-----END PUBLIC KEY-----")
    # The PEM round-trips back to a usable public key.
    serialization.load_pem_public_key(public_pem.encode())
    assert private_key.key_size == 2048


def test_encrypt_decrypt_roundtrip():
    private_key, public_pem = _generate_device_keypair()
    pub = serialization.load_pem_public_key(public_pem.encode())
    raw = "usr_abcdefghijklmnopqrstuvwxyz012345"
    # Mirror the server's OAEP-SHA256 params exactly.
    ciphertext = pub.encrypt(
        raw.encode(),
        padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
    )
    assert _decrypt_api_key(private_key, base64.b64encode(ciphertext).decode()) == raw


def test_module_oaep_matches_decrypt_path():
    # The shared _OAEP constant is what _decrypt_api_key uses; encrypting with it
    # must round-trip, so a future edit that desyncs the params fails here.
    private_key, public_pem = _generate_device_keypair()
    pub = serialization.load_pem_public_key(public_pem.encode())
    ciphertext = pub.encrypt(b"usr_token", _OAEP)
    assert _decrypt_api_key(private_key, base64.b64encode(ciphertext).decode()) == "usr_token"


def test_decrypt_rejects_garbage():
    private_key, _ = _generate_device_keypair()
    # Not valid base64 / not valid ciphertext — must raise so the setup flow can
    # surface a clean ClickException rather than writing a corrupt key.
    with pytest.raises(Exception):
        _decrypt_api_key(private_key, "this is not base64 ciphertext!!!")


@pytest.mark.parametrize("location", ["config-file", "keychain-unreadable"])
def test_connected_line_names_the_resolved_config_path(tmp_path, monkeypatch, capsys, location):
    """The success line names the real config.json, not a literal ~/.poppy.

    And only when the key is actually there: a session that cannot read the
    keychain does not know where the key landed, so it claims no location.
    """
    import httpx

    import poppy.config as config_module
    from poppy import keychain
    from poppy.setup import trags as setup_trags

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    # No usable backend, so the key falls back to config.json and the line that
    # names a path is the one under test.
    monkeypatch.setattr(keychain, "available", lambda: False)
    monkeypatch.setattr(setup_trags.webbrowser, "open", lambda *_a, **_k: True)
    if location != "config-file":
        monkeypatch.setattr(config_module, "trags_api_key_location", lambda _poppy_dir: location)

    key = "usr_from_device_flow"
    device_pubkey = {}

    class _Response:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self.text = ""
            self._payload = payload

        def json(self):
            return self._payload

    def fake_post(url, **kwargs):
        device_pubkey["key"] = serialization.load_pem_public_key(kwargs["json"]["device_pubkey"].encode())
        return _Response(201, {"code": "WXYZ", "setup_url": "https://example.test/cli", "poll_interval_seconds": 0.01})

    def fake_get(url, **kwargs):
        ciphertext = device_pubkey["key"].encrypt(key.encode(), _OAEP)
        return _Response(200, {"api_key_encrypted": base64.b64encode(ciphertext).decode()})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)

    setup_trags.run_device_code_flow()

    out = capsys.readouterr().out
    if location == "config-file":
        assert f"Your API key is saved in {tmp_path / 'config.json'} (0600)." in out
    else:
        assert "This session could not confirm where the key was stored" in out
        assert "saved in" not in out
    assert "~/.poppy/config.json" not in out
    assert key not in out
