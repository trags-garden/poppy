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
