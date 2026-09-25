# Security Policy

## Local data at rest

By default Poppy's local store (`~/.poppy/memories.db`) is a plaintext SQLite
file protected by owner-only file permissions. You can optionally encrypt it at
rest with a key in the OS keychain: install the extra (`pipx inject poppy-memory
sqlcipher3`, or `pip install 'poppy-memory[encryption]'` for pip-based installs)
then run `poppy encrypt enable`. This is local encryption at rest, not
zero-knowledge or end-to-end encryption; the key lives in your keychain and
anything running as your user that can read it can decrypt the store. See the
README section "Encryption at rest" for details.

## Supported versions

Only the latest release of `poppy-memory` receives security fixes. If you are
on an older version, upgrade to the latest release first.

## Reporting a vulnerability

Please do not report security vulnerabilities through public GitHub issues.

Report them privately, either way works:

- Open a private report on GitHub:
  https://github.com/trags-garden/poppy/security/advisories/new
- Email **security@trags.ai**

Include the Poppy version (`poppy --version`), reproduction steps, and the
impact as you understand it.

You can expect an initial response within a few days. We will keep you
informed while the report is investigated and fixed.
