import os
from pathlib import Path


def store_environment() -> dict[str, str]:
    """Capture an explicit store override for a client or service process."""
    value = os.environ.get("POPPY_DIR")
    return {"POPPY_DIR": str(Path(value).resolve())} if value else {}
