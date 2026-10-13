#!/usr/bin/env python
"""Re-download the lab images listed in data/images.json and verify their SHA-256.

The images are already committed in ``data/images``; this script only exists so the
data can be restored (and its integrity checked) from the original sources.
"""

from __future__ import annotations

import hashlib
import sys

import requests

from xai_lab import config, data


def main() -> int:
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    status = 0
    for spec in data.load_manifest():
        if not spec.path.exists():
            r = requests.get(spec.source_url, headers={"User-Agent": "uvg-xai-lab/1.0 (student project)"}, timeout=60)
            r.raise_for_status()
            spec.path.write_bytes(r.content)
            print(f"downloaded {spec.filename}")
        digest = hashlib.sha256(spec.path.read_bytes()).hexdigest()
        ok = digest == spec.sha256
        print(f"{'OK ' if ok else 'BAD'} {spec.filename}")
        status |= 0 if ok else 1
    return status


if __name__ == "__main__":
    sys.exit(main())
