"""Verify a downloaded HF model is complete: every shard present, sizes and
sha256 matching the Hub's LFS metadata. Exit 1 on any mismatch.

    python verify_weights.py Qwen/Qwen3-8B
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main(model: str) -> int:
    local = Path(snapshot_download(model, local_files_only=True))
    info = HfApi().model_info(model, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}

    index = json.loads((local / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    print(f"{model} @ {info.sha}: {len(shards)} shards listed in the index")

    bad = 0
    for name in shards + ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        path = local / name
        meta = remote.get(name)
        if not path.exists():
            print(f"  MISSING {name}")
            bad += 1
            continue
        size = path.stat().st_size
        if meta is not None and meta.size is not None and size != meta.size:
            print(f"  SIZE MISMATCH {name}: {size} local vs {meta.size} hub")
            bad += 1
            continue
        if meta is not None and meta.lfs is not None:
            digest = sha256(path)
            if digest != meta.lfs.sha256:
                print(f"  SHA256 MISMATCH {name}")
                bad += 1
                continue
            print(f"  ok {name} ({size / 1e9:.2f} GB, sha256 verified)")
        else:
            print(f"  ok {name} ({size} bytes)")
    print("ALL WEIGHTS VERIFIED" if not bad else f"{bad} PROBLEM(S) FOUND")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"))
