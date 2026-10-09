"""Regenerate SHA-256 digests after a deliberate evidence update."""
from hashlib import sha256
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "evidence"
manifest = root / "SHA256SUMS.txt"
files = sorted(
    p for p in root.rglob("*")
    if p.is_file() and p != manifest and p.name != "sha256_manifest.json"
)
manifest.write_text("".join(
    f"{sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}\n"
    for p in files
), encoding="utf-8")
print(f"Recorded {len(files)} evidence files")
