"""Check saved evidence against the committed SHA-256 manifest."""

from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
MANIFEST = EVIDENCE / "SHA256SUMS.txt"


def main() -> None:
    recorded = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, path = line.split("  ", 1)
        if path in recorded:
            raise ValueError(f"duplicate manifest path: {path}")
        recorded[path] = digest
    actual_files = {
        p.relative_to(EVIDENCE).as_posix(): p
        for p in EVIDENCE.rglob("*")
        if p.is_file() and p != MANIFEST and p.name != "sha256_manifest.json"
    }
    if set(recorded) != set(actual_files):
        missing = sorted(set(recorded) - set(actual_files))
        extra = sorted(set(actual_files) - set(recorded))
        raise ValueError(f"manifest/file mismatch: missing={missing[:5]}, extra={extra[:5]}")
    for path, expected in recorded.items():
        actual = sha256(actual_files[path].read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch: {path}")
    print(f"Verified {len(recorded)} evidence files.")


if __name__ == "__main__":
    main()
