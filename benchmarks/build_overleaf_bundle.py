"""Package the manuscript for Overleaf: one flat folder, figure paths rewritten, zipped."""
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "paper" / "overleaf"


def main():
    if DEST.exists():
        shutil.rmtree(DEST)
    (DEST / "figures").mkdir(parents=True)
    tex = (ROOT / "paper" / "gp_resource_allocation.tex").read_text(encoding="utf-8").replace("../figures/", "figures/")
    (DEST / "main.tex").write_text(tex, encoding="utf-8")
    for name in ("novelty_tables.tex", "novelty_numbers.tex"):
        shutil.copy(ROOT / "paper" / name, DEST / name)
    for figure in (ROOT / "figures").glob("novelty_*.pdf"):
        shutil.copy(figure, DEST / "figures" / figure.name)
    archive = ROOT / "paper" / "gp_resource_allocation_overleaf.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in DEST.rglob("*"):
            if path.is_file():
                handle.write(path, path.relative_to(DEST))
    print(archive, sorted(str(p.relative_to(DEST)) for p in DEST.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main()
