import subprocess
from pathlib import Path

FIGURES_DIR = Path("figures")

for source in sorted(FIGURES_DIR.glob("fig-*.typ")):
    output = source.with_suffix(".pdf")
    subprocess.run(["quarto", "typst", "compile", str(source), str(output)], check=True)
    print(f"Built {output}")
