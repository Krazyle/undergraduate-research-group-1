import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

os.environ["MINERU_LMDEPLOY_BACKEND"] = "turbomind"

INDEX_PATH = Path("papers/index.json")
PDF_DIR = Path("papers/pdf")
MD_DIR = Path("papers/markdown")
MD_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 10

papers = json.loads(INDEX_PATH.read_text(encoding="utf-8"))

pending = []
for paper in papers:
    key = paper["citation_key"]
    pdf_file = PDF_DIR / f"{key}.pdf"
    md_file = MD_DIR / f"{key}.md"

    if pdf_file.exists() and not md_file.exists():
        pending.append((key, paper.get("title", ""), pdf_file, md_file))

if not pending:
    print("Found 0 paper(s) to convert.")
    exit(0)

print(f"Found {len(pending)} paper(s) to convert.\n")

for i in range(0, len(pending), BATCH_SIZE):
    batch = pending[i : i + BATCH_SIZE]
    batch_num = (i // BATCH_SIZE) + 1
    total_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"--- Batch [{batch_num}/{total_batches}] ({len(batch)} papers) ---")
    for key, title, _, _ in batch:
        print(f"  • [{key}] {title}")

    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="mineru_batch_") as tmp:
        tmp_dir = Path(tmp)
        pdf_paths = [str(pdf_file) for _, _, pdf_file, _ in batch]

        cmd = [
            "mineru-kit",
            "parse",
            "-o",
            str(tmp_dir),
            *pdf_paths,
            "--tier",
            "standard",
            "--disable-image-analysis",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if res.returncode != 0:
            print(f"  ✗ Batch failed ({res.returncode}): {res.stderr.strip()[:300]}")
            print("  Falling back to individual processing for this batch...")
            for key, title, pdf_file, md_file in batch:
                t_sub = time.time()
                with tempfile.TemporaryDirectory(prefix=f"mineru_fb_{key}_") as fb_tmp:
                    fb_dir = Path(fb_tmp)
                    sub_cmd = [
                        "mineru-kit",
                        "parse",
                        "-o",
                        str(fb_dir),
                        str(pdf_file),
                        "--tier",
                        "standard",
                        "--disable-image-analysis",
                    ]
                    sub_res = subprocess.run(sub_cmd, capture_output=True, text=True, check=False)
                    candidates = list(fb_dir.rglob("*.md"))
                    if sub_res.returncode == 0 and candidates:
                        shutil.copy2(candidates[0], md_file)
                        print(f"    ✓ Converted [{key}] in {time.time() - t_sub:.1f}s")
                    else:
                        print(f"    ✗ Failed [{key}]")
            continue

        for key, _, _, md_file in batch:
            generated = tmp_dir / f"{key}.md"
            if not generated.exists():
                candidates = list(tmp_dir.rglob(f"{key}.md"))
                generated = candidates[0] if candidates else None

            if generated and generated.exists():
                shutil.copy2(generated, md_file)
                print(f"  ✓ Saved: {md_file.name}")
            else:
                print(f"  ✗ Missing output for: {key}")

        elapsed = time.time() - t0
        print(f"Batch [{batch_num}/{total_batches}] finished in {elapsed:.1f}s\n")

print("\nFinished converting PDFs.")
