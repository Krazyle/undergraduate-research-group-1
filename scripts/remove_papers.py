#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

index_path = Path("papers/index.json")
bib_path = Path("bibliography.bib")
pdf_dir = Path("papers/pdf")
md_dir = Path("papers/markdown")

target_keys = set(sys.argv[1:])

if target_keys:
    if index_path.exists():
        papers = json.loads(index_path.read_text(encoding="utf-8"))
        papers = [p for p in papers if p.get("citation_key") not in target_keys]
        index_path.write_text(json.dumps(papers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if bib_path.exists():
        entries = [e.strip() for e in re.split(r"\n(?=@)", bib_path.read_text(encoding="utf-8")) if e.strip()]
        kept = []
        for e in entries:
            m = re.match(r"@\w+\{([^,]+),", e)
            if not (m and m.group(1).strip() in target_keys):
                kept.append(e)
        bib_path.write_text("\n\n".join(kept).strip() + "\n", encoding="utf-8")

active_keys = set()
if index_path.exists():
    papers = json.loads(index_path.read_text(encoding="utf-8"))
    active_keys = {p["citation_key"] for p in papers}

if pdf_dir.exists():
    for f in pdf_dir.glob("*.pdf"):
        if f.stem not in active_keys:
            f.unlink()
            print(f"Deleted {f}")

if md_dir.exists():
    for f in md_dir.glob("*.md"):
        if f.stem not in active_keys:
            f.unlink()
            print(f"Deleted {f}")
