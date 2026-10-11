import json
import time
import urllib.request
from pathlib import Path

MD_DIR, PDF_DIR = Path("papers/markdown"), Path("papers/pdf")
PDF_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
papers = json.loads(Path("papers/index.json").read_text(encoding="utf-8"))

SPECIAL_MAP = {
    "alam2021crisisbench": "https://ojs.aaai.org/index.php/ICWSM/article/download/18115/17918",
    "nguyen2017robust": "https://ojs.aaai.org/index.php/ICWSM/article/download/14950/14800",
    "ermino2022urgency": "https://stepacademic.net/ijcsr/article/download/289/114/",
    "caoile2025urdaneta": "https://theaspd.com/index.php/ijes/article/download/5695/4129",
    "yute2021satellite": "https://isprs-archives.copernicus.org/articles/XLVI-4-W6-2021/329/2021/isprs-archives-XLVI-4-W6-2021-329-2021.pdf",
}


def get_open_access_pdf_url(paper):
    doi, aid, key = paper.get("doi", ""), paper.get("arxiv_id"), paper["citation_key"]
    if aid:
        return f"https://arxiv.org/pdf/{aid}.pdf"
    if key in SPECIAL_MAP:
        return SPECIAL_MAP[key]
    if "10.18653/v1/" in doi:
        return f"https://aclanthology.org/{doi.split('10.18653/v1/')[-1]}.pdf"
    if doi.startswith("acl:"):
        return f"https://aclanthology.org/{doi.split('acl:')[-1]}.pdf"
    if "10.1371/journal.pone." in doi:
        return f"https://journals.plos.org/plosone/article/file?id={doi}&type=printable"
    if "jmlr:" in doi:
        v, n = doi.split("jmlr:")[-1].split("/")
        return f"https://www.jmlr.org/papers/{v.replace('v', 'volume')}/{n}/{n}.pdf"
    if "10.24963/ijcai." in doi:
        y, p = doi.split("ijcai.")[-1].split("/")
        return f"https://www.ijcai.org/proceedings/{y}/{int(p):04d}.pdf"
    if "10.1007/" in doi or "10.1023/" in doi:
        return f"https://link.springer.com/content/pdf/{doi}.pdf"
    return None


def fetch(url, timeout=30):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


downloaded, existing_markdown, existing_pdf, excluded_paywalled = [], [], [], []

for paper in papers:
    key, title, doi = paper["citation_key"], paper["title"], paper.get("doi", "")
    out_file = PDF_DIR / f"{key}.pdf"

    if (MD_DIR / f"{key}.md").exists():
        existing_markdown.append(key)
        continue
    if out_file.exists():
        existing_pdf.append(key)
        continue

    url = get_open_access_pdf_url(paper)
    if not url:
        excluded_paywalled.append((key, doi, title))
        continue

    content = fetch(url)
    if content:
        out_file.write_bytes(content)
        downloaded.append((key, len(content)))
        print(f"✓ Downloaded: [{key}] ({len(content) // 1024} KB)")
        time.sleep(1)
    else:
        print(f"✗ Failed to download [{key}] from {url}")

print("\n" + "=" * 50)
print(f"Summary: {len(downloaded)} downloaded, {len(existing_markdown)} already in Markdown, {len(existing_pdf)} already in PDF, {len(excluded_paywalled)} paywalled/excluded.")
if excluded_paywalled:
    print(f"\n[Paywalled / Protected - Excluded ({len(excluded_paywalled)})]:")
    for key, doi, title in excluded_paywalled:
        print(f"  - [{key}] ({doi}): {title}")
