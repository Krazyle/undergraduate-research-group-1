import json, re, time, urllib.request
from pathlib import Path
from bs4 import BeautifulSoup
import html2text

MD_DIR, PDF_DIR = Path("papers/markdown"), Path("papers/pdf")
MD_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
converter = html2text.HTML2Text()
converter.body_width, converter.skip_internal_links, converter.protect_links = 0, True, True
papers = json.loads(Path("papers/index.json").read_text(encoding="utf-8"))


def fetch(url, timeout=20):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


def parse_html_article(html):
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article", class_="ltx_document") or soup.find("article")
    if not article:
        return None
    for math in article.find_all("math"):
        ann = math.find("annotation", attrs={"encoding": "application/x-tex"})
        tex = ann.get_text().strip() if ann else math.get_text().strip()
        math.replace_with(f" ${tex}$ " if math.get("display") != "block" else f"\n$$\n{tex}\n$$\n")
    return re.sub(r"\n{3,}", "\n\n", converter.handle(str(article))).strip()


converted_html, downloaded_pdf, existing_markdown, unavailable, not_arxiv = [], [], [], [], []

for paper in papers:
    key, title, aid = paper["citation_key"], paper["title"], paper.get("arxiv_id")
    md_file, pdf_file = MD_DIR / f"{key}.md", PDF_DIR / f"{key}.pdf"

    if md_file.exists():
        existing_markdown.append(key)
        continue
    if not aid:
        not_arxiv.append((key, title))
        continue

    raw_html = fetch(f"https://arxiv.org/html/{aid}")
    md = parse_html_article(raw_html.decode("utf-8", "replace")) if raw_html else None
    if md:
        md_file.write_text(md, encoding="utf-8")
        converted_html.append((key, title))
        print(f"✓ Converted via HTML: [{key}] {title}")
        time.sleep(1)
        continue

    if pdf_file.exists():
        print(f"• PDF already present: [{key}] {title}")
        downloaded_pdf.append((key, title))
        continue

    raw_pdf = fetch(f"https://arxiv.org/pdf/{aid}.pdf")
    if raw_pdf and raw_pdf.startswith(b"%PDF"):
        pdf_file.write_bytes(raw_pdf)
        downloaded_pdf.append((key, title))
        print(f"✓ Downloaded PDF: [{key}] {title}")
        time.sleep(1)
        continue

    unavailable.append((aid, key, title))
    print(f"✗ Skipped (HTML and PDF unavailable): [{key}] {title}")

print("\n" + "=" * 50)
print(f"Summary: {len(converted_html)} converted via HTML, {len(downloaded_pdf)} PDF saved/present, {len(existing_markdown)} already in Markdown, {len(unavailable)} unavailable/skipped, {len(not_arxiv)} non-arXiv.")
for header, items in [
    ("[arXiv HTML and PDF Unavailable - Skipped]:", [(k, f"({a}): {t}") for a, k, t in unavailable]),
    (f"[Non-arXiv Papers - Skipped ({len(not_arxiv)})]:", [(k, t) for k, t in not_arxiv]),
]:
    if items:
        print(f"\n{header}")
        for k, info in items:
            print(f"  - [{k}] {info}")
