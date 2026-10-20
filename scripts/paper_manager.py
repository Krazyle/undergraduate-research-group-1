#!/usr/bin/env python3

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Annotated, Any

import arxiv
import pypandoc
import typer
from rich.console import Console
from rich.table import Table

ROOT_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT_DIR / "papers" / "index.json"
BIB_PATH = ROOT_DIR / "bibliography.bib"
ENV_PATH = ROOT_DIR / ".env"
PDF_DIR = ROOT_DIR / "papers" / "pdf"
MD_DIR = ROOT_DIR / "papers" / "markdown"
BIB_ENTRY_START = re.compile(r"^@\w+\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)

DEFAULT_TIMEOUT = 30
HEADERS = {"User-Agent": "ThesisLiteraturePipeline/1.0 (academic research)"}

app = typer.Typer(
    name="paper-manager",
    help="Research Literature Pipeline Manager",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def ensure_directories() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    MD_DIR.mkdir(parents=True, exist_ok=True)


def load_papers() -> list[dict[str, Any]]:
    if not INDEX_PATH.exists():
        return []
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def get_unpaywall_email() -> str:
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                if key.strip() == "UNPAYWALL_EMAIL":
                    return val.strip().strip('"').strip("'")
    return os.environ.get("UNPAYWALL_EMAIL", "researcher@localhost")


def fetch_url(url: str, timeout: int = DEFAULT_TIMEOUT) -> bytes | None:
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.URLError, TimeoutError):
        return None


def get_open_access_pdf_url(paper: dict[str, Any]) -> str | None:
    doi = paper.get("doi", "")
    aid = paper.get("arxiv_id")

    if aid:
        return f"https://arxiv.org/pdf/{aid}.pdf"

    if doi:
        email = get_unpaywall_email()
        clean_doi = doi.strip()
        unpaywall_url = f"https://api.unpaywall.org/v2/{clean_doi}?email={email}"
        resp_bytes = fetch_url(unpaywall_url, timeout=DEFAULT_TIMEOUT)
        if resp_bytes:
            try:
                data = json.loads(resp_bytes.decode("utf-8"))
                best = data.get("best_oa_location")
                if best and best.get("url_for_pdf"):
                    return str(best["url_for_pdf"])
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        if "10.18653/v1/" in doi:
            return f"https://aclanthology.org/{doi.split('10.18653/v1/')[-1]}.pdf"

    return None


def parse_arxiv_html(raw_html_bytes: bytes) -> str | None:
    html_text = raw_html_bytes.decode("utf-8", "replace")
    try:
        return pypandoc.convert_text(html_text, "gfm", format="html").strip()
    except (OSError, RuntimeError, ValueError):
        return None


@app.command("status")
def status_cmd() -> None:
    ensure_directories()
    papers = load_papers()
    if not papers:
        console.print("[yellow]No papers found in index.[/yellow]")
        return

    total = len(papers)
    has_md: list[str] = []
    has_pdf_only: list[str] = []
    missing: list[str] = []

    for p in papers:
        key = p["citation_key"]
        md_exists = (MD_DIR / f"{key}.md").exists()
        pdf_exists = (PDF_DIR / f"{key}.pdf").exists()

        if md_exists:
            has_md.append(key)
        elif pdf_exists:
            has_pdf_only.append(key)
        else:
            missing.append(key)

    pct_md = (len(has_md) / total * 100) if total else 0.0
    pct_pdf = (len(has_pdf_only) / total * 100) if total else 0.0
    pct_missing = (len(missing) / total * 100) if total else 0.0

    table = Table(title="Literature Repository Status", show_header=True, header_style="bold cyan")
    table.add_column("Category", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")
    table.add_column("Status", style="magenta")

    table.add_row("Total Indexed", str(total), "100.0%", "Cataloged")
    table.add_row("Ready in Markdown", str(len(has_md)), f"{pct_md:.1f}%", "[green]Complete[/green]")
    table.add_row("PDF Only", str(len(has_pdf_only)), f"{pct_pdf:.1f}%", "[yellow]Pending Conversion[/yellow]" if has_pdf_only else "[dim]None[/dim]")
    table.add_row("Missing / Unfetched", str(len(missing)), f"{pct_missing:.1f}%", "[red]Action Needed[/red]" if missing else "[dim]None[/dim]")

    console.print(table)

    if has_pdf_only:
        console.print(f"\n[bold yellow]Pending PDF Conversion ({len(has_pdf_only)}):[/bold yellow]")
        for k in has_pdf_only[:5]:
            console.print(f"  • {k}")
        if len(has_pdf_only) > 5:
            console.print(f"  [dim]... and {len(has_pdf_only) - 5} more[/dim]")

    if missing:
        console.print(f"\n[bold red]Unfetched Papers ({len(missing)}):[/bold red]")
        for k in missing[:5]:
            console.print(f"  • {k}")
        if len(missing) > 5:
            console.print(f"  [dim]... and {len(missing) - 5} more[/dim]")


@app.command("fetch")
def fetch_cmd(
    source: Annotated[str, typer.Option("--source", "-s", help="Source filter: all, arxiv, general")] = "all",
    key: Annotated[str | None, typer.Option("--key", "-k", help="Target a specific citation key")] = None,
) -> None:
    ensure_directories()
    papers = load_papers()
    if not papers:
        console.print("[yellow]No papers found in index.[/yellow]")
        return

    if key:
        papers = [p for p in papers if p.get("citation_key") == key]
        if not papers:
            console.print(f"[bold red]Error:[/bold red] Citation key [{key}] not found in index.")
            raise typer.Exit(1)

    arxiv_client = arxiv.Client(page_size=1, delay_seconds=2.0, num_retries=3)
    converted_html = 0
    downloaded_pdf = 0
    skipped_existing = 0
    failed: list[str] = []

    for paper in papers:
        pkey = paper["citation_key"]
        title = paper.get("title", "")
        aid = paper.get("arxiv_id")
        md_file = MD_DIR / f"{pkey}.md"
        pdf_file = PDF_DIR / f"{pkey}.pdf"

        if md_file.exists():
            skipped_existing += 1
            continue

        if aid and source in ("all", "arxiv"):
            raw_html = fetch_url(f"https://arxiv.org/html/{aid}", timeout=DEFAULT_TIMEOUT)
            if raw_html:
                md_content = parse_arxiv_html(raw_html)
                if md_content:
                    md_file.write_text(md_content, encoding="utf-8")
                    converted_html += 1
                    console.print(f"[green]✓[/green] Converted HTML: [bold]{pkey}[/bold] {title}")
                    time.sleep(1)
                    continue

            if pdf_file.exists():
                skipped_existing += 1
                continue

            try:
                search = arxiv.Search(id_list=[aid])
                paper_result = next(arxiv_client.results(search), None)
                if paper_result and paper_result.pdf_url:
                    content = fetch_url(paper_result.pdf_url, timeout=DEFAULT_TIMEOUT)
                    if content and content.startswith(b"%PDF"):
                        pdf_file.write_bytes(content)
                        downloaded_pdf += 1
                        console.print(f"[green]✓[/green] Downloaded arXiv PDF: [bold]{pkey}[/bold] {title}")
                        time.sleep(1)
                        continue
            except (arxiv.ArxivError, urllib.error.URLError, TimeoutError, OSError):
                pass

            failed.append(pkey)
            console.print(f"[red]✗[/red] Failed arXiv: [bold]{pkey}[/bold] {title}")
            continue

        if source in ("all", "general"):
            if pdf_file.exists():
                skipped_existing += 1
                continue

            url = get_open_access_pdf_url(paper)
            if not url:
                continue

            content = fetch_url(url, timeout=DEFAULT_TIMEOUT)
            if content and content.startswith(b"%PDF"):
                pdf_file.write_bytes(content)
                downloaded_pdf += 1
                console.print(f"[green]✓[/green] Downloaded PDF: [bold]{pkey}[/bold] ({len(content) // 1024} KB)")
                time.sleep(1)
            else:
                failed.append(pkey)
                console.print(f"[red]✗[/red] Failed PDF: [bold]{pkey}[/bold] {url}")

    if converted_html == 0 and downloaded_pdf == 0 and not failed:
        console.print(f"[dim]{len(papers)} papers checked (all up to date)[/dim]")
    else:
        console.print(f"Summary: {converted_html} converted via HTML, {downloaded_pdf} PDF saved, {len(failed)} failed")


@app.command("convert")
def convert_cmd(
    batch_size: Annotated[int, typer.Option("--batch-size", "-b", help="Batch size for MinerU parser")] = 10,
    key: Annotated[str | None, typer.Option("--key", "-k", help="Target a specific citation key")] = None,
    backend: Annotated[str, typer.Option("--backend", help="LMDeploy backend for MinerU")] = "turbomind",
    tier: Annotated[str | None, typer.Option("--tier", "-t", help="MinerU parse tier: flash, basic, standard, advanced")] = None,
    mineru_bin: Annotated[str | None, typer.Option("--mineru-bin", help="Path to mineru-kit executable")] = None,
) -> None:
    ensure_directories()
    papers = load_papers()
    if not papers:
        console.print("[yellow]No papers found in index.[/yellow]")
        return

    if key:
        papers = [p for p in papers if p.get("citation_key") == key]
        if not papers:
            console.print(f"[bold red]Error:[/bold red] Citation key [{key}] not found in index.")
            raise typer.Exit(1)

    os.environ["MINERU_LMDEPLOY_BACKEND"] = backend
    executable = mineru_bin or shutil.which("mineru-kit") or str(Path.home() / ".local" / "bin" / "mineru-kit")

    pending: list[dict[str, Any]] = []
    for paper in papers:
        pkey = paper["citation_key"]
        if (PDF_DIR / f"{pkey}.pdf").exists() and not (MD_DIR / f"{pkey}.md").exists():
            pending.append(paper)

    if not pending:
        console.print("[dim]No pending PDF conversions needed.[/dim]")
        return

    console.print(f"Converting {len(pending)} pending PDFs (batch size {batch_size})...")

    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(pending) + batch_size - 1) // batch_size
        console.print(f"\n[cyan]Batch {batch_num}/{total_batches} ({len(batch)} papers)[/cyan]")

        with tempfile.TemporaryDirectory() as temp_input_dir:
            temp_input_path = Path(temp_input_dir)
            temp_output_dir = temp_input_path / "output"

            for paper in batch:
                pkey = paper["citation_key"]
                shutil.copy2(PDF_DIR / f"{pkey}.pdf", temp_input_path / f"{pkey}.pdf")

            cmd = [
                executable,
                "parse",
                str(temp_input_path),
                "-o", str(temp_output_dir),
            ]
            if tier:
                cmd += ["--tier", tier]

            try:
                subprocess.run(cmd, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                console.print(f"[bold red]MinerU processing error in batch {batch_num}:[/bold red] {e}")
                continue

            for paper in batch:
                pkey = paper["citation_key"]
                md_out = temp_output_dir / f"{pkey}.md"
                if not md_out.is_file():
                    console.print(f"[red]✗[/red] No Markdown output: [bold]{pkey}[/bold]")
                    continue

                shutil.copy2(md_out, MD_DIR / f"{pkey}.md")
                console.print(f"[green]✓[/green] Saved Markdown: [bold]{pkey}[/bold]")

    console.print("[bold green]Conversion process finished.[/bold green]")


@app.command("pipeline")
def pipeline_cmd(
    source: Annotated[str, typer.Option("--source", "-s", help="Source filter: all, arxiv, general")] = "all",
    batch_size: Annotated[int, typer.Option("--batch-size", "-b", help="Batch size for MinerU parser")] = 10,
    key: Annotated[str | None, typer.Option("--key", "-k", help="Target a specific citation key")] = None,
) -> None:
    console.print("\n[bold cyan][1/3] Fetching literature[/bold cyan]")
    fetch_cmd(source=source, key=key)

    console.print("\n[bold cyan][2/3] Converting PDF documents[/bold cyan]")
    convert_cmd(batch_size=batch_size, key=key)

    console.print("\n[bold cyan][3/3] Repository status[/bold cyan]")
    status_cmd()


def remove_bib_entries(text: str, keys: set[str]) -> tuple[str, int]:
    pieces: list[str] = []
    removed = 0
    pos = 0
    for match in BIB_ENTRY_START.finditer(text):
        if match.start() < pos or match.group(1) not in keys:
            continue
        depth = 0
        end = None
        for i in range(text.index("{", match.start()), len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            raise ValueError(f"Unbalanced braces in entry {match.group(1)}")
        while end < len(text) and text[end] == "\n":
            end += 1
        pieces.append(text[pos : match.start()])
        pos = end
        removed += 1
    pieces.append(text[pos:])
    result = "".join(pieces)
    if removed and text.endswith("\n"):
        result = result.rstrip("\n") + "\n"
    return result, removed


@app.command("remove")
def remove_cmd(
    keys: Annotated[list[str], typer.Argument(help="Citation keys to remove from the repository")],
) -> None:
    target_keys = set(keys)
    if not target_keys:
        console.print("[yellow]No citation keys specified for removal.[/yellow]")
        return

    active_keys: set[str] = set()
    if INDEX_PATH.exists():
        papers = load_papers()
        remaining = [p for p in papers if p.get("citation_key") not in target_keys]
        INDEX_PATH.write_text(
            json.dumps(remaining, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        active_keys = {p["citation_key"] for p in remaining}
        console.print(f"Updated index (removed {len(papers) - len(remaining)} entries)")

    if BIB_PATH.exists():
        try:
            bib_text = BIB_PATH.read_text(encoding="utf-8")
            new_text, removed_count = remove_bib_entries(bib_text, target_keys)
            if removed_count > 0:
                BIB_PATH.write_text(new_text, encoding="utf-8")
            console.print(f"Updated bibliography (removed {removed_count} entries)")
        except (OSError, ValueError) as e:
            console.print(f"[bold red]Failed to update bibliography: {e}[/bold red]")

    if PDF_DIR.exists():
        for f in PDF_DIR.glob("*.pdf"):
            if f.stem not in active_keys:
                f.unlink()
                console.print(f"[dim]Deleted orphan PDF: {f.name}[/dim]")

    if MD_DIR.exists():
        for f in MD_DIR.glob("*.md"):
            if f.stem not in active_keys:
                f.unlink()
                console.print(f"[dim]Deleted orphan Markdown: {f.name}[/dim]")

    images_dir = MD_DIR / "images"
    if images_dir.exists():
        for item in images_dir.iterdir():
            if item.is_dir() and item.name not in active_keys:
                shutil.rmtree(item, ignore_errors=True)
                console.print(f"[dim]Deleted orphan images directory: {item.name}[/dim]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
