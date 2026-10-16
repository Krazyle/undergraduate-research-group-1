#!/usr/bin/env python3

import difflib
import io
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Annotated, Any

import typer
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseUpload
from rich.console import Console
from rich.table import Table

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
QUARTO_YML = ROOT_DIR / "_quarto.yml"
SERVICE_ACCOUNT_PATH = ROOT_DIR / "service_account.json"
BIB_PATH = ROOT_DIR / "bibliography.bib"

SCOPES = ["https://www.googleapis.com/auth/drive"]

app = typer.Typer(
    name="gdocs-manager",
    help="Google Docs Thesis Synchronization and Feedback Manager",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def get_env_var(var_name: str, default: str = "") -> str:
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                if key.strip() == var_name:
                    return val.strip().strip('"').strip("'")
    return os.environ.get(var_name, default).strip()


def extract_id_from_url_or_id(raw_val: str) -> str:
    cleaned = raw_val.strip()
    match = re.search(r"/document/d/([a-zA-Z0-9-_]+)", cleaned)
    if match:
        return match.group(1)
    return cleaned


def get_document_id(cli_doc_id: str | None = None) -> str:
    raw_id = cli_doc_id or get_env_var("GDOCS_DOCUMENT_ID")
    if raw_id:
        return extract_id_from_url_or_id(raw_id)
    console.print(
        "[bold red]Error:[/bold red] Missing GDOCS_DOCUMENT_ID in .env or via --doc-id option."
    )
    raise typer.Exit(1)


def handle_google_error(error: HttpError, target_id: str) -> None:
    code = error.resp.status if hasattr(error, "resp") else 500
    if code == 404:
        console.print(
            f"[bold red]Error (404):[/bold red] Document [{target_id}] not found. "
            "Verify the document ID or web link in .env."
        )
    elif code == 403:
        console.print(
            f"[bold red]Error (403):[/bold red] Access denied for document [{target_id}]. "
            "Ensure the Google Doc is shared with your account or service account."
        )
    elif code == 401:
        console.print(
            "[bold red]Error (401):[/bold red] Authentication expired or invalid. "
            "Delete token.json and re-authenticate."
        )
    else:
        console.print(f"[bold red]Google API error ({code}):[/bold red] {error}")
    raise typer.Exit(1)


def get_credentials() -> Any:
    configured_path = get_env_var("GOOGLE_CREDENTIALS_PATH", "service_account.json")
    target_path = ROOT_DIR / configured_path
    if not target_path.exists() and SERVICE_ACCOUNT_PATH.exists():
        target_path = SERVICE_ACCOUNT_PATH
    if not target_path.exists():
        console.print("[bold red]Error:[/bold red] Google credentials file not found.")
        console.print(
            "Place [bold]service_account.json[/bold] in the project root, "
            "or set [bold]GOOGLE_CREDENTIALS_PATH[/bold] in .env."
        )
        raise typer.Exit(1)
    return service_account.Credentials.from_service_account_file(
        str(target_path), scopes=SCOPES
    )


def get_drive() -> Any:
    return build("drive", "v3", credentials=get_credentials())


def get_docs() -> Any:
    return build("docs", "v1", credentials=get_credentials())


def get_document_mode(doc_id: str) -> str:
    try:
        docs = get_docs()
        doc = (
            docs.documents()
            .get(documentId=doc_id, fields="documentStyle.documentFormat.documentMode")
            .execute()
        )
        return doc.get("documentStyle", {}).get("documentFormat", {}).get("documentMode", "PAGES")
    except (HttpError, OSError):
        return "PAGES"


def set_document_mode(doc_id: str, mode: str) -> None:
    try:
        docs = get_docs()
        docs.documents().batchUpdate(
            documentId=doc_id,
            body={
                "requests": [
                    {
                        "updateDocumentStyle": {
                            "documentStyle": {
                                "documentFormat": {
                                    "documentMode": mode,
                                }
                            },
                            "fields": "documentFormat.documentMode",
                        }
                    }
                ]
            },
        ).execute()
    except (HttpError, OSError):
        pass


def get_manuscript_chapters() -> list[Path]:
    chapters: list[Path] = []
    if QUARTO_YML.exists():
        in_chapters = False
        for line in QUARTO_YML.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("chapters:"):
                in_chapters = True
                continue
            if in_chapters:
                if stripped.startswith("- "):
                    fname = stripped.removeprefix("- ").strip().strip('"').strip("'")
                    target = ROOT_DIR / fname
                    if target.exists():
                        chapters.append(target)
                elif stripped and not stripped.startswith("#"):
                    break
    if not chapters:
        chapters = sorted(ROOT_DIR.glob("*.qmd"))
    return chapters


DEFAULT_IMAGE_BASE_URL = (
    "https://raw.githubusercontent.com/edwardducy/undergraduate-research-group-1/main/figures"
)


def get_image_base_url() -> str:
    configured = get_env_var("GDOCS_IMAGE_BASE_URL")
    return configured.rstrip("/") if configured else DEFAULT_IMAGE_BASE_URL


def ensure_png_figures() -> None:
    figures_dir = ROOT_DIR / "figures"
    if not figures_dir.exists():
        return
    typst_bin = shutil.which("typst") or str(
        Path.home() / ".local" / "share" / "mise" / "shims" / "typst"
    )
    if not Path(typst_bin).exists() and not shutil.which("typst"):
        return
    for typ_file in figures_dir.glob("fig-*.typ"):
        png_output = typ_file.with_suffix(".png")
        if not png_output.exists() or typ_file.stat().st_mtime > png_output.stat().st_mtime:
            try:
                subprocess.run(
                    [typst_bin, "compile", "--ppi", "300", str(typ_file), str(png_output)],
                    check=True,
                    capture_output=True,
                )
            except (subprocess.CalledProcessError, OSError):
                pass


def get_png_dimensions(png_path: Path) -> tuple[int, int]:
    try:
        with png_path.open("rb") as f:
            f.seek(16)
            width, height = struct.unpack(">II", f.read(8))
            return width, height
    except (OSError, struct.error):
        return 1800, 1200


def calculate_figure_width(png_path: Path, max_w: float = 6.0, max_h: float = 7.0) -> float:
    w_px, h_px = get_png_dimensions(png_path)
    natural_w = w_px / 300.0
    natural_h = h_px / 300.0

    scale = 1.0
    if natural_w * scale > max_w:
        scale = max_w / natural_w
    if natural_h * scale > max_h:
        scale = max_h / natural_h

    return round(natural_w * scale, 2)


def extract_media_registries(
    chapter_paths: list[Path],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    figures: dict[str, dict[str, str]] = {}
    tables: dict[str, dict[str, str]] = {}
    fig_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)\{#(fig-[^}]+)\}")
    tbl_pattern = re.compile(r"^:\s*(.*?)\s*\{#(tbl-[^}\s]+)", re.MULTILINE)

    for chap in chapter_paths:
        chap_num_match = re.search(r"\d+", chap.stem)
        chap_num = int(chap_num_match.group(0)) if chap_num_match else 1
        content = chap.read_text(encoding="utf-8")

        fig_idx = 1
        for m in fig_pattern.finditer(content):
            caption, path, fig_id = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            figures[fig_id] = {
                "label": f"Figure {chap_num}.{fig_idx}",
                "caption": caption,
                "path": path,
            }
            fig_idx += 1

        tbl_idx = 1
        for m in tbl_pattern.finditer(content):
            caption, tbl_id = m.group(1).strip(), m.group(2).strip()
            tables[tbl_id] = {
                "label": f"Table {chap_num}.{tbl_idx}",
                "caption": caption,
            }
            tbl_idx += 1

    return figures, tables


def preprocess_cross_references(
    content: str,
    figures: dict[str, dict[str, str]],
    tables: dict[str, dict[str, str]],
) -> str:
    def repl_fig(m: re.Match[str]) -> str:
        fig_id = m.group(1)
        return figures.get(fig_id, {}).get("label", f"Figure {fig_id}")

    def repl_tbl(m: re.Match[str]) -> str:
        tbl_id = m.group(1)
        return tables.get(tbl_id, {}).get("label", f"Table {tbl_id}")

    content = re.sub(r"@([a-zA-Z0-9_-]*fig-[a-zA-Z0-9_-]+)", repl_fig, content)
    content = re.sub(r"@([a-zA-Z0-9_-]*tbl-[a-zA-Z0-9_-]+)", repl_tbl, content)
    return content


def format_apa_figures(
    html_content: str,
    base_url: str,
    figures: dict[str, dict[str, str]],
) -> str:
    figures_dir = ROOT_DIR / "figures"
    figure_block_pattern = re.compile(
        r'<figure id="(?P<id>[^"]+)">\s*<embed src="figures/(?P<name>[^"/]+)\.pdf"\s*/>\s*<figcaption[^>]*>(?P<caption>.*?)</figcaption>\s*</figure>',
        re.DOTALL,
    )

    def repl_block(m: re.Match[str]) -> str:
        fig_id = m.group("id")
        name = m.group("name")
        raw_caption = m.group("caption")
        clean_caption = " ".join(re.sub(r"<[^>]+>", "", raw_caption).split())
        label = figures.get(fig_id, {}).get("label", "Figure")
        png_path = figures_dir / f"{name}.png"
        width_in = calculate_figure_width(png_path)
        return (
            f'<p><strong>{label}</strong><br />\n'
            f'<em>{clean_caption}</em></p>\n'
            f'<p style="text-align: center;"><img src="{base_url}/{name}.png" '
            f'alt="{label}: {clean_caption}" style="width: {width_in}in;" /></p>'
        )

    html_content = figure_block_pattern.sub(repl_block, html_content)

    fallback_pattern = re.compile(r'<embed src="figures/([^"/]+)\.pdf"\s*/>')

    def repl_fallback(m: re.Match[str]) -> str:
        name = m.group(1)
        png_path = figures_dir / f"{name}.png"
        width_in = calculate_figure_width(png_path)
        return (
            f'<p style="text-align: center;"><img src="{base_url}/{name}.png" '
            f'alt="{name}" style="width: {width_in}in;" /></p>'
        )

    return fallback_pattern.sub(repl_fallback, html_content)


def format_apa_tables(
    html_content: str,
    tables: dict[str, dict[str, str]],
) -> str:
    tbl_pattern = re.compile(
        r'(<table id="(?P<id>tbl-[^"]+)"[^>]*>\s*)<caption>(?P<caption>.*?)</caption>',
        re.DOTALL,
    )

    def repl_tbl(m: re.Match[str]) -> str:
        prefix = m.group(1)
        tbl_id = m.group("id")
        raw_caption = m.group("caption")
        clean_caption = " ".join(re.sub(r"<[^>]+>", "", raw_caption).split())
        label = tables.get(tbl_id, {}).get("label", "Table")
        return f"{prefix}<caption><strong>{label}</strong><br />\n<em>{clean_caption}</em></caption>"

    return tbl_pattern.sub(repl_tbl, html_content)


def compile_manuscript(output_format: str = "html") -> str:
    quarto_bin = shutil.which("quarto") or str(
        Path.home() / ".local" / "share" / "mise" / "shims" / "quarto"
    )
    pandoc_bin = shutil.which("pandoc") or str(
        Path.home() / ".local" / "share" / "mise" / "shims" / "pandoc"
    )

    chapter_paths = [p for p in get_manuscript_chapters() if p.exists()]
    if not chapter_paths:
        console.print("[bold red]Error:[/bold red] No manuscript chapters found.")
        raise typer.Exit(1)

    if output_format == "html":
        ensure_png_figures()

    if Path(quarto_bin).exists():
        cmd = [quarto_bin, "pandoc", "-s"]
    elif Path(pandoc_bin).exists():
        cmd = [pandoc_bin, "-s"]
    else:
        console.print("[bold red]Error:[/bold red] Neither quarto nor pandoc executable found.")
        raise typer.Exit(1)

    if BIB_PATH.exists():
        cmd.extend(["--citeproc", f"--bibliography={BIB_PATH}"])
    cmd.extend(["-t", output_format])
    if output_format == "plain":
        cmd.append("--wrap=none")

    figures, tables = extract_media_registries(chapter_paths)
    processed_parts = [
        preprocess_cross_references(p.read_text(encoding="utf-8"), figures, tables)
        for p in chapter_paths
    ]
    combined_input = "\n\n".join(processed_parts)

    try:
        result = subprocess.run(
            cmd,
            input=combined_input,
            check=True,
            capture_output=True,
            text=True,
        )
        output = result.stdout
        if output_format == "html":
            base_url = get_image_base_url()
            output = format_apa_figures(output, base_url, figures)
            output = format_apa_tables(output, tables)
        return output
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Compilation error:[/bold red] {e.stderr}")
        raise typer.Exit(1)


@app.command("status")
def status_cmd(
    doc_id: Annotated[str | None, typer.Option("--doc-id", help="Google Doc identifier or URL")] = None,
) -> None:
    target_id = get_document_id(doc_id)
    drive = get_drive()

    try:
        file_meta = (
            drive.files()
            .get(fileId=target_id, fields="id, name, modifiedTime, version, owners")
            .execute()
        )
        comments_meta = (
            drive.comments()
            .list(fileId=target_id, fields="comments(id, resolved)")
            .execute()
        )
    except HttpError as error:
        handle_google_error(error, target_id)

    comments = comments_meta.get("comments", [])
    open_count = sum(1 for c in comments if not c.get("resolved"))
    resolved_count = sum(1 for c in comments if c.get("resolved"))

    owners = ", ".join(o.get("displayName", "") for o in file_meta.get("owners", []))
    modified = file_meta.get("modifiedTime", "Unknown")

    table = Table(title="Google Doc Status", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    layout_mode = get_document_mode(target_id)

    table.add_row("Document Name", file_meta.get("name", "Unknown"))
    table.add_row("Document ID", target_id)
    table.add_row("Owner", owners)
    table.add_row("Layout Mode", f"[cyan]{layout_mode.capitalize()}[/cyan]")
    table.add_row("Last Modified", modified)
    table.add_row(
        "Open Comments",
        f"[yellow]{open_count}[/yellow]" if open_count else "[dim]0[/dim]",
    )
    table.add_row(
        "Resolved Comments",
        f"[green]{resolved_count}[/green]" if resolved_count else "[dim]0[/dim]",
    )

    console.print(table)


@app.command("comments")
def comments_cmd(
    open_only: Annotated[bool, typer.Option("--open-only", help="Filter for open comments")] = False,
    doc_id: Annotated[str | None, typer.Option("--doc-id", help="Google Doc identifier or URL")] = None,
) -> None:
    target_id = get_document_id(doc_id)
    drive = get_drive()

    try:
        response = (
            drive.comments()
            .list(
                fileId=target_id,
                fields="comments(id, content, author(displayName), createdTime, resolved, quotedFileContent(value), replies(author(displayName), content))",
            )
            .execute()
        )
    except HttpError as error:
        handle_google_error(error, target_id)

    comments = response.get("comments", [])
    if open_only:
        comments = [c for c in comments if not c.get("resolved")]

    if not comments:
        console.print("[dim]No comments found.[/dim]")
        return

    table = Table(title="Reviewer Comments", show_header=True, header_style="bold cyan")
    table.add_column("Reviewer", style="bold magenta", width=20)
    table.add_column("Status", width=10)
    table.add_column("Highlighted Passage", style="italic yellow", width=35)
    table.add_column("Feedback", width=45)

    for c in comments:
        author = c.get("author", {}).get("displayName", "Anonymous")
        status = "[green]Resolved[/green]" if c.get("resolved") else "[bold yellow]Open[/bold yellow]"
        quoted = c.get("quotedFileContent", {}).get("value", "[No quote]")
        content = c.get("content", "")

        replies = c.get("replies", [])
        if replies:
            reply_lines = [
                f"\n[dim]↳ {r.get('author', {}).get('displayName')}: {r.get('content')}[/dim]"
                for r in replies
            ]
            content += "".join(reply_lines)

        table.add_row(author, status, quoted, content)

    console.print(table)


@app.command("sync")
def sync_cmd(
    doc_id: Annotated[str | None, typer.Option("--doc-id", help="Google Doc identifier or URL")] = None,
    pageless: Annotated[bool | None, typer.Option("--pageless/--pages", help="Force pageless or pages layout")] = None,
) -> None:
    target_id = get_document_id(doc_id)
    drive = get_drive()

    current_mode = get_document_mode(target_id)
    env_pageless = get_env_var("GDOCS_PAGELESS").lower() in ("true", "1", "yes")

    if pageless is True:
        target_mode = "PAGELESS"
    elif pageless is False:
        target_mode = "PAGES"
    elif env_pageless or current_mode == "PAGELESS":
        target_mode = "PAGELESS"
    else:
        target_mode = "PAGES"

    console.print("Compiling Quarto manuscript to HTML...")
    html_content = compile_manuscript("html")

    console.print(f"Uploading content to Google Doc ({target_id})...")
    stream = io.BytesIO(html_content.encode("utf-8"))
    media = MediaIoBaseUpload(stream, mimetype="text/html", resumable=True)

    try:
        drive.files().update(
            fileId=target_id,
            media_body=media,
        ).execute()
    except HttpError as error:
        handle_google_error(error, target_id)

    if target_mode == "PAGELESS":
        set_document_mode(target_id, "PAGELESS")

    console.print(f"[bold green]✓[/bold green] Document synchronized: https://docs.google.com/document/d/{target_id}/edit")
    if target_mode == "PAGELESS":
        console.print("[dim]Pageless layout mode preserved.[/dim]")


@app.command("diff")
def diff_cmd(
    doc_id: Annotated[str | None, typer.Option("--doc-id", help="Google Doc identifier or URL")] = None,
) -> None:
    target_id = get_document_id(doc_id)
    drive = get_drive()

    try:
        remote_bytes = drive.files().export(fileId=target_id, mimeType="text/plain").execute()
    except HttpError as error:
        handle_google_error(error, target_id)

    remote_text = remote_bytes.decode("utf-8") if isinstance(remote_bytes, bytes) else str(remote_bytes)
    remote_lines = [line.strip() for line in remote_text.splitlines() if line.strip()]

    console.print("Compiling local manuscript to plain text for comparison...")
    local_text = compile_manuscript("plain")
    local_lines = [line.strip() for line in local_text.splitlines() if line.strip()]

    diff_lines = list(
        difflib.unified_diff(
            remote_lines,
            local_lines,
            fromfile="gdocs/document",
            tofile="local/manuscript",
            n=1,
            lineterm="",
        )
    )

    if not diff_lines:
        console.print("[bold green]Local manuscript and Google Doc match. No differences detected.[/bold green]")
        return

    added = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    deleted = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))

    console.print(f"[bold cyan]Text Differences Detected[/bold cyan] ([green]+{added}[/green] / [red]-{deleted}[/red]):\n")
    for line in diff_lines[:40]:
        if line.startswith("+"):
            console.print(f"[green]{line}[/green]")
        elif line.startswith("-"):
            console.print(f"[red]{line}[/red]")
        elif line.startswith("@"):
            console.print(f"[cyan]{line}[/cyan]")
        else:
            console.print(f"[dim]{line}[/dim]")

    if len(diff_lines) > 40:
        console.print(f"\n[dim]... and {len(diff_lines) - 40} more lines[/dim]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
