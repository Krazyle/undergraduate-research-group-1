#!/usr/bin/env python3

import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Annotated, Any

import pypandoc
import requests
import typer
from canvasapi import Canvas
from canvasapi.course import Course
from canvasapi.exceptions import CanvasException
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
DEFAULT_COURSE_KEYWORD = "Undergraduate Research 1"

CourseIdOpt = Annotated[
    int | None,
    typer.Option("--course-id", "-c", help="Override course ID"),
]

app = typer.Typer(
    name="canvas-manager",
    help="Canvas LMS Course Manager",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def load_config() -> tuple[str, str]:
    config: dict[str, str] = {}
    if ENV_PATH.exists():
        with open(ENV_PATH, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    config[key.strip()] = val.strip().strip('"').strip("'")

    base_url = (
        config.get("CANVAS_BASE_URL")
        or os.environ.get("CANVAS_BASE_URL")
        or "https://ceu.instructure.com"
    ).rstrip("/")
    token = config.get("CANVAS_API_TOKEN") or os.environ.get("CANVAS_API_TOKEN") or ""
    return base_url, token


def init_canvas() -> Canvas:
    base_url, token = load_config()
    if not token:
        console.print("[bold red]Error:[/bold red] CANVAS_API_TOKEN missing from .env or environment")
        raise typer.Exit(1)
    return Canvas(base_url, token)


def html_to_markdown(raw_html: str | None) -> str:
    if not raw_html:
        return ""
    converted = pypandoc.convert_text(
        raw_html,
        "gfm-raw_html",
        format="html",
        extra_args=["--wrap=none"],
    ).strip()
    return re.sub(r"\n{3,}", "\n\n", converted)


def slugify(text: str) -> str:
    if not text:
        return "untitled"
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s_]+", "-", text).strip("-")
    return text[:60] if text else "untitled"


def download_file(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ThesisCanvasManager/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            dest.write_bytes(resp.read())
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        console.print(f"[yellow]Warning: Failed to download {dest.name}: {err}[/yellow]")
        return False


def get_course(
    canvas: Canvas,
    course_id: int | None = None,
    keyword: str = DEFAULT_COURSE_KEYWORD,
) -> Course:
    try:
        if course_id:
            return canvas.get_course(course_id)
        courses = canvas.get_courses(enrollment_state="active")
        for course in courses:
            name = getattr(course, "name", "")
            code = getattr(course, "course_code", "")
            if keyword.lower() in name.lower() or keyword.lower() in code.lower():
                return course
    except (CanvasException, requests.RequestException) as err:
        console.print(f"[bold red]Error communicating with Canvas:[/bold red] {err}")
        raise typer.Exit(1)

    console.print("[bold red]Error:[/bold red] Could not locate course automatically.")
    raise typer.Exit(1)


@app.command("modules")
def list_modules_cmd(course_id: CourseIdOpt = None) -> None:
    canvas = init_canvas()
    course = get_course(canvas, course_id)

    console.print(f"\n[bold green]Course:[/bold green] {course.name} [dim](ID: {course.id})[/dim]\n")
    modules = list(course.get_modules(include=["items"]))
    if not modules:
        console.print("[yellow]No modules found.[/yellow]")
        return

    table = Table(title="Course Modules and Items", show_header=True, header_style="bold cyan")
    table.add_column("Module", style="bold")
    table.add_column("Item ID", style="dim", justify="right")
    table.add_column("Type", style="magenta")
    table.add_column("Title")

    for module in modules:
        items = getattr(module, "items", [])
        if not items:
            table.add_row(module.name, "-", "-", "[dim]No items[/dim]")
        for idx, item in enumerate(items):
            mod_label = module.name if idx == 0 else ""
            table.add_row(
                mod_label,
                str(item.get("id", "-")),
                str(item.get("type", "-")),
                str(item.get("title", "Untitled")),
            )

    console.print(table)


@app.command("module")
def inspect_module_cmd(
    module_id: Annotated[int, typer.Argument(help="ID of the module to inspect")],
    course_id: CourseIdOpt = None,
) -> None:
    canvas = init_canvas()
    course = get_course(canvas, course_id)

    try:
        module = course.get_module(module_id)
        items = list(module.get_module_items())
    except (CanvasException, requests.RequestException) as err:
        console.print(f"[bold red]Error fetching module {module_id}:[/bold red] {err}")
        raise typer.Exit(1)

    console.print(f"\n[bold green]Module:[/bold green] {module.name} [dim](ID: {module.id})[/dim]")
    console.print(f"[bold]Total Items:[/bold] {len(items)}\n")

    for item in items:
        itype = getattr(item, "type", "")
        ititle = getattr(item, "title", "")
        content_id = getattr(item, "content_id", None)
        extra = ""

        try:
            if itype == "Assignment" and content_id:
                assign = course.get_assignment(content_id)
                extra = f"Points: {getattr(assign, 'points_possible', 'None')} | Due: {getattr(assign, 'due_at', 'None')}"
            elif itype == "File" and content_id:
                f = course.get_file(content_id)
                extra = f"Download: {getattr(f, 'url', '')}"
        except (CanvasException, requests.RequestException) as err:
            extra = f"Notice: Could not fetch details ({err})"

        console.print(f"- [bold]{ititle}[/bold] [magenta]({itype})[/magenta] [dim]ID: {item.id}[/dim]")
        if extra:
            console.print(f"  [dim]{extra}[/dim]")


@app.command("item")
def inspect_item_cmd(
    item_id: Annotated[int, typer.Argument(help="ID of the module item to inspect")],
    course_id: CourseIdOpt = None,
) -> None:
    canvas = init_canvas()
    course = get_course(canvas, course_id)
    modules = list(course.get_modules(include=["items"]))

    target_item: dict[str, Any] | None = None
    target_module_name: str = ""

    for module in modules:
        for item in getattr(module, "items", []):
            if str(item.get("id")) == str(item_id):
                target_item = item
                target_module_name = getattr(module, "name", "")
                break
        if target_item:
            break

    if not target_item:
        console.print(f"[bold red]Error:[/bold red] Item ID {item_id} not found.")
        raise typer.Exit(1)

    ititle = target_item.get("title", "")
    itype = target_item.get("type", "")
    content_id = target_item.get("content_id")

    console.print(f"\n[bold green]{ititle}[/bold green] [magenta]({itype})[/magenta]")
    console.print(f"[dim]Module: {target_module_name} | Item ID: {item_id}[/dim]\n")

    try:
        if itype == "Assignment" and content_id:
            assign = course.get_assignment(content_id)
            console.print(f"[bold]Due Date:[/bold] {getattr(assign, 'due_at', 'None')}")
            console.print(f"[bold]Points:[/bold] {getattr(assign, 'points_possible', 'None')}")
            desc = html_to_markdown(getattr(assign, "description", ""))
            console.print(Panel(desc, title="Instructions", border_style="blue"))
        elif itype == "Page":
            page_url = target_item.get("page_url")
            if page_url:
                page = course.get_page(page_url)
                body = html_to_markdown(getattr(page, "body", ""))
                console.print(Panel(body, title="Page Content", border_style="green"))
        elif itype == "File" and content_id:
            f = course.get_file(content_id)
            console.print(f"[bold]File Name:[/bold] {getattr(f, 'display_name', '')}")
            console.print(f"[bold]File Size:[/bold] {getattr(f, 'size', '')} bytes")
            console.print(f"[bold]Download URL:[/bold] {getattr(f, 'url', '')}")
        elif itype == "Quiz" and content_id:
            quiz = course.get_quiz(content_id)
            console.print(f"[bold]Due Date:[/bold] {getattr(quiz, 'due_at', 'None')}")
            console.print(f"[bold]Points:[/bold] {getattr(quiz, 'points_possible', 'None')}")
    except (CanvasException, requests.RequestException) as err:
        console.print(f"[bold red]Error fetching item details:[/bold red] {err}")
        raise typer.Exit(1)


@app.command("announcements")
def list_announcements_cmd(course_id: CourseIdOpt = None) -> None:
    canvas = init_canvas()
    course = get_course(canvas, course_id)
    announcements = list(canvas.get_announcements(context_codes=[f"course_{course.id}"]))

    if not announcements:
        console.print("[yellow]No announcements found.[/yellow]")
        return

    console.print(f"\n[bold green]Course Announcements[/bold green] [dim]({len(announcements)} found)[/dim]\n")
    for a in announcements:
        posted = getattr(a, "posted_at", "Undated")
        author = getattr(a, "user_name", "Instructor")
        body = html_to_markdown(getattr(a, "message", ""))
        attachments = getattr(a, "attachments", [])

        att_text = ""
        if attachments:
            att_items = [
                f"- {att.get('display_name')} ({att.get('size', 0)} bytes)"
                for att in attachments
            ]
            att_text = "\n\n[bold cyan]Attachments:[/bold cyan]\n" + "\n".join(att_items)

        console.print(
            Panel(
                f"[dim]Date: {posted} | Author: {author}[/dim]\n\n{body}{att_text}",
                title=f"[bold]{a.title}[/bold]",
                border_style="cyan",
            )
        )


@app.command("export")
def export_cmd(
    output_dir: Annotated[str, typer.Option("--output-dir", "-o", help="Target export directory")] = "canvas_docs",
    course_id: CourseIdOpt = None,
) -> None:
    canvas = init_canvas()
    course = get_course(canvas, course_id)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with console.status("[bold blue]Fetching course structure...[/bold blue]"):
        modules = list(course.get_modules(include=["items"]))
        announcements = list(canvas.get_announcements(context_codes=[f"course_{course.id}"]))

    console.print(f"[1/3] Loaded {len(modules)} modules and {len(announcements)} announcements for {course.name}")

    index_lines = [
        f"# Course Documentation: {course.name}\n\n",
        f"- Course ID: {course.id}\n",
        f"- Modules: {len(modules)}\n",
        f"- Announcements: {len(announcements)}\n\n",
        "## Announcements\n\n",
    ]

    ann_dir = out_path / "00-announcements"
    ann_dir.mkdir(parents=True, exist_ok=True)
    att_dir = ann_dir / "attachments"

    with console.status("[bold blue]Exporting announcements...[/bold blue]"):
        for a_idx, a in enumerate(announcements, start=1):
            posted = getattr(a, "posted_at", "")
            file_slug = f"{a_idx:02d}-{slugify(a.title)}.md"
            ann_file = ann_dir / file_slug

            index_lines.append(
                f"- [{a.title}](00-announcements/{file_slug}) ({posted[:10] if posted else 'Undated'})\n"
            )

            body_md = html_to_markdown(getattr(a, "message", ""))
            attachments = getattr(a, "attachments", [])

            doc_lines = [
                f"# {a.title}\n\n",
                f"- Date: {posted}\n",
                f"- Author: {getattr(a, 'user_name', 'Instructor')}\n",
            ]

            if attachments:
                att_dir.mkdir(parents=True, exist_ok=True)
                doc_lines.append(f"- Attachments: {len(attachments)}\n")
                for att in attachments:
                    raw_name = att.get("display_name") or "attachment"
                    safe_name = f"{att.get('id', a_idx)}_{Path(raw_name).name}"
                    furl = att.get("url")
                    saved_att = att_dir / safe_name

                    if furl:
                        download_file(furl, saved_att)

                    doc_lines.append(
                        f"  - [{raw_name}](attachments/{safe_name}) ({att.get('size', 0)} bytes)\n"
                    )

            doc_lines.append("\n## Announcement Content\n\n")
            doc_lines.append(f"{body_md}\n")
            ann_file.write_text("".join(doc_lines), encoding="utf-8")

    console.print(f"[2/3] Exported {len(announcements)} announcements and attachments")

    index_lines.append("\n## Course Modules\n\n")

    with console.status("[bold blue]Exporting modules...[/bold blue]"):
        for m_idx, module in enumerate(modules, start=1):
            mid = module.id
            mname = getattr(module, "name", "Unnamed Module")
            folder_slug = f"{m_idx:02d}-{slugify(mname)}"
            module_dir = out_path / folder_slug
            module_dir.mkdir(parents=True, exist_ok=True)
            files_dir = module_dir / "files"

            index_lines.append(f"### {mname} (ID: {mid})\n\n")
            items = getattr(module, "items", [])

            for i_idx, item in enumerate(items, start=1):
                iid = item.get("id")
                ititle = item.get("title", "Untitled")
                itype = item.get("type", "")
                content_id = item.get("content_id")
                file_slug = f"{i_idx:02d}-{slugify(ititle)}.md"
                file_path = module_dir / file_slug

                index_lines.append(f"- [{ititle}]({folder_slug}/{file_slug}) ({itype})\n")

                content_lines = [
                    f"# {ititle}\n\n",
                    f"- Type: {itype}\n",
                    f"- Item ID: {iid}\n",
                    f"- Module ID: {mid}\n",
                ]

                if itype == "Assignment" and content_id:
                    try:
                        assign = course.get_assignment(content_id)
                        content_lines.append(f"- Due Date: {getattr(assign, 'due_at', 'None')}\n")
                        content_lines.append(f"- Points: {getattr(assign, 'points_possible', 'None')}\n\n")
                        content_lines.append("## Instructions\n\n")
                        content_lines.append(f"{html_to_markdown(getattr(assign, 'description', ''))}\n")
                    except (CanvasException, requests.RequestException) as err:
                        content_lines.append(f"\nNotice: Failed to fetch assignment details: {err}\n")
                elif itype == "Page":
                    page_url = item.get("page_url")
                    if page_url:
                        try:
                            page = course.get_page(page_url)
                            content_lines.append("\n## Content\n\n")
                            content_lines.append(f"{html_to_markdown(getattr(page, 'body', ''))}\n")
                        except (CanvasException, requests.RequestException) as err:
                            content_lines.append(f"\nNotice: Failed to fetch page details: {err}\n")
                elif itype == "File" and content_id:
                    try:
                        f = course.get_file(content_id)
                        raw_fname = getattr(f, "display_name", "") or f"file_{content_id}"
                        safe_fname = f"{content_id}_{Path(raw_fname).name}"
                        furl = getattr(f, "url", "")

                        content_lines.append(f"- File Name: {raw_fname}\n")
                        if furl:
                            files_dir.mkdir(parents=True, exist_ok=True)
                            saved_file = files_dir / safe_fname
                            if download_file(furl, saved_file):
                                content_lines.append(f"- Local File: [files/{safe_fname}](files/{safe_fname})\n")
                            content_lines.append(f"- Download URL: {furl}\n")
                    except (CanvasException, requests.RequestException) as err:
                        content_lines.append(f"\nNotice: Failed to fetch file details: {err}\n")
                elif itype == "Quiz" and content_id:
                    try:
                        quiz = course.get_quiz(content_id)
                        content_lines.append(f"- Due Date: {getattr(quiz, 'due_at', 'None')}\n")
                        content_lines.append(f"- Points: {getattr(quiz, 'points_possible', 'None')}\n\n")
                        content_lines.append("## Quiz Description\n\n")
                        content_lines.append(f"{html_to_markdown(getattr(quiz, 'description', ''))}\n")
                    except (CanvasException, requests.RequestException) as err:
                        content_lines.append(f"\nNotice: Failed to fetch quiz details: {err}\n")

                file_path.write_text("".join(content_lines), encoding="utf-8")

            index_lines.append("\n")

    index_path = out_path / "index.md"
    index_path.write_text("".join(index_lines), encoding="utf-8")
    console.print(f"[3/3] Export complete: [bold green]{output_dir}/[/bold green]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
