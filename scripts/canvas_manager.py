#!/usr/bin/env python3

import os
import sys
import json
import re
import html
import argparse
import urllib.request
import urllib.error

ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
DEFAULT_COURSE_KEYWORD = "Undergraduate Research 1"

def load_config():
    config = {}
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    config[key.strip()] = val.strip().strip('"').strip("'")
    base_url = config.get("CANVAS_BASE_URL", "https://ceu.instructure.com").rstrip("/")
    token = config.get("CANVAS_API_TOKEN", "")
    return base_url, token

def clean_html(raw_html):
    if not raw_html:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", raw_html, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<li[^>]*>", "\n* ", text, flags=re.IGNORECASE)
    text = re.sub(r"<h[1-6][^>]*>(.*?)</h[1-6]>", r"\n\n## \1\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def canvas_get(endpoint_or_url, base_url, token):
    if endpoint_or_url.startswith("http"):
        url = endpoint_or_url
    else:
        url = f"{base_url}/api/v1/{endpoint_or_url.lstrip('/')}"

    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "ThesisCanvasManager/1.0"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        print(f"Error {error.code}: {url}")
        return None
    except Exception as error:
        print(f"Network error: {error}")
        return None

def get_course_id(base_url, token, keyword=DEFAULT_COURSE_KEYWORD):
    courses = canvas_get("courses?enrollment_state=active&per_page=50", base_url, token)
    if not courses:
        return None, None
    for course in courses:
        name = course.get("name", "")
        code = course.get("course_code", "")
        if keyword.lower() in name.lower() or keyword.lower() in code.lower():
            return course.get("id"), name
    return None, None

def get_modules(course_id, base_url, token):
    modules = canvas_get(f"courses/{course_id}/modules?include[]=items&per_page=50", base_url, token)
    if not modules:
        print("No modules found.")
        return

    for module in modules:
        mid = module.get("id")
        mname = module.get("name")
        items = module.get("items", [])
        print(f"Module ID: {mid}")
        print(f"Module Name: {mname}")
        print(f"Item Count: {len(items)}")
        for item in items:
            iid = item.get("id")
            ititle = item.get("title")
            itype = item.get("type")
            print(f"  - Item ID: {iid} | Type: {itype} | Title: {ititle}")
        print()

def get_module(course_id, module_id, base_url, token):
    items = canvas_get(f"courses/{course_id}/modules/{module_id}/items?per_page=50", base_url, token)
    if not items:
        print(f"No items found for module {module_id}.")
        return

    print(f"Module ID: {module_id}")
    for item in items:
        iid = item.get("id")
        ititle = item.get("title")
        itype = item.get("type")
        api_url = item.get("url")
        print(f"Item ID: {iid}")
        print(f"Title: {ititle}")
        print(f"Type: {itype}")
        if api_url:
            data = canvas_get(api_url, base_url, token)
            if data:
                if itype == "Assignment":
                    print(f"Due Date: {data.get('due_at')}")
                    print(f"Points Possible: {data.get('points_possible')}")
                elif itype == "File":
                    print(f"Download URL: {data.get('url')}")
        print()

def get_module_item(course_id, item_id, base_url, token):
    modules = canvas_get(f"courses/{course_id}/modules?include[]=items&per_page=50", base_url, token)
    target_item = None
    for module in modules:
        for item in module.get("items", []):
            if str(item.get("id")) == str(item_id):
                target_item = item
                break
        if target_item:
            break

    if not target_item:
        print(f"Item ID: {item_id} not found.")
        return

    ititle = target_item.get("title")
    itype = target_item.get("type")
    api_url = target_item.get("url")

    print(f"Item ID: {item_id}")
    print(f"Title: {ititle}")
    print(f"Type: {itype}")

    if api_url:
        data = canvas_get(api_url, base_url, token)
        if data:
            if itype == "Assignment":
                print(f"Due Date: {data.get('due_at')}")
                print(f"Points: {data.get('points_possible')}")
                desc = clean_html(data.get("description", ""))
                print(f"Instructions:\n{desc}")
            elif itype == "Page":
                body = clean_html(data.get("body", ""))
                print(f"Content:\n{body}")
            elif itype == "File":
                print(f"File Name: {data.get('display_name')}")
                print(f"File Size: {data.get('size')} bytes")
                print(f"Download URL: {data.get('url')}")

def get_announcements(course_id, base_url, token):
    notices = canvas_get(f"courses/{course_id}/discussion_topics?only_announcements=true&per_page=15", base_url, token)
    if not notices:
        print("No announcements found.")
        return

    for notice in notices:
        title = notice.get("title")
        posted = notice.get("posted_at")
        body = clean_html(notice.get("message", ""))
        print(f"Title: {title}")
        print(f"Date: {posted}")
        print(f"Message:\n{body}\n")

def slugify(text):
    if not text:
        return "untitled"
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s_]+", "-", text).strip("-")
    return text[:60] if text else "untitled"

def export_markdown(course_id, output_dir, base_url, token):
    os.makedirs(output_dir, exist_ok=True)
    modules = canvas_get(f"courses/{course_id}/modules?include[]=items&per_page=50", base_url, token)
    if not modules:
        print("Export failed: no modules retrieved.")
        return

    index_lines = ["# Course Modules Index\n\n"]
    for m_idx, module in enumerate(modules, start=1):
        mid = module.get("id")
        mname = module.get("name", "Unnamed Module")
        folder_slug = f"{m_idx:02d}-{slugify(mname)}"
        module_dir = os.path.join(output_dir, folder_slug)
        os.makedirs(module_dir, exist_ok=True)

        index_lines.append(f"## {mname} (ID: {mid})\n")
        items = module.get("items", [])
        for i_idx, item in enumerate(items, start=1):
            iid = item.get("id")
            ititle = item.get("title", "Untitled")
            itype = item.get("type")
            api_url = item.get("url")
            file_slug = f"{i_idx:02d}-{slugify(ititle)}.md"
            file_path = os.path.join(module_dir, file_slug)

            index_lines.append(f"- [{ititle}]({folder_slug}/{file_slug}) ({itype})\n")

            content_lines = [
                f"# {ititle}\n\n",
                f"- Type: {itype}\n",
                f"- Item ID: {iid}\n",
                f"- Module ID: {mid}\n",
            ]
            if api_url:
                data = canvas_get(api_url, base_url, token)
                if data:
                    if itype == "Assignment":
                        content_lines.append(f"- Due Date: {data.get('due_at')}\n")
                        content_lines.append(f"- Points: {data.get('points_possible')}\n\n")
                        content_lines.append("## Instructions\n\n")
                        content_lines.append(clean_html(data.get("description", "")))
                    elif itype == "Page":
                        content_lines.append("\n## Content\n\n")
                        content_lines.append(clean_html(data.get("body", "")))
                    elif itype == "File":
                        content_lines.append(f"- File Name: {data.get('display_name')}\n")
                        content_lines.append(f"- Download URL: {data.get('url')}\n")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(content_lines))

        index_lines.append("\n")

    with open(os.path.join(output_dir, "index.md"), "w", encoding="utf-8") as f:
        f.write("".join(index_lines))
    print(f"Export completed: {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description="Canvas LMS Course Manager")
    parser.add_argument("--modules", action="store_true", help="List all course modules and items")
    parser.add_argument("--module", type=int, help="Inspect a specific module by ID")
    parser.add_argument("--item", type=int, help="Inspect a specific module item by ID")
    parser.add_argument("--announcements", action="store_true", help="List course announcements")
    parser.add_argument("--export", action="store_true", help="Export modules and guidelines to Markdown")
    parser.add_argument("--output-dir", default="canvas_docs", help="Target directory for Markdown export")
    parser.add_argument("--course-id", type=int, help="Override course identifier")
    args = parser.parse_args()

    base_url, token = load_config()
    if not token:
        print("Error: CANVAS_API_TOKEN missing from .env")
        sys.exit(1)

    course_id = args.course_id
    if not course_id:
        course_id, course_name = get_course_id(base_url, token)
        if not course_id:
            print("Error: Could not locate course automatically.")
            sys.exit(1)
        print(f"Course: {course_name} (ID: {course_id})")

    if args.modules:
        get_modules(course_id, base_url, token)
    elif args.module:
        get_module(course_id, args.module, base_url, token)
    elif args.item:
        get_module_item(course_id, args.item, base_url, token)
    elif args.announcements:
        get_announcements(course_id, base_url, token)
    elif args.export:
        export_markdown(course_id, args.output_dir, base_url, token)
    else:
        get_modules(course_id, base_url, token)

if __name__ == "__main__":
    main()
