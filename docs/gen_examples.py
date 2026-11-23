"""Generate example documentation pages and screenshots.

Run this script before building or serving the docs:

    python docs/gen_examples.py

For each Python file in examples/, this script:
  - Executes it with snx.show/run patched to capture an offscreen screenshot
  - Parses its module-level docstring and code
  - Writes docs/examples/{name}.md with title, description, screenshot, and code block

docs/examples/index.md is also regenerated as a gallery of all examples.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import scenex as snx

if TYPE_CHECKING:
    import numpy as np

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
DOCS_EXAMPLES_DIR = Path(__file__).parent / "examples"
IMAGES_DIR = DOCS_EXAMPLES_DIR / "images"


def _parse_example(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
        docstring = ast.get_docstring(tree) or ""
        if docstring and tree.body and isinstance(tree.body[0], ast.Expr):
            end_line = tree.body[0].end_lineno
            code = "\n".join(source.splitlines()[end_line:]).lstrip("\n")
        else:
            code = source
    except SyntaxError:
        docstring = ""
        code = source
    return {
        "name": path.stem,
        "title": path.stem.replace("_", " ").title(),
        "docstring": docstring,
        "code": code,
    }


def _save_png(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(img).save(path)


def _take_screenshot(path: Path) -> bool:
    """Execute an example with patched snx.show/run and save a PNG screenshot.

    Returns True if the PNG was written successfully.
    """
    name = path.stem
    output_path = IMAGES_DIR / f"{name}.png"
    source = path.read_text(encoding="utf-8")

    captured: list[snx.Canvas] = []
    original_show = snx.show

    def patched_show(obj: Any, **kwargs: Any) -> snx.Canvas:
        canvas = original_show(obj, **kwargs)
        captured.append(canvas)
        return canvas

    def patched_run() -> None:
        if not captured:
            return
        canvas = captured[-1]
        try:
            _save_png(canvas.render(), output_path)
        except Exception as e:
            print(f"  render failed: {e}")
        finally:
            for c in captured:
                try:
                    c.close()
                except Exception:
                    pass

    namespace: dict = {"__name__": "__main__", "__file__": str(path)}
    with (
        patch.object(snx, "show", side_effect=patched_show),
        patch.object(snx, "run", side_effect=patched_run),
    ):
        try:
            exec(compile(source, str(path), "exec"), namespace)
        except SystemExit:
            pass
        except Exception as e:
            print(f"  exec failed: {e}")
            return False

    return output_path.exists()


def _write_example_md(info: dict) -> None:
    name = info["name"]
    lines: list[str] = [f"# {info['title']}", ""]
    if info["docstring"]:
        lines += [info["docstring"], ""]
    img = IMAGES_DIR / f"{name}.png"
    if img.exists():
        lines += [f"![Screenshot of {info['title']}](images/{name}.png)", ""]
    lines += ["```python", info["code"].rstrip(), "```", ""]
    (DOCS_EXAMPLES_DIR / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")


def _update_nav(examples: list[dict]) -> None:
    toml_path = Path(__file__).parent.parent / "zensical.toml"
    content = toml_path.read_text(encoding="utf-8")
    pages = ["examples/index.md"] + [f"examples/{ex['name']}.md" for ex in examples]
    items = ", ".join(f'"{p}"' for p in pages)
    new_entry = f"{{Examples = [{items}]}}"
    content = re.sub(r"\{Examples = \[.*?\]\}", new_entry, content)
    toml_path.write_text(content, encoding="utf-8")


def _write_index_md(examples: list[dict]) -> None:
    lines: list[str] = ["---", "icon: material/flask-outline", "---", "# Examples", ""]
    for ex in examples:
        lines += [
            "---",
            "",
            f"## [{ex['title']}]({ex['name']}.md)",
            "",
            ex["docstring"] or "*No description.*",
            "",
        ]
    (DOCS_EXAMPLES_DIR / "index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Generate a documentation page for each example."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    examples: list[dict] = []
    screenshots_ok = screenshots_failed = 0

    for path in sorted(EXAMPLES_DIR.glob("*.py")):
        if path.name == "conftest.py":
            continue
        print(f"{path.stem}...")
        if _take_screenshot(path):
            screenshots_ok += 1
        else:
            screenshots_failed += 1
        info = _parse_example(path)
        _write_example_md(info)
        examples.append(info)

    _write_index_md(examples)
    _update_nav(examples)

    n = len(examples)
    print(f"\n{n} example pages written")
    print(f"Screenshots: {screenshots_ok} ok, {screenshots_failed} failed")


if __name__ == "__main__":
    main()
