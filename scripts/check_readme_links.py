"""Validate local Markdown links in the main README against the working tree."""
from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def broken_local_links(markdown: str, root: Path) -> list[str]:
    broken = []
    for match in LINK_PATTERN.finditer(markdown):
        target = match.group(1).strip().split(" ", 1)[0].strip("<>")
        if not target or target.startswith("#"):
            continue
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or target.startswith("//"):
            continue
        path = unquote(parsed.path)
        if not path:
            continue
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
            broken.append(target)
    return sorted(set(broken))


def main():
    broken = broken_local_links((ROOT / "README.md").read_text(encoding="utf-8"), ROOT)
    if broken:
        print("Broken or unsafe local README links:")
        for link in broken:
            print(f"  {link}")
        raise SystemExit(1)
    print("README local links resolve to files within the repository.")


if __name__ == "__main__":
    main()
