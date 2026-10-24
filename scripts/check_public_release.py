"""Basic pre-publish screening of the working folder.

A quick safety net, not a security audit: it cannot see secrets inside
images or anything already stored in earlier Git commits.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOT_FILES = {"README.md", ".gitignore", "requirements.txt", "LICENSE"}
ALLOWED_TOP_DIRS = {"src", "docs", "scripts", "tests", "notebooks", ".github", "data", "outputs"}
ALLOWED_SUFFIXES = {".md", ".py", ".csv", ".png", ".yml", ".yaml", ".ipynb", ".txt", ""}
DENY_NAME_PATTERNS = (".env", "token", "password", "credential", "secret", ".pdf", ".xlsx",
                      ".xls", ".docx", ".zip", ".pem", ".key", "private", "handoff")
LOCAL_ONLY_DIRS = {"data/raw", "outputs/local"}
SKIP_DIR_NAMES = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
TEXT_SUFFIXES = {".md", ".py", ".csv", ".yml", ".yaml", ".ipynb", ".txt"}

# Each pattern needs an actual VALUE, so prose that merely mentions
# "API key" or "password" does not trigger it.
CONTENT_PATTERNS = {
    "API/write key value": re.compile(r"(api[_-]?key|write[_-]?key|read[_-]?key)\s*[:=]\s*['\"]?[A-Za-z0-9]{8,}", re.I),
    "Wi-Fi credential value": re.compile(r"(ssid|wifi[_-]?pass(word)?|password)\s*[:=]\s*['\"][^'\"]{3,}['\"]", re.I),
    "Private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "Email address": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]*[A-Za-z]{2,}\b"),
    "Local user path": re.compile(r"([A-Za-z]:\\Users\\[^\\\s]+|/Users/[^/\s]+|/home/[^/\s]+)"),
}
CONTENT_SCAN_EXEMPT = {"scripts/check_public_release.py", "tests/test_release_check.py"}


def _scan_content(p: Path, rel: str):
    if p.suffix not in TEXT_SUFFIXES or rel in CONTENT_SCAN_EXEMPT:
        return []
    text = p.read_text(encoding="utf-8", errors="ignore")
    problems = [f"Content looks like '{name}': {rel}"
                for name, pattern in CONTENT_PATTERNS.items() if pattern.search(text)]
    if p.suffix == ".ipynb":
        try:
            if any(cell.get("outputs") for cell in json.loads(text).get("cells", [])):
                problems.append(f"Notebook has saved outputs (clear them before committing): {rel}")
        except json.JSONDecodeError:
            problems.append(f"Notebook is not valid JSON: {rel}")
    return problems


def scan(root: Path = ROOT):
    """Return a sorted list of things to review before publishing."""
    problems = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel_path = p.relative_to(root)
        rel, parts = rel_path.as_posix(), rel_path.parts
        if any(part in SKIP_DIR_NAMES for part in parts):
            continue
        if any(rel.startswith(d + "/") for d in LOCAL_ONLY_DIRS):
            if p.name != ".gitkeep":
                problems.append(f"Local-only data present (keep it out of Git): {rel}")
            continue
        if any(bad in p.name.lower() for bad in DENY_NAME_PATTERNS):
            problems.append(f"Suspicious filename: {rel}")
        if len(parts) == 1 and rel not in ALLOWED_ROOT_FILES:
            problems.append(f"Unexpected root file: {rel}")
        elif len(parts) > 1 and parts[0] not in ALLOWED_TOP_DIRS:
            problems.append(f"Unexpected folder: {rel}")
        elif parts[0] == "data" and len(parts) > 1 and rel != "data/README.md" and not rel.startswith("data/demo/"):
            problems.append(f"Unexpected data file: {rel}")
        elif parts[0] == "outputs" and not rel.startswith("outputs/demo/"):
            problems.append(f"Unexpected output file: {rel}")
        if p.suffix not in ALLOWED_SUFFIXES:
            problems.append(f"Unexpected file type: {rel}")
        if p.stat().st_size > 10_000_000:
            problems.append(f"Large file (>10 MB): {rel}")
        problems.extend(_scan_content(p, rel))
    return sorted(set(problems))


if __name__ == "__main__":
    found = scan()
    if found:
        print("REVIEW REQUIRED before publishing:")
        for item in found:
            print(" -", item)
        raise SystemExit(1)
    print("PASS (basic screening only). Still review files and commit history yourself.")
