"""Conservative secret-pattern scan of blobs reachable from Git refs.

Reports object IDs and rule names only; never prints candidate secret values.
This is a heuristic, not a complete credential or licensing audit.
"""
from __future__ import annotations

import re
import subprocess


PATTERNS = {
    "private-key-header": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"),
    "github-token": re.compile(rb"\b(?:ghp_|gho_|ghu_|ghs_|ghr_)[A-Za-z0-9_]{30,}\b"),
    "github-fine-grained-token": re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    "aws-access-key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "slack-token": re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{15,}\b"),
}


def detect_secret_patterns(content: bytes) -> list[str]:
    return sorted(name for name, pattern in PATTERNS.items() if pattern.search(content))


def main() -> None:
    objects = subprocess.check_output(
        ["git", "rev-list", "--objects", "--all"], text=True,
    )
    blob_ids = set()
    synthetic_fixture_blobs = set()
    for line in objects.splitlines():
        oid, _, path = line.partition(" ")
        if path == "tests/test_history_secrets.py":
            synthetic_fixture_blobs.add(oid)
        if subprocess.check_output(
            ["git", "cat-file", "-t", oid], text=True,
        ).strip() == "blob":
            blob_ids.add(oid)

    findings = []
    skipped = 0
    for oid in sorted(blob_ids - synthetic_fixture_blobs):
        size = int(subprocess.check_output(
            ["git", "cat-file", "-s", oid], text=True,
        ).strip())
        if size > 2_000_000:
            skipped += 1
            continue
        content = subprocess.check_output(["git", "cat-file", "blob", oid])
        for rule in detect_secret_patterns(content):
            findings.append((oid, rule))

    if findings:
        print("Potential secret patterns in reachable Git blobs (values redacted):")
        for oid, rule in findings:
            print(f"  blob {oid}: {rule}")
        raise SystemExit(1)
    print(f"Reachable Git blob pattern scan passed ({len(blob_ids) - len(synthetic_fixture_blobs)} scanned; {len(synthetic_fixture_blobs)} synthetic-fixture blobs excluded; {skipped} oversized skipped).")
    print("Synthetic secret-scanner test fixture history is excluded to avoid known demonstration-pattern false positives.")
    print("Heuristic only: not proof of absence of credentials or restricted content.")


if __name__ == "__main__":
    main()
