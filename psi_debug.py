"""Find where the NASA PSI API lists ALL the files of an investigation.

Usage:  python3 psi_debug.py PSI-69
Tries several API addresses and writes the results to psi_debug_<PSI>.txt (plus the raw answers as .json).
Nothing else is changed.
"""
import json
import sys

import psi_remote as pr

psi = sys.argv[1] if len(sys.argv) > 1 else "PSI-69"
lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def count_files(data):
    try:
        return len(pr.all_files(data if isinstance(data, dict) else {"x": data}))
    except Exception:
        return 0


base = pr.investigation(psi)
version = int(str(base.get("version") or 1).split(".")[0])
log(f"{psi}: current version {version}, totalFileSize {base.get('totalFileSize')}, files in the current answer: {count_files(base)}")

tries = [f"{pr.API}/repo/investigations/{psi}?version={v}" for v in range(1, version + 1)]
tries += [
    f"{pr.API}/repo/investigations/{psi}/versions",
    f"{pr.API}/repo/investigations/{psi}/files",
    f"{pr.API}/studies/{psi}/files",
    f"{pr.API}/repo/files/{psi}",
    f"{pr.API}/studies/{psi}",
    f"https://psi.nasa.gov/osdr/data/osd/files/{psi.split('-')[-1]}",
]
for url in tries:
    try:
        raw = pr._get(url, timeout=120)
        try:
            data = json.loads(raw)
        except ValueError:
            log(f"[not JSON, {len(raw)} bytes] {url}")
            continue
        n = count_files(data)
        extra = ""
        if isinstance(data, dict):
            extra = f" | version={data.get('version')} added={len(data.get('added') or [])} removed={len(data.get('removed') or [])}"
        log(f"[{n:5} files] {url}{extra}")
        safe = url.split("/ws/")[-1].replace("/", "_").replace("?", "_").replace("=", "")
        with open(f"psi_debug_{safe}.json", "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1, default=str)
    except Exception as e:
        log(f"[error] {url} -> {str(e)[:120]}")

with open(f"psi_debug_{psi}.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"\nSaved the summary to psi_debug_{psi}.txt")
