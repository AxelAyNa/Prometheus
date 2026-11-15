"""Maintainer tool: puts the NASA reference PDFs inside the project, in data/NASA References/.

Run it once on your computer before committing, so the PDFs are part of the repository and of the published
site (users never download anything; the dashboard reads them from that folder):
    python3 fetch_references.py
Files that are already there are skipped. To add a document, add a line to DOCS and run it again.
"""
import ssl
import sys
import urllib.request
from pathlib import Path

try:
    import certifi
    CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    CTX = ssl.create_default_context()

FOLDER = "NASA References"
DEST = Path(__file__).resolve().parent / "data" / FOLDER

# key, file name, short label, url (or several urls: they are tried in order until one gives a PDF)
DOCS = [
    ("guide", "Combustion Science - A Researcher's Guide (ISS).pdf", "Combustion Science: A Researcher's Guide (ISS)",
     "https://www.nasa.gov/wp-content/uploads/2016/05/np-2015-10-034-jsc_combustion_in_reduced_gravity-iss-mini-book-011116-508.pdf"),
    ("apollo1_senate", "Apollo 1 - Senate Report on the Apollo 204 Accident.pdf", "Apollo 1: Senate report",
     "https://history.nasa.gov/as204_senate_956.pdf"),
    ("apollo13_board", "Apollo 13 - Review Board Report.pdf", "Apollo 13: Review Board report",
     ("https://www.nasa.gov/wp-content/uploads/static/history/afj/ap13fj/pdf/report-of-a13-review-board-19700615-19700076776.pdf",
      "https://ntrs.nasa.gov/api/citations/20170000913/downloads/20170000913.pdf")),
    ("apollo13_mission", "Apollo 13 - Mission Report.pdf", "Apollo 13: Mission report",
     ("https://www.nasa.gov/wp-content/uploads/static/history/alsj/a13/a13_missionreport.pdf",
      "https://nasa.gov/wp-content/uploads/static/history/alsj/a13/a13_missionreport.pdf",
      "https://ntrs.nasa.gov/api/citations/19710003598/downloads/19710003598.pdf",
      "https://www.nasa.gov/wp-content/uploads/static/history/afj/ap13fj/pdf/a13-mission-report-197007.pdf")),
]

README = """---
title: NASA References
---
Reference documents about fire safety in crewed spaceflight, downloaded from NASA and kept inside the dashboard.

- Combustion Science - A Researcher's Guide (ISS): NASA guide to combustion research on the International Space Station: why low gravity matters, past research, lessons learned and future opportunities.
- Apollo 1 - Senate Report: U.S. Senate report on the 1967 Apollo 204 cabin fire, which happened in a 100% oxygen atmosphere during a launch pad test.
- Apollo 13 - Review Board Report and Mission Report: investigation and mission report of the 1970 oxygen tank explosion (an oxygen-system failure, not a cabin fire).
"""


def status():
    out = []
    for key, name, label, url in DOCS:
        p = DEST / name
        out.append({"key": key, "file": name, "label": label, "id": f"{FOLDER}/{name}",
                    "local": p.exists() and p.stat().st_size > 0, "size": p.stat().st_size if p.exists() else 0})
    return out


def fetch_all(progress=None):
    """Download the missing documents. progress(name) is called before each download. Returns a list of errors."""
    DEST.mkdir(parents=True, exist_ok=True)
    readme = DEST / "README.md"
    if not readme.exists():
        readme.write_text(README, encoding="utf-8")
    errors = []
    for key, name, label, url in DOCS:
        path = DEST / name
        if path.exists() and path.stat().st_size > 0:
            continue
        if progress:
            progress(label)
        tried = []
        for u in ((url,) if isinstance(url, str) else url):
            try:
                req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=300, context=CTX) as r:
                    data = r.read()
                if not data.startswith(b"%PDF"):
                    raise ValueError("the answer is not a PDF")
                tmp = path.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(path)
                tried = []
                break
            except Exception as e:
                tried.append(f"{u} -> {e}")
        if tried:
            errors.append(f"{label}: " + " | ".join(tried))
    return errors


if __name__ == "__main__":
    errs = fetch_all(lambda n: print("  downloading:", n))
    for s in status():
        print(f"  {'OK ' if s['local'] else '-- '} {s['file']}  {s['size'] / 1048576:.1f} MB")
    for e in errs:
        print("  FAILED", e)
    sys.exit(1 if errs else 0)
