import json, os, re, ssl, sys, time, urllib.parse, urllib.request
from collections import Counter

try:
    import certifi
    CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    CTX = ssl.create_default_context()

API = "https://psi.nasa.gov/geode-py/ws"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
OK_EXT = {".pdf", ".csv", ".tsv", ".xlsx", ".xlsm", ".txt", ".md", ".json"} | IMG_EXT
MAX_BYTES = 20 * 1024 * 1024   # documents and tables
MAX_IMG = 7 * 1024 * 1024      # Gemini image limit used by server.py
MAX_FILES = 30                 # per PSI
SKIP_DIRS = {"raw data", "metadata"}
PRIORITY = ("report", "engineering", "experimental table", "representative", "image")
_CACHE = {}


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.read()


def _folder(f):
    # /Home/psi-data/PSI-159/<folder>/file
    parts = f.get("remote_path", "").split("/")
    return parts[4] if len(parts) > 4 else ""


def investigation(psi_id, ttl=3600):
    hit = _CACHE.get(psi_id)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    data = json.loads(_get(f"{API}/repo/investigations/{psi_id}", timeout=120))
    _CACHE[psi_id] = (time.time(), data)
    return data


def psi_info(psi_id):
    """Title, objective and description as plain text."""
    data = investigation(psi_id)
    out = {}
    for k in ("title", "objective", "description"):
        v = data.get(k)
        if v:
            out[k] = " ".join(re.sub(r"<[^>]+>", " ", str(v)).split())
    return out


def psi_folders(psi_id):
    return Counter(_folder(f) for f in investigation(psi_id).get("added", []))


def _rank(f):
    d = _folder(f).lower()
    for i, k in enumerate(PRIORITY):
        if k in d:
            return i
    return len(PRIORITY)


def psi_files(psi_id):
    """Useful files only: no Raw Data, readable types, sane sizes, at most MAX_FILES."""
    out = []
    for f in investigation(psi_id).get("added", []):
        if f.get("restricted") or not f.get("visible", True):
            continue
        ext = os.path.splitext(f["file_name"])[1].lower()
        if ext not in OK_EXT or _folder(f).lower() in SKIP_DIRS:
            continue
        if f.get("file_size", 0) > (MAX_IMG if ext in IMG_EXT else MAX_BYTES):
            continue
        out.append(f)
    out.sort(key=lambda f: (_rank(f), f.get("file_size", 0)))
    return out[:MAX_FILES]


def psi_download(psi_id, f):
    """Download a file into memory and return its bytes. Nothing is written to disk."""
    q = urllib.parse.urlencode({"file": f["file_name"], "version": f.get("version", 1), "redirect": "false"})
    signed = _get(f"{API}/studies/{psi_id}/download?{q}").decode().strip().strip('"')
    return _get(signed, timeout=180)


if __name__ == "__main__":
    psi = sys.argv[1] if len(sys.argv) > 1 else "PSI-159"
    print(f"Folders in {psi}:")
    for name, n in psi_folders(psi).most_common():
        print(f"  {name or '(root)'}: {n} files")
    files = psi_files(psi)
    print(f"\n{len(files)} useful files in {psi}")
    for f in files:
        print(" -", _folder(f), "|", f["file_name"], f["file_size"])
    print("\nInfo:", psi_info(psi))
