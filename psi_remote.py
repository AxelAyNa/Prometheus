import json, os, re, ssl, sys, time, urllib.parse, urllib.request
from collections import Counter

try:
    import certifi
    CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    CTX = ssl.create_default_context()

API = "https://psi.nasa.gov/geode-py/ws"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
OK_EXT = {".pdf", ".csv", ".tsv", ".xlsx", ".xlsm", ".xls", ".docx", ".txt", ".md", ".json"} | IMG_EXT
MB = 1024 * 1024
MAX_PDF = 50 * MB              # reports can be large
MAX_BYTES = 20 * MB            # tables and other documents
MAX_IMG = 7 * MB               # Gemini image limit used by server.py
MAX_FILES = 30                 # per PSI, not counting Raw Data
MAX_RAW = 20                   # Raw Data: only PDFs (any subfolder), at most the 20 most relevant
MAX_PRES = 3                   # presentations are often repeated talks
SKIP_DIRS = {"metadata"}
PRIORITY = (("report", 0), ("experimental table", 1), ("analyzed data", 2), ("engineering", 3),
            ("science doc", 4), ("representative", 5), ("image", 5), ("reference data", 6), ("presentation", 8))
_CACHE = {}
_VCACHE = {}
RAW_KEYS = ("report", "final", "summary", "overview", "requirement", "eird", "srd", "results", "paper", "journal",
            "proceedings", "conference", "meeting", "symposium", "combustion", "flame", "fire", "test plan", "analysis")


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
    return Counter(_folder(f) for f in current_files(psi_id))


def _rank(f):
    d = _folder(f).lower()
    if d == "raw data":
        return 9
    for k, r in PRIORITY:
        if k in d:
            return r
    return 7


def _in_subfolder(f):
    """True when the file sits in a subfolder of its NASA folder (e.g. Raw Data/Run 3/file.csv)."""
    parts = [p for p in f.get("remote_path", "").split("/")]
    rest = parts[5:]
    if rest and rest[-1] == f.get("file_name"):
        rest = rest[:-1]
    return any(p.strip() for p in rest)


def _date(f):
    v = (f.get("date_modified") or {})
    v = v.get("$date", 0) if isinstance(v, dict) else v
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _norm(name):
    """Name without copy/version marks, so 'Report (1).pdf', 'Report v2.pdf' and 'Report_Final.pdf' match."""
    base = os.path.splitext(name)[0].lower().strip()
    pats = (r"\s*\(\d+\)$", r"[\s_\-]*(copy|final|draft)$", r"[\s_\-]*v(er)?\.?\s?\d+(\.\d+)*$", r"[\s_\-]*rev\.?\s?\d*$")
    prev = None
    while prev != base:
        prev = base
        for p in pats:
            base = re.sub(p, "", base).strip()
    return re.sub(r"[\s_\-]+", " ", base)


def all_files(data):
    """Every file entry in the investigation JSON. Most PSIs list them under "added", but some use other keys
    or nest them, so walk the whole document and keep each file once (by path, else by name)."""
    found, seen = [], set()

    def walk(v, depth):
        if depth > 5:
            return
        if isinstance(v, dict):
            if v.get("file_name") and isinstance(v.get("file_name"), str):
                key = v.get("remote_path") or v["file_name"]
                if key not in seen:
                    seen.add(key)
                    found.append(v)
                return
            for x in v.values():
                walk(x, depth + 1)
        elif isinstance(v, list):
            for x in v:
                walk(x, depth + 1)

    walk(data.get("added", []), 0)
    walk(data, 0)
    return found


def current_files(psi_id, ttl=3600):
    """Every file the PSI has now. The investigation answer only lists what its LATEST version added, so a PSI
    updated later (e.g. a metadata-only version) looks empty. /versions has every version: replay them in order
    (added and modified in, removed out). Falls back to the plain answer if /versions is not available."""
    hit = _VCACHE.get(psi_id)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    cur = {}
    key = lambda f: f.get("remote_path") or f.get("file_name")
    try:
        vs = json.loads(_get(f"{API}/repo/investigations/{psi_id}/versions", timeout=180)).get("versions") or []
        for v in sorted(vs, key=lambda v: float(str(v.get("version") or 0))):
            for f in (v.get("added") or []) + (v.get("modified") or []):
                if isinstance(f, dict) and f.get("file_name"):
                    cur[key(f)] = f
            for f in v.get("removed") or []:
                cur.pop(key(f) if isinstance(f, dict) else f, None)
    except Exception:
        pass
    for f in all_files(investigation(psi_id)):
        cur.setdefault(key(f), f)
    files = list(cur.values())
    _VCACHE[psi_id] = (time.time(), files)
    return files


def _sub(f):
    """Subfolders inside the NASA folder, from 'subcategory' (e.g. 'Legacy/FLEX HEP/FLEX 266...')."""
    return [p for p in str(f.get("subcategory") or "").replace("&quot;", '"').strip().strip("/").split("/") if p.strip()]


def _raw_score(f):
    """Importance of a Raw Data PDF: reports, publications and summaries first, shallow before deep, bigger first."""
    sub = _sub(f)
    text = (f.get("file_name", "") + " " + " ".join(sub)).lower()
    score = -2 * len(sub)
    score += 3 * sum(k in text for k in RAW_KEYS)
    if any(re.match(r"\d{4}\s*-\s*\S", p) for p in sub):
        score += 4
    return score + min((f.get("file_size") or 0) / MB, 10) / 2


def display_name(f, psi_id):
    """Name shown in the dashboard. NASA names usually start with 'PSI-N_<folder>_'; Raw Data files often do not
    (e.g. '20140006750.pdf'), so prefix them with their folder and the name of their subfolder, which is the title."""
    name = f.get("file_name", "")
    if re.match(r"PSI-?\d+_", name):
        return name
    sub = _sub(f)
    title = re.sub(r"[\\/:*?<>|]+", " ", sub[-1]).strip() if sub else ""
    title = re.sub(r"\s+", " ", title)[:90].strip()
    base, ext = os.path.splitext(name)
    label = f"{title} · {base}" if title and title.lower() != base.lower() else base
    return f"{psi_id}_{_folder(f) or 'Files'}_{label}{ext}"


def psi_select(psi_id):
    """Pick the useful files of a PSI and explain every file left out.

    Returns (files, skipped) where skipped is a list of {"name", "reason"}.
    Raw Data: only PDF files (at any depth), and at most the MAX_RAW most relevant ones.
    """
    skipped, cand, raw_other = [], [], 0
    entries = current_files(psi_id)
    if not entries:
        return [], [{"name": psi_id, "reason": "NASA lists no downloadable files for this PSI (it may only have external links, or data on request)"}]
    for f in entries:
        name, folder = f.get("file_name", ""), _folder(f)
        fl, ext, size = folder.lower(), os.path.splitext(f.get("file_name", ""))[1].lower(), f.get("file_size", 0) or 0
        if f.get("restricted") or not f.get("visible", True):
            skipped.append({"name": name, "reason": "restricted on NASA"})
            continue
        if fl in SKIP_DIRS:
            skipped.append({"name": name, "reason": f"{folder} folder"})
            continue
        if fl == "raw data" and ext != ".pdf":
            raw_other += 1
            continue
        if ext not in OK_EXT:
            skipped.append({"name": name, "reason": f"{ext or 'no extension'} files are not supported"})
            continue
        limit = MAX_IMG if ext in IMG_EXT else MAX_PDF if ext == ".pdf" else MAX_BYTES
        if size > limit:
            skipped.append({"name": name, "reason": f"too big (over {limit // MB} MB)"})
            continue
        cand.append(f)
    groups = {}
    for f in cand:
        groups.setdefault((_folder(f).lower(), _norm(f["file_name"]), os.path.splitext(f["file_name"])[1].lower()), []).append(f)
    unique = []
    for fs in groups.values():
        fs.sort(key=lambda f: (_date(f), f.get("file_size", 0)), reverse=True)
        unique.append(fs[0])
        for f in fs[1:]:
            same = f.get("file_size") == fs[0].get("file_size")
            skipped.append({"name": f["file_name"], "reason": "duplicate copy" if same else "older version of another file"})
    unique.sort(key=lambda f: (_rank(f), -(_raw_score(f) if _rank(f) == 9 else 0), -_date(f), -(f.get("file_size") or 0)))
    out, n_main, n_raw, n_pres = [], 0, 0, 0
    for f in unique:
        r = _rank(f)
        if r == 9:
            if n_raw >= MAX_RAW:
                skipped.append({"name": f["file_name"], "reason": f"Raw Data: not among the {MAX_RAW} most relevant PDFs"})
                continue
            n_raw += 1
        else:
            if r == 8 and n_pres >= MAX_PRES:
                skipped.append({"name": f["file_name"], "reason": f"extra presentation (max {MAX_PRES})"})
                continue
            if n_main >= MAX_FILES:
                skipped.append({"name": f["file_name"], "reason": f"over the {MAX_FILES}-file limit"})
                continue
            n_main += 1
            n_pres += r == 8
        out.append(f)
    if raw_other:
        skipped.append({"name": f"{raw_other} files", "reason": "Raw Data: only PDF files are loaded (data, text, zip and video files are skipped)"})
    return out, skipped


def psi_files(psi_id):
    """Useful files only (see psi_select)."""
    return psi_select(psi_id)[0]


def psi_download(psi_id, f):
    """Download a file into memory and return its bytes. Nothing is written to disk."""
    q = urllib.parse.urlencode({"file": f["file_name"], "version": f.get("version", 1), "redirect": "false"})
    signed = _get(f"{API}/studies/{psi_id}/download?{q}").decode().strip().strip('"')
    return _get(signed, timeout=180)


if __name__ == "__main__":
    psi = sys.argv[1] if len(sys.argv) > 1 else "PSI-159"
    data = investigation(psi)
    print("Top-level keys:", {k: (f"list[{len(v)}]" if isinstance(v, list) else type(v).__name__) for k, v in data.items()})
    print(f"Folders in {psi}:")
    for name, n in psi_folders(psi).most_common():
        print(f"  {name or '(root)'}: {n} files")
    files, skipped = psi_select(psi)
    print(f"\n{len(files)} useful files in {psi}")
    for f in files:
        print(" -", _folder(f), "|", f["file_name"], f["file_size"])
    print(f"\n{len(skipped)} left out:")
    for s in skipped:
        print(" -", s["reason"], "|", s["name"])
    print("\nInfo:", psi_info(psi))
