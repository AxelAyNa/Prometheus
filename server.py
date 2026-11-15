import hashlib
import io
import json
import mimetypes
import os
import pickle
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

import psi_remote

BASE = Path(__file__).parent
load_dotenv(BASE / ".env")
DATA = Path(os.getenv("PSI_DATA", BASE / "data"))
DATA.mkdir(exist_ok=True)
CACHE_FILE = BASE / "ai_cache.json"
MODEL = os.getenv("GeminiModel", "gemini-3.5-flash")
EXT = {
    ".csv": "csv", ".tsv": "csv", ".xlsx": "xlsx", ".xlsm": "xlsx", ".xls": "xlsx",
    ".pdf": "pdf", ".docx": "docx", ".txt": "text", ".md": "text", ".json": "text",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image", ".gif": "image",
}
MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp", "gif": "image/gif"}
INDEX = Path(os.getenv("PSI_INDEX", BASE / "index.txt"))
PSI_URL = "https://psi.nasa.gov/physci/repo/search?q={q}&data_source=psi&data_type=investigation"
TABLES = {}
FIRE_TERMS = "fire, flame, flammab, combust, ignition, smoke, extinguish, suppress, detect"
ROWS_PER_CHUNK = 50
PREVIEW_ROWS = 40
AI_LIMIT = 40000
RESEARCH_LIMIT = int(os.getenv("ResearchLimit", "400000"))
INDEX_VERSION = 4
INDEX_CACHE = BASE / "index_cache.pkl"
MAX_SEARCH_CHUNKS = 60
CAT_RE = re.compile(r"^(fuel|fuel type|flame type|material|mission|hardware|platform|category)$", re.I)
O2_RE = re.compile(r"o2", re.I)
P_RE = re.compile(r"(atm|psi|kpa|pressure)", re.I)

SYSTEM = (
    "You are a spacecraft fire safety analyst. Use only the content provided below, taken from "
    "NASA Physical Sciences Informatics (PSI) files. Do not invent values, results or facts; "
    "if something is not in the content, say so. Answer in {lang}. Be concise. "
    "Never use LaTeX or $ math notation: write R², ±, ≈, ×, °C as plain text characters."
)
ML_EXPLAIN = (
    "Below are the results of a machine learning model trained just now, on the fly, only with the rows of one NASA PSI "
    "table and evaluated with 5-fold cross-validation (scores come from rows the model had not seen). Explain it to a "
    "non-expert, with simple words and concrete numbers, using exactly this Markdown structure:\n"
    "**In one sentence:** how useful the model is.\n"
    "Then one short block per metric. For regression: "
    "**R² = <value> — <Strong if ≥0.8, Moderate if ≥0.5, Weak if ≥0.2, otherwise Not predictive>**: say what share of the "
    "variation of the target the inputs explain (0.67 → about 67%), and that 1 would be perfect and 0 the same as always "
    "guessing the average; say what that means in practice (good for trends vs exact numbers). "
    "**MAE = <value> — typical error**: the average miss, in the units of the target; give a concrete example using the "
    "target's real range (for example: if it predicts 0.25, the real value is usually between 0.22 and 0.28); compare it "
    "with the baseline MAE (always guessing the average) and say whether the model clearly adds value. "
    "For classification, instead: **Accuracy = <x%> vs <baseline%> baseline** with the same kind of plain explanation.\n"
    "**What drives it:** 2-3 bullets with the most important inputs and what they may mean physically for spacecraft fire safety.\n"
    "**Limitations:** 2 bullets (sample size, data range, near-copies of the target, correlation is not causation).\n"
    "**In short:** one closing sentence. Then **Next step:** one line. Do not overstate the results."
)
ML_PRED_EXPLAIN = (
    "A user typed their own input values into a machine learning model that was trained on the fly with one NASA PSI "
    "table, and got the prediction below. Explain it to a non-expert, with simple words and concrete numbers, using "
    "exactly this Markdown structure:\n"
    "**Prediction:** the predicted value or class. For a number, turn the typical error into a range "
    "(predicted ± MAE → 'the real value would most likely be between A and B'), and compare with the typical case.\n"
    "**How reliable:** one or two sentences using R² (share of variation explained, with the same Strong/Moderate/Weak/"
    "Not predictive label) or accuracy vs baseline.\n"
    "**What moved it:** bullets with the user's values that pushed the prediction up or down vs a typical case, and why "
    "that may make physical sense for spacecraft fire safety.\n"
    "**⚠️ Outside the data:** only if any value is outside the training range, say the prediction is an extrapolation.\n"
    "**In short:** one closing sentence. Do not present it as a validated result."
)
TASKS = {
    "summary": "In 5 short bullets summarize the objective, fuels and conditions, key measurements and main findings.",
    "lunar": "In 5 short bullets explain what this implies for fire safety in a crewed lunar habitat (30% O2, 10.2 psi) and name one data gap.",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1048576
DOCS = {}
LOCK = threading.Lock()
STATE = {"checked": 0.0}
CLIENTS = {}

try:
    CACHE = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
except Exception:
    CACHE = {}

CACHE_LOCK = threading.Lock()


def save_cache():
    with CACHE_LOCK:
        CACHE_FILE.write_text(json.dumps(CACHE, ensure_ascii=False), encoding="utf-8")


def call(fn):
    for i in range(4):
        try:
            return fn()
        except Exception as e:
            retry = any(x in str(e) for x in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
            if i == 3 or not retry:
                raise
            time.sleep(2 * 2 ** i)


def read_text(path):
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return ""


def read_csv(path, data=None):
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    last = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data) if data is not None else path, sep=sep, dtype=str, encoding=enc)
        except UnicodeDecodeError as e:
            last = e
    raise last


def table_chunks(df):
    df = df.fillna("").astype(str)
    stop = min(len(df), ROWS_PER_CHUNK * MAX_SEARCH_CHUNKS)
    return [
        (f"rows {i + 1}-{min(i + ROWS_PER_CHUNK, len(df))}", df.iloc[i:i + ROWS_PER_CHUNK].to_csv(index=False))
        for i in range(0, stop, ROWS_PER_CHUNK)
    ]


def preview(df):
    df = df.fillna("").astype(str)
    return {
        "headers": [str(c) for c in df.columns][:30],
        "rows": df.head(PREVIEW_ROWS).iloc[:, :30].values.tolist(),
    }


def front_matter(text):
    m = re.match(r"^\ufeff?---\s*\n(.*?)\n---", text.replace("\r", ""), re.S)
    out = {}
    if m:
        for line in m.group(1).splitlines():
            k, sep, v = line.partition(":")
            if sep and k.strip():
                out[k.strip().lower()] = v.strip()
    return out


def merge_span(a, b):
    if not a:
        return b
    if not b:
        return a
    return [min(a[0], b[0]), max(a[1], b[1])]


def p_divisor(name):
    n = name.lower()
    if "psi" in n:
        return 14.696
    if "kpa" in n:
        return 101.325
    return 1.0


def facets_of(df):
    out = {"cats": {}, "o2": None, "p": None}
    for c in df.columns:
        name = str(c).strip()
        s = df[c].dropna().astype(str).str.strip()
        s = s[s != ""]
        if s.empty:
            continue
        num = pd.to_numeric(s, errors="coerce").dropna()
        is_num = len(num) / len(s) >= 0.5
        if not is_num and CAT_RE.match(name) and s.nunique() <= 12:
            out["cats"][name.lower()] = sorted(s.unique())
        elif is_num and O2_RE.search(name):
            v = num * 100 if num.max() <= 1.5 else num
            out["o2"] = merge_span(out["o2"], [float(v.min()), float(v.max())])
        elif is_num and P_RE.search(name):
            v = num / p_divisor(name)
            out["p"] = merge_span(out["p"], [float(v.min()), float(v.max())])
    return out


def merge_facets(a, b):
    cats = {k: sorted(set(a["cats"].get(k, [])) | set(b["cats"].get(k, []))) for k in set(a["cats"]) | set(b["cats"])}
    return {"cats": cats, "o2": merge_span(a["o2"], b["o2"]), "p": merge_span(a["p"], b["p"])}


def docx_text(data):
    """Plain text of a .docx (Word) file using only the standard library."""
    import html
    import zipfile
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    xml = re.sub(r"<w:tab/>", "\t", xml)
    xml = re.sub(r"<w:br[^>]*/>", "\n", xml)
    paras = [html.unescape(re.sub(r"<[^>]+>", "", p)).strip() for p in xml.split("</w:p>")]
    return "\n".join(p for p in paras if p)


def load(path, rel, sig, data=None):
    parts = rel.split("/")
    kind = EXT[path.suffix.lower()]
    doc = {
        "id": rel, "name": path.name, "repo": parts[0] if len(parts) > 1 else "General",
        "type": kind, "ext": path.suffix.lower().lstrip("."), "size": sig[1], "mtime": sig[0],
        "chunks": [], "rows": 0, "cols": 0, "pages": 0, "sheets": [],
        "preview": None, "excerpt": "", "error": "",
        "meta": {}, "facets": {"cats": {}, "o2": None, "p": None}, "chars": 0,
    }
    try:
        if kind == "csv":
            df = read_csv(path, data)
            doc["rows"], doc["cols"] = int(df.shape[0]), int(df.shape[1])
            doc["chunks"] = table_chunks(df)
            doc["preview"] = preview(df)
            doc["facets"] = facets_of(df)
        elif kind == "xlsx":
            try:
                sheets = pd.read_excel(io.BytesIO(data) if data is not None else path, sheet_name=None, dtype=str,
                                       engine="xlrd" if path.suffix.lower() == ".xls" else None)
            except ImportError:
                raise RuntimeError("Old Excel (.xls) files need the xlrd package: run  pip install xlrd , restart the server and load the PSI again")
            doc["sheets"] = [str(s) for s in sheets]
            for n, (name, df) in enumerate(sheets.items()):
                doc["rows"] += len(df)
                doc["cols"] = max(doc["cols"], int(df.shape[1]))
                doc["facets"] = merge_facets(doc["facets"], facets_of(df))
                doc["chunks"] += [(f"{name} · {label}", text) for label, text in table_chunks(df)]
                if n == 0:
                    doc["preview"] = preview(df)
        elif kind == "image":
            pass
        elif kind == "docx":
            text = docx_text(data if data is not None else path.read_bytes())
            doc["chunks"] = [(f"part {i // 3000 + 1}", text[i:i + 3000]) for i in range(0, len(text), 3000)]
            doc["excerpt"] = text[:4000]
            if not text.strip():
                doc["error"] = "No readable text in this Word file"
        elif kind == "pdf":
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data) if data is not None else str(path))
            doc["pages"] = len(reader.pages)
            doc["chunks"] = [(f"p.{i + 1}", page.extract_text() or "") for i, page in enumerate(reader.pages)]
            if not any(t.strip() for _, t in doc["chunks"]):
                doc["error"] = "No extractable text (scanned PDF?)"
        else:
            text = data.decode("utf-8-sig", errors="replace") if data is not None else read_text(path)
            doc["chunks"] = [("text", text)]
            doc["excerpt"] = text[:4000]
            doc["meta"] = front_matter(text)
    except Exception as e:
        doc["error"] = str(e)
    doc["chars"] = sum(len(t) for _, t in doc["chunks"])
    doc["chunks"] = [(label, text, text.lower()) for label, text in doc["chunks"]]
    return doc


def save_index():
    try:
        with open(INDEX_CACHE, "wb") as f:
            pickle.dump({"v": INDEX_VERSION, "docs": dict(DOCS)}, f)
    except Exception:
        pass


def load_index_cache():
    try:
        with open(INDEX_CACHE, "rb") as f:
            data = pickle.load(f)
        if data.get("v") == INDEX_VERSION:
            DOCS.update(data["docs"])
    except Exception:
        pass


def scan(force=False):
    with LOCK:
        now = time.time()
        if not force and now - STATE["checked"] < 2:
            return
        STATE["checked"] = now
        remote = {k: v for k, v in DOCS.items() if v.get("remote")}
        if force:
            DOCS.clear()
        seen = {}
        changed = force
        for p in sorted(DATA.rglob("*")):
            rel_parts = p.relative_to(DATA).parts
            if not p.is_file() or p.suffix.lower() not in EXT or any(x.startswith((".", "~")) for x in rel_parts):
                continue
            rel = "/".join(rel_parts)
            st = p.stat()
            sig = (st.st_mtime_ns, st.st_size)
            old = DOCS.get(rel)
            if old and (old["mtime"], old["size"]) == sig:
                seen[rel] = old
            else:
                seen[rel] = load(p, rel, sig)
                changed = True
        for k, v in remote.items():
            seen.setdefault(k, v)
        if set(seen) != set(DOCS):
            changed = True
        DOCS.clear()
        DOCS.update(seen)
        if changed:
            save_index()


def terms(s):
    return [t for t in re.split(r"[\s,]+", (s or "").lower()) if t]


def hits(doc, ts):
    return sum(low.count(t) for _, _, low in doc["chunks"] for t in ts)


def match_all(doc, qs):
    head = (doc["name"] + " " + doc["repo"]).lower()
    return all(q in head or any(q in low for _, _, low in doc["chunks"]) for q in qs)


def snippets(doc, ts, limit=3):
    out = []
    for label, text, low in doc["chunks"]:
        for t in ts:
            i = low.find(t)
            if i > -1:
                out.append({"loc": label, "text": " ".join(text[max(0, i - 80):i + len(t) + 120].split())})
                break
        if len(out) >= limit:
            break
    return out


def brief(d):
    return {k: d[k] for k in ("id", "name", "repo", "type", "ext", "size", "rows", "cols", "pages", "error")}


def repo_facets():
    out = {}
    for d in DOCS.values():
        r = out.setdefault(d["repo"], {"cats": {}, "o2": None, "p": None})
        f = d["facets"]
        for k, v in f["cats"].items():
            r["cats"][k] = sorted(set(r["cats"].get(k, [])) | set(v))
        for k, v in d["meta"].items():
            if CAT_RE.match(k):
                vals = [x.strip() for x in re.split(r"[,;]", v) if x.strip()]
                r["cats"][k] = sorted(set(r["cats"].get(k, [])) | set(vals))
        r["o2"] = merge_span(r["o2"], f["o2"])
        r["p"] = merge_span(r["p"], f["p"])
    for name, r in out.items():
        if " PSI-" in name:
            r["cats"]["group"] = [name.split(" PSI-")[0]]
    return out


def parse_filters():
    cats = {}
    for item in request.args.getlist("cat"):
        k, _, v = item.partition(":")
        if k and v:
            cats.setdefault(k, []).append(v)

    def num(n):
        try:
            return float(request.args.get(n, ""))
        except ValueError:
            return None

    return cats, {"o2min": num("o2min"), "o2max": num("o2max"), "pmin": num("pmin"), "pmax": num("pmax")}


def passes(r, cats, rng):
    r = r or {"cats": {}, "o2": None, "p": None}
    for k, vals in cats.items():
        if not set(vals) & set(r["cats"].get(k, [])):
            return False
    for key, lo, hi in (("o2", rng["o2min"], rng["o2max"]), ("p", rng["pmin"], rng["pmax"])):
        if lo is None and hi is None:
            continue
        sp = r[key]
        if not sp:
            return False
        if lo is not None and sp[1] < lo:
            return False
        if hi is not None and sp[0] > hi:
            return False
    return True


def repo_summary():
    out = {}
    rf = repo_facets()
    for d in DOCS.values():
        r = out.setdefault(d["repo"], {"name": d["repo"], "files": 0, "tables": 0, "pdfs": 0, "rows": 0, "desc": "", "facets": rf.get(d["repo"]), "remote": False, "chars": 0})
        r["files"] += 1
        r["chars"] += d.get("chars", 0)
        r["remote"] = r["remote"] or bool(d.get("remote"))
        r["tables"] += d["type"] in ("csv", "xlsx")
        r["pdfs"] += d["type"] == "pdf"
        r["rows"] += d["rows"]
        if d["name"].lower() in ("readme.md", "readme.txt") and d["chunks"]:
            body = re.sub(r"^\ufeff?---.*?---\s*", "", d["chunks"][0][1].replace("\r", ""), flags=re.S)
            body = body.split("## Files not loaded from NASA")[0]
            r["desc"] = " ".join(body.split())[:240]
    return sorted(out.values(), key=lambda r: r["name"].lower())


@app.get("/api/facets")
def facets():
    rf = repo_facets()
    opts = {}
    for r in rf.values():
        for k, v in r["cats"].items():
            opts.setdefault(k, set()).update(v)
    return jsonify(ok=True, options={k: sorted(v) for k, v in sorted(opts.items())}, repos=rf)


@app.get("/api/repo")
def repo_detail():
    name = request.args.get("name", "")
    docs = [d for d in DOCS.values() if d["repo"] == name]
    if not docs:
        return jsonify(ok=False, error="Repository not found"), 404
    desc = next((r["desc"] for r in repo_summary() if r["name"] == name), "")
    files = sorted(({**brief(d), "sheets": d["sheets"]} for d in docs), key=lambda r: (r["type"], r["name"].lower()))
    return jsonify(ok=True, name=name, desc=desc, facets=repo_facets().get(name), files=files)


def doc_text(d):
    return "\n".join(t for _, t, _ in d["chunks"])


def api_key():
    """Key sent by the visitor's browser (X-Gemini-Key); falls back to the server .env if one is set."""
    try:
        user = (request.headers.get("X-Gemini-Key") or "").strip()
    except RuntimeError:
        user = ""
    return user or os.getenv("GeminiAPIKey") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def gemini(prompt):
    text = call(lambda: get_client().models.generate_content(model=MODEL, contents=prompt)).text
    if not text:
        raise RuntimeError("Gemini returned an empty answer")
    return text


def answer(key, instruction, content, lang, refresh):
    if key and not refresh and key in CACHE:
        return CACHE[key], True
    head = f"{SYSTEM.format(lang=lang)}\n{instruction}\n\n"
    text = gemini([head] + content if isinstance(content, list) else head + content)
    if key:
        CACHE[key] = text
        save_cache()
    return text, False


@app.before_request
def refresh_index():
    if request.path.startswith("/api/") and request.path != "/api/reindex":
        scan()


@app.get("/")
def home():
    return send_file(BASE / "dashboard.html")


@app.get("/api/logos")
def logos():
    d = BASE / "logos"
    exts = (".png", ".svg", ".webp", ".jpg", ".jpeg")
    files = sorted(p.name for p in d.iterdir() if p.suffix.lower() in exts) if d.exists() else []
    return jsonify(ok=True, logos=files[:2])


@app.get("/favicon.ico")
def favicon():
    return send_from_directory(BASE / "logos", "prometheus-icon.png", mimetype="image/png")


@app.get("/logos/<path:name>")
def logo_file(name):
    return send_from_directory(BASE / "logos", name)


@app.get("/api/index")
def index():
    return jsonify(ok=True, repos=repo_summary(), files=len(DOCS), ai=bool(api_key()), model=MODEL, fire=FIRE_TERMS, data=str(DATA))


@app.post("/api/reindex")
def reindex():
    scan(True)
    return jsonify(ok=True, files=len(DOCS))


@app.get("/api/search")
def search():
    qs = terms(request.args.get("q"))
    fs = terms(request.args.get("fire"))
    repo = request.args.get("repo", "")
    kind = request.args.get("type", "all")
    cats, rng = parse_filters()
    rf = repo_facets()
    base = [
        d for d in DOCS.values()
        if (kind == "all" or d["type"] == kind) and match_all(d, qs) and passes(rf.get(d["repo"]), cats, rng)
    ]
    by_repo = {}
    for d in base:
        by_repo[d["repo"]] = by_repo.get(d["repo"], 0) + 1
    rows = [
        {**brief(d), "score": round(hits(d, fs) * 10000 / max(d["chars"], 1000), 2), "snippets": snippets(d, qs or fs)}
        for d in base if not repo or d["repo"] == repo
    ]
    rows.sort(key=lambda r: (-r["score"], r["repo"].lower(), r["name"].lower()))
    return jsonify(ok=True, total=len(rows), by_repo=by_repo, results=rows[:200])


@app.get("/api/file")
def file_detail():
    d = DOCS.get(request.args.get("id", ""))
    if not d:
        return jsonify(ok=False, error="File not found"), 404
    return jsonify(ok=True, file={**brief(d), "sheets": d["sheets"], "preview": d["preview"], "excerpt": d["excerpt"]})


@app.get("/api/raw")
def raw():
    d = DOCS.get(request.args.get("id", ""))
    if d and d.get("remote"):
        try:
            data = file_bytes(d)
        except Exception as e:
            return jsonify(ok=False, error=str(e)), 404
        mime = mimetypes.guess_type(d["name"])[0] or "application/octet-stream"
        return send_file(io.BytesIO(data), mimetype=mime, download_name=d["name"], as_attachment=request.args.get("dl") == "1")
    return send_from_directory(DATA, request.args.get("id", ""), as_attachment=request.args.get("dl") == "1")


def image_parts(d, label):
    from google.genai import types
    data = file_bytes(d)
    if len(data) > 7 * 1048576:
        raise ValueError(d["name"] + " is larger than 7 MB")
    return [
        f"{label}: {d['name']} (repository {d['repo']})",
        types.Part.from_bytes(data=data, mime_type=MIME[d["ext"]]),
    ]


def parse_index():
    if not INDEX.exists():
        return {}
    text = INDEX.read_text(encoding="utf-8-sig", errors="replace")
    head, _, body = text.partition("Datasets")
    desc = {}
    for line in head.splitlines():
        m = re.match(r"^(.*?\(([\w\-]+)\))\s{2,}(.+)$", line.strip())
        if m:
            desc[m.group(2).upper()] = m.group(3)
    items = {}
    group = ""
    for line in body.splitlines():
        g = re.match(r"^([A-Za-z][A-Za-z ]*):\s*$", line)
        if g:
            group = g.group(1).strip()
            continue
        m = re.match(r"^\s+PSI-(\d+)\s+(.+?)\s*$", line)
        if m:
            item = items.setdefault(int(m.group(1)), {"group": group, "titles": []})
            item["titles"].append(m.group(2))
    for item in items.values():
        item["desc"] = desc.get(item["group"].upper(), "")
    return items


def folder_name(n, group):
    g = "OTHER" if group.lower() == "unorganized" else group.upper()
    return f"{g} PSI-{n}"


@app.get("/api/catalog")
def catalog():
    counts = {}
    for d in DOCS.values():
        counts[d["repo"]] = counts.get(d["repo"], 0) + 1
    out = []
    for n, v in sorted(parse_index().items()):
        folder = folder_name(n, v["group"])
        out.append({
            "n": n, "group": v["group"], "titles": v["titles"], "desc": v["desc"],
            "folder": folder, "files": counts.get(folder, 0), "url": PSI_URL.format(q=f"PSI-{n}"),
        })
    return jsonify(ok=True, items=out, index=INDEX.exists(), repo_url=PSI_URL.format(q=""))


@app.post("/api/upload")
def upload():
    psi = re.search(r"\d+", request.form.get("psi") or "")
    repo = remote_folder(int(psi.group())) if psi else (request.form.get("repo") or "").strip()
    if not re.fullmatch(r"[\w\- .]{1,80}", repo) or repo.startswith("."):
        return jsonify(ok=False, error="Invalid repository folder"), 400
    folder = DATA / repo
    folder.mkdir(parents=True, exist_ok=True)
    saved = 0
    skipped = []
    for f in request.files.getlist("files"):
        name = secure_filename(f.filename or "")
        if not name or Path(name).suffix.lower() not in EXT:
            skipped.append(f.filename)
            continue
        f.save(folder / name)
        saved += 1
    STATE["checked"] = 0.0
    scan()
    return jsonify(ok=True, saved=saved, skipped=skipped)


BYTES = {}
REMOTE = {}


def file_bytes(d):
    """Bytes of a file: from disk, or downloaded from NASA PSI into memory (small cache)."""
    r = d.get("remote")
    if not r:
        return (DATA / d["id"]).read_bytes()
    if d["id"] not in BYTES:
        if not r.get("file"):
            raise ValueError("No file behind " + d["name"])
        data = psi_remote.psi_download(r["psi"], r["file"])
        if len(BYTES) >= 6:
            BYTES.pop(next(iter(BYTES)))
        BYTES[d["id"]] = data
    return BYTES[d["id"]]


def remote_folder(n):
    v = parse_index().get(n)
    return folder_name(n, v["group"] if v else "Unorganized")


def psi_number(v):
    m = re.search(r"\d+", str(v or ""))
    n = int(m.group()) if m else 0
    return n if 1 <= n <= 9999 else None


def remote_load(n):
    st = REMOTE[n]
    try:
        psi, folder = f"PSI-{n}", remote_folder(n)
        info = psi_remote.psi_info(psi)
        files, left = psi_remote.psi_select(psi)
        if not info and not files and not left:
            raise RuntimeError(f"{psi} was not found or has no public files")
        st["total"] = len(files)
        st["left"] = left
        st["bytes_total"] = sum(f.get("file_size") or 0 for f in files)
        new = {}
        slock = threading.Lock()

        def fetch(f):
            rid = f"{folder}/{psi_remote.display_name(f, psi)}"
            if (DATA / rid).exists():
                return rid, None
            with slock:
                st["current"].append(f["file_name"])
            try:
                data = psi_remote.psi_download(psi, f)
                sig = (psi_remote._date(f), len(data))
                doc = load(Path(rid), rid, sig, data)
                doc["remote"] = {"psi": psi, "file": f}
                return rid, doc
            finally:
                with slock:
                    st["current"].remove(f["file_name"])
                    st["bytes_done"] += f.get("file_size") or 0

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(fetch, f): f for f in files}
            for fut in as_completed(futures):
                f = futures[fut]
                try:
                    rid, doc = fut.result()
                    if doc:
                        with LOCK:
                            DOCS[rid] = doc
                except Exception as e:
                    with slock:
                        st["skipped"] += 1
                    left.append({"name": f["file_name"], "reason": "Download failed (" + str(e)[:80] + ")"})
                with slock:
                    st["done"] += 1
        body = "\n\n".join(f"{k.title()}: {info[k]}" for k in ("title", "objective", "description") if info.get(k))
        for x in left:
            x["reason"] = x["reason"][:1].upper() + x["reason"][1:]
        if left:
            groups = {}
            for x in left:
                groups.setdefault(x["reason"], []).append(x["name"])
            body += "\n\n## Files not loaded from NASA\n" + "\n".join(
                (f"- {r}: {ns[0]}" if len(ns) == 1 and re.fullmatch(r"\d+ files", ns[0]) else
                 f"- {r} ({len(ns)}): " + ", ".join(ns[:8]) + (" …" if len(ns) > 8 else "")) for r, ns in groups.items())
        if body.strip():
            rid = f"{folder}/README.md"
            new[rid] = load(Path(rid), rid, (0, len(body)), body[:8000].encode())
            new[rid]["remote"] = {"psi": psi, "file": None}
        with LOCK:
            DOCS.update(new)
            save_index()
        st["state"] = "done"
    except Exception as e:
        st.update(state="error", error=str(e))


@app.post("/api/remote/add")
def remote_add():
    n = psi_number((request.get_json(force=True) or {}).get("psi"))
    if not n:
        return jsonify(ok=False, error="Write a PSI number, for example 159"), 400
    if REMOTE.get(n, {}).get("state") != "loading":
        REMOTE[n] = {"state": "loading", "done": 0, "total": 0, "skipped": 0, "error": "", "left": [],
                     "bytes_done": 0, "bytes_total": 0, "current": []}
        threading.Thread(target=remote_load, args=(n,), daemon=True).start()
    return jsonify(ok=True, n=n, folder=remote_folder(n))


@app.get("/api/remote/status")
def remote_status():
    return jsonify(ok=True, items=[{"n": n, "folder": remote_folder(n), **s} for n, s in REMOTE.items()])


@app.post("/api/remote/remove")
def remote_remove():
    folder = (request.get_json(force=True) or {}).get("repo", "")
    with LOCK:
        gone = [k for k, d in DOCS.items() if d["repo"] == folder and d.get("remote")]
        for k in gone:
            DOCS.pop(k)
            BYTES.pop(k, None)
        save_index()
    for n in [n for n in REMOTE if remote_folder(n) == folder]:
        REMOTE.pop(n)
    return jsonify(ok=True, removed=len(gone))


@app.post("/api/recommend")
def recommend():
    b = request.get_json(force=True)
    msgs = [m for m in (b.get("messages") or []) if m.get("role") in ("user", "assistant") and str(m.get("text") or "").strip()][-10:]
    if not msgs or msgs[-1]["role"] != "user":
        return jsonify(ok=False, error="Write a question first"), 400
    items = parse_index()
    if not items:
        return jsonify(ok=False, error="index.txt not found next to server.py"), 400
    lang = "Spanish" if b.get("lang") == "es" else "English"
    system = (
        "You help a user choose which NASA Physical Sciences Informatics (PSI) low-gravity combustion investigations "
        "to load into a spacecraft fire safety dashboard. Recommend 1 to 4 investigations from the list below that "
        "best fit what the user wants, using only the titles and group descriptions. Never invent results or numbers. "
        "If the request is too vague, ask one short question and return no picks. Rockets and propellants are out of scope. "
        'Return only JSON: {"reply": plain text of at most 70 words in ' + lang + ', "picks": [{"psi": number, '
        '"reason": one sentence of at most 20 words in ' + lang + "}]}.\n\nInvestigations:\n" + catalog_text()
    )
    try:
        from google.genai import types
        contents = [
            types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part(text=str(m["text"])[:3000])])
            for m in msgs
        ]
        resp = call(lambda: get_client().models.generate_content(
            model=MODEL, contents=contents,
            config=types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json"),
        ))
        data = json.loads(resp.text)
        if not isinstance(data, dict):
            data = {}
        counts = {}
        for d in DOCS.values():
            counts[d["repo"]] = counts.get(d["repo"], 0) + 1
        picks = []
        for r in (data.get("picks") or [])[:4]:
            try:
                n = int(r["psi"])
            except (KeyError, TypeError, ValueError):
                continue
            if n in items and all(p["n"] != n for p in picks):
                v = items[n]
                folder = folder_name(n, v["group"])
                picks.append({"n": n, "group": v["group"], "title": " · ".join(v["titles"]),
                              "reason": str(r.get("reason", "")), "folder": folder, "files": counts.get(folder, 0)})
        return jsonify(ok=True, text=str(data.get("reply", "")), picks=picks)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.get("/api/files")
def files():
    rows = sorted(({**brief(d), "sheets": d["sheets"]} for d in DOCS.values()), key=lambda r: (r["repo"].lower(), r["name"].lower()))
    return jsonify(ok=True, files=rows)


def table(d, sheet=None):
    key = (d["id"], sheet)
    hit = TABLES.get(key)
    if hit and hit[0] == d["mtime"]:
        return hit[1]
    path = DATA / d["id"]
    src = file_bytes(d) if d.get("remote") else None
    if d["type"] == "csv":
        df = read_csv(path, src)
    else:
        df = pd.read_excel(io.BytesIO(src) if src is not None else path, sheet_name=sheet if sheet in d["sheets"] else 0, dtype=str,
                           engine="xlrd" if d["ext"] == "xls" else None)
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    if len(TABLES) >= 8:
        TABLES.pop(next(iter(TABLES)))
    TABLES[key] = (d["mtime"], df)
    return df


def tabular(arg):
    d = DOCS.get(request.args.get(arg, ""))
    return d if d and d["type"] in ("csv", "xlsx") else None


@app.get("/api/columns")
def columns():
    d = tabular("id")
    if not d:
        return jsonify(ok=False, error="Table not found"), 404
    sheet = request.args.get("sheet") or None
    df = table(d, sheet)
    cols = []
    for c in df.columns:
        s = df[c].dropna()
        num = pd.to_numeric(s, errors="coerce")
        cols.append({"name": c, "numeric": bool(len(s)) and bool(num.notna().mean() >= 0.5), "unique": int(s.nunique())})
    return jsonify(ok=True, columns=cols, rows=len(df), sheets=d["sheets"])


@app.get("/api/series")
def series():
    d = tabular("id")
    if not d:
        return jsonify(ok=False, error="Table not found"), 404
    sheet = request.args.get("sheet") or None
    df = table(d, sheet)
    x, y, g = request.args.get("x", ""), request.args.get("y", ""), request.args.get("g", "")
    if x not in df.columns or y not in df.columns:
        return jsonify(ok=False, error="Choose valid X and Y columns"), 400
    xs = pd.to_numeric(df[x], errors="coerce")
    ys = pd.to_numeric(df[y], errors="coerce")
    ok = xs.notna() & ys.notna()
    groups = df[g].fillna("").astype(str) if g in df.columns else pd.Series("All", index=df.index)
    names = sorted(groups[ok].value_counts().index[:12])
    out = []
    stats = []
    for name in names:
        m = ok & (groups == name)
        pts = list(zip(xs[m].tolist(), ys[m].tolist()))
        out.append({"name": name or "(empty)", "points": pts[::max(1, len(pts) // 3000)]})
        v = ys[m]
        stats.append({"group": name or "(empty)", "n": int(m.sum()), "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())})
    return jsonify(ok=True, series=out, stats=stats, skipped=int((~ok).sum()))


ML_MAX_ROWS = 5000
ML_STORE = {}


def _ml_fit(kind, Xtr, ytr, ncls):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    Z = (Xtr - mu) / sd
    if kind == "ridge":
        Zb = np.hstack([Z, np.ones((len(Z), 1))])
        Y = np.eye(ncls)[ytr] if ncls else ytr
        reg = np.eye(Zb.shape[1])
        reg[-1, -1] = 0
        w = np.linalg.solve(Zb.T @ Zb + 1.0 * reg, Zb.T @ Y)
        return {"kind": kind, "mu": mu, "sd": sd, "w": w, "ncls": ncls}
    k = int(min(15, max(3, round(len(Z) ** 0.5 / 2))))
    return {"kind": kind, "mu": mu, "sd": sd, "Z": Z, "y": ytr, "k": min(k, len(Z)), "ncls": ncls}


def _ml_predict(m, X):
    Z = (X - m["mu"]) / m["sd"]
    if m["kind"] == "ridge":
        out = np.hstack([Z, np.ones((len(Z), 1))]) @ m["w"]
        return out.argmax(1) if m["ncls"] else out
    preds = []
    for i in range(0, len(Z), 500):
        d = ((Z[i:i + 500, None, :] - m["Z"][None, :, :]) ** 2).sum(-1)
        nn = np.argpartition(d, m["k"] - 1, axis=1)[:, :m["k"]]
        ny = m["y"][nn]
        if m["ncls"]:
            preds.append(np.array([np.bincount(r, minlength=m["ncls"]).argmax() for r in ny]))
        else:
            preds.append(ny.mean(1))
    return np.concatenate(preds)


def _ml_scores(m, X):
    Z = (X - m["mu"]) / m["sd"]
    if m["kind"] == "ridge":
        return np.hstack([Z, np.ones((len(Z), 1))]) @ m["w"]
    d = ((Z[:, None, :] - m["Z"][None, :, :]) ** 2).sum(-1)
    nn = np.argpartition(d, m["k"] - 1, axis=1)[:, :m["k"]]
    return np.stack([np.bincount(r, minlength=m["ncls"]) for r in m["y"][nn]]) / m["k"]


def _ml_score(ncls, y, p):
    if ncls:
        return float((y == p).mean())
    ss = float(((y - y.mean()) ** 2).sum())
    return 1 - float(((y - p) ** 2).sum()) / ss if ss else 0.0


@app.post("/api/ml")
def ml_train():
    b = request.get_json(force=True)
    d = DOCS.get(b.get("id", ""))
    if not d or d["type"] not in ("csv", "xlsx"):
        return jsonify(ok=False, error="Table not found"), 404
    df = table(d, b.get("sheet") or None)
    target = b.get("target", "")
    if target not in df.columns:
        return jsonify(ok=False, error="Choose a column to predict"), 400
    t0 = time.time()
    raw = df[target]
    txt = raw.fillna("").astype(str).str.strip()
    filled = txt != ""
    ynum = pd.to_numeric(raw, errors="coerce")
    if filled.sum() and ynum.notna().sum() >= 0.5 * filled.sum() and ynum.nunique() > 12:
        task, keep, classes = "regression", ynum.notna(), []
    else:
        vc = txt[filled].value_counts()
        vc = vc[vc >= 2]
        if not 2 <= len(vc) <= 12:
            return jsonify(ok=False, error=f"“{target}” can't be predicted: pick a numeric column, or one with 2 to 12 categories."), 400
        task, classes = "classification", [str(c) for c in vc.index]
        keep = txt.isin(classes)
    data = df[keep]
    if len(data) > ML_MAX_ROWS:
        data = data.sample(ML_MAX_ROWS, random_state=42)
    n = len(data)
    if n < 20:
        return jsonify(ok=False, error=f"Only {n} usable rows for “{target}”. The model needs at least 20."), 400
    mats, groups, used, dropped, spec = [], [], [], [], []
    for c in df.columns:
        if c == target:
            continue
        s = data[c]
        st = s.fillna("").astype(str).str.strip()
        nf = int((st != "").sum())
        if nf < 0.4 * n:
            dropped.append(f"{c} (mostly empty)")
            continue
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().sum() >= 0.8 * nf:
            if num.nunique() < 2:
                dropped.append(f"{c} (constant)")
                continue
            mats.append(num.fillna(num.median()).to_numpy(float)[:, None])
            spec.append({"name": c, "type": "number", "median": float(num.median()), "min": float(num.min()), "max": float(num.max())})
        else:
            nu = st.nunique()
            if not 2 <= nu <= 12:
                dropped.append(f"{c} ({'constant' if nu < 2 else 'text or ID'})")
                continue
            dm = pd.get_dummies(st)
            mats.append(dm.to_numpy(float))
            nz = st[st != ""]
            spec.append({"name": c, "type": "choice", "options": [str(o) for o in dm.columns],
                         "mode": str(nz.mode().iloc[0]) if len(nz) else str(dm.columns[0])})
        start = sum(m.shape[1] for m in mats[:-1])
        groups.append((c, start, start + mats[-1].shape[1]))
        used.append(c)
    if not mats:
        return jsonify(ok=False, error="No usable input columns besides the one to predict."), 400
    X = np.hstack(mats)
    if task == "regression":
        y = ynum[data.index].to_numpy(float)
        ncls = 0
    else:
        lab = txt[data.index]
        y = np.array([classes.index(v) for v in lab])
        ncls = len(classes)
    rng = np.random.default_rng(42)
    folds = np.array_split(rng.permutation(n), 5)
    names = {"ridge": "Linear (ridge)", "knn": "Nearest neighbours (k-NN)"}
    res = {}
    for kind in names:
        oof = np.zeros(n) if not ncls else np.zeros(n, dtype=int)
        imp = np.zeros(len(groups))
        for te in folds:
            tr = np.setdiff1d(np.arange(n), te)
            m = _ml_fit(kind, X[tr], y[tr], ncls)
            p = _ml_predict(m, X[te])
            oof[te] = p
            base = _ml_score(ncls, y[te], p)
            for gi, (_, a, z) in enumerate(groups):
                Xp = X[te].copy()
                Xp[:, a:z] = Xp[rng.permutation(len(te))][:, a:z]
                imp[gi] += base - _ml_score(ncls, y[te], _ml_predict(m, Xp))
        res[kind] = (oof, imp / len(folds))
    out_models = []
    for kind, (oof, _) in res.items():
        if ncls:
            out_models.append({"key": kind, "name": names[kind], "accuracy": _ml_score(ncls, y, oof)})
        else:
            out_models.append({"key": kind, "name": names[kind], "r2": _ml_score(0, y, oof),
                               "mae": float(np.abs(y - oof).mean()), "rmse": float(np.sqrt(((y - oof) ** 2).mean()))})
    metric = "accuracy" if ncls else "r2"
    best = max(out_models, key=lambda m: m[metric])
    oof, imp = res[best["key"]]
    imp = np.clip(imp, 0, None)
    tot = imp.sum()
    importance = sorted(({"feature": g[0], "share": float(v / tot) if tot else 0.0} for g, v in zip(groups, imp)), key=lambda r: -r["share"])[:12]
    out = {"ok": True, "task": task, "target": target, "rows": n, "features": used, "dropped": dropped,
           "models": out_models, "best": best, "importance": importance,
           "trained_at": time.strftime("%H:%M:%S"), "seconds": round(time.time() - t0, 2),
           "file": d["name"], "sheet": b.get("sheet") or "first"}
    token = hashlib.md5(f"{time.time()}|{d['id']}|{target}".encode()).hexdigest()[:16]
    if len(ML_STORE) >= 20:
        ML_STORE.pop(next(iter(ML_STORE)))
    ML_STORE[token] = {"model": _ml_fit(best["key"], X, y, ncls), "spec": spec, "groups": groups, "ncls": ncls,
                       "classes": classes, "target": target, "task": task, "best": best, "file": d["name"]}
    out["token"] = token
    out["inputs"] = spec
    if ncls:
        maj = np.bincount(y, minlength=ncls)
        out["baseline"] = float(maj.max() / n)
        out["classes"] = classes
        cm = np.zeros((ncls, ncls), dtype=int)
        for a_, p_ in zip(y, oof):
            cm[a_, p_] += 1
        out["confusion"] = cm.tolist()
    else:
        out["baseline_mae"] = float(np.abs(y - y.mean()).mean())
        out["target_range"] = [float(y.min()), float(y.max())]
        out["target_mean"] = float(y.mean())
        step = max(1, n // 1500)
        out["points"] = [[float(a_), float(p_)] for a_, p_ in zip(y[::step], oof[::step])]
    return jsonify(out)


@app.post("/api/ml/predict")
def ml_predict():
    b = request.get_json(force=True)
    st = ML_STORE.get(b.get("token", ""))
    if not st:
        return jsonify(ok=False, error="This model is no longer in memory (the server restarted). Press Train model again."), 410
    vals = b.get("values") or {}
    width = st["groups"][-1][2]
    typical, x, outside, shown = np.zeros(width), np.zeros(width), [], []
    for sp, (name, a, z) in zip(st["spec"], st["groups"]):
        if sp["type"] == "number":
            typical[a] = sp["median"]
            raw = str(vals.get(name, "")).strip()
            try:
                v = float(raw) if raw else sp["median"]
            except ValueError:
                return jsonify(ok=False, error=f"“{name}” needs a number."), 400
            if not np.isfinite(v):
                return jsonify(ok=False, error=f"“{name}” needs a number."), 400
            x[a] = v
            if v < sp["min"] or v > sp["max"]:
                outside.append({"feature": name, "value": v, "min": sp["min"], "max": sp["max"]})
            shown.append((name, v, sp["median"]))
        else:
            opts = sp["options"]
            typical[a + opts.index(sp["mode"])] = 1
            v = str(vals.get(name, sp["mode"]))
            if v not in opts:
                v = sp["mode"]
            x[a + opts.index(v)] = 1
            shown.append((name, v, sp["mode"]))
    m, ncls = st["model"], st["ncls"]
    effects = []
    if ncls:
        sc = _ml_scores(m, x[None])[0]
        k = int(sc.argmax())
        sup = np.clip(sc, 0, None)
        sup = sup / sup.sum() if sup.sum() > 0 else np.eye(ncls)[k]
        order = np.argsort(-sup)[:3]
        result = {"label": st["classes"][k], "support": [{"label": st["classes"][i], "share": float(sup[i])} for i in order]}
        base = sc[k]
    else:
        base = float(_ml_predict(m, x[None])[0])
        result = {"value": base, "typical_value": float(_ml_predict(m, typical[None])[0]), "typical_error": st["best"]["mae"]}
    for (name, v, t), (_, a, z) in zip(shown, st["groups"]):
        if v == t or (isinstance(v, float) and abs(v - t) <= 1e-4 * max(1.0, abs(t))):
            continue
        x2 = x.copy()
        x2[a:z] = typical[a:z]
        other = _ml_scores(m, x2[None])[0][k] if ncls else float(_ml_predict(m, x2[None])[0])
        effects.append({"feature": name, "value": v, "typical": t, "effect": float(base - other)})
    effects.sort(key=lambda e: -abs(e["effect"]))
    return jsonify(ok=True, task=st["task"], target=st["target"], model=st["best"]["name"], result=result,
                   effects=effects[:6], outside=outside, inputs={n: v for n, v, _ in shown})


@app.post("/api/compare")
def compare():
    b = request.get_json(force=True)
    a, c = DOCS.get(b.get("a")), DOCS.get(b.get("b"))
    if not a or not c:
        return jsonify(ok=False, error="File not found"), 404
    if a["id"] == c["id"]:
        return jsonify(ok=False, error="Choose two different files"), 400
    lang = "Spanish" if b.get("lang") == "es" else "English"
    try:
        parts = []
        for label, d in (("A", a), ("B", c)):
            if d["type"] == "image":
                parts += image_parts(d, "File " + label)
                continue
            text = doc_text(d).strip()
            if not text:
                return jsonify(ok=False, error=d["name"] + " has no readable text"), 400
            parts.append(f"File {label}: {d['name']} (repository {d['repo']})\n{text[:20000]}")
        instruction = (
            "Compare File A and File B in 6 short bullets: what each one shows, fuels and conditions, "
            "key numeric or visual differences, and what the difference implies for spacecraft fire safety. "
            "Say when something cannot be determined from the content. "
            "Don't use markdown symbols or any kind of format, you can just use bold text."
        )
        key = f"cmp|{a['id']}|{a['mtime']}|{c['id']}|{c['mtime']}|{lang}"
        text, cached = answer(key, instruction, parts, lang, bool(b.get("refresh")))
        return jsonify(ok=True, text=text, cached=cached)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


LENSES = {
    "habitat": "relevance to the fire safety of habitable crew environments",
    "modeling": "how much its data supports predictive modeling of flammability and fire scenarios",
    "controls": "how much its research informs next-generation fire safety controls and equipment (detection, suppression, materials screening, operations)",
    "earth": "potential applications of its findings to fire safety on Earth",
}
ENVS = {
    "any": "any crewed environment",
    "spacecraft": "a crewed spacecraft such as the ISS or a visiting vehicle",
    "station": "a space station",
    "lunar": "a lunar surface base",
    "mars": "a Mars surface base",
    "rover": "a pressurized rover",
}
HISTORY = {
    "cases": [
        {
            "id": "apollo1", "year": "1967", "date": "January 27, 1967", "title": "Apollo 1 (AS-204)", "kind": "Fire",
            "where": "Command Module, launch pad test",
            "what": "During a preflight test on the pad, a fire broke out in the cabin while the crew was suited, the hatches "
                    "were closed and the cabin was pressurized with 100% oxygen. The pure oxygen made the fire spread so fast "
                    "that the crew could not escape. No single ignition source was proven, but there was evidence of electrical "
                    "arcing in wiring near the environmental control system.",
            "lesson": "Oxygen-rich atmospheres turn small ignitions into fast fires. NASA strictly limited combustible materials "
                      "in the spacecraft, adopted fire-resistant suits and a hatch that opens in seconds.",
            "topics": ["Oxygen-rich atmosphere", "Material flammability", "Electrical ignition", "Crew escape"],
            "ask": "Which PSI investigations study how oxygen concentration and pressure change flame spread and material "
                   "flammability, the risks behind the Apollo 1 fire? What do their results say for cabin atmospheres?",
            "source": "https://sma.nasa.gov/sma-disciplines/accllp/apollo-1",
            "docs": [{"label": "Senate report", "ref": "apollo1_senate"}],
        },
        {
            "id": "mir", "year": "1997", "date": "February 24, 1997", "title": "Fire aboard space station Mir", "kind": "Fire",
            "where": "Kvant-1 module, in orbit",
            "what": "A solid-fuel oxygen generator canister ignited while a cosmonaut was activating it. The flame burned for "
                    "several minutes, filled the station with dense smoke and blocked access to one Soyuz escape vehicle. The "
                    "crew wore oxygen masks and used extinguishers until it went out; contamination of the canister (possibly "
                    "latex glove material) was suspected.",
            "lesson": "In orbit, smoke spreads through the whole station and escape routes can be cut off. The event shaped crew "
                      "emergency training, procedures and the handling of oxygen generators later used on the ISS.",
            "topics": ["Smoke and detection", "Suppression", "Oxygen generators", "Escape routes"],
            "ask": "Which PSI investigations help with smoke detection, fire suppression and oxygen-generator fire risks in "
                   "microgravity, like the 1997 Mir fire? What data could improve detectors or extinguishers?",
            "source": "https://www.nasa.gov/history/25-years-ago-fire-aboard-space-station-mir/",
            "docs": [],
        },
        {
            "id": "apollo13", "year": "1970", "date": "April 1970", "title": "Apollo 13", "kind": "Related",
            "note": "An explosion in an oxygen system, not a cabin fire",
            "where": "Service Module oxygen tank, in flight",
            "what": "On the third day of the mission, oxygen tank 2 exploded and cut power from the fuel cells; the crew used the "
                    "Lunar Module as a lifeboat and returned safely. Heaters in the tank had overheated during a prelaunch test "
                    "and damaged the Teflon insulation of its wiring.",
            "lesson": "Oxygen systems and the insulation around them are fire and explosion hazards in their own right. NASA "
                      "redesigned the oxygen storage, added a third tank and sheathed wiring in stainless steel.",
            "topics": ["Oxygen systems", "Wire insulation", "Overheating"],
            "ask": "Which PSI investigations study ignition and flammability of wire insulation or materials in oxygen-rich "
                   "conditions, related to the Apollo 13 oxygen tank failure?",
            "source": "https://sma.nasa.gov/sma-disciplines/accllp/apollo-13",
            "docs": [{"label": "Review Board report", "ref": "apollo13_board"}, {"label": "Mission report", "ref": "apollo13_mission"}],
        },
    ],
    "spinoffs": [
        {"year": "1976", "title": "Lighter firefighter air tanks", "what": "A firefighter air pack about 13 pounds lighter, "
         "with 30 minutes of air at twice the usual pressure, developed with NASA Johnson Space Center.",
         "source": "https://spinoff.nasa.gov/node/9582"},
        {"year": "1978", "title": "Early-warning fire detector", "what": "An air-sampling sensor that recognizes combustion "
         "particles while ignoring harmless dust, warning before flames or smoke appear; it descends from a detector used "
         "on the Space Shuttle orbiters.", "source": "https://spinoff.nasa.gov/node/9407"},
        {"year": "1982", "title": "Better gear for firefighters", "what": "Project FIRES, with the U.S. Fire Administration, "
         "used lightweight fire- and heat-resistant materials first developed for spacesuits and spacecraft thermal protection.",
         "source": "https://spinoff.nasa.gov/node/9918"},
    ],
}


def history_text():
    out = [f"- {c['title']} ({c['date']}, {c['kind']}{': ' + c['note'] if c.get('note') else ''}): {c['what']} Lesson: {c['lesson']}"
           for c in HISTORY["cases"]]
    out += [f"- Spinoff {s['year']}, {s['title']}: {s['what']}" for s in HISTORY["spinoffs"]]
    return "\n".join(out)


HIST_CHAT_SYSTEM = (
    "You are a historian of spaceflight fire safety inside a NASA fire safety dashboard. Answer ONLY about historical "
    "fire and oxygen-related incidents in crewed spaceflight, the lessons NASA learned from them, and the related "
    "spinoffs, using the case summaries and the excerpts of NASA reports given below. If the answer is not in them, "
    "say so plainly and do not invent dates, numbers or names. Remember that Apollo 13 was an oxygen tank explosion, "
    "not a cabin fire. When you use a report, mention its name in the sentence (no brackets or page codes). Write for "
    "a non-expert, with short paragraphs or bullets and bold for key facts. Never use LaTeX. Answer in {lang}.\n\n"
    "The user is currently reading: {current}\n\nCases and spinoffs:\n{history}"
)


@app.post("/api/history/chat")
def history_chat():
    import fetch_references as fr
    b = request.get_json(force=True)
    msgs = [m for m in (b.get("messages") or []) if m.get("role") in ("user", "assistant") and str(m.get("text") or "").strip()][-10:]
    if not msgs or msgs[-1]["role"] != "user":
        return jsonify(ok=False, error="Write a question first"), 400
    lang = "Spanish" if b.get("lang") == "es" else "English"
    current = str(b.get("current") or "nothing in particular")[:300]
    focus = set(b.get("docs") or [])
    question = str(msgs[-1]["text"])[:4000]
    ts = [t for t in re.findall(r"[\w\-\.%]+", (question + " " + current).lower()) if len(t) > 2 and t not in STOP]
    units = []
    for r in fr.status():
        d = DOCS.get(r["id"])
        if not r["local"] or not d:
            continue
        for label, text, low in d["chunks"]:
            if text.strip():
                score = sum(low.count(t) for t in ts) + (25 if r["id"] in focus else 0)
                units.append((score, d["name"], label, text.strip()))
    units.sort(key=lambda u: -u[0])
    picked, used = [], 0
    for u in units:
        if used + len(u[3]) > 60000:
            continue
        picked.append(u)
        used += len(u[3])
        if used > 55000:
            break
    excerpts = "\n\n".join(f"### {n} ({l})\n{t}" for _, n, l, t in picked) or "(no reports available)"
    try:
        from google.genai import types
        system = HIST_CHAT_SYSTEM.format(lang=lang, current=current, history=history_text())
        last = question + "\n\nExcerpts of NASA reports:\n" + excerpts
        contents = [types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part(text=str(m["text"])[:4000])])
                    for m in msgs[:-1]] + [types.Content(role="user", parts=[types.Part(text=last)])]
        resp = call(lambda: get_client().models.generate_content(
            model=MODEL, contents=contents, config=types.GenerateContentConfig(system_instruction=system)))
        if not resp.text:
            raise RuntimeError("Gemini returned an empty answer")
        return jsonify(ok=True, text=resp.text, used=sorted({n for _, n, _, _ in picked}))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


REF_META = {
    "guide": {"year": "2015", "title": "Combustion Science: A Researcher's Guide",
              "desc": "NASA's guide to combustion research on the ISS: why low gravity matters, past experiments, lessons and open questions.",
              "related": "ISS research"},
    "apollo1_senate": {"year": "1968", "title": "Apollo 204 Accident: Senate Report",
                       "desc": "U.S. Senate report on the Apollo 1 cabin fire in a 100% oxygen atmosphere during a launch pad test.",
                       "related": "Apollo 1"},
    "apollo13_board": {"year": "1970", "title": "Report of the Apollo 13 Review Board",
                       "desc": "Official investigation of the oxygen tank failure: cause, analysis, findings and recommendations.",
                       "related": "Apollo 13"},
    "apollo13_mission": {"year": "1970", "title": "Apollo 13 Mission Report",
                         "desc": "NASA's technical report of the mission, the oxygen tank anomaly and how the crew returned.",
                         "related": "Apollo 13"},
}


@app.get("/api/history")
def history():
    import fetch_references as fr
    refs = [{**r, **REF_META.get(r["key"], {}), "pages": (DOCS.get(r["id"]) or {}).get("pages", 0)}
            for r in fr.status() if r["local"]]
    by = {r["key"]: r for r in refs}
    cases = [{**c, "docs": [{**d, **by[d["ref"]]} for d in c["docs"] if d["ref"] in by]} for c in HISTORY["cases"]]
    return jsonify(ok=True, cases=cases, spinoffs=HISTORY["spinoffs"], refs=refs, repo=fr.FOLDER)


CHAT_SYSTEM = (
    "You are a fire safety research assistant inside a dashboard about NASA Physical Sciences Informatics (PSI) "
    "low-gravity combustion investigations. Help users (1) identify investigations relevant to crewed habitats: "
    "spacecraft, space stations, lunar and Mars surface bases and pressurized rovers; (2) find data that supports "
    "predictive modeling of flammability and fire scenarios; (3) find research that informs next-generation fire "
    "safety controls and equipment; (4) see applications to fire safety on Earth, such as fine water mist "
    "extinguishers, smoke detection and firefighter gear. Use the investigation list below and search the web for "
    "details of specific investigations, results and papers. Cite PSI numbers. Never invent numeric results; if you "
    "cannot find something, say so. Separate what comes from the list, from web sources and from your own inference. "
    "Fire safety of rockets and propellants is outside the scope of PSI data. Be concise and answer in {lang}.\n\n"
    "Investigations:\n{catalog}\n\nFiles loaded in the dashboard:\n{files}\n\n"
    "Historical space fire incidents and spinoffs shown in the dashboard (use them to connect PSI research with real "
    "events; Apollo 13 was an oxygen tank explosion, not a cabin fire):\n{history}\n\n"
    "Don't use markdown symbols or any kind of format, you can just use bold text."
)


def get_client():
    key = api_key()
    if not key:
        raise RuntimeError("Add your Gemini API key in the sidebar (above Status & settings) to use the Super Intelligence (SI) features.")
    if key not in CLIENTS:
        from google import genai
        if len(CLIENTS) > 50:
            CLIENTS.clear()
        CLIENTS[key] = genai.Client(api_key=key)
    return CLIENTS[key]


@app.post("/api/key/check")
def key_check():
    try:
        call(lambda: get_client().models.get(model=MODEL))
        return jsonify(ok=True, model=MODEL)
    except Exception as e:
        msg = str(e)
        if any(x in msg for x in ("API_KEY_INVALID", "API key not valid", "PERMISSION_DENIED", "401", "403")):
            msg = "That API key was rejected by Gemini. Check it and try again."
        return jsonify(ok=False, error=msg), 400


def catalog_text():
    groups = {}
    lines = []
    for n, v in sorted(parse_index().items()):
        lines.append(f"PSI-{n} [{v['group']}] " + " / ".join(v["titles"]))
        groups[v["group"]] = v["desc"]
    extra = [f"{g}: {d}" for g, d in groups.items() if d]
    return "\n".join(lines) + ("\n\nGroup descriptions:\n" + "\n".join(extra) if extra else "")


def sources_of(resp):
    out = []
    seen = set()
    try:
        for c in resp.candidates[0].grounding_metadata.grounding_chunks or []:
            if c.web and c.web.uri and c.web.uri not in seen:
                seen.add(c.web.uri)
                out.append({"title": c.web.title or c.web.uri, "url": c.web.uri})
    except Exception:
        pass
    return out[:8]


@app.post("/api/rank")
def rank():
    b = request.get_json(force=True)
    lens, env = b.get("lens"), b.get("env")
    env_text = " ".join(str(b.get("env_text") or "").split())[:200]
    lens_text = " ".join(str(b.get("lens_text") or "").split())[:200]
    extra = " ".join(str(b.get("extra") or "").split())[:400]
    if env == "custom" and not env_text:
        return jsonify(ok=False, error="Describe your environment, or pick one from the list"), 400
    if lens == "custom" and not lens_text:
        return jsonify(ok=False, error="Describe what to rank by, or pick a focus from the list"), 400
    if (lens not in LENSES and lens != "custom") or (env not in ENVS and env != "custom"):
        return jsonify(ok=False, error="Choose an environment and a focus"), 400
    atm = []
    try:
        if str(b.get("o2") or "").strip():
            o2 = float(b["o2"])
            if not 0 < o2 <= 100:
                raise ValueError
            atm.append(f"{o2:g}% oxygen")
        if str(b.get("pressure") or "").strip():
            pr = float(b["pressure"])
            if not 0 < pr <= 50:
                raise ValueError
            atm.append(f"{pr:g} psi total pressure")
    except ValueError:
        return jsonify(ok=False, error="Oxygen must be 1-100 % and pressure 0.1-50 psi"), 400
    env_desc = env_text if env == "custom" else ENVS[env]
    lens_desc = lens_text if lens == "custom" else LENSES[lens]
    if atm:
        env_desc += " with a cabin atmosphere of " + " and ".join(atm)
    items = parse_index()
    if not items:
        return jsonify(ok=False, error="index.txt not found next to server.py"), 400
    lang = "Spanish" if b.get("lang") == "es" else "English"
    custom = hashlib.md5(f"{env_desc}|{lens_desc}|{extra}".encode()).hexdigest()[:12]
    key = f"rank|{lens}|{env}|{custom}|{lang}|{int(INDEX.stat().st_mtime)}"
    try:
        if key in CACHE and not b.get("refresh"):
            raw = CACHE[key]
        else:
            from google.genai import types
            prompt = (
                "You are a spacecraft fire safety analyst. Below is the list of NASA PSI combustion investigations. "
                f"Rank every one of them by {lens_desc}, for {env_desc}. "
                + (f"Also take into account these extra criteria from the user: {extra}. " if extra else "")
                + "Base each judgement on the titles, the group "
                "descriptions and well-established public knowledge of these NASA experiments; if you are not sure what "
                "an investigation measured, say so and score it lower. Do not invent results or numbers. Fire safety of "
                "rockets and propellants is out of scope. Return only a JSON array where each element is "
                '{"psi": number, "score": integer 0-100, "reason": one sentence of at most 25 words in ' + lang + "}. "
                "Include every investigation once.\n\n" + catalog_text()
            )
            resp = call(lambda: get_client().models.generate_content(
                model=MODEL, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            ))
            raw = resp.text
            data = json.loads(raw)
            CACHE[key] = raw
            save_cache()
        data = json.loads(raw)
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [])
        counts = {}
        for d in DOCS.values():
            counts[d["repo"]] = counts.get(d["repo"], 0) + 1
        best = {}
        rf = repo_facets()
        for r in data:
            n = int(r["psi"])
            if n in items:
                score = max(0, min(100, int(r["score"])))
                if n not in best or score > best[n]["score"]:
                    v = items[n]
                    folder = folder_name(n, v["group"])
                    best[n] = {
                        "n": n, "score": score, "reason": str(r.get("reason", "")), "group": v["group"],
                        "title": " · ".join(v["titles"]), "folder": folder, "files": counts.get(folder, 0),
                        "url": PSI_URL.format(q=f"PSI-{n}"),
                        "o2": (rf.get(folder) or {}).get("o2"), "p": (rf.get(folder) or {}).get("p"),
                        "fuels": ((rf.get(folder) or {}).get("cats") or {}).get("fuel", []),
                    }
        return jsonify(ok=True, items=sorted(best.values(), key=lambda x: -x["score"]))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/chat")
def chat():
    b = request.get_json(force=True)
    msgs = [m for m in (b.get("messages") or []) if m.get("role") in ("user", "assistant") and str(m.get("text") or "").strip()][-12:]
    if not msgs or msgs[-1]["role"] != "user":
        return jsonify(ok=False, error="Write a question first"), 400
    lang = "Spanish" if b.get("lang") == "es" else "English"
    last = str(msgs[-1]["text"])[:4000]
    d = DOCS.get(b.get("file") or "")
    if d and d["type"] != "image":
        t = doc_text(d).strip()[:20000]
        if t:
            last += f"\n\n[Attached file {d['name']} (repository {d['repo']})]\n{t}"
    try:
        from google.genai import types
        contents = [
            types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part(text=str(m["text"])[:4000])])
            for m in msgs[:-1]
        ] + [types.Content(role="user", parts=[types.Part(text=last)])]
        files = "\n".join(sorted(f"{x['repo']}/{x['name']}" for x in DOCS.values())[:150])
        system = CHAT_SYSTEM.format(lang=lang, catalog=catalog_text() or "(index.txt not loaded)", files=files or "(none)", history=history_text())
        client = get_client()
        resp = None
        grounded = False
        if b.get("search", True):
            try:
                resp = call(lambda: client.models.generate_content(
                    model=MODEL, contents=contents,
                    config=types.GenerateContentConfig(system_instruction=system, tools=[types.Tool(google_search=types.GoogleSearch())]),
                ))
                grounded = True
            except Exception:
                resp = None
        if resp is None:
            resp = call(lambda: client.models.generate_content(model=MODEL, contents=contents, config=types.GenerateContentConfig(system_instruction=system)))
        if not resp.text:
            raise RuntimeError("Gemini returned an empty answer")
        return jsonify(ok=True, text=resp.text, sources=sources_of(resp), grounded=grounded)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.post("/api/ai")
def ai():
    b = request.get_json(force=True)
    task = b.get("task", "summary")
    lang = "Spanish" if b.get("lang") == "es" else "English"
    refresh = bool(b.get("refresh"))
    try:
        if task == "chart":
            content = (b.get("payload") or "")[:AI_LIMIT]
            instruction = (
                "These are statistics and settings of a chart built from a NASA PSI table. In 5 short bullets "
                "describe the trends or differences between groups and what they imply for spacecraft fire safety."
            )
            key = None
        elif task == "ml":
            content = (b.get("payload") or "")[:AI_LIMIT]
            instruction = ML_EXPLAIN
            key = None
        elif task == "mlpred":
            content = (b.get("payload") or "")[:AI_LIMIT]
            instruction = ML_PRED_EXPLAIN
            key = None
        elif task == "repo":
            docs = [d for d in DOCS.values() if d["repo"] == b.get("repo")]
            if not docs:
                return jsonify(ok=False, error="Repository not found"), 404
            rf = repo_facets().get(b.get("repo")) or {}
            body = "\n\n".join(f"### {d['name']}\n{doc_text(d)[:2500]}" for d in docs)
            content = (f"Conditions detected in the tables: {json.dumps(rf, ensure_ascii=False)}\n\n" + body)[:AI_LIMIT]
            instruction = (
                "In 6 short bullets summarize this repository: which investigations or experiments it covers, "
                "fuels and conditions, and how it relates to spacecraft fire safety. Then name one data gap."
            )
            stamp = hashlib.md5("|".join(sorted(f"{d['id']}{d['mtime']}" for d in docs)).encode()).hexdigest()
            key = f"repo2|{b.get('repo')}|{stamp}|{lang}"
        else:
            d = DOCS.get(b.get("id"))
            if not d:
                return jsonify(ok=False, error="File not found"), 404
            if d["type"] == "image":
                content = image_parts(d, "File")
            else:
                text = doc_text(d).strip()
                if not text:
                    return jsonify(ok=False, error=d["error"] or "This file has no readable text"), 400
                content = f"File: {d['name']} (repository {d['repo']})\n{text[:AI_LIMIT]}"
            if task == "ask":
                question = (b.get("question") or "").strip()
                if not question:
                    return jsonify(ok=False, error="Write a question first"), 400
                instruction = "Answer this question using only the content: " + question
                key = None
            elif task in TASKS:
                instruction = TASKS[task]
                key = f"{task}|{d['id']}|{d['mtime']}|{lang}"
            else:
                return jsonify(ok=False, error="Unknown task"), 400
        text, cached = answer(key, instruction, content, lang, refresh)
        return jsonify(ok=True, text=text, cached=cached)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


STOP = set("""a an and are as at be by can de del do does el en es for from has have how in is it la las los of on or que the to was were what when where which who why with y""".split())


@app.post("/api/research")
def research():
    b = request.get_json(force=True)
    repo = b.get("repo")
    question = (b.get("question") or "").strip()
    lang = "Spanish" if b.get("lang") == "es" else "English"
    if not question:
        return jsonify(ok=False, error="Write what you want to research first"), 400
    docs = sorted((d for d in DOCS.values() if d["repo"] == repo), key=lambda d: d["name"].lower())
    if not docs:
        return jsonify(ok=False, error="Repository not found"), 404
    try:
        ts = [t for t in re.findall(r"[\w\-\.%]+", question.lower()) if len(t) > 2 and t not in STOP]
        units = []
        for d in docs:
            if d["type"] == "image":
                continue
            for n, (label, text, low) in enumerate(d["chunks"]):
                if text.strip():
                    score = sum(low.count(t) for t in ts) + sum(3 for t in ts if t in d["name"].lower())
                    units.append({"doc": d, "n": n, "label": label, "text": text.strip(), "score": score})
        if not units:
            return jsonify(ok=False, error="This repository has no readable text"), 400
        total = sum(len(u["text"]) for u in units)
        if total <= RESEARCH_LIMIT:
            chosen, full = units, True
        else:
            chosen, used, full = [], 0, False
            for u in sorted(units, key=lambda u: -u["score"]):
                if used + len(u["text"]) > RESEARCH_LIMIT:
                    continue
                chosen.append(u)
                used += len(u["text"])
            order = {d["id"]: i for i, d in enumerate(docs)}
            chosen.sort(key=lambda u: (order[u["doc"]["id"]], u["n"]))
        body = "\n\n".join(f"### [{u['doc']['name']} | {u['label']}]\n{u['text']}" for u in chosen)
        content = [f"Repository {repo}. Excerpts below are labelled [file | page or rows].\n\n" + body]
        imgs = [d for d in docs if d["type"] == "image"][:3]
        for d in imgs:
            try:
                content += image_parts(d, "Image")
            except Exception:
                pass
        instruction = (
            "DEEP RESEARCH REQUEST. Ignore the usual brevity: be thorough and precise. Research this question across "
            "ALL the documents provided: " + question + "\n\n"
            "Structure the answer as: 1) Short answer (2-3 sentences). 2) Key findings: bullets with concrete numbers, "
            "conditions (fuel, O2, pressure, flow, gravity) and results. Do not add source citations or brackets. 3) Disagreements or caveats between sources, if any. 4) What the documents do "
            "not answer. Quote numbers exactly as written; never guess."
        )
        stamp = hashlib.md5("|".join(sorted(f"{d['id']}{d['mtime']}" for d in docs)).encode()).hexdigest()
        key = f"research|{repo}|{stamp}|{lang}|{hashlib.md5(question.lower().encode()).hexdigest()}"
        text, cached = answer(key, instruction, content, lang, bool(b.get("refresh")))
        files = len({u["doc"]["id"] for u in chosen})
        chars = sum(len(u["text"]) for u in chosen)
        note = (f"Read {files} files, {len(chosen)} pages/sections, {chars:,} characters"
                + (" (all the text in this PSI)" if full else f" of {total:,} (the most relevant parts)")
                + (f" + {len(imgs)} images" if imgs else "") + ".")
        return jsonify(ok=True, text=text, cached=cached, note=note)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


if __name__ == "__main__":
    load_index_cache()
    scan()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5050"))
    threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(host=host, port=port, debug=False)