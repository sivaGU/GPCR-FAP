"""
GPCR Class A Functional Activity Prediction Streamlit GUI.

Run from this folder (project root):
  streamlit run streamlit_app.py
"""
import gc as _gc
import http.cookiejar
import json as _json
import os
import re
import shutil
import subprocess as _subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

# Ensure project root (this folder) is on path for src.gpcr
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _artifact_tree_has_models(artifacts_dir: Path) -> bool:
    """True when ``artifacts/manuscript/`` contains loadable exported models."""
    if not artifacts_dir.is_dir():
        return False
    root = artifacts_dir.parent if artifacts_dir.name == "artifacts" else artifacts_dir
    try:
        from src.gpcr.manuscript_bundle import manuscript_bundle_available

        return bool(manuscript_bundle_available(root))
    except Exception:
        return False


def _resolve_handoff_dir() -> Path:
    """
    Resolve the project directory for manuscript models and data.

    Order: ./artifacts in this repo → sibling **artifact sahith** → ../artifacts → cwd.
    """
    local_art = PROJECT_ROOT / "artifacts"
    if _artifact_tree_has_models(local_art):
        return PROJECT_ROOT
    sahith = PROJECT_ROOT.parent / "artifact sahith"
    if _artifact_tree_has_models(sahith):
        return sahith
    parent_art = PROJECT_ROOT.parent / "artifacts"
    if _artifact_tree_has_models(parent_art):
        return PROJECT_ROOT.parent
    return PROJECT_ROOT


HANDOFF_DIR = _resolve_handoff_dir()


def _is_valid_gpcr_data_root(path: Path) -> bool:
    return (path / "Josh_Receptor_Features").is_dir()


def _is_manuscript_ready_gpcr_root(path: Path) -> bool:
    """Full training tree: pocket CSVs plus ML_code (shared_utilities, *_NEW.xlsx)."""
    return _is_valid_gpcr_data_root(path) and (path / "ML_code").is_dir()


def _bundled_ligand_lookup_path() -> Path:
    return HANDOFF_DIR / "artifacts" / "manuscript" / "ligand_feature_lookup.joblib"


def _bundled_ligand_lookup_sqlite_path() -> Path:
    return HANDOFF_DIR / "artifacts" / "manuscript" / "ligand_feature_lookup.sqlite"


def _has_bundled_ligand_lookup() -> bool:
    p = _bundled_ligand_lookup_path()
    if p.is_file() and p.stat().st_size > 1_000_000:
        return True
    s = _bundled_ligand_lookup_sqlite_path()
    return s.is_file() and s.stat().st_size > 100_000


# With ligand_feature_lookup.joblib in git LFS, Cloud only needs pockets + ML_code (~137 MB zip).
_CLOUD_MAX_ZIP_BYTES = 450_000_000


def _is_inference_ready_gpcr_root(path: Path) -> bool:
    """Pockets + ML_code, or pockets + bundled ligand lookup (no per-receptor xlsx on disk)."""
    if _is_manuscript_ready_gpcr_root(path):
        return True
    return _is_valid_gpcr_data_root(path) and _has_bundled_ligand_lookup()


def _apply_gpcr_data_root(root: Path) -> None:
    """Set GPCR_DATA_ROOT and MANUSCRIPT_ML_ROOT when layout is recognized."""
    root = root.resolve()
    os.environ["GPCR_DATA_ROOT"] = str(root)
    ml_code = root / "ML_code"
    if ml_code.is_dir():
        os.environ["MANUSCRIPT_ML_ROOT"] = str(ml_code.resolve())


def _find_gpcr_data_root(base: Path, subdir_hint: str = "") -> Optional[Path]:
    """Locate folder containing Josh_Receptor_Features under base (or nested one level)."""
    if subdir_hint:
        hinted = base / subdir_hint
        if _is_valid_gpcr_data_root(hinted):
            return hinted.resolve()
    if _is_valid_gpcr_data_root(base):
        return base.resolve()
    if not base.is_dir():
        return None
    for child in sorted(base.iterdir()):
        if child.is_dir() and _is_valid_gpcr_data_root(child):
            return child.resolve()
    return None


def _ensure_default_gpcr_data_root() -> None:
    """
    Default **GPCR_DATA_ROOT** to the folder that contains Josh_Receptor_Features (pocket CSVs).

    Prefer sibling **GUI_Folder** (training layout) when present; else project-local bundle.
    Cloud download runs later (after Streamlit import) via _bootstrap_cloud_gpcr_data().
    """
    existing = os.environ.get("GPCR_DATA_ROOT", "").strip()
    if existing and _is_valid_gpcr_data_root(Path(existing)):
        _apply_gpcr_data_root(Path(existing))
        return
    gui = PROJECT_ROOT.parent / "GUI_Folder"
    if _is_valid_gpcr_data_root(gui):
        _apply_gpcr_data_root(gui)
        return
    if _is_valid_gpcr_data_root(PROJECT_ROOT):
        _apply_gpcr_data_root(PROJECT_ROOT)
        return
    if _is_manuscript_ready_gpcr_root(gui):
        _apply_gpcr_data_root(gui)
        return
    if _is_manuscript_ready_gpcr_root(PROJECT_ROOT):
        _apply_gpcr_data_root(PROJECT_ROOT)
        return
    if _is_inference_ready_gpcr_root(PROJECT_ROOT):
        _apply_gpcr_data_root(PROJECT_ROOT)
        return
    # Pocket CSVs only (no ML_code): do not set GPCR_DATA_ROOT here — cloud bootstrap
    # must download the full zip when DATA_ZIP_URL is configured.


_ensure_default_gpcr_data_root()

import streamlit as st


def _read_deploy_cfg(key: str, default: str = "") -> str:
    """Read Streamlit secret or environment variable."""
    val = os.environ.get(key, "").strip()
    if val:
        return val
    try:
        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return default


def _google_drive_file_id(url: str) -> str:
    """Parse file id from common Google Drive share / uc URLs."""
    url = url.strip()
    for pattern in (
        r"[?&]id=([a-zA-Z0-9_-]+)",
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"/open\?id=([a-zA-Z0-9_-]+)",
    ):
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not parse Google Drive file id from: {url}")


def _resolve_google_drive_file_id(url_or_id: str) -> str:
    raw = url_or_id.strip()
    if raw.startswith("http"):
        return _google_drive_file_id(raw)
    return raw


def _google_drive_download_url(url_or_id: str) -> str:
    file_id = _resolve_google_drive_file_id(url_or_id)
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _looks_like_zip_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 4:
        return False
    with open(path, "rb") as fh:
        head = fh.read(512)
    lower = head.lower()
    if b"<html" in lower or b"<!doctype" in lower:
        return False
    return head[:2] == b"PK" or path.stat().st_size > 50_000_000


def _cloud_zip_byte_limit() -> Optional[int]:
    """Max download bytes on Streamlit Cloud when ligand lookup is in the repo."""
    if _is_streamlit_cloud() and _has_bundled_ligand_lookup():
        return _CLOUD_MAX_ZIP_BYTES
    return None


def _check_download_size_limit(dest: Path, max_bytes: Optional[int]) -> None:
    if max_bytes is None or not dest.is_file():
        return
    size = dest.stat().st_size
    if size > max_bytes:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download is {size / 1e9:.2f} GB — too large for Streamlit Cloud (~1 GB RAM). "
            "Upload **GPCRtryagain-inference-slim.zip** (~137 MB: Josh_Receptor_Features + ML_code only) "
            "from `GPCR-FAP-main/GPCRtryagain-inference-slim.zip` and update **DATA_DRIVE_FILE_ID**. "
            "Do not use the ~1.3 GB zip with all *_NEW.xlsx workbooks."
        )


def _stream_url_to_file(
    opener: urllib.request.OpenerDirector,
    download_url: str,
    dest: Path,
    max_bytes: Optional[int] = None,
) -> None:
    total = 0
    with opener.open(download_url, timeout=3600) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                dest.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Download exceeded {max_bytes / 1e6:.0f} MB — aborting to avoid Cloud OOM. "
                    "Use the **137 MB** slim zip, not the ~1.3 GB file."
                )
            out.write(chunk)


def _is_google_drive_ref(url_or_id: str) -> bool:
    raw = url_or_id.strip()
    if not raw:
        return False
    lower = raw.lower()
    if raw.startswith("http"):
        return (
            "drive.google.com" in lower
            or "drive.usercontent.google.com" in lower
            or "docs.google.com" in lower
        )
    return bool(re.fullmatch(r"[a-zA-Z0-9_-]{15,}", raw))


def _format_drive_download_error(errors: list[str]) -> str:
    details = "; ".join(errors)
    lower = details.lower()
    if "too many users" in lower or "download quota" in lower:
        return (
            "Google Drive blocked automated downloads (shared-file daily quota). "
            "Fix: (1) In Drive, right-click the zip -> Make a copy -> share the copy "
            "and set DATA_DRIVE_FILE_ID to the new file ID; (2) wait up to 24 hours; "
            "(3) host the zip on Hugging Face or S3 and set DATA_ZIP_URL to that direct https link. "
            f"Details: {details}"
        )
    return (
        "Google Drive download failed. Share the zip as Anyone with the link (Viewer), "
        "then set secrets to DATA_DRIVE_FILE_ID or DATA_ZIP_URL. "
        f"Details: {details}"
    )


def _download_http_archive(url: str, dest_zip: Path) -> None:
    """Download a zip from a direct HTTPS URL (Hugging Face /resolve/, S3, etc.)."""
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    max_bytes = _cloud_zip_byte_limit()
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; GPCR-FAP/1.0)"},
    )
    total = 0
    with urllib.request.urlopen(req, timeout=3600) as resp, open(dest_zip, "wb") as out:
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                dest_zip.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Download exceeded {max_bytes / 1e6:.0f} MB — aborting to avoid Cloud OOM. "
                    "Use the **137 MB** slim zip, not the ~1.3 GB file."
                )
            out.write(chunk)
    _check_download_size_limit(dest_zip, max_bytes)
    if not _looks_like_zip_file(dest_zip):
        dest_zip.unlink(missing_ok=True)
        raise RuntimeError(
            f"URL did not return a zip file ({url[:96]}). "
            "Use a direct download link, e.g. Hugging Face .../resolve/main/file.zip"
        )


def _download_data_archive(url_or_id: str, dest_zip: Path) -> None:
    if _is_google_drive_ref(url_or_id):
        _download_google_drive(url_or_id, dest_zip)
        return
    url = url_or_id.strip()
    if not url.startswith("http"):
        url = f"https://{url}"
    _download_http_archive(url, dest_zip)


def _gdown_download_file(file_id: str, dest_zip: Path) -> None:
    """gdown 4.x–6.x compatible (no fuzzy= — removed in gdown 6)."""
    import gdown

    dest = str(dest_zip)
    uc_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    last_error: Optional[Exception] = None

    def _try(fn) -> bool:
        nonlocal last_error
        try:
            fn()
            if _looks_like_zip_file(dest_zip):
                return True
            dest_zip.unlink(missing_ok=True)
        except Exception as exc:
            last_error = exc
            dest_zip.unlink(missing_ok=True)
        return False

    # gdown 6: avoid /file/d/.../view URLs (they trigger removed fuzzy= internally).
    max_bytes = _cloud_zip_byte_limit()
    if _try(lambda: gdown.download(id=file_id, output=dest, quiet=False)):
        _check_download_size_limit(dest_zip, max_bytes)
        return
    if _try(lambda: gdown.download(uc_url, dest, quiet=False)):
        _check_download_size_limit(dest_zip, max_bytes)
        return

    if last_error is not None:
        raise last_error
    raise RuntimeError("gdown did not produce a zip file")

def _download_google_drive_with_cookies(download_url: str, dest_zip: Path) -> None:
    """urllib + cookies fallback (large Drive files need confirm token)."""
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")]

    resp = opener.open(download_url, timeout=3600)
    peek = resp.read(65536)

    confirm: Optional[str] = None
    for cookie in cookie_jar:
        if cookie.name.startswith("download_warning"):
            confirm = cookie.value
            break

    text = peek.decode("utf-8", errors="ignore")
    needs_confirm = b"<html" in peek.lower() or "virus scan" in text.lower()
    if confirm is None and needs_confirm:
        for pattern in (
            r"confirm=([0-9A-Za-z_-]+)",
            r"uuid=([0-9a-fA-F-]{36})",
        ):
            match = re.search(pattern, text)
            if match:
                confirm = match.group(1)
                break
        if confirm is None:
            confirm = "t"

    max_bytes = _cloud_zip_byte_limit()
    if confirm:
        resp.close()
        sep = "&" if "?" in download_url else "?"
        final_url = f"{download_url}{sep}confirm={confirm}"
        _stream_url_to_file(opener, final_url, dest_zip, max_bytes=max_bytes)
        _check_download_size_limit(dest_zip, max_bytes)
        return

    total = len(peek)
    with open(dest_zip, "wb") as out:
        out.write(peek)
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                dest_zip.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Download exceeded {max_bytes / 1e6:.0f} MB — aborting to avoid Cloud OOM. "
                    "Use the **137 MB** slim zip, not the ~1.3 GB file."
                )
            out.write(chunk)
    resp.close()
    _check_download_size_limit(dest_zip, max_bytes)


def _download_google_drive(url: str, dest_zip: Path) -> None:
    """Download a Google Drive file (large zip: virus-scan confirm handled)."""
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    file_id = _resolve_google_drive_file_id(url)
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    usercontent_url = (
        f"https://drive.usercontent.google.com/download?id={file_id}"
        "&export=download&confirm=t"
    )
    errors: list[str] = []

    try:
        _gdown_download_file(file_id, dest_zip)
        return
    except ImportError:
        errors.append("gdown not installed")
    except Exception as exc:
        errors.append(f"gdown: {exc}")
        dest_zip.unlink(missing_ok=True)

    for cookie_url in (usercontent_url, download_url):
        try:
            _download_google_drive_with_cookies(cookie_url, dest_zip)
            if _looks_like_zip_file(dest_zip):
                return
            errors.append(f"cookie download from {cookie_url[:48]}... saved HTML or tiny file")
            dest_zip.unlink(missing_ok=True)
        except Exception as exc:
            errors.append(f"urllib: {exc}")
            dest_zip.unlink(missing_ok=True)

    raise RuntimeError(_format_drive_download_error(errors))




def _extract_data_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract zip with low RAM use (system unzip on Linux, else one file at a time)."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        import subprocess

        try:
            subprocess.run(
                ["unzip", "-q", "-o", str(zip_path), "-d", str(extract_dir)],
                check=True,
                timeout=3600,
            )
            print(f"[gpcr-data] extracted via unzip -> {extract_dir}")
            return
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(f"[gpcr-data] unzip failed ({exc}), falling back to zipfile")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for i, member in enumerate(members):
            zf.extract(member, extract_dir)
            if i % 200 == 0:
                _gc.collect()
    print(f"[gpcr-data] extracted {len(members)} zip members -> {extract_dir}")


def _merge_extracted_tree(staging: Path, base: Path) -> None:
    """Move top-level entries from staging into base (replace existing names)."""
    base.mkdir(parents=True, exist_ok=True)
    for child in staging.iterdir():
        dest = base / child.name
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(child), str(dest))


@st.cache_resource(show_spinner=False)
def _prepare_cloud_gpcr_data(zip_url: str, data_dir_name: str, subdir_hint: str) -> str:
    """
    Download/extract GPCR training data once per Streamlit container.
    Returns resolved GPCR_DATA_ROOT as a string, or "" on failure.
    """
    base = (PROJECT_ROOT / data_dir_name).resolve()
    marker = base / ".gpcr_data_ready"
    root = _find_gpcr_data_root(base, subdir_hint=subdir_hint)
    if root and _is_inference_ready_gpcr_root(root):
        marker.parent.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            marker.write_text(str(root), encoding="utf-8")
        return str(root)

    staging = base / "_gpcr_staging"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    print("[gpcr-data] downloading training zip (first run only)...")
    with tempfile.TemporaryDirectory(prefix="gpcr_zip_") as tmp:
        zip_path = Path(tmp) / "gpcr_data.zip"
        _download_data_archive(zip_url, zip_path)
        zip_bytes = zip_path.stat().st_size
        print(f"[gpcr-data] download complete ({zip_bytes / 1e9:.2f} GB), extracting...")
        if _is_streamlit_cloud() and _has_bundled_ligand_lookup() and zip_bytes > _CLOUD_MAX_ZIP_BYTES:
            raise RuntimeError(
                f"Downloaded zip is {zip_bytes / 1e9:.2f} GB — too large for Streamlit Cloud (~1 GB RAM). "
                "Your secrets still point at the **full** inference zip (~1.3 GB). "
                "Upload **GPCRtryagain-inference-slim.zip** (~137 MB: Josh_Receptor_Features + ML_code only) "
                "to Google Drive and set **DATA_DRIVE_FILE_ID** to that file's id. "
                "Git LFS only speeds up cloning the lookup joblib; it does not fix unzip OOM."
            )
        _gc.collect()
        _extract_data_zip(zip_path, staging)
        print("[gpcr-data] extract complete, merging into runtime_data...")
        _gc.collect()

    root = _find_gpcr_data_root(staging, subdir_hint=subdir_hint)
    if not root:
        shutil.rmtree(staging, ignore_errors=True)
        raise FileNotFoundError(
            f"Extracted data under {staging} but no Josh_Receptor_Features folder found. "
            "Set DATA_EXTRACTED_SUBDIR in secrets to the folder inside the zip."
        )
    if not _is_inference_ready_gpcr_root(root):
        shutil.rmtree(staging, ignore_errors=True)
        raise FileNotFoundError(
            f"Extracted data at {root} is not inference-ready. "
            "Need **Josh_Receptor_Features** plus **ML_code**, or pockets plus "
            "**ligand_feature_lookup.joblib** in the repo."
        )

    _merge_extracted_tree(staging, base)
    shutil.rmtree(staging, ignore_errors=True)

    root = _find_gpcr_data_root(base, subdir_hint=subdir_hint)
    if not root or not _is_inference_ready_gpcr_root(root):
        raise FileNotFoundError(f"Data merge into {base} did not produce an inference-ready tree.")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(root), encoding="utf-8")
    return str(root)


def _log_manuscript_debug(prefix: str = "post-bootstrap") -> None:
    """Stdout-only status (no joblib load)."""
    try:
        from src.gpcr.manuscript_features import ligand_lookup_meta, ligand_lookup_entry_count

        gpcr = os.environ.get("GPCR_DATA_ROOT", "")
        ml = os.environ.get("MANUSCRIPT_ML_ROOT", "")
        meta = ligand_lookup_meta(PROJECT_ROOT)
        print(
            f"[manuscript-debug:{prefix}] "
            f"gpcr_data_root={gpcr} ml_root={ml} "
            f"lookup_entries={ligand_lookup_entry_count(PROJECT_ROOT)} "
            f"lookup_source={meta.get('source', '?')}"
        )
    except Exception as exc:
        print(f"[manuscript-debug:{prefix}] status unavailable: {exc}")


def _resolve_inference_gpcr_root(data_dir_name: str, subdir_hint: str) -> Optional[Path]:
    """Find an on-disk GPCR data tree without downloading."""
    current = os.environ.get("GPCR_DATA_ROOT", "").strip()
    if current:
        p = Path(current)
        if _is_inference_ready_gpcr_root(p):
            return p
    if _is_inference_ready_gpcr_root(PROJECT_ROOT):
        return PROJECT_ROOT.resolve()
    base = (PROJECT_ROOT / data_dir_name).resolve()
    for candidate in (base, PROJECT_ROOT):
        root = _find_gpcr_data_root(candidate, subdir_hint=subdir_hint)
        if root and _is_inference_ready_gpcr_root(root):
            return root
    return None


def _apply_inference_root_if_ready(data_dir_name: str, subdir_hint: str) -> bool:
    root = _resolve_inference_gpcr_root(data_dir_name, subdir_hint)
    if root is None:
        return False
    _apply_gpcr_data_root(root)
    _log_manuscript_debug("ready")
    return True


def _resolve_data_download_source() -> tuple[str, str]:
    """Return (url_or_id, secret_key_used). DATA_ZIP_URL wins if both are set."""
    zip_url = _read_deploy_cfg("DATA_ZIP_URL")
    drive_id = _read_deploy_cfg("DATA_DRIVE_FILE_ID")
    if zip_url and drive_id:
        print(
            "[gpcr-data] WARNING: both DATA_ZIP_URL and DATA_DRIVE_FILE_ID are set; "
            "using DATA_ZIP_URL only. Remove the old one from secrets."
        )
    if zip_url:
        return zip_url, "DATA_ZIP_URL"
    if drive_id:
        return drive_id, "DATA_DRIVE_FILE_ID"
    return "", ""


def _bootstrap_cloud_gpcr_data() -> bool:
    """
    Ensure GPCR_DATA_ROOT is set. On Cloud, never auto-download — user must click a button.
    """
    data_dir_name = _read_deploy_cfg("DATA_DIR", "runtime_data")
    subdir_hint = _read_deploy_cfg("DATA_EXTRACTED_SUBDIR", "")

    if _apply_inference_root_if_ready(data_dir_name, subdir_hint):
        return True

    zip_url, data_source_key = _resolve_data_download_source()

    if _has_bundled_ligand_lookup():
        print(
            "[gpcr-data] ligand_feature_lookup.joblib is in the repo — "
            "use a slim Drive zip (Josh_Receptor_Features + ML_code only), not the 1.3GB inference zip."
        )

    if not zip_url:
        _log_manuscript_debug("no-zip-url")
        root = _resolve_inference_gpcr_root(data_dir_name, subdir_hint)
        if root is None and _has_bundled_ligand_lookup():
            st.error(
                "Ligand lookup is in the repo, but **Josh_Receptor_Features** is missing. "
                "Set **DATA_DRIVE_FILE_ID** to a **slim** zip (pockets + ML_code only), "
                "or add pocket CSVs under the app."
            )
        return root is not None

    st.warning(
        "Pocket CSVs are not on disk yet. On Streamlit Cloud (~1 GB RAM), download only the "
        "**~137 MB** slim zip — not a **~1.3 GB** full bundle."
    )
    if st.button("Download pocket data (one-time)", type="primary", key="gpcr_download_pockets_btn"):
        try:
            with st.status("Downloading GPCR data (first run may take several minutes)...", expanded=True):
                preview = zip_url if len(zip_url) <= 72 else zip_url[:69] + "..."
                st.caption(f"Data source: **{data_source_key}** → `{preview}`")
                resolved = _prepare_cloud_gpcr_data(zip_url, data_dir_name, subdir_hint)
                _apply_gpcr_data_root(Path(resolved))
                st.write(f"Using **GPCR_DATA_ROOT**: `{resolved}`")
                ml = os.environ.get("MANUSCRIPT_ML_ROOT", "")
                if ml:
                    st.write(f"Using **MANUSCRIPT_ML_ROOT**: `{ml}`")
                _log_manuscript_debug("after-download")
                if not os.environ.get("MANUSCRIPT_ML_ROOT", "").strip():
                    st.warning(
                        "Downloaded data has no **ML_code** folder — receptor features use pocket CSV fallback."
                    )
            st.rerun()
        except (urllib.error.URLError, zipfile.BadZipFile, RuntimeError, FileNotFoundError, OSError) as exc:
            st.error(f"Could not prepare GPCR data from {data_source_key}: {exc}")
            err_text = str(exc).lower()
            if "too large for streamlit cloud" in err_text or "1.3 gb" in err_text or "too large for cloud" in err_text:
                st.error(
                    "Zip is too large for Cloud RAM. Upload **GPCRtryagain-inference-slim.zip** (~137 MB)."
                )
            if "too many users" in err_text or "download quota" in err_text:
                st.warning(
                    "**Google Drive quota:** make a **copy** of the zip in Drive (new file ID) or use Hugging Face **DATA_ZIP_URL**."
                )
            st.info(
                "Secrets: `DATA_DRIVE_FILE_ID` or `DATA_ZIP_URL`, `DATA_DIR`, "
                "`DATA_EXTRACTED_SUBDIR` = `GPCRtryagain-inference-slim`."
            )
    return False


import pandas as pd
from rdkit import Chem


def _is_streamlit_cloud() -> bool:
    """True on Streamlit Community Cloud (tight ~1 GB RAM)."""
    if os.environ.get("GPCR_FORCE_CLOUD_MODE", "").strip().lower() in ("1", "true", "yes"):
        return True
    if Path("/mount/src").is_dir():
        return True
    return str(os.environ.get("STREAMLIT_RUNTIME_ENVIRONMENT", "")).strip().lower() == "cloud"


_CLOUD = _is_streamlit_cloud()
if _CLOUD:
    os.environ.setdefault("GPCR_JOBLIB_MMAP", "1")
    os.environ.setdefault("GPCR_CLOUD_LITE", "1")
    os.environ.setdefault("GPCR_POCKET_FEATURES_ONLY", "1")
    # Per-SMILES SQLite lookup avoids loading the full joblib dict into RAM.
    _sq = _bundled_ligand_lookup_sqlite_path()
    if not (_sq.is_file() and _sq.stat().st_size > 100_000):
        os.environ.setdefault("GPCR_SKIP_LIGAND_LOOKUP", "1")

from src.gpcr.predict import (
    predict_single,
    load_predictor,
    get_available_receptors,
    get_gpcr_data_root,
)
from src.gpcr.receptor_names import receptor_display_options, resolve_receptor_folder
from src.gpcr.manuscript_bundle import manuscript_bundle_available, scan_manuscript_artifacts
from src.gpcr.manuscript_features import manuscript_debug_status
from src.gpcr.structure_view import py3dmol_available
from src.gpcr.docking import (
    dock_grid_display_defaults,
    ensure_docking_files_folder,
    run_single_receptor_docking,
)

try:
    import streamlit.components.v1 as st_components
except ImportError:
    st_components = None

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RECEPTORS_FILE = DATA_DIR / "gpcr_class_a_receptors.txt"


def extract_smiles_from_file(file_content: bytes, file_extension: str) -> Optional[str]:
    """
    Extract SMILES string from various molecular file formats.
    Supported formats: SDF, PDB, PDBQT, MOL, MOL2, CSV (first row only).
    """
    try:
        ext = file_extension.lower()
        if ext == ".sdf":
            from io import StringIO
            sdf_data = StringIO(file_content.decode("utf-8"))
            supplier = Chem.SDMolSupplier(sdf_data)
            for m in supplier:
                if m is not None:
                    return Chem.MolToSmiles(m, canonical=True)
        elif ext == ".mol":
            mol = Chem.MolFromMolBlock(file_content.decode("utf-8"))
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        elif ext == ".pdb":
            mol = Chem.MolFromPDBBlock(file_content.decode("utf-8"))
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
            lines = file_content.decode("utf-8").split("\n")
            for line in lines:
                if "SMILES" in line.upper():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "SMILES" in part.upper() and i + 1 < len(parts):
                            potential = parts[i + 1]
                            mol = Chem.MolFromSmiles(potential)
                            if mol:
                                return Chem.MolToSmiles(mol, canonical=True)
        elif ext == ".pdbqt":
            mol = Chem.MolFromPDBBlock(file_content.decode("utf-8"))
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
            lines = file_content.decode("utf-8").split("\n")
            for line in lines:
                if "SMILES" in line.upper():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "SMILES" in part.upper() and i + 1 < len(parts):
                            potential = parts[i + 1]
                            mol = Chem.MolFromSmiles(potential)
                            if mol:
                                return Chem.MolToSmiles(mol, canonical=True)
        elif ext == ".mol2":
            try:
                mol = Chem.MolFromMol2Block(file_content.decode("utf-8"))
                if mol:
                    return Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                pass
        elif ext == ".csv":
            from io import BytesIO
            df = pd.read_csv(BytesIO(file_content))
            col = next((c for c in df.columns if c.lower() in ("smiles", "smi") or c == "SMILES"), None)
            if col and len(df) > 0:
                return str(df[col].iloc[0]).strip()
    except Exception:
        pass
    return None


# ============================================================================
# CONFIGURATION
# ============================================================================

PAGE_HOME = "Home"
PAGE_DOCS = "Documentation"
PAGE_PREDICT = "GPCR Ligand Functional Activity Prediction"

NAV_PAGES = [PAGE_HOME, PAGE_DOCS, PAGE_PREDICT]

st.set_page_config(
    page_title="GPCR Class A Functional Activity Prediction",
    page_icon=None,
    layout="wide",
    menu_items={
        "About": "GPCR Class A Functional Activity Prediction GUI - Predicts Agonist/Antagonist/Inactive for receptor-ligand pairs.",
    },
)

# MBind-style palette: higher contrast gradient, white content card, dark sidebar
_LIGHT_POWDER_BLUE = "#6B9FC0"
_SOFT_SKY_BLUE = "#4A7FA8"
_LIGHT_AZURE = "#2D6B94"
_MEDIUM_STEEL_BLUE = "#1C5C8F"
_DEEP_CERULEAN = "#0D4F7A"
_RICH_TEAL_BLUE = "#004566"
_MIDNIGHT_AZURE = "#003A5F"
_WHITE = "#FFFFFF"

st.markdown(
    f"""
<style>
.stApp {{
    background: linear-gradient(135deg, {_LIGHT_POWDER_BLUE} 0%, {_SOFT_SKY_BLUE} 35%,
        {_LIGHT_AZURE} 70%, {_MIDNIGHT_AZURE} 100%);
}}

.block-container {{
    background-color: rgba(255, 255, 255, 0.97) !important;
    padding: 2.25rem 2.5rem 4rem 2.5rem !important;
    border-radius: 18px;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.14);
    max-width: 1300px;
    font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}}

.block-container .stButton > button,
.block-container button {{
    color: {_WHITE} !important;
}}
.block-container .stButton > button *,
.block-container button * {{
    color: {_WHITE} !important;
}}

[data-testid="stSidebar"] {{
    background: {_MIDNIGHT_AZURE} !important;
    color: {_WHITE};
    width: 300px !important;
}}
[data-testid="stSidebar"] * {{
    color: {_WHITE} !important;
}}

.stButton > button, .stDownloadButton > button {{
    background: {_MEDIUM_STEEL_BLUE} !important;
    color: {_WHITE} !important;
    border: none;
    border-radius: 999px;
    padding: 0.4rem 1.1rem;
    font-weight: 600;
    font-size: 1rem !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    background: {_DEEP_CERULEAN} !important;
    color: {_WHITE} !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.18);
}}

[data-testid="stSidebar"] .stButton > button {{
    background: {_LIGHT_AZURE} !important;
    color: {_WHITE} !important;
    border: 2px solid {_SOFT_SKY_BLUE} !important;
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    margin-bottom: 0.5rem !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
}}
[data-testid="stSidebar"] .stButton > button * {{
    color: {_WHITE} !important;
    font-weight: 700 !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: {_SOFT_SKY_BLUE} !important;
    border-color: {_LIGHT_AZURE} !important;
    color: {_WHITE} !important;
}}

[data-testid="stFileUploader"],
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small {{
    color: #000000 !important;
}}
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button * {{
    color: #000000 !important;
}}

input[type="radio"], input[type="checkbox"] {{
    accent-color: {_MEDIUM_STEEL_BLUE};
}}

/* Page titles (st.title) and section headings */
.block-container [data-testid="stHeading"] h1,
.block-container h1 {{
    color: {_RICH_TEAL_BLUE} !important;
    font-size: 2.35rem !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem !important;
    padding-bottom: 0.15rem !important;
    border-bottom: 3px solid {_SOFT_SKY_BLUE};
}}
.block-container [data-testid="stHeading"] h2,
.block-container h2 {{
    color: {_RICH_TEAL_BLUE} !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    line-height: 1.25 !important;
    margin-top: 0.5rem !important;
}}
.block-container [data-testid="stHeading"] h3,
.block-container h3 {{
    color: {_RICH_TEAL_BLUE} !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    line-height: 1.3 !important;
    margin-top: 1.35rem !important;
    margin-bottom: 0.45rem !important;
}}
.block-container [data-testid="stHeading"] h4,
.block-container h4,
.block-container h5 {{
    color: {_DEEP_CERULEAN} !important;
    font-size: 1.3rem !important;
    font-weight: 600 !important;
    margin-top: 1rem !important;
    margin-bottom: 0.35rem !important;
}}

.block-container p, .block-container li {{
    color: {_MIDNIGHT_AZURE} !important;
    font-size: 1.0625rem !important;
    line-height: 1.65;
}}
.block-container label, .block-container [data-testid="stWidgetLabel"] p {{
    color: {_MIDNIGHT_AZURE} !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}}
[data-testid="stCaptionContainer"], .stCaption {{
    color: {_DEEP_CERULEAN} !important;
    font-size: 0.98rem !important;
    line-height: 1.5;
}}
[data-testid="stMetricValue"] {{
    color: {_RICH_TEAL_BLUE} !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricLabel"] {{
    color: {_MIDNIGHT_AZURE} !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}}

.block-container table {{
    color: {_MIDNIGHT_AZURE} !important;
    font-size: 1.05rem !important;
}}
.block-container th {{
    color: {_RICH_TEAL_BLUE} !important;
    font-weight: 700 !important;
    font-size: 1.08rem !important;
}}

[data-testid="stAlert"] {{
    font-size: 1.05rem !important;
    line-height: 1.55;
}}

[data-testid="stSidebar"] [data-testid="stHeading"] h1,
[data-testid="stSidebar"] [data-testid="stHeading"] h2,
[data-testid="stSidebar"] [data-testid="stHeading"] h3,
[data-testid="stSidebar"] h3 {{
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em;
    text-transform: none;
    margin-bottom: 0.75rem !important;
}}

hr {{
    margin: 1.75rem 0 !important;
    border: none;
    border-top: 1px solid rgba(0, 58, 95, 0.15) !important;
}}
</style>
""",
    unsafe_allow_html=True,
)


def _page_header(title: str, description: str) -> None:
    """One page title and intro paragraph — consistent across tabs."""
    st.title(title)
    if description:
        st.write(description)

# ============================================================================
# PREDICTOR (single in-memory slot — avoids OOM on Streamlit Cloud)
# ============================================================================

def _reset_predictor_session() -> None:
    for key in (
        "_predictor_loaded",
        "_active_predictor",
        "_predictor_key",
        "_cloud_load_requested",
    ):
        st.session_state.pop(key, None)
    _gc.collect()


def _cloud_model_artifact_ready(
    evaluation_regime: Optional[str],
    model_type: str,
    seed: int,
) -> bool:
    if not evaluation_regime:
        return False
    from src.gpcr.cloud_predict import cloud_model_ready

    return cloud_model_ready(
        HANDOFF_DIR,
        evaluation_regime,
        model_type,
        int(seed),
    )


def _cloud_predict_ephemeral(
    receptor: str,
    ligand_smiles: str,
    evaluation_regime: Optional[str],
    seed: int,
    model_type: str,
) -> dict:
    """Load one cloud model, predict, free RAM (RF / XGB / LGB only)."""
    from src.gpcr.cloud_predict import predict_cloud_manuscript

    _gc.collect()
    result = predict_cloud_manuscript(
        HANDOFF_DIR,
        receptor,
        ligand_smiles,
        evaluation_regime=evaluation_regime or "independent_ligand",
        seed=int(seed),
        model_type=model_type,
    )
    st.session_state.pop("_active_predictor", None)
    st.session_state.pop("_predictor_key", None)
    _gc.collect()
    return {
        "is_valid": result.is_valid,
        "receptor": result.receptor,
        "canonical_smiles": result.canonical_smiles,
        "predicted_class": result.predicted_class,
        "class_id": int(result.class_id),
        "prob_agonist": float(result.prob_agonist),
        "prob_antagonist": float(result.prob_antagonist),
        "prob_inactive": float(result.prob_inactive),
        "error": result.error,
    }


def load_active_predictor(
    model_type: Optional[str] = None,
    evaluation_regime: Optional[str] = None,
    seed: int = 42,
):
    """
    Load at most one predictor per session.

    ``@st.cache_resource`` kept every model type in RAM after a dropdown change;
    ensemble + RF together exceeds Streamlit Cloud's ~1 GiB limit.
    """
    key = (evaluation_regime or "", model_type or "", int(seed))
    cached = st.session_state.get("_active_predictor")
    if st.session_state.get("_predictor_key") == key and cached is not None:
        return cached

    st.session_state.pop("_active_predictor", None)
    _gc.collect()

    label = (model_type or "rf").replace("_", " ").title()
    with st.spinner(f"Loading {label} model…"):
        predictor = load_predictor(
            HANDOFF_DIR,
            model_type=model_type,
            evaluation_regime=evaluation_regime,
            seed=seed,
        )
    st.session_state["_predictor_key"] = key
    st.session_state["_active_predictor"] = predictor
    return predictor

# ============================================================================
# PAGES
# ============================================================================

def render_home_page():
    """Render the home/dashboard page."""
    _page_header(
        "Welcome to GPCR-FAP",
        None,
    )
    st.write(
        "**GPCR-FAP** (GPCR Class A Functional Activity Prediction) scores **Class A GPCR** receptor–ligand pairs "
        "with manuscript-aligned machine learning "
        "models, then optionally docks **your** query ligand with SMINA for a structural sanity check. "
        f"Open **{PAGE_PREDICT}** in the sidebar to start; use **{PAGE_DOCS}** for the step-by-step workflow "
        "and technical reference."
    )

    st.subheader("Highlights")
    st.markdown(
        "- **Manuscript models:** **6,633** features per pair (ligand + receptor pocket + interaction)\n"
        "- **68** Class A receptors aligned with Supplementary File 12\n"
        "- **Multiclass prediction:** Agonist, Antagonist, or Inactive with class probabilities\n"
        "- **Algorithms:** Random Forest, LightGBM, XGBoost (+ stacking ensemble on local runs)\n"
        "- **Evaluation regimes:** independent ligand, scaffold split, and LORO (when exported)\n"
        "- **Optional SMINA docking** and **py3Dmol** 3D view of the top pose after prediction"
    )

    st.subheader("Output guide")
    st.markdown(
        "| Output | What you see |\n"
        "| --- | --- |\n"
        "| Predicted class | Agonist, Antagonist, or Inactive |\n"
        "| Class probabilities | P(Agonist), P(Antagonist), P(Inactive) |\n"
        "| Top docking pose | SMINA score (kcal/mol), when docking is run |\n"
        "| 3D viewer | Receptor + docked ligand in py3Dmol |\n"
        "| Pocket contacts | Closest receptor residues to the docked ligand |"
    )

    st.subheader("Contact Information")
    st.markdown(
        "**For Contact:** Dr. Sivanesan Dakshanamurthy, PhD, MBA. "
        "[sivanesan@innsciteai.com](mailto:sivanesan@innsciteai.com); "
        "[sd233@georgetown.edu](mailto:sd233@georgetown.edu)"
    )


def render_documentation_page():
    """Render the documentation page."""
    _page_header(
        "Documentation",
        "Workflow, models, and docking for the GPCR Class A Functional Activity Prediction GUI.",
    )

    st.subheader("Workflow")
    st.markdown(
        f"1. Open **{PAGE_PREDICT}**.\n"
        "2. Choose **evaluation regime**, **model**, and **seed**.\n"
        "3. Select a **receptor** (68 targets).\n"
        "4. Enter **SMILES** or upload SDF, MOL, PDB, PDBQT, MOL2, or CSV.\n"
        "5. Click **Predict** — class label and P(Agonist), P(Antagonist), P(Inactive).\n"
        "6. Optionally set the docking box and **Run docking and show top pose**."
    )

    st.subheader("Docking ligand")
    st.write(
        "Docking uses **your** predicted ligand (canonical SMILES → 3D → SMINA), not the co-crystal ligand. "
        "Grid defaults come from `*_ligand_only.pdb` or `docking_assets/receptor_grid_boxes.json`. "
        "Receptor structure: `*_receptor_only.pdb` (Cloud: `docking_assets/receptor_pdbs/`)."
    )

    st.subheader("Models and features")
    st.markdown(
        "- **Classes:** Agonist (0), Antagonist (1), Inactive (2)\n"
        "- **Algorithms:** Random Forest, LightGBM, XGBoost; **ensemble** (local only, more RAM)\n"
        "- **Regimes:** independent ligand, scaffold split, LORO (when exported)"
    )
    st.markdown(
        "| Block | Count | Source |\n"
        "| --- | ---: | --- |\n"
        "| Ligand | 6,588 | `manifest.json` / ligand lookup |\n"
        "| Receptor | 31 | `Josh_Receptor_Features` pocket CSVs |\n"
        "| Interaction | 14 | `INT_*` ligand × receptor products |\n"
        "| **Total** | **6,633** | Manuscript `.pkl` models |"
    )

    if not _is_streamlit_cloud():
        st.subheader("Local setup")
        st.markdown(
            "1. `pip install -r requirements.txt`\n"
            "2. **Josh_Receptor_Features** in the repo or **`GPCR_DATA_ROOT`**\n"
            "3. **`artifacts/manuscript/`** with models, `manifest.json`, `ligand_feature_lookup.sqlite`\n"
            "4. `streamlit run streamlit_app.py`"
        )

    st.subheader("SMINA defaults")
    st.markdown(
        "- `exhaustiveness=64`, `num_modes=10`, `seed=42`, scoring `vina`\n"
        "- Grid: centroid ± padded co-crystal ligand box (15–20 Å per axis), editable in the UI"
    )


def _load_receptor_select_options():
    """
    (folder_name, display_label) for the receptor dropdown.
    Uses Josh_Receptor_Features folder names; labels include gene aliases (e.g. beta2 (ADRB2)).
    """
    options = receptor_display_options(get_gpcr_data_root())
    if options:
        return options
    if not RECEPTORS_FILE.exists():
        return []
    try:
        lines = RECEPTORS_FILE.read_text(encoding="utf-8").strip().splitlines()
        return [(s.strip(), s.strip()) for s in lines if s.strip()]
    except Exception:
        return []


def render_gpcr_prediction_page():
    """Render the GPCR Ligand Functional Activity Prediction page."""
    if not _bootstrap_cloud_gpcr_data():
        st.error(
            "Training data is not ready. If **ligand_feature_lookup.joblib** is in the repo, set "
            "**DATA_DRIVE_FILE_ID** to a **slim** zip (Josh_Receptor_Features + ML_code only). "
            "Do not use the 1.3 GB full inference zip on Cloud — it OOMs during unzip."
        )
        return

    _page_header(
        PAGE_PREDICT,
        "Select a receptor, provide SMILES or upload a structure file, and obtain "
        "Agonist / Antagonist / Inactive probabilities. Optionally generate a SMINA docked pose and 3D view.",
    )

    _artifact_scan = scan_manuscript_artifacts(HANDOFF_DIR)
    _has_manuscript = manuscript_bundle_available(HANDOFF_DIR)
    _ms_regimes = _artifact_scan.get("regimes") or {}
    _ms_seeds = _artifact_scan.get("seeds") or [42]

    if not _has_manuscript:
        st.error(
            "Manuscript models are not available. Deploy `artifacts/manuscript/` "
            "(models, `manifest.json`, and `ligand_feature_lookup.sqlite`)."
        )
        return

    st.markdown("#### Evaluation regime")
    _regime_labels = {
        "independent_ligand": "Independent ligand test (paper: dev80-trained)",
        "scaffold": "Scaffold split",
        "loro": "Leave-one-receptor-out (LORO)",
    }
    regime_map: dict = {}
    regime_options: list = []
    for key, label in _regime_labels.items():
        if _ms_regimes.get(key):
            regime_options.append(label)
            regime_map[label] = key

    if not regime_options:
        st.error(
            "No exported evaluation regimes found under `artifacts/manuscript/`. "
            "Deploy at least one of: independent_ligand, scaffold, loro."
        )
        return

    regime_label = st.selectbox(
        "Evaluation regime",
        regime_options,
        index=0,
        key="gpcr_eval_regime",
        help="Only regimes with exported artifacts are listed.",
    )
    evaluation_regime = regime_map[regime_label]

    _model_labels = {
        "rf": "Random Forest",
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "ensemble": "Ensemble (stacking)",
    }
    model_type_map = {v: k for k, v in _model_labels.items()}

    if evaluation_regime and _ms_regimes.get(evaluation_regime):
        available_models = list(_ms_regimes[evaluation_regime].keys())
        cloud_ok = (
            ("rf", "lightgbm", "xgboost")
            if _is_streamlit_cloud()
            else ("rf", "lightgbm", "xgboost", "ensemble")
        )
        if _is_streamlit_cloud():
            dropped = [m for m in available_models if m not in cloud_ok]
            if dropped:
                st.caption(
                    f"On Streamlit Cloud, **{', '.join(dropped)}** is hidden (~1 GB RAM). "
                    "RF, LightGBM, and XGBoost load **one at a time** per Predict."
                )
        available_models = [m for m in available_models if m in cloud_ok]
        if _is_streamlit_cloud() and "rf" not in available_models:
            st.error(
                "**Random Forest** is not deployed (only a placeholder `model_seed42.pkl` on GitHub, or LFS pull failed). "
                "Deploy `artifacts/manuscript/independent_ligand/rf/model_seed42_cloud.pkl` "
                "(or full `model_seed42.pkl`) via Git LFS."
            )
            return
        # RF first (default) — ensemble loads RF+XGB+LGB+meta and OOMs on Cloud (~1 GB RAM).
        model_options = [_model_labels[m] for m in ("rf", "lightgbm", "xgboost", "ensemble") if m in available_models]
        default_model_ix = 0
    else:
        st.error(f"No models listed for regime `{evaluation_regime}`. Re-export manuscript artifacts.")
        return

    if not model_options:
        st.error(
            f"No models available for regime `{evaluation_regime}` on this deployment. "
            "Export RF, XGBoost, and/or LightGBM for this regime and redeploy."
        )
        return

    model_type_label = st.selectbox(
        "Model type",
        model_options,
        index=default_model_ix,
        key="gpcr_pred_model",
        help="Only models present under artifacts/manuscript/ are shown.",
    )
    model_type = model_type_map[model_type_label]

    if evaluation_regime == "loro" and model_type == "ensemble":
        st.warning("Manuscript stacking ensemble was evaluated on the **independent ligand** split, not LORO.")

    seed = 42
    if evaluation_regime and evaluation_regime in _ms_regimes:
        regime_seeds = sorted(
            set(_ms_regimes[evaluation_regime].get(model_type, [])) & set(_ms_seeds)
        ) or [42]
        seed = st.selectbox(
            "Random seed",
            regime_seeds,
            index=0,
            key="gpcr_model_seed",
            help="Seeds exported to artifacts/manuscript/ (default: 42).",
        )

    if _is_streamlit_cloud() and model_type == "ensemble":
        st.warning("Ensemble needs RF+XGB+LGB together — not available on Streamlit Cloud (~1 GB RAM).")
        return

    predictor = None
    cloud_ephemeral_mode = False
    if _is_streamlit_cloud():
        if not _cloud_model_artifact_ready(evaluation_regime, model_type, seed):
            from src.gpcr.cloud_predict import cloud_model_path

            expected = cloud_model_path(
                HANDOFF_DIR,
                evaluation_regime or "independent_ligand",
                model_type,
                int(seed),
            )
            st.error(
                f"**Missing** `{expected.name}` for **{model_type_label}**. "
                + (
                    "Deploy `model_seed42_cloud.pkl` for RF on Cloud."
                    if model_type == "rf"
                    else f"Export and `git lfs push` to `artifacts/manuscript/.../{model_type}/`."
                )
            )
            return
        cloud_ephemeral_mode = True
        _cloud_file = (
            f"model_seed{seed}_cloud.pkl"
            if model_type == "rf"
            else f"model_seed{seed}.pkl"
        )
        st.caption(
            f"**Cloud:** **{model_type_label}** (`{_cloud_file}`) loads only when you click **Predict**, "
            "then memory is freed. Ensemble is not available here."
        )
    else:
        try:
            predictor = load_active_predictor(model_type, evaluation_regime=evaluation_regime, seed=seed)
        except Exception as e:
            st.error(f"Could not load {model_type_label} model: {e}")
            st.info(
                f"Expected: `artifacts/manuscript/{evaluation_regime}/{model_type}/model_seed{seed}.pkl`"
            )
            return

    if predictor is None and not cloud_ephemeral_mode:
        return

    st.sidebar.markdown("### Model Info")
    if cloud_ephemeral_mode:
        st.sidebar.info(
            f"**Regime:** {regime_label}\n\n"
            f"**Model:** {model_type_label} (cloud, load per predict)\n\n"
            f"**Seed:** {seed}\n\n"
            f"**Feature mode:** manuscript"
        )
        _mode = "manuscript"
    else:
        _mode = getattr(predictor, "feature_mode", "manuscript")
    if not cloud_ephemeral_mode:
        st.sidebar.info(
            f"**Regime:** {regime_label}\n\n"
            f"**Model:** {model_type_label}\n\n"
            f"**Seed:** {seed}\n\n"
            f"**Feature mode:** {_mode}\n\n"
            f"**Loaded:** {len(predictor.models)} estimator(s)\n\n"
            f"**Classes:** {', '.join(predictor.class_names)}"
        )
    _gdata = os.environ.get("GPCR_DATA_ROOT", "").strip()
    _efd = getattr(predictor, "expected_feature_dim", None) if predictor is not None else 6633
    st.caption(
        f"**ML bundle:** `{HANDOFF_DIR}` · **Pocket data:** `{_gdata or 'default'}` · "
        f"**Features:** {_efd if _efd is not None else '—'} dims"
    )
    if _mode == "manuscript" and not _is_streamlit_cloud():
        st.caption("Install **mordred** for best ligand descriptor parity with enriched training CSVs.")
        dbg = manuscript_debug_status(HANDOFF_DIR)
        print(
            "[manuscript-debug] "
            f"gpcr_data_root={dbg['gpcr_data_root']} "
            f"ml_root={dbg['ml_root']} "
            f"ligand_lookup_entries={dbg['ligand_lookup_entries']}"
        )
        with st.sidebar.expander("Manuscript feature diagnostics", expanded=False):
            st.write(f"GPCR data root: `{dbg['gpcr_data_root']}`")
            st.write(f"ligand lookup entries: `{dbg['ligand_lookup_entries']}`")
    elif _mode == "manuscript" and _is_streamlit_cloud():
        _sqlite = _bundled_ligand_lookup_sqlite_path()
        if _sqlite.is_file() and _sqlite.stat().st_size > 100_000:
            st.caption(
                "**Cloud:** SQLite ligand lookup + one model (RF / XGB / LGB) loaded per **Predict**."
            )
        else:
            st.warning(
                "**ligand_feature_lookup.sqlite** not deployed — commit it via Git LFS and redeploy. "
                "Until then, Cloud uses Mordred-only ligand features (less accurate)."
            )

    st.subheader("Ligand input")

    receptor_options = _load_receptor_select_options()
    if not receptor_options:
        st.warning(
            "No receptors found under **Josh_Receptor_Features**. "
            "Set **GPCR_DATA_ROOT** to the folder that contains **Josh_Receptor_Features**, "
            "or place **GUI_Folder** next to this project (see README)."
        )
    folder_names = [f for f, _ in receptor_options]
    display_labels = [lbl for _, lbl in receptor_options]
    receptor_pick = st.selectbox(
        "GPCR Class A Receptor",
        options=["Select receptor..."] + display_labels,
        key="receptor_input",
        help="Pocket folder name with optional gene symbol (e.g. beta2 = ADRB2).",
    )
    if receptor_pick and receptor_pick != "Select receptor...":
        idx = display_labels.index(receptor_pick)
        receptor_selected = folder_names[idx]
    else:
        receptor_selected = ""

    ligand_input = st.text_input(
        "Ligand SMILES (or upload a structure file below)",
        placeholder="e.g. CCO, c1ccccc1",
        key="ligand_input",
    )
    
    st.write("Or upload a ligand structure file:")
    structure_file = st.file_uploader(
        "Upload ligand structure file",
        type=["sdf", "mol", "pdb", "pdbqt", "mol2", "csv"],
        key="structure_upload",
        help="Supported: SDF, MOL, PDB, PDBQT, MOL2, CSV (first smiles row).",
    )
    
    ligand_to_use = None
    if structure_file:
        content = structure_file.read()
        ext = os.path.splitext(structure_file.name)[1]
        extracted = extract_smiles_from_file(content, ext)
        if extracted:
            ligand_to_use = extracted
            st.success(f"Extracted SMILES from {structure_file.name}")
        else:
            st.error(f"Could not extract SMILES from {ext.upper()} file. Try SMILES input instead.")
    elif ligand_input and ligand_input.strip():
        ligand_to_use = ligand_input.strip()

    st.caption(
        "Workflow: run FAP prediction first, then click the docking button below to generate and visualize "
        "a top docking pose."
    )

    def _render_single_prediction_from_session(pred: dict) -> None:
        """Render persisted single-prediction outputs so docking reruns do not reset the panel."""
        st.success("Valid input")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Class", str(pred["predicted_class"]))
        with col2:
            st.metric("Receptor", str(pred["receptor"]))
        with col3:
            st.metric("Class ID", int(pred["class_id"]))

        st.subheader("Probability Distributions")
        prob_col1, prob_col2, prob_col3 = st.columns(3)
        with prob_col1:
            st.metric("P(Agonist)", f"{float(pred['prob_agonist']):.4f}")
        with prob_col2:
            st.metric("P(Antagonist)", f"{float(pred['prob_antagonist']):.4f}")
        with prob_col3:
            st.metric("P(Inactive)", f"{float(pred['prob_inactive']):.4f}")

        probs = {
            "Agonist": float(pred["prob_agonist"]),
            "Antagonist": float(pred["prob_antagonist"]),
            "Inactive": float(pred["prob_inactive"]),
        }
        if _CLOUD:
            st.bar_chart(probs, height=320)
        else:
            import plotly.graph_objects as go

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=list(probs.keys()),
                        y=list(probs.values()),
                        marker_color=["#7E57C2", "#673AB7", "#512DA8"],
                        text=[f"{v:.3f}" for v in probs.values()],
                        textposition="auto",
                    )
                ]
            )
            fig.update_layout(
                title="Class Probability Distribution",
                xaxis_title="Class",
                yaxis_title="Probability",
                yaxis_range=[0, 1],
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    def _render_docking_section(last_pred: dict) -> None:
        """Docking + SMINA + py3Dmol (same flow as legacy working app)."""
        st.divider()
        st.subheader("Docking + receptor-ligand visualization")
        st.caption(
            "**Recommended** grid center and size come from this receptor's `<id>_ligand_only.pdb` "
            "(centroid and padded extent, each axis clipped to 15–20 Å). You can override these in the panel below. "
            "Pose generation uses SMINA with defaults: exhaustiveness=64, num_modes=10, seed=42."
        )

        dock_folder = str(last_pred["receptor"])
        data_root = get_gpcr_data_root()
        dock_folder_resolved = (
            resolve_receptor_folder(dock_folder, data_root) or dock_folder
        )
        ensure_docking_files_folder(HANDOFF_DIR)
        rec_center, rec_size, has_recommended, grid_help = dock_grid_display_defaults(
            dock_folder_resolved,
            project_root=HANDOFF_DIR,
        )

        with st.expander(
            "Docking search box (recommended vs. custom)",
            expanded=not has_recommended,
        ):
            if not has_recommended:
                st.warning(grid_help)
                st.caption(
                    "Automatic grid from `*_ligand_only.pdb` is unavailable (often Git LFS on Cloud). "
                    "Use the values below — or open **Reset box to recommended** after redeploying real PDBs."
                )
            else:
                st.caption(
                    "These defaults follow the co-crystal ligand geometry (or precomputed "
                    "`docking_assets/receptor_grid_boxes.json`). Edited values are passed to SMINA as "
                    "`--center_*` and `--size_*`."
                )
            cx, cy, cz = st.columns(3)
            with cx:
                st.number_input(
                    "Center X (Å)",
                    format="%.3f",
                    step=0.1,
                    value=float(rec_center[0]),
                    key=f"dock_cx_{dock_folder}",
                )
            with cy:
                st.number_input(
                    "Center Y (Å)",
                    format="%.3f",
                    step=0.1,
                    value=float(rec_center[1]),
                    key=f"dock_cy_{dock_folder}",
                )
            with cz:
                st.number_input(
                    "Center Z (Å)",
                    format="%.3f",
                    step=0.1,
                    value=float(rec_center[2]),
                    key=f"dock_cz_{dock_folder}",
                )
            sx, sy, sz = st.columns(3)
            with sx:
                st.number_input(
                    "Size X (Å)",
                    format="%.3f",
                    step=0.5,
                    min_value=1.0,
                    max_value=80.0,
                    value=float(rec_size[0]),
                    key=f"dock_sx_{dock_folder}",
                )
            with sy:
                st.number_input(
                    "Size Y (Å)",
                    format="%.3f",
                    step=0.5,
                    min_value=1.0,
                    max_value=80.0,
                    value=float(rec_size[1]),
                    key=f"dock_sy_{dock_folder}",
                )
            with sz:
                st.number_input(
                    "Size Z (Å)",
                    format="%.3f",
                    step=0.5,
                    min_value=1.0,
                    max_value=80.0,
                    value=float(rec_size[2]),
                    key=f"dock_sz_{dock_folder}",
                )
            if has_recommended and st.button(
                "Reset box to recommended", key=f"dock_reset_grid_{dock_folder}"
            ):
                st.session_state[f"dock_cx_{dock_folder}"] = float(rec_center[0])
                st.session_state[f"dock_cy_{dock_folder}"] = float(rec_center[1])
                st.session_state[f"dock_cz_{dock_folder}"] = float(rec_center[2])
                st.session_state[f"dock_sx_{dock_folder}"] = float(rec_size[0])
                st.session_state[f"dock_sy_{dock_folder}"] = float(rec_size[1])
                st.session_state[f"dock_sz_{dock_folder}"] = float(rec_size[2])
                st.rerun()

        if st.button("Run docking and show top pose", key="btn_single_docking", type="secondary"):
            with st.spinner("Running docking..."):
                grid_kw = {
                    "grid_center": (
                        float(st.session_state[f"dock_cx_{dock_folder}"]),
                        float(st.session_state[f"dock_cy_{dock_folder}"]),
                        float(st.session_state[f"dock_cz_{dock_folder}"]),
                    ),
                    "grid_size": (
                        float(st.session_state[f"dock_sx_{dock_folder}"]),
                        float(st.session_state[f"dock_sy_{dock_folder}"]),
                        float(st.session_state[f"dock_sz_{dock_folder}"]),
                    ),
                }
                dock_res = run_single_receptor_docking(
                    receptor_folder=dock_folder_resolved,
                    canonical_smiles=str(last_pred["canonical_smiles"]),
                    **grid_kw,
                )
            st.session_state["last_docking_result"] = dock_res.__dict__

        dock_result = st.session_state.get("last_docking_result")
        if dock_result and dock_result.get("receptor_name") in (
            dock_folder,
            dock_folder_resolved,
        ):
            if dock_result.get("ok"):
                if not py3dmol_available():
                    st.info("Install **py3Dmol** to render the docked complex: `pip install py3Dmol`")
                elif st_components is None:
                    st.warning("streamlit.components is unavailable; cannot embed the docked 3D viewer.")
                elif dock_result.get("html"):
                    st.markdown(
                        "**3D viewer:** white receptor cartoon, green ligand sticks; the **three closest residues** "
                        "to the docked ligand are emphasized with **dashed cylinders** from the best contact atom on "
                        "each residue to the ligand (MBind-style; teal ≈ polar, green ≈ aromatic C–C, slate ≈ other)."
                    )
                    st_components.html(str(dock_result["html"]), height=560, scrolling=False)
                    st.caption(
                        "**3D viewer:** drag to rotate the scene • **Ctrl+drag** or **middle mouse** drag to pan "
                        "(move left/right and up/down) • scroll to zoom."
                    )
                    score = dock_result.get("score_kcal_mol")
                    st.markdown(
                        f"**Top Pose Docking Score (kcal/mol):** "
                        f"{float(score):.3f}" if score is not None else "**Top Pose Docking Score (kcal/mol):** N/A"
                    )
                    gc = dock_result.get("center")
                    gs = dock_result.get("size")
                    if gc and gs and len(gc) == 3 and len(gs) == 3:
                        st.caption(
                            f"**Search box used:** center ({float(gc[0]):.3f}, {float(gc[1]):.3f}, {float(gc[2]):.3f}) Å · "
                            f"size ({float(gs[0]):.3f}, {float(gs[1]):.3f}, {float(gs[2]):.3f}) Å"
                        )
                    contacts = dock_result.get("contact_summary")
                    if contacts:
                        st.markdown("**Closest residue contacts (≤5 Å, best heavy-atom pair per residue):**")
                        for line in contacts:
                            st.markdown(f"- {line}")
                else:
                    st.warning("Docking succeeded, but the 3D viewer payload was empty.")
            else:
                st.error(str(dock_result.get("message", "Docking failed.")))

    def _run_single_predict() -> None:
        if receptor_selected and ligand_to_use:
            if _CLOUD:
                _gc.collect()
            try:
                if cloud_ephemeral_mode:
                    with st.spinner("Running prediction…"):
                        payload = _cloud_predict_ephemeral(
                            receptor_selected,
                            ligand_to_use,
                            evaluation_regime,
                            seed,
                            model_type,
                        )
                    is_valid = bool(payload.get("is_valid"))
                    err_msg = str(payload.get("error") or "")
                else:
                    result = predict_single(
                        receptor_selected,
                        ligand_to_use,
                        predictor=predictor,
                    )
                    is_valid = result.is_valid
                    err_msg = result.error
                    payload = {
                        "receptor": result.receptor,
                        "canonical_smiles": result.canonical_smiles,
                        "predicted_class": result.predicted_class,
                        "class_id": int(result.class_id),
                        "prob_agonist": float(result.prob_agonist),
                        "prob_antagonist": float(result.prob_antagonist),
                        "prob_inactive": float(result.prob_inactive),
                    }
            except (RuntimeError, FileNotFoundError, MemoryError, OSError) as exc:
                st.session_state.pop("last_single_prediction", None)
                st.error(f"Prediction failed: {exc}")
                if _CLOUD:
                    _gc.collect()
                return
            if is_valid:
                st.session_state["last_single_prediction"] = {
                    "receptor": payload["receptor"],
                    "canonical_smiles": payload["canonical_smiles"],
                    "predicted_class": payload["predicted_class"],
                    "class_id": int(payload["class_id"]),
                    "prob_agonist": float(payload["prob_agonist"]),
                    "prob_antagonist": float(payload["prob_antagonist"]),
                    "prob_inactive": float(payload["prob_inactive"]),
                }
                st.session_state.pop("last_docking_result", None)
                st.rerun()
            else:
                st.session_state.pop("last_single_prediction", None)
                st.session_state.pop("last_docking_result", None)
                st.error(err_msg or "Prediction failed")
            if _CLOUD:
                _gc.collect()
        else:
            st.warning("Please select a GPCR Class A receptor and provide ligand SMILES or upload a structure file.")

    if st.button("Predict", type="primary", key="btn_single"):
        _run_single_predict()

    last_pred = st.session_state.get("last_single_prediction")
    if last_pred:
        _render_single_prediction_from_session(last_pred)
        _render_docking_section(last_pred)

    st.divider()
    st.caption(
        "GPCR Class A Functional Activity Prediction. Multi-class classification: Agonist/Antagonist/Inactive."
    )


# ============================================================================
# MAIN - NAVIGATION
# ============================================================================

def _render_sidebar_nav() -> None:
    """Sidebar navigation (original styling)."""
    st.sidebar.markdown("### Navigation")
    current = st.session_state.current_page

    if st.sidebar.button(PAGE_HOME, use_container_width=True, key="nav_home"):
        if current != PAGE_HOME:
            st.session_state.current_page = PAGE_HOME
            st.rerun()

    if st.sidebar.button(PAGE_DOCS, use_container_width=True, key="nav_docs"):
        if current != PAGE_DOCS:
            st.session_state.current_page = PAGE_DOCS
            st.rerun()

    if st.sidebar.button(PAGE_PREDICT, use_container_width=True, key="nav_prediction"):
        if current != PAGE_PREDICT:
            st.session_state.current_page = PAGE_PREDICT
            if _is_streamlit_cloud():
                _reset_predictor_session()
            st.rerun()

    st.sidebar.markdown("---")


def main():
    """Main app entry point with navigation."""
    if "current_page" not in st.session_state:
        st.session_state.current_page = PAGE_HOME
    if st.session_state.current_page not in NAV_PAGES:
        st.session_state.current_page = PAGE_HOME

    _render_sidebar_nav()

    page = st.session_state.current_page
    if page == PAGE_HOME:
        render_home_page()
    elif page == PAGE_DOCS:
        render_documentation_page()
    elif page == PAGE_PREDICT:
        render_gpcr_prediction_page()


if __name__ == "__main__":
    main()
