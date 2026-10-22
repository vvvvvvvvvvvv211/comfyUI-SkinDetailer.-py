"""Small, dependency-free model downloader used by the Skin Detailer node pack.

ComfyUI 0.22.3 ships no general-purpose model download helper (only
``app/frontend_management.py`` knows how to fetch frontend release zips), so we
implement the minimum here: resolve a file inside a ``folder_paths`` model
folder, and fetch it over HTTPS with a ComfyUI ``ProgressBar`` if it is absent.
"""

from __future__ import annotations

import logging
import hashlib
import os
import tempfile
import urllib.error
import urllib.request
from typing import Iterable

import comfy.utils
import comfy.model_management
import folder_paths

log = logging.getLogger("SkinDetailer")

_USER_AGENT = "ComfyUI-SkinDetailer/1.0 (+https://github.com/)"
_CHUNK = 1 << 20  # 1 MiB


class ModelDownloadError(RuntimeError):
    """Raised when a required model file is missing and cannot be fetched."""


def find_in_folder(folder_name: str, filename: str) -> str | None:
    """Return the full path of ``filename`` inside a ComfyUI model folder.

    Matching is case-insensitive and also accepts entries living in
    subdirectories (``get_filename_list`` returns relative paths).
    """
    direct = folder_paths.get_full_path(folder_name, filename)
    if direct is not None:
        return direct

    target = filename.lower()
    try:
        candidates: Iterable[str] = folder_paths.get_filename_list(folder_name)
    except KeyError:
        return None
    for entry in candidates:
        if os.path.basename(entry).lower() == target:
            return folder_paths.get_full_path(folder_name, entry)
    return None


def default_folder_path(folder_name: str) -> str:
    """Pick the directory a downloaded model should land in.

    ``get_folder_paths`` can return several roots when ``extra_model_paths`` maps
    in other installs, and the first entry is not necessarily the active one --
    on a Desktop install with a migrated second tree it is often the *old* one.
    Prefer whichever candidate lives under this instance's ``models_dir`` so
    downloads land where the user actually expects them.
    """
    paths = folder_paths.get_folder_paths(folder_name)
    if not paths:
        raise ModelDownloadError(
            f"[Skin Detailer] ComfyUI has no '{folder_name}' model folder configured."
        )

    models_dir = os.path.normcase(os.path.abspath(folder_paths.models_dir))
    chosen = paths[0]
    for candidate in paths:
        candidate_abs = os.path.normcase(os.path.abspath(candidate))
        try:
            in_active_models = os.path.commonpath([models_dir, candidate_abs]) == models_dir
        except ValueError:  # Different drives on Windows.
            in_active_models = False
        if in_active_models:
            chosen = candidate
            break

    if chosen != paths[0]:
        log.info(
            "[Skin Detailer] '%s' resolves to %d folders; downloading into the active "
            "install at %s",
            folder_name,
            len(paths),
            chosen,
        )
    os.makedirs(chosen, exist_ok=True)
    return chosen


def _download_one(
    url: str, dest_path: str, label: str, expected_sha256: str | None = None
) -> None:
    """Stream ``url`` to ``dest_path`` atomically, driving a ComfyUI ProgressBar."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=".skin_detailer_", suffix=".part", dir=os.path.dirname(dest_path)
    )
    os.close(tmp_fd)
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type or "application/json" in content_type:
                raise ModelDownloadError(f"[Skin Detailer] {label} URL returned a web page, not weights.")
            total = int(response.headers.get("Content-Length") or 0)
            pbar = comfy.utils.ProgressBar(max(total, 1))
            done = 0
            digest = hashlib.sha256() if expected_sha256 else None
            log.info(
                "[Skin Detailer] downloading %s (%.1f MB) from %s",
                label,
                total / 1e6 if total else 0.0,
                url,
            )
            with open(tmp_path, "wb") as handle:
                while True:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    if digest is not None:
                        digest.update(chunk)
                    if total:
                        pbar.update_absolute(min(done, total), total)
            if done == 0 or (total and done != total):
                raise ModelDownloadError(
                    f"[Skin Detailer] truncated download for {label} "
                    f"({done} of {total} bytes)."
                )
            if digest is not None and digest.hexdigest() != expected_sha256.lower():
                raise ModelDownloadError(f"[Skin Detailer] checksum mismatch for {label}; download discarded.")
        os.replace(tmp_path, dest_path)
        log.info("[Skin Detailer] saved %s -> %s", label, dest_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def ensure_model(
    folder_name: str,
    filename: str,
    urls: list[str],
    label: str,
    manual_hint: str = "",
    expected_sha256: str | None = None,
) -> str:
    """Return a local path for ``filename``, downloading it on first use.

    ``urls`` is tried in order so that mirrors / alternative repository layouts
    can be listed without the caller caring which one exists.
    """
    existing = find_in_folder(folder_name, filename)
    if existing is not None and os.path.exists(existing):
        if expected_sha256:
            digest = hashlib.sha256()
            with open(existing, "rb") as handle:
                while chunk := handle.read(_CHUNK):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    digest.update(chunk)
            if digest.hexdigest() != expected_sha256.lower():
                raise ModelDownloadError(
                    f"[Skin Detailer] '{existing}' does not match the published model checksum.\n"
                    "Move it aside and re-run to download the original weights.\n" + manual_hint
                )
        return existing

    target_dir = default_folder_path(folder_name)
    dest_path = os.path.join(target_dir, filename)

    errors: list[str] = []
    for url in urls:
        try:
            _download_one(url, dest_path, label, expected_sha256)
            return dest_path
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ModelDownloadError) as exc:
            errors.append(f"  {url}\n    -> {type(exc).__name__}: {exc}")
            log.warning("[Skin Detailer] download attempt failed for %s: %s", url, exc)

    detail = "\n".join(errors) if errors else "  (no URLs configured)"
    raise ModelDownloadError(
        f"[Skin Detailer] could not obtain '{filename}' ({label}).\n"
        f"Tried:\n{detail}\n\n"
        f"Fix: download the file manually and place it in:\n  {target_dir}\n"
        + (f"{manual_hint}\n" if manual_hint else "")
        + "Then re-run the node (no restart needed)."
    )
