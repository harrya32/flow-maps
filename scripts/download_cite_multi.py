#!/usr/bin/env python3
"""Download and inspect the CITE/Multi data used by Metric Flow Matching.

The release is Mendeley Data dataset ``hhny5ff7yj``, version 1.  Its two
files used by this repository are downloaded into ``metric-flow-matching/data``
by default.  The inspection is backed/read-only: the large expression matrix
is not loaded into memory.
"""

from __future__ import print_function

import argparse
import html
import http.cookiejar
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


DATASET_ID = "hhny5ff7yj"
DATASET_VERSION = 1
DATASET_URL = "https://data.mendeley.com/datasets/hhny5ff7yj/1"
LEGACY_ARCHIVE_URL = (
    "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/"
    "hhny5ff7yj-1.zip"
)
METADATA_URLS = (
    "https://data.mendeley.com/api/datasets/hhny5ff7yj/versions/1",
    "https://data.mendeley.com/api/datasets/hhny5ff7yj/versions/1/files",
    "https://data.mendeley.com/api/datasets/hhny5ff7yj/1",
    "https://data.mendeley.com/api/datasets/hhny5ff7yj/1/files",
    "https://data.mendeley.com/api/datasets/hhny5ff7yj/files?version=1",
    "https://data.mendeley.com/public-api/datasets/hhny5ff7yj/versions/1",
    "https://api.mendeley.com/datasets/hhny5ff7yj/versions/1",
)
TARGET_FILES = (
    "op_cite_inputs_0.h5ad",
    "op_train_multi_targets_0.h5ad",
)
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _format_bytes(value: Optional[int]) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return "{:.1f} {}".format(size, unit)
        size /= 1024.0
    return "{} B".format(value)


def _progress(downloaded: int, total: Optional[int]) -> None:
    if total:
        percent = min(100.0, 100.0 * downloaded / total)
        text = "Downloading: {:6.2f}% ({}/{})".format(
            percent, _format_bytes(downloaded), _format_bytes(total)
        )
    else:
        text = "Downloading: {}".format(_format_bytes(downloaded))
    print("\r" + text, end="", file=sys.stderr, flush=True)


def _opener() -> urllib.request.OpenerDirector:
    cookies = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookies))


def download_file(
    opener: urllib.request.OpenerDirector,
    url: str,
    destination: Path,
    *,
    expect_zip: bool = False,
    expect_hdf5: bool = False,
) -> Path:
    """Download *url* atomically, resuming a partial HTTP download when possible."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        print("Already downloaded: {}".format(destination))
        return destination

    partial = destination.with_name(destination.name + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT, "Referer": DATASET_URL}
    if existing:
        headers["Range"] = "bytes={}-".format(existing)

    request = urllib.request.Request(url, headers=headers)
    try:
        response = opener.open(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and existing and zipfile.is_zipfile(str(partial)):
            os.replace(str(partial), str(destination))
            return destination
        raise RuntimeError("Download failed with HTTP {}: {}".format(exc.code, url))
    except urllib.error.URLError as exc:
        raise RuntimeError("Could not download {}: {}".format(url, exc.reason))

    with response:
        status = getattr(response, "status", None) or response.getcode()
        resumed = bool(existing and status == 206)
        if existing and not resumed:
            print("Server did not accept resume request; restarting download.")
        mode = "ab" if resumed else "wb"
        downloaded = existing if resumed else 0
        raw_length = response.headers.get("Content-Length")
        remaining = int(raw_length) if raw_length is not None else None
        total = downloaded + remaining if remaining is not None else None

        with partial.open(mode) as output:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                _progress(downloaded, total)
    print(file=sys.stderr)

    if expect_zip and not zipfile.is_zipfile(str(partial)):
        raise RuntimeError(
            "The downloaded file is not a ZIP archive. Kept it for diagnosis at {}".format(
                partial
            )
        )
    if expect_hdf5:
        with partial.open("rb") as downloaded_file:
            signature = downloaded_file.read(8)
        if signature != b"\x89HDF\r\n\x1a\n":
            raise RuntimeError(
                "The downloaded file is not HDF5. Kept it for diagnosis at {}".format(
                    partial
                )
            )
    os.replace(str(partial), str(destination))
    return destination


def _walk_dicts(value: Any) -> Iterator[Dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _walk_key_values(value: Any) -> Iterator[Tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, str):
                yield str(key), child
            else:
                yield from _walk_key_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_key_values(child)


def _record_filename(record: Dict[str, Any]) -> Optional[str]:
    filename_keys = ("filename", "file_name", "fileName", "name")
    for key in filename_keys:
        value = record.get(key)
        if isinstance(value, str):
            basename = PurePosixPath(value).name
            if basename in TARGET_FILES:
                return basename
    return None


def _any_record_filename(record: Dict[str, Any]) -> Optional[str]:
    for key in ("filename", "file_name", "fileName", "name"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return PurePosixPath(value).name
    return None


def _record_download_url(record: Dict[str, Any], base_url: str) -> Optional[str]:
    candidates = []
    for key, value in _walk_key_values(record):
        if value.startswith(("https://", "http://", "/")):
            score = 0
            lowered_key = key.lower()
            lowered_value = value.lower()
            if "download" in lowered_key:
                score += 10
            if "file_downloaded" in lowered_value:
                score += 10
            if "public-files" in lowered_value:
                score += 5
            if lowered_value.endswith(".h5ad"):
                score += 3
            if lowered_key in ("url", "href", "contenturl", "content_url"):
                score += 1
            if score:
                candidates.append((score, urllib.parse.urljoin(base_url, value)))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    file_ids = []
    for key, value in _walk_key_values(record):
        if "id" in key.lower() and (
            re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}", value)
            or (key.lower() in ("fileid", "file_id") and len(value) >= 8)
        ):
            file_ids.append(value)
    if file_ids:
        return (
            "https://data.mendeley.com/public-files/datasets/{}/files/{}/"
            "file_downloaded".format(DATASET_ID, file_ids[0])
        )
    return None


def _downloads_from_json(value: Any, base_url: str) -> Dict[str, str]:
    downloads = {}
    for record in _walk_dicts(value):
        filename = _record_filename(record)
        if filename is None:
            continue
        url = _record_download_url(record, base_url)
        if url is not None:
            downloads[filename] = url
    return downloads


def _filenames_from_json(value: Any) -> List[str]:
    filenames = []
    for record in _walk_dicts(value):
        filename = _any_record_filename(record)
        if filename and "." in filename:
            filenames.append(filename)
    return list(dict.fromkeys(filenames))


def _archive_urls_from_json(value: Any, base_url: str) -> List[str]:
    candidates = []
    archive_suffixes = (".zip", ".tar", ".tar.gz", ".tgz")
    for record in _walk_dicts(value):
        filename = (_any_record_filename(record) or "").lower()
        if filename.endswith(archive_suffixes):
            url = _record_download_url(record, base_url)
            if url is not None:
                candidates.append(url)
    for key, value_string in _walk_key_values(value):
        if not value_string.startswith(("https://", "http://", "/")):
            continue
        lowered_key = key.lower()
        lowered_value = value_string.lower()
        if (
            "zipfiles" in lowered_value
            or ".zip" in urllib.parse.urlsplit(lowered_value).path
            or ("download" in lowered_key and "archive" in lowered_key)
        ):
            candidates.append(urllib.parse.urljoin(base_url, value_string))
    return list(dict.fromkeys(candidates))


class _MendeleyPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.scripts: List[str] = []
        self.links: List[Tuple[Dict[str, str], str]] = []
        self._script_parts: Optional[List[str]] = None
        self._anchor_attrs: Optional[Dict[str, str]] = None
        self._anchor_text: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attributes = {key: value or "" for key, value in attrs}
        if tag == "script":
            self._script_parts = []
        elif tag == "a":
            self._anchor_attrs = attributes
            self._anchor_text = []

    def handle_data(self, data: str) -> None:
        if self._script_parts is not None:
            self._script_parts.append(data)
        if self._anchor_attrs is not None:
            self._anchor_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._script_parts is not None:
            self.scripts.append("".join(self._script_parts))
            self._script_parts = None
        elif tag == "a" and self._anchor_attrs is not None:
            self.links.append((self._anchor_attrs, "".join(self._anchor_text).strip()))
            self._anchor_attrs = None
            self._anchor_text = []


def _downloads_from_html(page: str, base_url: str) -> Tuple[Dict[str, str], List[str]]:
    parser = _MendeleyPageParser()
    parser.feed(page)
    downloads: Dict[str, str] = {}
    archive_urls: List[str] = []

    for attributes, text in parser.links:
        href = attributes.get("href", "")
        download_name = attributes.get("download", "")
        for candidate in (download_name, text):
            basename = PurePosixPath(candidate).name
            if basename in TARGET_FILES and href:
                downloads[basename] = urllib.parse.urljoin(base_url, href)
        lowered_href = href.lower()
        if href and (
            "download all" in text.lower()
            or "zipfiles" in lowered_href
            or ".zip" in urllib.parse.urlsplit(lowered_href).path
        ):
            archive_urls.append(urllib.parse.urljoin(base_url, href))

    for script in parser.scripts:
        stripped = script.strip()
        if not stripped or stripped[0] not in "[{":
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        downloads.update(_downloads_from_json(payload, base_url))
        archive_urls.extend(_archive_urls_from_json(payload, base_url))

    # Some frameworks HTML-escape or slash-escape the JSON in an inline script.
    unescaped = html.unescape(page).replace(r"\/", "/")
    for target in TARGET_FILES:
        if target in downloads:
            continue
        position = unescaped.find(target)
        if position < 0:
            continue
        window = unescaped[max(0, position - 2000) : position + 4000]
        matches = re.findall(
            r'https?://[^"\'<> ]+(?:file_downloaded|\.h5ad[^"\'<> ]*)', window
        )
        if matches:
            downloads[target] = matches[0]
    archive_urls.extend(
        re.findall(r'https?://[^"\'<> ]+(?:zipfiles[^"\'<> ]*|\.zip[^"\'<> ]*)', unescaped)
    )
    return downloads, list(dict.fromkeys(archive_urls))


def _read_url(
    opener: urllib.request.OpenerDirector, url: str, accept: str
) -> Tuple[bytes, str]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": accept, "Referer": DATASET_URL},
    )
    with opener.open(request, timeout=60) as response:
        return response.read(), response.geturl()


def discover_downloads(
    opener: urllib.request.OpenerDirector,
) -> Tuple[Dict[str, str], Optional[str]]:
    """Resolve current public download URLs instead of relying on an expired S3 URL."""
    errors = []
    archive_urls: List[str] = []
    observed_filenames: List[str] = []

    try:
        raw_page, final_url = _read_url(opener, DATASET_URL, "text/html")
        page = raw_page.decode("utf-8", errors="replace")
        downloads, page_archives = _downloads_from_html(page, final_url)
        archive_urls.extend(page_archives)
        if all(name in downloads for name in TARGET_FILES):
            return downloads, None
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        errors.append("{}: {}".format(DATASET_URL, exc))

    downloads = {}
    for metadata_url in METADATA_URLS:
        try:
            raw_json, final_url = _read_url(opener, metadata_url, "application/json")
            payload = json.loads(raw_json.decode("utf-8"))
            observed_filenames.extend(_filenames_from_json(payload))
            downloads.update(_downloads_from_json(payload, final_url))
            archive_urls.extend(_archive_urls_from_json(payload, final_url))
            if all(name in downloads for name in TARGET_FILES):
                return downloads, None
        except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            errors.append("{}: {}".format(metadata_url, exc))

    if archive_urls:
        return downloads, archive_urls[0]

    missing = [name for name in TARGET_FILES if name not in downloads]
    detail = " | ".join(errors)
    observed = ", ".join(dict.fromkeys(observed_filenames)) or "none"
    raise RuntimeError(
        "Could not discover public URL(s) for {} from the Mendeley record. {} "
        "File-like names returned by its metadata endpoints: {}. "
        "Open {} in a browser, copy its Download All link, and rerun with "
        "--archive-url URL. The legacy cache URL is {}.".format(
            ", ".join(missing),
            detail,
            observed,
            DATASET_URL,
            LEGACY_ARCHIVE_URL,
        )
    )


def _target_members(
    archive: zipfile.ZipFile, targets: Iterable[str]
) -> Dict[str, zipfile.ZipInfo]:
    wanted = set(targets)
    matches: Dict[str, zipfile.ZipInfo] = {}
    duplicates: Dict[str, List[str]] = {}
    for member in archive.infolist():
        if member.is_dir():
            continue
        basename = PurePosixPath(member.filename).name
        if basename not in wanted:
            continue
        if basename in matches:
            duplicates.setdefault(basename, [matches[basename].filename]).append(
                member.filename
            )
        else:
            matches[basename] = member

    if duplicates:
        details = "; ".join(
            "{}: {}".format(name, paths) for name, paths in sorted(duplicates.items())
        )
        raise RuntimeError("Archive contains ambiguous duplicate filenames: " + details)
    return matches


def extract_datasets(
    archive_path: Path, output_dir: Path, force: bool = False
) -> List[Path]:
    """Extract only the H5AD files consumed by this repository."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(archive_path)) as archive:
        members = _target_members(archive, TARGET_FILES)
        missing = [name for name in TARGET_FILES if name not in members]
        if missing:
            available = sorted(
                PurePosixPath(item.filename).name
                for item in archive.infolist()
                if not item.is_dir()
            )
            preview = ", ".join(available[:20])
            raise RuntimeError(
                "Mendeley archive is missing expected file(s): {}. "
                "First archive entries: {}".format(", ".join(missing), preview)
            )

        extracted: List[Path] = []
        for name in TARGET_FILES:
            destination = output_dir / name
            extracted.append(destination)
            if destination.exists() and not force:
                print("Already present: {}".format(destination))
                continue

            partial = destination.with_name(destination.name + ".part")
            if partial.exists():
                partial.unlink()
            member = members[name]
            print(
                "Extracting {} ({})".format(name, _format_bytes(member.file_size))
            )
            try:
                with archive.open(member) as source, partial.open("wb") as output:
                    shutil.copyfileobj(source, output, length=DOWNLOAD_CHUNK_BYTES)
                os.replace(str(partial), str(destination))
            except Exception:
                if partial.exists():
                    partial.unlink()
                raise
    return extracted


def _cell_type_columns(columns: Sequence[str]) -> List[str]:
    candidates = []
    for column in columns:
        normalized = "".join(character for character in column.lower() if character.isalnum())
        if "celltype" in normalized or "cellannotation" in normalized:
            candidates.append(column)
    return candidates


def inspect_h5ad(path: Path) -> Tuple[bool, List[str]]:
    """Report the trajectory fields and return whether cell types are present."""
    try:
        import anndata as ad
    except ImportError:
        raise RuntimeError(
            "Checking H5AD metadata requires anndata. Activate the repository's "
            "flowmaps environment or install it with: pip install anndata"
        )

    adata = ad.read_h5ad(str(path), backed="r")
    try:
        obs_columns = [str(column) for column in adata.obs.columns]
        obsm_keys = [str(key) for key in adata.obsm.keys()]
        cell_type_columns = _cell_type_columns(obs_columns)

        print("\n{}".format(path))
        print("  cells x features: {} x {}".format(adata.n_obs, adata.n_vars))
        if "X_pca" in adata.obsm:
            print("  X_pca: present, shape={}".format(tuple(adata.obsm["X_pca"].shape)))
        else:
            print("  X_pca: MISSING (available obsm keys: {})".format(obsm_keys))

        if "day" in adata.obs:
            days = adata.obs["day"]
            values = [str(value) for value in days.dropna().unique().tolist()]
            print("  day: present, values={}".format(values))
        else:
            print("  day: MISSING")

        if not cell_type_columns:
            print("  cell types: MISSING")
            print("  obs columns: {}".format(obs_columns))
            return False, obs_columns

        print("  cell types: present in {}".format(cell_type_columns))
        for column in cell_type_columns:
            values = adata.obs[column]
            examples = [
                str(value) for value in values.dropna().unique().tolist()[:12]
            ]
            print(
                "    {}: {} labelled, {} missing, examples={}".format(
                    column,
                    int(values.notna().sum()),
                    int(values.isna().sum()),
                    examples,
                )
            )
        return True, obs_columns
    finally:
        adata.file.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_root() / "metric-flow-matching" / "data",
        help="Destination for the two H5AD files (default: %(default)s)",
    )
    parser.add_argument(
        "--archive-url",
        default=None,
        help=(
            "Use a copied Mendeley Download All URL instead of discovering "
            "the two public file URLs."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Inspect files already in --output-dir without downloading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload/re-extract files that already exist.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded ZIP after successful extraction.",
    )
    parser.add_argument(
        "--require-cell-types",
        action="store_true",
        help="Exit unsuccessfully if either H5AD has no cell-type column.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    paths = [output_dir / name for name in TARGET_FILES]

    if args.check_only:
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise RuntimeError("Cannot check missing file(s): {}".format(", ".join(missing)))
    elif args.force or any(not path.is_file() for path in paths):
        opener = _opener()
        direct_urls: Dict[str, str] = {}
        archive_url = args.archive_url
        if archive_url is None:
            try:
                direct_urls, archive_url = discover_downloads(opener)
            except RuntimeError as discovery_error:
                print(
                    "Mendeley file discovery failed; retrying its legacy archive "
                    "with browser headers.\n  {}".format(discovery_error),
                    file=sys.stderr,
                )
                archive_url = LEGACY_ARCHIVE_URL

        if archive_url:
            cache_dir = _repo_root() / ".cache" / "mendeley"
            archive_path = cache_dir / "{}-{}.zip".format(
                DATASET_ID, DATASET_VERSION
            )
            if args.force and archive_path.exists():
                archive_path.unlink()
            download_file(
                opener, archive_url, archive_path, expect_zip=True
            )
            paths = extract_datasets(archive_path, output_dir, force=args.force)
            if not args.keep_archive:
                archive_path.unlink()
                print("Removed archive after extraction: {}".format(archive_path))
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            for path in paths:
                if path.exists() and not args.force:
                    print("Already present: {}".format(path))
                    continue
                if path.exists():
                    path.unlink()
                print("Downloading {}".format(path.name))
                download_file(
                    opener,
                    direct_urls[path.name],
                    path,
                    expect_hdf5=True,
                )
    else:
        print("Both datasets already exist; skipping download.")

    results = []
    for path in paths:
        has_cell_types, _ = inspect_h5ad(path)
        results.append((path.name, has_cell_types))

    missing_annotations = [name for name, present in results if not present]
    print("\nSummary")
    if missing_annotations:
        print("  Cell-type annotations missing from: {}".format(", ".join(missing_annotations)))
        print(
            "  Join the official NeurIPS Open Problems metadata.csv by cell_id/obs_names."
        )
        return 2 if args.require_cell_types else 0

    print("  Both datasets contain cell-type annotations.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        sys.exit(1)
