from __future__ import annotations

import hashlib
import re
import urllib.request
from typing import Callable
from urllib.parse import urljoin


WIKIMEDIA_ROOT = "https://dumps.wikimedia.org/other/cirrus_search_index/"
AOZORA_METADATA_URL = (
    "https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip"
)
AOZORA_SOURCE_ROOT_URL = "https://www.aozora.gr.jp/"
# Kept as an import-compatible alias for callers of the original CLI/API.
AOZORA_REPOSITORY_URL = AOZORA_SOURCE_ROOT_URL
JMDICT_URL = "https://www.edrdg.org/pub/Nihongo/JMdict_e.gz"

DATE_LINK_RE = re.compile(r'href=["\'](\d{8}/)["\']')
SHARD_LINK_RE = re.compile(
    r'href=["\'](jawiki_content-(\d{8})-(\d{5})\.json\.bz2)["\']'
)


def fetch_text(url: str) -> str:
    request = urllib.request.Request(
        url, headers={"User-Agent": "KazumaProject-JapaneseCorpus/0.1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "KazumaProject-JapaneseCorpus/0.1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def fetch_headers(url: str) -> dict[str, str]:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "KazumaProject-JapaneseCorpus/0.1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return {key.lower(): value for key, value in response.headers.items()}


def discover_wikipedia(
    root_url: str = WIKIMEDIA_ROOT,
    fetcher: Callable[[str], str] = fetch_text,
) -> dict[str, object]:
    root_html = fetcher(root_url)
    dates = sorted(set(DATE_LINK_RE.findall(root_html)), reverse=True)
    if not dates:
        raise RuntimeError(f"No dated dumps found under {root_url}")

    for date_link in dates:
        dump_date = date_link.rstrip("/")
        index_url = urljoin(
            urljoin(root_url, date_link), "index_name=jawiki_content/"
        )
        try:
            index_html = fetcher(index_url)
        except Exception:
            continue
        if 'href="_SUCCESS"' not in index_html and "href='_SUCCESS'" not in index_html:
            continue

        shard_matches = SHARD_LINK_RE.findall(index_html)
        shards = [
            {
                "name": name,
                "sequence": sequence,
                "url": urljoin(index_url, name),
            }
            for name, match_date, sequence in shard_matches
            if match_date == dump_date
        ]
        shards.sort(key=lambda shard: str(shard["name"]))
        if shards:
            return {
                "dump_date": dump_date,
                "index_url": index_url,
                "shards": shards,
            }

    raise RuntimeError("No completed Japanese Wikipedia content dump found")


def discover_aozora(
    source_root: str = AOZORA_SOURCE_ROOT_URL,
    metadata_url: str = AOZORA_METADATA_URL,
    fetcher: Callable[[str], bytes] = fetch_bytes,
) -> dict[str, str]:
    """Discover the official source version without relying on a Git mirror.

    The former GitHub repository is no longer publicly reachable.  The official
    metadata archive is served alongside the source files, so its digest is a
    stable and reproducible version identifier for the selected source set.
    """
    metadata_sha256 = hashlib.sha256(fetcher(metadata_url)).hexdigest()
    return {
        "source_url": source_root,
        "source_version": metadata_sha256,
        "metadata_url": metadata_url,
        "metadata_sha256": metadata_sha256,
    }


def discover_jmdict(
    url: str = JMDICT_URL,
    header_fetcher: Callable[[str], dict[str, str]] = fetch_headers,
) -> dict[str, object]:
    headers = header_fetcher(url)
    etag = headers.get("etag", "")
    last_modified = headers.get("last-modified", "")
    content_length = headers.get("content-length", "")
    if not etag and not last_modified:
        raise RuntimeError(f"JMdict source has no ETag or Last-Modified header: {url}")
    try:
        bytes_value = int(content_length) if content_length else 0
    except ValueError as error:
        raise RuntimeError(
            f"JMdict source has invalid Content-Length: {content_length}"
        ) from error
    return {
        "url": url,
        "etag": etag,
        "last_modified": last_modified,
        "bytes": bytes_value,
    }


def discover_sources(
    wikipedia_root: str = WIKIMEDIA_ROOT,
    aozora_source_root: str = AOZORA_SOURCE_ROOT_URL,
    aozora_metadata: str = AOZORA_METADATA_URL,
    jmdict_url: str = JMDICT_URL,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "wikipedia": discover_wikipedia(wikipedia_root),
        "aozora": discover_aozora(aozora_source_root, aozora_metadata),
        "jmdict": discover_jmdict(jmdict_url),
    }
