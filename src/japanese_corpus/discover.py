from __future__ import annotations

import re
import subprocess
import urllib.request
from typing import Callable
from urllib.parse import urljoin


WIKIMEDIA_ROOT = "https://dumps.wikimedia.org/other/cirrus_search_index/"
AOZORA_METADATA_URL = (
    "https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip"
)
AOZORA_REPOSITORY_URL = "https://github.com/aozorabunko/aozorabunko.git"

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


def discover_git_head(repository_url: str = AOZORA_REPOSITORY_URL) -> str:
    result = subprocess.run(
        ["git", "ls-remote", repository_url, "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = result.stdout.strip().split()
    if len(fields) != 2 or fields[1] != "HEAD":
        raise RuntimeError(f"Could not discover HEAD for {repository_url}")
    return fields[0]


def discover_sources(
    wikipedia_root: str = WIKIMEDIA_ROOT,
    aozora_repository: str = AOZORA_REPOSITORY_URL,
    aozora_metadata: str = AOZORA_METADATA_URL,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "wikipedia": discover_wikipedia(wikipedia_root),
        "aozora": {
            "repository_url": aozora_repository,
            "repository_commit": discover_git_head(aozora_repository),
            "metadata_url": aozora_metadata,
        },
    }
