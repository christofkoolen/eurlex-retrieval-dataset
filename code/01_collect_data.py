#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect CJEU judgment metadata and text from EUR-Lex.

Edit the IDE configuration near the end of the file and run the complete
script. Credentials are read from EURLEX_USERNAME and EURLEX_PASSWORD.
"""

from types import SimpleNamespace
import math
import os
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import pandas as pd
import requests
from bs4 import BeautifulSoup

from pipeline_common import (
    find_project_root,
    resolve_project_path,
    set_reproducible_seed,
)


# ============================================================
# CONSTANTS
# ============================================================
ENDPOINT = "https://eur-lex.europa.eu/EURLexWebService"
SOAP_HEADERS = {"Content-Type": "application/soap+xml; charset=utf-8"}
UUID_RE = re.compile(r"cellar:([0-9a-fA-F-]{36})")
DEFAULT_BASE_QUERY = (
    "DTS_SUBDOM = EU_CASE_LAW AND FM_CODED = JUDG AND CASE_LAW_SUMMARY = false "
    "AND DD >= 01/01/1900 AND DD <= 31/12/2025"
)
DEFAULT_DATA_COLUMNS = ["cellar_url", "html_source_url", "html_content", "raw_content"]


# ============================================================
# CONFIG
# ============================================================
@dataclass
class SearchConfig:
    username: str
    password: str
    page_size: int = 100
    sleep_between_calls: float = 1.0
    year_split_guard: int = 9500
    base_query: str = DEFAULT_BASE_QUERY
    start_year: int = 1970
    end_year: int = 2025
    dedupe_subset: str = "celex"


@dataclass
class CellarConfig:
    timeout: float = 1.0
    max_tries: int = 3
    base_sleep: float = 0.1
    delay_between_rows: float = 0.1
    language: str = "ENG"
    max_rows: int | None = None
    raw_reset: bool = False
    collapse_double_spaces: bool = True


# ============================================================
# SMALL HELPERS
# ============================================================
def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df


def load_or_resume_dataframe(input_file: str, output_file: str, max_rows: int | None = None) -> pd.DataFrame:
    if os.path.exists(output_file):
        data = pd.read_pickle(output_file)
        print(f"Resuming from {output_file}")
    else:
        data = pd.read_pickle(input_file)
        print(f"Loaded {input_file}")

    if max_rows is not None:
        data = data.head(max_rows).copy()

    return require_columns(data, DEFAULT_DATA_COLUMNS)


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    df.to_pickle(path)


def print_summary(df: pd.DataFrame, label: str) -> None:
    print(f"\n{label}")
    print(f"Rows: {len(df)}")
    for col in ["html_content", "raw_content"]:
        if col in df.columns:
            print(f"{col} non-null: {df[col].notna().sum()}")


# ============================================================
# SOAP SEARCH LAYER
# ============================================================
def get_soap_request(username: str, password: str, page: int, page_size: int, expert_query: str) -> str:
    return f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope" xmlns:sear="http://eur-lex.europa.eu/search">
  <soap:Header>
    <wsse:Security xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd" soap:mustUnderstand="true">
      <wsse:UsernameToken xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0.xsd" wsu:Id="UsernameToken-1">
        <wsse:Username>{username}</wsse:Username>
        <wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordText">{password}</wsse:Password>
      </wsse:UsernameToken>
    </wsse:Security>
  </soap:Header>
  <soap:Body>
    <sear:searchRequest>
      <sear:expertQuery><![CDATA[{expert_query}]]></sear:expertQuery>
      <sear:page>{page}</sear:page>
      <sear:pageSize>{page_size}</sear:pageSize>
      <sear:searchLanguage>en</sear:searchLanguage>
      <sear:showDocumentsAvailableIn>en</sear:showDocumentsAvailableIn>
    </sear:searchRequest>
  </soap:Body>
</soap:Envelope>
"""


def parse_results(xml_string: str) -> list[dict]:
    ns = {
        "s": "http://www.w3.org/2003/05/soap-envelope",
        "search": "http://eur-lex.europa.eu/search",
    }
    root = ET.fromstring(xml_string)
    results: list[dict] = []

    for result in root.findall(".//search:result", ns):
        rank = result.findtext("search:rank", default="", namespaces=ns)
        reference = result.findtext("search:reference", default="", namespaces=ns)

        html_links = [
            dl.text for dl in result.findall("search:document_link", ns)
            if dl.attrib.get("type") == "html" and dl.text
        ]

        title = result.find(".//search:EXPRESSION_TITLE/search:VALUE", ns)
        ecli = result.find(".//search:ECLI/search:VALUE", ns)
        celex = result.find(".//search:ID_CELEX/search:VALUE", ns)
        court = result.find(".//search:WORK_CREATED_BY_AGENT/search:PREFLABEL", ns)
        date = result.find(".//search:WORK_DATE_DOCUMENT/search:VALUE", ns)
        resource_type = result.find(".//search:WORK_HAS_RESOURCE-TYPE/search:PREFLABEL", ns)
        year_decision = result.find(".//search:CONTAINER_CASE-LAW_YEAR_DECISION/search:VALUE", ns)
        case_number = result.find(".//search:CONTAINER_CASE-LAW_NUMBER/search:VALUE", ns)
        case_chamber = result.find(".//search:CONTAINER_CASE-LAW_COURT/search:VALUE", ns)

        languages = sorted({
            lang.text for lang in result.findall(".//search:EXPRESSION_USES_LANGUAGE/search:OP-CODE", ns)
            if lang.text
        })

        manifestation_uris = [
            u.text for u in result.findall(".//search:NOTICE/search:MANIFESTATION/search:URI/search:VALUE", ns)
            if u.text
        ]
        sameas = [
            u.text for u in result.findall(".//search:NOTICE/search:MANIFESTATION/search:SAMEAS/search:URI/search:VALUE", ns)
            if u.text
        ]

        agents: list[str] = []
        court_name = court.text if court is not None else ""
        for agent in result.findall(".//search:WORK_CREATED_BY_AGENT", ns):
            label = agent.find("search:PREFLABEL", ns)
            if label is not None and label.text and label.text != court_name:
                agents.append(label.text)

        results.append({
            "rank": rank,
            "reference": reference,
            "document_link": html_links[0] if html_links else "",
            "title": title.text if title is not None else "",
            "ecli": ecli.text if ecli is not None else "",
            "celex": celex.text if celex is not None else "",
            "court": court_name,
            "date": date.text if date is not None else "",
            "resource_type": resource_type.text if resource_type is not None else "",
            "case_law_year_decision": year_decision.text if year_decision is not None else "",
            "case_number": case_number.text if case_number is not None else "",
            "case_chamber": case_chamber.text if case_chamber is not None else "",
            "languages_available": ", ".join(languages),
            "manifestation_uris": ", ".join(manifestation_uris),
            "sameas_links": ", ".join(sameas),
            "agents": ", ".join(agents),
        })

    return results


def parse_total_hits(xml_string: str) -> int:
    ns = {"search": "http://eur-lex.europa.eu/search"}
    root = ET.fromstring(xml_string)
    node = root.find(".//search:totalhits", namespaces=ns)
    if node is None or not (node.text or "").strip():
        raise RuntimeError("Could not find <totalhits> in response.")
    return int(node.text)


def post_search(config: SearchConfig, expert_query: str, page: int) -> requests.Response:
    soap = get_soap_request(config.username, config.password, page, config.page_size, expert_query)
    if config.sleep_between_calls:
        time.sleep(config.sleep_between_calls)
    response = requests.post(ENDPOINT, data=soap.encode("utf-8"), headers=SOAP_HEADERS)
    response.raise_for_status()
    return response


def fetch_query_all_pages(config: SearchConfig, expert_query: str) -> tuple[list[dict], int]:
    first_resp = post_search(config, expert_query, 1)
    first_xml = first_resp.text

    total_hits = parse_total_hits(first_xml)
    total_pages = math.ceil(total_hits / config.page_size) if total_hits else 0
    records = parse_results(first_xml)

    for page in range(2, total_pages + 1):
        resp = post_search(config, expert_query, page)
        xml = resp.text
        new = parse_results(xml)

        if not new:
            print(f"    • page {page}/{total_pages} returned 0 results -> stopping early (cap/truncation)")
            break

        records.extend(new)
        print(f"    • fetched page {page}/{total_pages} (rows={len(records)})")

    return records, total_hits


def run_search(config: SearchConfig, output_file: str) -> pd.DataFrame:
    all_records: list[dict] = []

    for year in range(config.start_year, config.end_year + 1):
        year_query = f"{config.base_query} AND DD_YEAR = {year}"
        print(f"\nYEAR {year}")
        year_records, year_hits = fetch_query_all_pages(config, year_query)

        if year_hits == 0:
            continue

        if year_hits > config.year_split_guard:
            print(f"  -> year hits={year_hits} (large), re-fetching by month...")
            for month in range(1, 13):
                month_query = f"{config.base_query} AND DD_YEAR = {year} AND DD_MONTH = {month:02d}"
                month_records, month_hits = fetch_query_all_pages(config, month_query)
                if month_hits == 0:
                    continue
                all_records.extend(month_records)
                print(f"  -> added {len(month_records)} rows for {year}-{month:02d} (total={len(all_records)})")
        else:
            all_records.extend(year_records)
            print(f"  -> added {len(year_records)} rows for {year} (total={len(all_records)})")

    df = pd.DataFrame(all_records)

    if config.dedupe_subset in df.columns:
        df = df.drop_duplicates(subset=[config.dedupe_subset])

    save_dataframe(df, output_file)
    print("\nDONE")
    print("Total rows:", len(df))
    print(f"Saved: {output_file}")
    return df


# ============================================================
# CELLAR HTML + RAW LAYER
# ============================================================
def extract_cellar_uuid(document_link: str) -> str | None:
    if not isinstance(document_link, str):
        return None
    match = UUID_RE.search(document_link)
    return match.group(1) if match else None


def canonical_cellar_url(uuid: str) -> str:
    return f"https://publications.europa.eu/resource/cellar/{uuid}"


def force_https(url: str) -> str:
    if isinstance(url, str) and url.startswith("http://"):
        return "https://" + url[len("http://"):]
    return url


def request_with_retries(
    session: requests.Session,
    url: str,
    timeout: float,
    max_tries: int,
    base_sleep: float,
    headers: dict | None = None,
) -> tuple[requests.Response | None, int | None, str | None]:
    last_status: int | None = None

    for attempt in range(1, max_tries + 1):
        try:
            response = session.get(url, timeout=timeout, allow_redirects=True, headers=headers)
            last_status = response.status_code

            if response.status_code in (202, 429, 503):
                time.sleep(base_sleep * attempt)
                continue

            if response.status_code >= 400:
                return None, response.status_code, None

            return response, response.status_code, response.headers.get("Content-Type", "")
        except requests.RequestException:
            time.sleep(base_sleep * attempt)

    return None, last_status, None


def fetch_branch_notice_xml(
    session: requests.Session,
    uuid: str,
    language: str,
    timeout: float,
    max_tries: int,
    base_sleep: float,
) -> str | None:
    url = f"https://publications.europa.eu/resource/cellar/{uuid}?language={language}"
    headers = {
        "Accept": "application/xml;notice=branch",
        "User-Agent": session.headers.get("User-Agent", ""),
    }

    response, status, content_type = request_with_retries(
        session=session,
        url=url,
        timeout=timeout,
        max_tries=max_tries,
        base_sleep=base_sleep,
        headers=headers,
    )
    if not response or status != 200:
        return None

    if content_type and "html" in content_type.lower():
        return None

    text = response.text or ""
    if len(text) < 200:
        return None
    return text


def extract_candidate_urls_from_notice(xml_text: str) -> list[str]:
    urls = sorted(set(re.findall(r"https?://publications\.europa\.eu/resource/[^\s\"'<]+", xml_text)))
    return [force_https(url) for url in urls]


def rank_html_candidate(url: str) -> tuple[int, int]:
    """
    Lower is better.
    Prefer .ENG.xhtml then .ENG.html then .ENG.SPEC.print.
    Avoid DOC streams by penalty.
    """
    ul = url.lower()
    is_eng = ".eng." in ul
    is_doc = "/doc_" in ul
    is_xhtml = ul.endswith(".xhtml")
    is_html = ul.endswith(".html") or ul.endswith(".spec.print")

    if not (is_xhtml or is_html):
        return (999, len(url))

    if is_xhtml:
        preference = 0
    elif ul.endswith(".html"):
        preference = 1
    else:
        preference = 2

    lang_penalty = 0 if is_eng else 10
    doc_penalty = 100 if is_doc else 0
    return (preference + lang_penalty + doc_penalty, len(url))


def looks_like_rdf_xml(text: str) -> bool:
    head = (text or "").lstrip()[:400].lower()
    return head.startswith("<rdf:rdf") or "<rdf:rdf" in head


def download_best_html(
    session: requests.Session,
    candidate_urls: list[str],
    timeout: float,
    max_tries: int,
    base_sleep: float,
) -> tuple[str | None, str | None]:
    candidates = sorted(candidate_urls, key=rank_html_candidate)
    headers = {
        "Accept": "application/xhtml+xml,application/xml;q=0.9,text/html;q=0.8,*/*;q=0.1",
        "User-Agent": session.headers.get("User-Agent", ""),
    }

    for url in candidates[:80]:
        if rank_html_candidate(url)[0] >= 999:
            continue

        response, status, _ = request_with_retries(
            session=session,
            url=url,
            timeout=timeout,
            max_tries=max_tries,
            base_sleep=base_sleep,
            headers=headers,
        )
        if not response or status != 200:
            continue

        markup = response.text or ""
        if not markup.strip():
            continue
        if looks_like_rdf_xml(markup):
            continue

        return markup, url

    return None, None


def extract_raw_from_html(html: str, collapse_double_spaces: bool = True) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=False)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\u202f", " ")
    text = text.replace("\u2007", " ")
    text = text.strip("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    if collapse_double_spaces:
        text = re.sub(r" {2,}", " ", text)

    return text


def run_html(data: pd.DataFrame, output_file: str, config: CellarConfig) -> pd.DataFrame:
    needs = data["html_content"].isna() | (data["html_content"].astype(str).str.strip() == "")
    todo = data.index[needs]

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept-Language": "en",
        })

        total = len(todo)
        print(f"\nHTML RUN: {total} rows\n")

        for i, idx in enumerate(todo, start=1):
            link = data.at[idx, "document_link"] if "document_link" in data.columns else None
            uuid = extract_cellar_uuid(link)
            print(f"[{i}/{total}] {link}", flush=True)

            if not uuid:
                print("    Skipped (no cellar UUID)", flush=True)
                data.at[idx, "cellar_url"] = None
                data.at[idx, "html_source_url"] = None
                data.at[idx, "html_content"] = None
                save_dataframe(data, output_file)
                continue

            data.at[idx, "cellar_url"] = canonical_cellar_url(uuid)

            notice = fetch_branch_notice_xml(
                session=session,
                uuid=uuid,
                language=config.language,
                timeout=config.timeout,
                max_tries=config.max_tries,
                base_sleep=config.base_sleep,
            )
            if not notice:
                print("    Failed (no branch notice)", flush=True)
                data.at[idx, "html_source_url"] = None
                data.at[idx, "html_content"] = None
                save_dataframe(data, output_file)
                continue

            candidates = extract_candidate_urls_from_notice(notice)
            html_markup, html_src = download_best_html(
                session=session,
                candidate_urls=candidates,
                timeout=config.timeout,
                max_tries=config.max_tries,
                base_sleep=config.base_sleep,
            )

            if html_markup:
                data.at[idx, "html_content"] = html_markup
                data.at[idx, "html_source_url"] = html_src
                print(f"    HTML Success (chars={len(html_markup)}) via {html_src}", flush=True)
            else:
                data.at[idx, "html_content"] = None
                data.at[idx, "html_source_url"] = None
                print("    HTML Failed (no usable HTML/XHTML found)", flush=True)

            save_dataframe(data, output_file)
            time.sleep(config.delay_between_rows)

    return data


def run_raw_offline(data: pd.DataFrame, output_file: str, config: CellarConfig) -> pd.DataFrame:
    needs = data["raw_content"].isna() | (data["raw_content"].astype(str).str.strip() == "")
    todo = data.index[needs]

    total = len(todo)
    print(f"\nRAW RUN (offline from html_content): {total} rows\n")

    for i, idx in enumerate(todo, start=1):
        link = data.at[idx, "document_link"] if "document_link" in data.columns else None
        print(f"[{i}/{total}] {link}", flush=True)

        html = data.at[idx, "html_content"]
        if not isinstance(html, str) or not html.strip():
            print("    RAW Skipped (no html_content)", flush=True)
            data.at[idx, "raw_content"] = None
            save_dataframe(data, output_file)
            continue

        if looks_like_rdf_xml(html):
            print("    RAW Skipped (html_content is RDF/XML, not HTML)", flush=True)
            data.at[idx, "raw_content"] = None
            save_dataframe(data, output_file)
            continue

        raw = extract_raw_from_html(html, collapse_double_spaces=config.collapse_double_spaces)
        html_len = len(html)
        raw_len = len(raw)
        ratio = (raw_len / html_len) if html_len else 0.0

        data.at[idx, "raw_content"] = raw
        print(f"    RAW Success (chars={raw_len}) | HTML chars={html_len} | ratio={ratio:.2f}", flush=True)

        save_dataframe(data, output_file)
        time.sleep(min(config.delay_between_rows, 0.01))

    return data


# ============================================================
# COMMAND HANDLERS
# ============================================================
def get_credentials(username: str | None, password: str | None) -> tuple[str, str]:
    user = username or os.getenv("EURLEX_USERNAME", "")
    pw = password or os.getenv("EURLEX_PASSWORD", "")
    if not user or not pw:
        raise EnvironmentError(
            "Missing EUR-Lex credentials. Set EURLEX_USERNAME and "
            "EURLEX_PASSWORD in the environment."
        )
    return user, pw


def command_search(args: SimpleNamespace) -> int:
    username, password = get_credentials(args.username, args.password)
    config = SearchConfig(
        username=username,
        password=password,
        page_size=args.page_size,
        sleep_between_calls=args.sleep_between_calls,
        year_split_guard=args.year_split_guard,
        base_query=args.base_query,
        start_year=args.start_year,
        end_year=args.end_year,
        dedupe_subset=args.dedupe_subset,
    )
    run_search(config, args.output)
    return 0


def command_cellar_html(args: SimpleNamespace) -> int:
    config = CellarConfig(
        timeout=args.timeout,
        max_tries=args.max_tries,
        base_sleep=args.base_sleep,
        delay_between_rows=args.delay_between_rows,
        language=args.language,
        max_rows=args.max_rows,
        collapse_double_spaces=not args.keep_double_spaces,
    )
    data = load_or_resume_dataframe(args.input, args.output, config.max_rows)
    data = run_html(data, args.output, config)
    save_dataframe(data, args.output)
    print_summary(data, "DONE (HTML)")
    print(f"Saved to: {args.output}")
    return 0


def command_raw_clean(args: SimpleNamespace) -> int:
    config = CellarConfig(
        timeout=args.timeout,
        max_tries=args.max_tries,
        base_sleep=args.base_sleep,
        delay_between_rows=args.delay_between_rows,
        language=args.language,
        max_rows=args.max_rows,
        raw_reset=args.raw_reset,
        collapse_double_spaces=not args.keep_double_spaces,
    )
    data = load_or_resume_dataframe(args.input, args.output, config.max_rows)
    if config.raw_reset:
        data["raw_content"] = None
    data = run_raw_offline(data, args.output, config)
    save_dataframe(data, args.output)
    print_summary(data, "DONE (RAW only)")
    print(f"Saved to: {args.output}")
    return 0


def command_pipeline(args: SimpleNamespace) -> int:
    username, password = get_credentials(args.username, args.password)

    search_config = SearchConfig(
        username=username,
        password=password,
        page_size=args.page_size,
        sleep_between_calls=args.sleep_between_calls,
        year_split_guard=args.year_split_guard,
        base_query=args.base_query,
        start_year=args.start_year,
        end_year=args.end_year,
        dedupe_subset=args.dedupe_subset,
    )
    cellar_config = CellarConfig(
        timeout=args.timeout,
        max_tries=args.max_tries,
        base_sleep=args.base_sleep,
        delay_between_rows=args.delay_between_rows,
        language=args.language,
        max_rows=args.max_rows,
        collapse_double_spaces=not args.keep_double_spaces,
    )

    metadata_df = run_search(search_config, args.metadata_output)
    metadata_df = require_columns(metadata_df, DEFAULT_DATA_COLUMNS)
    save_dataframe(metadata_df, args.metadata_output)

    html_df = load_or_resume_dataframe(args.metadata_output, args.html_output, cellar_config.max_rows)
    html_df = run_html(html_df, args.html_output, cellar_config)
    save_dataframe(html_df, args.html_output)

    raw_df = load_or_resume_dataframe(args.html_output, args.raw_output, cellar_config.max_rows)
    raw_df = run_raw_offline(raw_df, args.raw_output, cellar_config)
    save_dataframe(raw_df, args.raw_output)

    print_summary(raw_df, "PIPELINE COMPLETE")
    print(f"Metadata output: {args.metadata_output}")
    print(f"HTML output: {args.html_output}")
    print(f"RAW output: {args.raw_output}")
    return 0


# ============================================================
# IDE execution
# ============================================================

# Bash commands

# =============================================================================
# IDE configuration
# =============================================================================

PROJECT_ROOT_OVERRIDE: Path | None = None
RUN_MODE = "pipeline"  # search, cellar_html, raw_clean, or pipeline

START_YEAR = 1954
END_YEAR = 2025
PAGE_SIZE = 100
SLEEP_BETWEEN_CALLS = 1.0
YEAR_SPLIT_GUARD = 9500
BASE_QUERY = DEFAULT_BASE_QUERY
DEDUPE_SUBSET = "celex"

REQUEST_TIMEOUT_SECONDS = 1.0
MAX_REQUEST_ATTEMPTS = 3
BASE_RETRY_SLEEP_SECONDS = 0.1
DELAY_BETWEEN_ROWS_SECONDS = 0.1
LANGUAGE = "ENG"
MAX_ROWS: int | None = None
KEEP_DOUBLE_SPACES = False
RAW_RESET = False

METADATA_OUTPUT = Path("data/raw/eurlex_metadata.pkl")
HTML_OUTPUT = Path("data/raw/eurlex_html.pkl")
RAW_OUTPUT = Path("data/raw/eurlex_raw.pkl")


def build_ide_arguments(project_root: Path) -> SimpleNamespace:
    metadata_output = resolve_project_path(project_root, METADATA_OUTPUT)
    html_output = resolve_project_path(project_root, HTML_OUTPUT)
    raw_output = resolve_project_path(project_root, RAW_OUTPUT)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    return SimpleNamespace(
        username=None,
        password=None,
        start_year=START_YEAR,
        end_year=END_YEAR,
        page_size=PAGE_SIZE,
        sleep_between_calls=SLEEP_BETWEEN_CALLS,
        year_split_guard=YEAR_SPLIT_GUARD,
        base_query=BASE_QUERY,
        dedupe_subset=DEDUPE_SUBSET,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_tries=MAX_REQUEST_ATTEMPTS,
        base_sleep=BASE_RETRY_SLEEP_SECONDS,
        delay_between_rows=DELAY_BETWEEN_ROWS_SECONDS,
        language=LANGUAGE,
        max_rows=MAX_ROWS,
        keep_double_spaces=KEEP_DOUBLE_SPACES,
        raw_reset=RAW_RESET,
        metadata_output=str(metadata_output),
        html_output=str(html_output),
        raw_output=str(raw_output),
        input=str(metadata_output),
        output=str(metadata_output),
    )


def main() -> int:
    set_reproducible_seed(123)
    project_root = find_project_root(
        PROJECT_ROOT_OVERRIDE,
        script_file=globals().get("__file__"),
    )
    args = build_ide_arguments(project_root)

    mode = RUN_MODE.strip().lower()
    try:
        if mode == "search":
            args.output = args.metadata_output
            return command_search(args)
        if mode == "cellar_html":
            args.input = args.metadata_output
            args.output = args.html_output
            return command_cellar_html(args)
        if mode == "raw_clean":
            args.input = args.html_output
            args.output = args.raw_output
            args.delay_between_rows = 0.0
            return command_raw_clean(args)
        if mode == "pipeline":
            return command_pipeline(args)
        raise ValueError(
            "RUN_MODE must be one of: search, cellar_html, raw_clean, pipeline."
        )
    except (EnvironmentError, FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
