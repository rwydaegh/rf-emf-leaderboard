"""Download the candidate corpus from OpenAlex into data/raw/*.jsonl."""
import json
import sys
import time
from pathlib import Path

import httpx

import os

API = "https://api.openalex.org/works"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
SELECT = ",".join([
    "id", "doi", "title", "publication_year", "publication_date", "type", "language",
    "authorships", "primary_location", "cited_by_count", "fwci", "topics", "primary_topic",
    "abstract_inverted_index", "keywords", "is_retracted", "counts_by_year",
])
BASE_FILTER = "type:article|review,publication_year:1980-2026"

KEYWORD_QUERIES = {
    "sar": '"specific absorption rate"',
    "apd": '"absorbed power density" OR "epithelial power density"',
    "ipd": '"incident power density"',
    "exposimeter": "exposimeter OR exposimeters OR exposimetry",
    "rfemf": '"RF-EMF" OR "RF EMF"',
    "rfexp": '"RF exposure" OR "radiofrequency exposure" OR "radio frequency exposure"',
    "emfexp": '"EMF exposure" OR "electromagnetic field exposure" OR "electromagnetic exposure"',
    "induced": '"induced electric field" OR "in situ electric field" OR "internal electric field"',
    "icnirp": 'ICNIRP OR "IEEE C95.1"',
    "basicrestr": '"basic restrictions"',
    "dos_em": "dosimetry AND electromagnetic",
    "dos_rf": "dosimetry AND (radiofrequency OR microwave)",
    "hexp_em": '"human exposure" AND (electromagnetic OR radiofrequency)',
    "ea_em": '"exposure assessment" AND (electromagnetic OR radiofrequency)',
    "bs_exp": '"base station" AND exposure',
    "mmw_exp": '("millimeter wave" OR "millimetre wave") AND exposure',
    "g5_exp": '5G AND exposure AND (electromagnetic OR radiation)',
    "phone_exp": '"mobile phone" AND exposure AND (radiation OR electromagnetic)',
    "wifi_exp": '(WiFi OR "Wi-Fi" OR WLAN) AND exposure AND electromagnetic',
    "elf_exp": '("extremely low frequency" OR "power frequency") AND exposure AND (field OR fields)',
    "mri_sar": "MRI AND SAR",
}


def reconstruct(inv):
    if not inv:
        return ""
    pos = {}
    for word, idx in inv.items():
        for i in idx:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def slim(w):
    w = dict(w)
    w["abstract"] = reconstruct(w.pop("abstract_inverted_index", None))
    return w


def get(client, params):
    if os.environ.get("OPENALEX_API_KEY"):
        params = params | {"api_key": os.environ["OPENALEX_API_KEY"]}
    for attempt in range(6):
        try:
            r = client.get(API, params=params, timeout=90)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            print(f"retry {attempt}: {e}", flush=True)
            time.sleep(3 * (attempt + 1))
    sys.exit("giving up")


def harvest(name, params, client):
    """Page-number paging (no API key needed; capped at 10k results per query, so callers split by year)."""
    out = RAW / f"{name}.jsonl"
    if out.exists():
        return
    tmp = out.with_suffix(".part")
    page, n, total = 1, 0, None
    with tmp.open("w") as f:
        while total is None or n < total:
            d = get(client, params | {"per_page": 200, "page": page, "select": SELECT})
            total = min(d["meta"]["count"], 10000)
            if not d["results"]:
                break
            for w in d["results"]:
                f.write(json.dumps(slim(w)) + "\n")
            n += len(d["results"])
            page += 1
    if d["meta"]["count"] > 10000:
        sys.exit(f"{name}: {d['meta']['count']} results exceed the 10k paging cap")
    tmp.rename(out)
    print(f"{name}: {n}", flush=True)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = {}
    for lo, hi in [(1980, 1989), (1990, 1996), (1997, 1999), (2000, 2005), (2006, 2009)] + [(y, y) for y in range(2010, 2027)]:
        jobs[f"topic_T10874_{lo}_{hi}"] = {"filter": f"topics.id:T10874,type:article|review,publication_year:{lo}-{hi}"}
    for k, q in KEYWORD_QUERIES.items():
        jobs[f"kw_{k}"] = {"filter": BASE_FILTER, "search.title_and_abstract": q}
    with httpx.Client(headers={"User-Agent": "emf-bibliometrics/0.1"}) as client:
        for name, params in jobs.items():
            harvest(name, params, client)
            time.sleep(1)


if __name__ == "__main__":
    main()
