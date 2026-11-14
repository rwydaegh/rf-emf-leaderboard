"""Merge raw OpenAlex downloads into a deduplicated corpus with disambiguated authors."""
import itertools
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, DATA = ROOT / "data" / "raw", ROOT / "data"
UGENT, IMEC = "I32597200", "I4210114974"
NOT_PEER_REVIEWED = re.compile(r"technical report|tech\. rep|技術研究報告|講演論文集|大会|proceedings|abstracts?\b", re.I)
WAVES_RE = re.compile(r"\bINTEC\b|\bWAVES\b|information technology|informatietechnologie|Technologiepark[- ]?(Zwijnaarde)?,? ?(1[25]|126)\b|iGent", re.I)


def short(x):
    return x.rsplit("/", 1)[-1] if x else None


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", " ", s).split()


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def load():
    works, sources = {}, defaultdict(set)
    for f in sorted(RAW.glob("*.jsonl")):
        tag = "topic" if f.name.startswith("topic_") else f.stem
        for line in f.open():
            w = json.loads(line)
            wid = short(w["id"])
            works.setdefault(wid, w)
            sources[wid].add(tag)
    return works, sources


def disambiguate(works):
    """Union OpenAlex author IDs that share an ORCID, a manual alias, or (surname, given name, institution);
    an initials-only variant joins a full-name cluster only when exactly one full given name fits."""
    uf, names, insts, size = UF(), {}, defaultdict(set), defaultdict(int)
    by_orcid = defaultdict(set)
    for w in works.values():
        for a in w["authorships"]:
            aid = short(a["author"]["id"])
            if not aid:
                continue
            uf.find(aid)
            size[aid] += 1
            names.setdefault(aid, a["author"]["display_name"])
            for o in [a["author"].get("orcid")] + (a["author"].get("observed_orcids") or []):
                if o:
                    by_orcid[o].add(aid)
            for i in a["institutions"]:
                insts[aid].add(short(i["id"]))
    for ids in by_orcid.values():
        ids = sorted(ids)
        for x in ids[1:]:
            uf.union(ids[0], x)
    aliases = json.loads((DATA / "author_aliases.json").read_text())
    for k, ids in aliases.items():
        if not k.startswith("_"):
            for x in ids[1:]:
                uf.union(ids[0], x)
    full, initial_only = defaultdict(list), defaultdict(list)
    for aid, nm in names.items():
        t = norm(nm)
        if len(t) >= 2:
            for inst in insts[aid]:
                if len(t[0]) > 1:
                    full[(t[-1], t[0], inst)].append(aid)
                else:
                    initial_only[(t[-1], t[0], inst)].append(aid)
    for ids in full.values():
        for x in ids[1:]:
            uf.union(ids[0], x)
    by_initial = defaultdict(set)
    for (sur, given, inst) in full:
        by_initial[(sur, given[0], inst)].add(given)
    for (sur, ini, inst), ids in initial_only.items():
        givens = by_initial.get((sur, ini, inst), set())
        if len(givens) == 1:
            target = full[(sur, givens.pop(), inst)][0]
            for x in ids:
                uf.union(target, x)
    coauth = defaultdict(set)
    for w in works.values():
        ids = [short(a["author"]["id"]) for a in w["authorships"] if a["author"]["id"]]
        for x in ids:
            coauth[x].update(ids)
    members, by_name = defaultdict(list), defaultdict(set)
    for aid, nm in names.items():
        members[uf.find(aid)].append(aid)
        t = norm(nm)
        if len(t) >= 2 and len(t[0]) > 1:
            by_name[" ".join(t)].add(uf.find(aid))
    for roots in by_name.values():
        if len(roots) < 2:
            continue
        roots = sorted(roots)
        ca = {r: {uf.find(c) for a in members[r] for c in coauth[a]} - {r} for r in roots}
        for r1, r2 in itertools.combinations(roots, 2):
            if len(ca[r1] & ca[r2]) >= 2:
                uf.union(r1, r2)
    canon = {aid: uf.find(aid) for aid in names}
    best = {}
    for aid, c in canon.items():
        if c not in best or size[aid] > size[best[c]]:
            best[c] = aid
    canon_name = {c: names[a] for c, a in best.items()}
    for k, ids in aliases.items():
        if not k.startswith("_"):
            canon_name[uf.find(ids[0])] = k
    return canon, canon_name


def resolve_missing(inst_names, ids):
    """Parent institutions appear only in lineage lists; look up their names once."""
    import os
    import httpx
    missing = sorted(i for i in ids if i and i not in inst_names)
    key = {"api_key": os.environ["OPENALEX_API_KEY"]} if os.environ.get("OPENALEX_API_KEY") else {}
    for k in range(0, len(missing), 50):
        chunk = missing[k:k + 50]
        r = httpx.get("https://api.openalex.org/institutions", timeout=60,
                      params={"filter": "openalex_id:" + "|".join(chunk), "per_page": 50,
                              "select": "id,display_name,type,country_code"} | key)
        for i in r.json().get("results", []):
            inst_names[short(i["id"])] = {"name": i["display_name"], "type": i["type"], "country": i["country_code"]}


def main():
    works, sources = load()
    canon, canon_name = disambiguate(works)
    wrows, arows, inst_names = [], [], {}
    for wid, w in works.items():
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        waves = False
        for a in w["authorships"]:
            aid = short(a["author"]["id"])
            inst = [short(i["id"]) for i in a["institutions"]]
            for i in a["institutions"]:
                inst_names.setdefault(short(i["id"]), {"name": i["display_name"], "type": i.get("type"), "country": i.get("country_code")})
            lineage = {short(x) for i in a["institutions"] for x in (i.get("lineage") or [])}
            raw = " | ".join(a.get("raw_affiliation_strings") or [])
            is_ugent = UGENT in lineage or UGENT in inst or "ghent univ" in raw.lower() or "universiteit gent" in raw.lower()
            is_waves = (is_ugent or IMEC in inst) and bool(WAVES_RE.search(raw))
            waves |= is_waves
            arows.append({
                "work": wid, "author_raw": aid, "author": canon.get(aid, aid),
                "name": canon_name.get(canon.get(aid), a["author"]["display_name"]),
                "position": a["author_position"], "corresponding": a.get("is_corresponding"),
                "institutions": inst, "lineage": sorted(lineage | set(inst)), "countries": a.get("countries") or [],
                "ugent": is_ugent, "waves": is_waves,
            })
        topics = {short(t["id"]): t["score"] for t in (w.get("topics") or [])}
        wrows.append({
            "work": wid, "doi": w.get("doi"), "title": w.get("title"), "year": w.get("publication_year"),
            "type": w.get("type"), "language": w.get("language"), "source": src.get("display_name"),
            "source_type": src.get("type"), "core": bool(src.get("is_core")),
            "peer_reviewed": (bool(src.get("is_core")) or src.get("display_name") in ("PubMed", "PubMed Central"))
                             and not NOT_PEER_REVIEWED.search(src.get("display_name") or ""), "cited_by": w.get("cited_by_count"), "fwci": w.get("fwci"),
            "retracted": w.get("is_retracted"), "n_authors": len(w["authorships"]),
            "in_topic": "topic" in sources[wid], "t10874_score": topics.get("T10874"),
            "primary_topic": short((w.get("primary_topic") or {}).get("id")),
            "hits": sorted(sources[wid]), "has_abstract": bool(w.get("abstract")), "waves": waves,
        })
    W, A = pd.DataFrame(wrows), pd.DataFrame(arows)
    resolve_missing(inst_names, {x for r in arows for x in r["lineage"]})
    (DATA / "institution_names.json").write_text(json.dumps(inst_names))
    W.to_parquet(DATA / "works.parquet")
    A.to_parquet(DATA / "authorships.parquet")
    with (DATA / "screen_input.jsonl").open("w") as f:
        for wid, w in works.items():
            f.write(json.dumps({"id": wid, "title": w.get("title"), "abstract": w.get("abstract"),
                                "publication_year": w.get("publication_year"),
                                "primary_location": {"source": {"display_name": ((w.get("primary_location") or {}).get("source") or {}).get("display_name")}}}) + "\n")
    print(f"{len(W)} works, {len(A)} authorships, {A.author.nunique()} canonical authors "
          f"(from {A.author_raw.nunique()} OpenAlex IDs); WAVES works: {W.waves.sum()}")


if __name__ == "__main__":
    main()
