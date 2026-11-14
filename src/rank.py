"""Leaderboards over many slices of the corpus, and the rank of the focal entities in each."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "results"
FOCAL_AUTHOR_RAW = "A5018341759"
UGENT, WAVES = "I32597200", "WAVES"
MODEL_PRIORITY = ["sonnet"]
MIN_WORKS = 30

WINDOWS = {"all-time": (1980, 2026), "2006-2025": (2006, 2025), "2016-2025": (2016, 2025), "2021-2025": (2021, 2025)}
BANDS = {"all frequencies": None, "RF (10 MHz-300 GHz)": {"rf", "mmw"}, "mmWave (>6 GHz)": {"mmw"}, "ELF/IF (<10 MHz)": {"elf", "if"}}
TYPES = {"all study types": None, "technical/dosimetric": "tech"}
TAGS = {None: "any topic", "measure": "field measurements", "exposimetry": "personal exposimetry",
        "numerical": "numerical dosimetry", "sar": "SAR & absorbed power", "5g": "5G/6G networks",
        "basestation": "base stations & networks", "phone": "phones & user devices", "wifi": "Wi-Fi/IoT/smart meters",
        "model": "exposure modelling & prediction", "nonhuman": "exposure of insects/animals/plants",
        "children": "children & pregnancy", "lowexp": "exposure-aware network design", "implant": "implants & in-body",
        "compliance": "compliance & limits", "occupational": "occupational", "elf_source": "ELF/IF sources",
        "wpt": "wireless power transfer"}
NATIVE = {
    "OpenAlex topic 'EMF & Biological Effects'": lambda W: W.in_topic,
    "keyword: RF-EMF exposure terms": lambda W: W.hits.apply(lambda h: bool({"kw_rfemf", "kw_rfexp", "kw_emfexp", "kw_exposimeter"} & set(h))),
    "keyword: SAR / absorbed / incident power density": lambda W: W.hits.apply(lambda h: bool({"kw_sar", "kw_apd", "kw_ipd"} & set(h))),
    "keyword: dosimetry + EM/RF": lambda W: W.hits.apply(lambda h: bool({"kw_dos_em", "kw_dos_rf"} & set(h))),
}
# Population in millions, World Bank 2023 (rounded); used only for the per-capita country board.
POP = {"US": 335, "CN": 1410, "JP": 124, "RU": 144, "IT": 59, "DE": 84, "GB": 68, "IR": 89, "FR": 68, "TR": 85,
       "IN": 1430, "KR": 52, "ES": 48, "CA": 40, "AU": 27, "PL": 37, "BR": 216, "CH": 8.8, "NL": 17.9, "BE": 11.8,
       "SE": 10.5, "AT": 9.1, "FI": 5.6, "DK": 5.9, "NO": 5.5, "GR": 10.4, "EG": 113, "SA": 37, "PK": 240,
       "TW": 23.4, "IL": 9.8, "HU": 9.6, "CZ": 10.9, "SK": 5.4, "RS": 6.6, "PT": 10.5, "IE": 5.3, "NZ": 5.2,
       "SG": 5.9, "MY": 34, "ID": 278, "TH": 72, "MX": 128, "AR": 46, "UA": 38, "RO": 19, "BG": 6.4, "HR": 3.9,
       "SI": 2.1, "LT": 2.9, "EE": 1.4, "IQ": 45, "NG": 224, "ZA": 60, "DZ": 46, "TN": 12.5, "MA": 37,
       "VN": 99, "HK": 7.5, "CY": 1.3, "LU": 0.66}


def load(peer_only=True):
    W = pd.read_parquet(DATA / "works.parquet")
    A = pd.read_parquet(DATA / "authorships.parquet")
    labs = {}
    for m in reversed(MODEL_PRIORITY):
        f = DATA / "labels" / f"{m}.jsonl"
        if f.exists():
            for line in f.open():
                d = json.loads(line)
                labs[d["id"]] = d | {"model": m}
    L = pd.DataFrame(labs.values()).rename(columns={"id": "work"})
    W = W[W.type.isin(["article", "review"]) & (W.peer_reviewed | (not peer_only)) & (W.retracted != True)].merge(L, on="work", how="left")
    W["rel"] = W.rel.eq(True)
    for c in ["fb", "tags"]:
        W[c] = W[c].apply(lambda x: frozenset(x) if isinstance(x, (list, np.ndarray)) else frozenset())
    A = A[A.work.isin(W.work)].copy()
    n_auth = A.groupby("work").work.transform("size")
    A["lead"] = A.position.isin(["first", "last"]) | n_auth.eq(1)
    return W, A


def h_index(df, key):
    d = df.sort_values([key, "cited_by"], ascending=[True, False])
    pos = d.groupby(key).cumcount() + 1
    return (d.cited_by >= pos).groupby(d[key]).sum()


def boards(ws, A, inst_meta):
    a = A[A.work.isin(ws.work)].merge(ws[["work", "cited_by"]], on="work")
    res = {}
    au = a.drop_duplicates(["work", "author"])
    g = au.groupby("author")
    res["author"] = pd.DataFrame({"papers": g.size(), "lead_papers": au[au.lead].groupby("author").size(),
                                  "citations": g.cited_by.sum(), "h_index": h_index(au, "author")}).fillna(0)
    inst = a[["work", "lineage", "cited_by"]].explode("lineage").dropna().drop_duplicates(["work", "lineage"])
    wv = a[a.waves].drop_duplicates("work")[["work", "cited_by"]].assign(lineage=WAVES)
    inst = pd.concat([inst, wv])
    gi = inst.groupby("lineage")
    res["institution"] = pd.DataFrame({"papers": gi.size(), "citations": gi.cited_by.sum(), "h_index": h_index(inst, "lineage")})
    edu = [k for k in res["institution"].index if inst_meta.get(k, {}).get("type") == "education"]
    res["university"] = res["institution"].loc[edu]
    co = a[["work", "countries", "cited_by"]].explode("countries").dropna().drop_duplicates(["work", "countries"])
    gc = co.groupby("countries")
    res["country"] = pd.DataFrame({"papers": gc.size(), "citations": gc.cited_by.sum()})
    pc = res["country"][res["country"].index.isin(POP.keys())]
    pc = pc[pc.papers >= 20]
    res["country_per_capita"] = pd.DataFrame({"papers_per_million": pc.papers / np.array([POP[k] for k in pc.index], dtype=float)})
    return res


def slices(W):
    for wn, (lo, hi) in WINDOWS.items():
        inw = W.year.between(lo, hi)
        for dn, f in NATIVE.items():
            yield {"family": "OpenAlex-native (no LLM)", "window": wn, "definition": dn, "band": "-", "type": "-", "topic": "-"}, inw & f(W)
        for bn, bset in BANDS.items():
            mb = inw & W.rel & (W.fb.apply(lambda x: bool(x & bset)) if bset else True)
            for tn, st in TYPES.items():
                mt = mb & (W.st.eq(st) if st else True)
                for tag, tagname in TAGS.items():
                    if tag and not st:
                        continue
                    m = mt & (W.tags.apply(lambda x: tag in x) if tag else True)
                    yield {"family": "LLM-screened", "window": wn, "definition": "screened relevant", "band": bn,
                           "type": tn, "topic": tagname}, m


def main():
    W, A = load()
    OUT.mkdir(exist_ok=True)
    LB = OUT / "leaderboards"
    LB.mkdir(exist_ok=True)
    inst_meta = json.loads((DATA / "institution_names.json").read_text())
    inst_meta[WAVES] = {"name": "WAVES (UGent-imec)", "type": "group", "country": "BE"}
    focal_author = A.loc[A.author_raw == FOCAL_AUTHOR_RAW, "author"].iloc[0]
    names = A.drop_duplicates("author").set_index("author").name.to_dict()
    focal = {"author": [focal_author], "institution": [WAVES, UGENT], "university": [UGENT],
             "country": ["BE"], "country_per_capita": ["BE"]}

    def label(et, k):
        if et == "author":
            return names.get(k, k)
        if et in ("institution", "university"):
            return inst_meta.get(k, {}).get("name", k)
        return k

    rows, slice_rows = [], []
    for i, (meta, m) in enumerate(slices(W)):
        ws = W[m]
        if len(ws) < MIN_WORKS:
            continue
        sid = f"s{i:04d}"
        slice_rows.append({"slice": sid, **meta, "n_works": len(ws)})
        for et, df in boards(ws, A, inst_meta).items():
            for metric in df.columns:
                s = df[metric].sort_values(ascending=False)
                r = s.rank(method="min", ascending=False).astype(int)
                top = s.head(30)
                pd.DataFrame({"rank": r[top.index].values, "id": top.index, "name": [label(et, k) for k in top.index],
                              metric: top.values}).to_csv(LB / f"{sid}__{et}__{metric}.csv", index=False)
                for f in focal[et]:
                    if f not in s.index:
                        continue
                    others = [k for k in s.index[:2] if k != f]
                    rows.append({"slice": sid, **meta, "n_works": len(ws), "entity_type": et, "entity": label(et, f),
                                 "metric": metric, "rank": int(r[f]), "value": round(float(s[f]), 2), "of": len(s),
                                 "top_other": label(et, others[0]), "top_other_value": round(float(s[others[0]]), 2)})
    S = pd.DataFrame(slice_rows)
    R = pd.DataFrame(rows)
    S.to_csv(OUT / "slices.csv", index=False)
    R.to_csv(OUT / "ranks.csv", index=False)
    print(f"{len(S)} slices, {len(R)} focal rankings, {(R['rank'] == 1).sum()} first places, {(R['rank'] <= 3).sum()} top-3")


if __name__ == "__main__":
    main()
