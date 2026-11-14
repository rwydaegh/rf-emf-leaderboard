"""The picked claims, each re-tested under alternative choices so a reader can see how robust it is."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rank import DATA, FOCAL_AUTHOR_RAW, NATIVE, OUT, POP, UGENT, WAVES, boards, load

RF = {"rf", "mmw"}
CLAIMS = [
    # id, entity type, focal, statement, window, selector
    ("rf_tech_10y", "author", "focal", "Most peer-reviewed technical/dosimetric RF-EMF studies, 2016-2025", (2016, 2025),
     lambda W: W.rel & W.st.eq("tech") & W.fb.apply(lambda f: bool(f & RF))),
    ("all_tech_10y", "author", "focal", "Most peer-reviewed technical/dosimetric EMF studies (all frequencies), 2016-2025", (2016, 2025),
     lambda W: W.rel & W.st.eq("tech")),
    ("rf_tech_all", "author", "focal", "Most peer-reviewed technical/dosimetric RF-EMF studies, all time", (1980, 2026),
     lambda W: W.rel & W.st.eq("tech") & W.fb.apply(lambda f: bool(f & RF))),
    ("rf_any_10y", "author", "focal", "Most peer-reviewed RF-EMF studies of any type (incl. biology, epidemiology), 2016-2025", (2016, 2025),
     lambda W: W.rel & W.fb.apply(lambda f: bool(f & RF))),
    ("measure_10y", "author", "focal", "Most peer-reviewed RF-EMF field-measurement studies, 2016-2025", (2016, 2025),
     lambda W: W.rel & W.st.eq("tech") & W.fb.apply(lambda f: bool(f & RF)) & W.tags.apply(lambda t: "measure" in t)),
    ("5g_10y", "author", "focal", "Most peer-reviewed 5G/6G exposure studies, 2016-2025", (2016, 2025),
     lambda W: W.rel & W.st.eq("tech") & W.fb.apply(lambda f: bool(f & RF)) & W.tags.apply(lambda t: "5g" in t)),
    ("bs_10y", "author", "focal", "Most peer-reviewed base-station / network exposure studies, 2016-2025", (2016, 2025),
     lambda W: W.rel & W.st.eq("tech") & W.fb.apply(lambda f: bool(f & RF)) & W.tags.apply(lambda t: "basestation" in t)),
    ("ugent_rf_tech_10y", "university", UGENT, "Ghent University: most RF-EMF dosimetric studies of any university, 2016-2025", (2016, 2025),
     lambda W: W.rel & W.st.eq("tech") & W.fb.apply(lambda f: bool(f & RF))),
    ("ugent_rf_tech_all", "university", UGENT, "Ghent University: most RF-EMF dosimetric studies of any university, all time", (1980, 2026),
     lambda W: W.rel & W.st.eq("tech") & W.fb.apply(lambda f: bool(f & RF))),
    ("waves_vs_univ_10y", "waves_vs_univ", WAVES, "WAVES alone out-publishes every other university on RF-EMF dosimetry, 2016-2025", (2016, 2025),
     lambda W: W.rel & W.st.eq("tech") & W.fb.apply(lambda f: bool(f & RF))),
    ("be_pc_rf_tech_10y", "country_per_capita", "BE", "Belgium: RF-EMF dosimetric studies per million inhabitants, 2016-2025", (2016, 2025),
     lambda W: W.rel & W.st.eq("tech") & W.fb.apply(lambda f: bool(f & RF))),
    ("native_rfemf_10y", "author", "focal", "No-LLM check: most papers using RF-EMF exposure terms, 2016-2025", (2016, 2025),
     lambda W: NATIVE["keyword: RF-EMF exposure terms"](W)),
]


def table(bd, et, inst_meta):
    if et == "waves_vs_univ":
        t = bd["institution"]
        keep = [k for k in t.index if k == WAVES or (inst_meta.get(k, {}).get("type") == "education" and k != UGENT)]
        return t.loc[keep]
    return bd[et]


def main():
    inst_meta = json.loads((DATA / "institution_names.json").read_text())
    variants = {"peer-reviewed journals": load(True), "all OpenAlex articles": load(False)}
    out = []
    for cid, et, focal, text, (lo, hi), sel in CLAIMS:
        rec = {"id": cid, "claim": text, "entity_type": et, "variants": {}}
        for vn, (W, A) in variants.items():
            f = A.loc[A.author_raw == FOCAL_AUTHOR_RAW, "author"].iloc[0] if focal == "focal" else focal
            ws = W[sel(W) & W.year.between(lo, hi)]
            names = A.drop_duplicates("author").set_index("author").name.to_dict()
            t = table(boards(ws, A, inst_meta), et, inst_meta)
            for metric in t.columns:
                s = t[metric].sort_values(ascending=False)
                r = s.rank(method="min", ascending=False)
                top = [{"name": names.get(k) or inst_meta.get(k, {}).get("name") or ("WAVES (UGent-imec)" if k == WAVES else k),
                        "value": round(float(v), 1), "focal": k == f} for k, v in s.head(10).items()]
                rec["variants"][f"{vn} | {metric}"] = {"n_works": len(ws), "rank": int(r.get(f, 0)), "value": round(float(s.get(f, 0)), 1), "top": top}
        main_v = rec["variants"].get("peer-reviewed journals | papers") or rec["variants"].get("peer-reviewed journals | papers_per_million")
        rec["headline_rank"] = main_v["rank"]
        rec["robust_top1"] = sum(v["rank"] == 1 for v in rec["variants"].values())
        rec["robust_top3"] = sum(0 < v["rank"] <= 3 for v in rec["variants"].values())
        rec["n_variants"] = len(rec["variants"])
        out.append(rec)
        print(f"{text[:85]:85s} #{rec['headline_rank']}  (#1 in {rec['robust_top1']}/{rec['n_variants']}, top-3 in {rec['robust_top3']}/{rec['n_variants']})")
    (OUT / "headlines.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
