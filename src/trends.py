"""Evolution over time: yearly output, rolling 5-year ranks and the gap to the closest competitors."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from rank import OUT, UGENT, WAVES, FOCAL_AUTHOR_RAW, DATA, load

SLICES = {
    "rf_tech": ("RF technical/dosimetric", lambda W: W.rel & W.st.eq("tech") & W.fb.apply(lambda f: bool(f & {"rf", "mmw"}))),
    "all_tech": ("technical/dosimetric, all frequencies", lambda W: W.rel & W.st.eq("tech")),
    "rf_all": ("RF, all study types", lambda W: W.rel & W.fb.apply(lambda f: bool(f & {"rf", "mmw"}))),
}
Y0, Y1, ROLL = 1995, 2025, 5


def yearly(a, key):
    return a.drop_duplicates(["work", key]).groupby([key, "year"]).size().unstack(fill_value=0)


def rolling_rank(tab, focal):
    out = {}
    for y in range(Y0 + ROLL - 1, Y1 + 1):
        s = tab.loc[:, [c for c in tab.columns if y - ROLL < c <= y]].sum(axis=1)
        r = s.rank(method="min", ascending=False)
        lead = s.drop(focal, errors="ignore")
        out[y] = {"rank": int(r.get(focal, 0)), "value": int(s.get(focal, 0)),
                  "best_other": lead.idxmax(), "best_other_value": int(lead.max())}
    return out


def main():
    W, A = load()
    inst = json.loads((DATA / "institution_names.json").read_text())
    names = A.drop_duplicates("author").set_index("author").name.to_dict()
    focal = A.loc[A.author_raw == FOCAL_AUTHOR_RAW, "author"].iloc[0]
    res = {}
    for key, (label, f) in SLICES.items():
        ws = W[f(W) & W.year.between(Y0, Y1)][["work", "year"]]
        a = A.merge(ws, on="work")
        au = yearly(a, "author")
        win = au.loc[:, [c for c in au.columns if c >= 2016]].sum(axis=1).sort_values(ascending=False)
        top = list(dict.fromkeys([focal] + list(win.head(8).index)))
        ins = a[["work", "year", "lineage"]].explode("lineage")
        ins = pd.concat([ins, a[a.waves][["work", "year"]].assign(lineage=WAVES)])
        it = yearly(ins.rename(columns={"lineage": "inst"}), "inst")
        unis = [i for i in it.index if inst.get(i, {}).get("type") == "education"]
        uw = it.loc[unis, [c for c in it.columns if c >= 2016]].sum(axis=1).sort_values(ascending=False)
        top_u = list(dict.fromkeys([UGENT] + list(uw.head(6).index)))
        grp = it.loc[[i for i in it.index if i == WAVES or inst.get(i, {}).get("type") == "education" and i != UGENT]]
        res[key] = {
            "label": label,
            "years": list(range(Y0, Y1 + 1)),
            "authors": {names[k]: [int(au.loc[k].get(y, 0)) for y in range(Y0, Y1 + 1)] for k in top},
            "universities": {inst[k]["name"]: [int(it.loc[k].get(y, 0)) for y in range(Y0, Y1 + 1)] for k in top_u},
            "waves_series": [int(it.loc[WAVES].get(y, 0)) for y in range(Y0, Y1 + 1)],
            "rolling_rank_author": {y: v | {"best_other": names[v["best_other"]]} for y, v in rolling_rank(au, focal).items()},
            "rolling_rank_ugent_among_universities": {y: v | {"best_other": inst[v["best_other"]]["name"]}
                                                      for y, v in rolling_rank(it.loc[unis], UGENT).items()},
            "rolling_rank_waves_vs_other_universities": {y: v | {"best_other": inst.get(v["best_other"], {}).get("name", v["best_other"])}
                                                         for y, v in rolling_rank(grp, WAVES).items()},
        }
    (OUT / "trends.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for key, r in res.items():
        print("==", r["label"])
        for y in range(2005, Y1 + 1, 2):
            a, u, g = r["rolling_rank_author"][y], r["rolling_rank_ugent_among_universities"][y], r["rolling_rank_waves_vs_other_universities"][y]
            print(f"{y - ROLL + 1}-{y}: Wout #{a['rank']} ({a['value']} vs {a['best_other']} {a['best_other_value']}) | "
                  f"UGent #{u['rank']} among univ ({u['value']} vs {u['best_other']} {u['best_other_value']}) | WAVES #{g['rank']}")


if __name__ == "__main__":
    main()
