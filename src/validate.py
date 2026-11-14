"""Quality checks: recall on the focal author's full record, Sonnet-vs-Haiku agreement, and a sheet for human spot-checks."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "results"


def labels(name):
    f = DATA / "labels" / f"{name}.jsonl"
    return {d["id"]: d for d in map(json.loads, f.open())} if f.exists() else {}


def kappa(a, b):
    a, b = pd.Series(a), pd.Series(b)
    po = (a == b).mean()
    pe = sum(a.eq(c).mean() * b.eq(c).mean() for c in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    S, H = labels("sonnet"), labels("haiku")
    W = pd.read_parquet(DATA / "works.parquet").set_index("work")
    res = {}

    wout = [json.loads(l) for l in (DATA / "validation" / "wout_screen.jsonl").open()]
    relw = [w for w in wout if S.get(w["id"], {}).get("rel")]
    found = [w for w in relw if w["id"] in W.index]
    res["recall_focal_author"] = {"relevant_journal_papers": len(relw), "in_corpus": len(found),
                                  "recall": round(len(found) / len(relw), 3)}

    both = [k for k in H if k in S]
    rel_s = [bool(S[k].get("rel")) for k in both]
    rel_h = [bool(H[k].get("rel")) for k in both]
    br = [k for k in both if S[k].get("rel") and H[k].get("rel")]
    res["sonnet_vs_haiku"] = {
        "n": len(both),
        "relevance_agreement": round(sum(x == y for x, y in zip(rel_s, rel_h)) / len(both), 3),
        "relevance_kappa": round(kappa(rel_s, rel_h), 3),
        "both_relevant": len(br),
        "study_type_agreement": round(sum(S[k].get("st") == H[k].get("st") for k in br) / len(br), 3),
        "study_type_kappa": round(kappa([S[k].get("st") for k in br], [H[k].get("st") for k in br]), 3),
        "rf_band_agreement": round(sum(bool({"rf", "mmw"} & set(S[k].get("fb") or [])) == bool({"rf", "mmw"} & set(H[k].get("fb") or [])) for k in br) / len(br), 3),
    }

    rows = []
    rel_ids = [k for k in S if S[k].get("rel") and k in W.index]
    non_ids = [k for k in S if not S[k].get("rel") and k in W.index]
    sample = pd.Series(rel_ids).sample(100, random_state=1).tolist() + pd.Series(non_ids).sample(50, random_state=1).tolist()
    for k in pd.Series(sample).sample(frac=1, random_state=2):
        w, s = W.loc[k], S[k]
        rows.append({"work": k, "doi": w.doi, "year": w.year, "title": w.title, "llm_relevant": s.get("rel"),
                     "llm_type": s.get("st"), "llm_bands": ",".join(s.get("fb") or []),
                     "human_relevant (y/n)": "", "human_type": "", "note": ""})
    (OUT / "validation").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "validation" / "human_check_sheet.csv", index=False)
    (OUT / "validation" / "quality.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
