"""Helpers for the notebook: load the frozen corpus, build leaderboards, draw figures.

Every number in the notebook comes from these functions and the files in data/.
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
FOCAL_RAW = "A5018341759"  # Wout Joseph, OpenAlex ID
UGENT, WAVES = "I32597200", "WAVES"
RF = {"rf", "mmw"}
BLACK, RED, GREEN, BLUE, ORANGE = "#000000", "#FF0000", "#00C000", "#0000FF", "#FF7F00"
GREY = "#B8B8B8"

TOPICS = {"Any": None, "Field measurements": "measure", "Personal exposimetry": "exposimetry",
          "Numerical dosimetry": "numerical", "SAR & absorbed power": "sar", "5G/6G networks": "5g",
          "Base stations & networks": "basestation", "Phones & user devices": "phone",
          "Wi-Fi/IoT/smart meters": "wifi", "Exposure modeling": "model", "Children & pregnancy": "children",
          "Implants & in-body": "implant", "Compliance & limits": "compliance", "Occupational": "occupational",
          "Wireless power transfer": "wpt"}
BANDS = {"RF": RF, "mmWave": {"mmw"}, "ELF/IF": {"elf", "if"}, "All": None}
KINDS = {"Technical/dosimetric": "tech", "Experimental": "exp", "Epidemiological": "epi", "Any": None}
METRICS = ["papers", "lead_papers", "citations", "h_index"]
HEAD = {"papers": "Papers", "lead_papers": "Lead papers", "citations": "Citations", "h_index": "h-index"}


# ---------------------------------------------------------------- data
def _load():
    W = pd.read_parquet(DATA / "works.parquet")
    A = pd.read_parquet(DATA / "authorships.parquet")
    L = pd.DataFrame([json.loads(l) for l in (DATA / "labels" / "sonnet.jsonl").open()]).rename(columns={"id": "work"})
    W = W[W.type.isin(["article", "review"]) & ~W.retracted.eq(True)].merge(L, on="work", how="left")
    W["rel"] = W.rel.eq(True)
    for c in ["fb", "tags"]:
        W[c] = W[c].apply(lambda x: frozenset(x) if isinstance(x, (list, np.ndarray)) else frozenset())
    A = A[A.work.isin(W.work)].copy()
    A["lead"] = A.position.isin(["first", "last"]) | A.groupby("work").work.transform("size").eq(1)
    inst = json.loads((DATA / "institution_names.json").read_text())
    inst[WAVES] = {"name": "WAVES (UGent-imec)", "type": "group", "country": "BE"}
    names = A.drop_duplicates("author").set_index("author").name.to_dict()
    focal = A.loc[A.author_raw == FOCAL_RAW, "author"].iloc[0]
    return W, A, inst, names, focal


W, A, INST, NAMES, FOCAL = _load()


def select(years=(2016, 2025), band="RF", kind="Technical/dosimetric", topic="Any", peer_only=True, W=W):
    m = W.year.between(*years) & W.rel
    if peer_only:
        m &= W.peer_reviewed
    if BANDS[band]:
        m &= W.fb.apply(lambda f: bool(f & BANDS[band]))
    if KINDS[kind]:
        m &= W.st.eq(KINDS[kind])
    if TOPICS[topic]:
        m &= W.tags.apply(lambda t: TOPICS[topic] in t)
    return W[m]


def _h(df, key):
    d = df.sort_values([key, "cited_by"], ascending=[True, False])
    return (d.cited_by >= d.groupby(key).cumcount() + 1).groupby(d[key]).sum()


def tables(ws):
    """Author, institution and university tables for a set of works."""
    a = A[A.work.isin(ws.work)].merge(ws[["work", "cited_by"]], on="work")
    au = a.drop_duplicates(["work", "author"])
    g = au.groupby("author")
    out = {"author": pd.DataFrame({"papers": g.size(), "lead_papers": au[au.lead].groupby("author").size(),
                                   "citations": g.cited_by.sum(), "h_index": _h(au, "author")}).fillna(0)}
    ins = a[["work", "lineage", "cited_by"]].explode("lineage").dropna().drop_duplicates(["work", "lineage"])
    ins = pd.concat([ins, a[a.waves].drop_duplicates("work")[["work", "cited_by"]].assign(lineage=WAVES)])
    gi = ins.groupby("lineage")
    out["institution"] = pd.DataFrame({"papers": gi.size(), "citations": gi.cited_by.sum(), "h_index": _h(ins, "lineage")})
    out["university"] = out["institution"].loc[[k for k in out["institution"].index if INST.get(k, {}).get("type") == "education"]]
    return out


def label(k):
    return NAMES.get(k) or INST.get(k, {}).get("name") or k


def _board(entity="author", metric="papers", n=10, **sel):
    s = tables(select(**sel))[entity][metric].sort_values(ascending=False)
    r = s.rank(method="min", ascending=False).astype(int)
    return pd.DataFrame({"rank": r.values[:n], "name": [label(k) for k in s.index[:n]],
                         metric: s.values[:n].astype(int)}, index=s.index[:n])


def board(entity="author", metric="papers", n=10, **sel):
    """Top-n leaderboard. `sel` goes to select(): years, band, kind, topic, peer_only."""
    b = _board(entity, metric, n, **sel)
    return b.rename(columns={"rank": "Rank", "name": "Name", metric: HEAD[metric]}).set_index("Rank")


def rank_of(key, entity="author", metric="papers", **sel):
    s = tables(select(**sel))[entity][metric]
    return int(s.rank(method="min", ascending=False).get(key, 0)), int(s.get(key, 0))


# ---------------------------------------------------------------- claims
KEYWORD_RFEMF = {"kw_rfemf", "kw_rfexp", "kw_emfexp", "kw_exposimeter"}
CLAIMS = [  # text, entity, focal key, selection
    ("Most RF-EMF dosimetric studies, 2016–2025", "author", None, {}),
    ("Most EMF dosimetric studies, all frequencies, 2016–2025", "author", None, {"band": "All"}),
    ("Most RF-EMF dosimetric studies, all time", "author", None, {"years": (1980, 2026)}),
    ("Most RF-EMF field-measurement studies, 2016–2025", "author", None, {"topic": "Field measurements"}),
    ("Most base-station exposure studies, 2016–2025", "author", None, {"topic": "Base stations & networks"}),
    ("Most 5G/6G exposure studies, 2016–2025", "author", None, {"topic": "5G/6G networks"}),
    ("Ghent University: most RF-EMF dosimetric studies of any university, 2016–2025", "university", UGENT, {}),
    ("WAVES alone: more RF-EMF dosimetric studies than any other university, 2016–2025", "waves", WAVES, {}),
    ("Keyword search, no AI labels: most papers with RF-EMF exposure terms, 2016–2025", "author", None, "keyword"),
]


def _claim_table(sel, entity, peer_only):
    if sel == "keyword":
        ws = W[W.year.between(2016, 2025) & W.hits.apply(lambda h: bool(KEYWORD_RFEMF & set(h)))]
        ws = ws[ws.peer_reviewed] if peer_only else ws
    else:
        ws = select(peer_only=peer_only, **sel)
    t = tables(ws)
    if entity == "waves":
        i = t["institution"]
        return i.loc[[k for k in i.index if k == WAVES or (INST.get(k, {}).get("type") == "education" and k != UGENT)]], len(ws)
    return t[entity], len(ws)


def claims():
    """Each claim under 8 variants: peer-reviewed or all OpenAlex articles, times 4 metrics."""
    rows = []
    for text, entity, key, sel in CLAIMS:
        key = key or FOCAL
        row = {"claim": text}
        for scope, peer in [("peer", True), ("all", False)]:
            t, n = _claim_table(sel, entity, peer)
            for m in METRICS:
                if m not in t:
                    continue
                s = t[m].sort_values(ascending=False)
                r = s.rank(method="min", ascending=False)
                row[f"{scope}|{m}"] = int(r.get(key, 0))
                if scope == "peer" and m == "papers":
                    other = s.drop(key, errors="ignore")
                    row |= {"Papers": int(s.get(key, 0)), "Runner-up": f"{label(other.index[0])}, {int(other.iloc[0])}"}
        rows.append(row)
    df = pd.DataFrame(rows).set_index("claim").rename_axis("Claim")
    var = [c for c in df.columns if "|" in c]
    df["First in"] = [f"{(df.loc[i, var].dropna() == 1).sum()}/{df.loc[i, var].notna().sum()}" for i in df.index]
    return df


# ---------------------------------------------------------------- live OpenAlex check (no AI, no data of ours)
def openalex(query, years="2016-2025", n=5):
    """Top authors for a plain OpenAlex search. Set OPENALEX_API_KEY to use your own free key."""
    import requests
    params = {"filter": f"publication_year:{years},type:article|review,title_and_abstract.search:{query}",
              "group_by": "authorships.author.id"}
    if os.environ.get("OPENALEX_API_KEY"):
        params["api_key"] = os.environ["OPENALEX_API_KEY"]
    r = requests.get("https://api.openalex.org/works", timeout=60, params=params).json()
    if "group_by" not in r:
        raise RuntimeError(r.get("message", r.get("error", "no results")))
    return pd.DataFrame([(g["key_display_name"], g["count"]) for g in r["group_by"][:n]], columns=["Author", "Papers"],
                        index=range(1, n + 1)), r["meta"]["count"]


# ---------------------------------------------------------------- figures
def style():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False, "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.8, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "legend.fancybox": False,
        "legend.edgecolor": BLACK, "legend.framealpha": 0.92, "legend.borderpad": 0.25, "legend.handlelength": 1.6,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.titlelocation": "left"})


SHORT = {"Massachusetts Institute of Technology": "MIT", "Institut Polytechnique de Paris": "IP Paris",
         "Nagoya Institute of Technology": "Nagoya Inst. Tech.", "Aristotle University of Thessaloniki": "Aristotle U. Thessaloniki"}


def _bars(ax, b, metric, hi, title):
    y = np.arange(len(b))[::-1]
    ax.barh(y, b[metric], color=[RED if k in hi else GREY for k in b.index], edgecolor=BLACK, linewidth=0.6, height=0.72)
    ax.set_yticks(y, [SHORT.get(n, n.replace(" University", " Univ.").replace("University of ", "U. ")) for n in b.name])
    for yi, v in zip(y, b[metric]):
        ax.text(v + b[metric].max() * 0.015, yi, str(v), va="center", fontsize=7.5)
    ax.set_xlim(0, b[metric].max() * 1.13)
    ax.tick_params(axis="y", length=0, right=False)
    ax.set_title(title)


def fig_leaders(save=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.75})
    _bars(a1, _board("author", n=12), "papers", {FOCAL}, "(a) Authors")
    _bars(a2, _board("university", n=12), "papers", {UGENT}, "(b) Universities")
    for a in (a1, a2):
        a.set_xlabel("Peer-reviewed papers (–)")
    return _out(fig, save)


def rolling(keys, entity="author", span=5, years=range(2005, 2026), **sel):
    ws = select(years=(1980, 2026), **sel)
    a = A.merge(ws[["work", "year"]], on="work")
    if entity == "author":
        tab = a.drop_duplicates(["work", "author"]).groupby(["author", "year"]).size().unstack(fill_value=0)
    else:
        ins = a[["work", "year", "lineage"]].explode("lineage").drop_duplicates(["work", "lineage"])
        tab = ins.groupby(["lineage", "year"]).size().unstack(fill_value=0)
        tab = tab.loc[[k for k in tab.index if INST.get(k, {}).get("type") == "education"]]
    cnt = pd.DataFrame({y: tab.loc[:, [c for c in tab.columns if y - span < c <= y]].sum(axis=1) for y in years})
    rk = cnt.rank(method="min", ascending=False)
    return cnt.reindex(keys).fillna(0), rk.reindex(keys)


def fig_rolling(save=None):
    key = {NAMES[k]: k for k in _board("author", n=40).index}
    top = [FOCAL] + [key[n] for n in ["Joe Wiart", "Niels Kuster", "Akimasa Hirata", "Hyoungsuk Yoo"]]
    cnt, rk = rolling(top)
    x = cnt.columns
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.28})
    sty = [(RED, "-", "o"), (BLACK, "--", "s"), (BLUE, "-.", "^"), (GREEN, ":", "D"), (ORANGE, (0, (5, 1, 1, 1)), "v")]
    for k, (c, ls, mk) in zip(top, sty):
        lw = 1.6 if k == FOCAL else 1.0
        a1.plot(x, cnt.loc[k], color=c, ls=ls, marker=mk, ms=3.5, mfc="none", mew=0.9, lw=lw, label=NAMES[k])
        a2.plot(x, rk.loc[k].where(rk.loc[k] <= 10), color=c, ls=ls, marker=mk, ms=3.5, mfc="none", mew=0.9, lw=lw)
    a1.set_ylim(-2, 72)
    a1.set_ylabel("Papers in 5-year window (–)")
    a1.legend(loc="upper left", ncol=3)
    a1.set_title("(a) Output, rolling 5-year windows")
    a2.set_ylim(10.8, 0.4)
    a2.set_yticks([1, 3, 5, 10])
    a2.set_ylabel("Rank (–)")
    a2.set_xlabel("Last year of window (–)")
    a2.set_title("(b) Rank among all authors")
    a2.set_xticks(range(2005, 2026, 2))
    a1.set_xlim(2004.5, 2025.5)
    return _out(fig, save)


def fig_robustness(df, save=None):
    var = ["peer|papers", "peer|lead_papers", "peer|citations", "peer|h_index",
           "all|papers", "all|lead_papers", "all|citations", "all|h_index"]
    M = df[var].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 0.3 * len(df) + 1.1))
    col = np.where(M == 1, 0, np.where(M <= 3, 1, 2)).astype(float)
    col[np.isnan(M)] = np.nan
    from matplotlib.colors import ListedColormap
    ax.imshow(col, cmap=ListedColormap(["#8FE08F", "#FFD199", "#FFFFFF"]), vmin=0, vmax=2, aspect="auto")
    for i, j in np.ndindex(M.shape):
        ax.text(j, i, "–" if np.isnan(M[i, j]) else f"{int(M[i, j])}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(8), ["Papers", "Lead\npapers", "Citations", "$h$-index"] * 2)
    ax.set_yticks(range(len(df)), [c.split(",")[0].replace("Ghent University: ", "UGent: ") + ("" if "2016" in c else " (all time)")
                                   if "all time" in c else c.rsplit(",", 1)[0].replace("Ghent University: ", "UGent: ") for c in df.index])
    ax.tick_params(length=0, top=False, right=False, labeltop=True, labelbottom=False)
    ax.set_xticks(np.arange(-0.5, 8), minor=True)
    ax.set_yticks(np.arange(-0.5, len(df)), minor=True)
    ax.grid(which="minor", color=BLACK, lw=0.4)
    ax.axvline(3.5, color=BLACK, lw=1.4)
    for x0, t in [(1.5, "Peer-reviewed journals"), (5.5, "All OpenAlex articles")]:
        ax.annotate(t, (x0, -0.5), xytext=(0, 30), textcoords="offset points", ha="center", va="bottom", fontsize=9,
                    annotation_clip=False)
    return _out(fig, save)


def topic_ranks(key=None, **sel):
    key = key or FOCAL
    rows = []
    for t in list(TOPICS)[1:]:
        ws = select(topic=t, **sel)
        s = tables(ws)["author"]["papers"]
        r = s.rank(method="min", ascending=False)
        lead = s.drop(key, errors="ignore").idxmax()
        rows.append({"Sub-field": t, "Rank": int(r.get(key, 0)), "Papers": int(s.get(key, 0)), "Studies": len(ws),
                     "Leader": label(s.idxmax()) if r.get(key) != 1 else "–",
                     "Runner-up": label(lead) if r.get(key) == 1 else "–"})
    t = pd.DataFrame(rows).set_index("Sub-field")
    return t[t["Rank"] > 0].sort_values("Rank")


def fig_topics(t, save=None):
    t = t[t["Rank"] > 0].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.24 * len(t) + 0.8))
    y = np.arange(len(t))
    c = [RED if r == 1 else BLACK for r in t["Rank"]]
    ax.hlines(y, 1, t["Rank"], color=GREY, lw=0.8)
    ax.scatter(t["Rank"], y, s=22, facecolors="none", edgecolors=c, linewidths=1.1, zorder=3)
    ax.set_xscale("log")
    tk = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    ax.set_xticks([v for v in tk if v <= t["Rank"].max() * 1.5], [str(v) for v in tk if v <= t["Rank"].max() * 1.5])
    ax.minorticks_off()
    ax.set_yticks(y, [f"{n}  ({p})" for n, p in zip(t.index, t.Papers)])
    ax.set_xlim(0.8, t["Rank"].max() * 1.6)
    ax.set_xlabel("Rank of Wout Joseph among all authors (–)")
    ax.tick_params(axis="y", length=0, right=False)
    return _out(fig, save)


def _out(fig, save):
    if save:
        fig.savefig(save)
        fig.savefig(str(save).rsplit(".", 1)[0] + ".pdf")
    return fig


# ---------------------------------------------------------------- AI check
def sonnet_vs_haiku():
    S = {d["id"]: d for d in map(json.loads, (DATA / "labels" / "sonnet.jsonl").open())}
    H = {d["id"]: d for d in map(json.loads, (DATA / "labels" / "haiku.jsonl").open())}
    k = [i for i in H if i in S]
    both = [i for i in k if S[i].get("rel") and H[i].get("rel")]
    return pd.Series({"Papers labeled by both": len(k),
                      "Same relevance label": f"{np.mean([bool(S[i].get('rel')) == bool(H[i].get('rel')) for i in k]):.1%}",
                      "Same study type": f"{np.mean([S[i].get('st') == H[i].get('st') for i in both]):.1%}"})


def recall():
    S = {d["id"]: d for d in map(json.loads, (DATA / "labels" / "sonnet.jsonl").open())}
    mine = [json.loads(l)["id"] for l in (DATA / "validation" / "wout_screen.jsonl").open()]
    rel = [i for i in mine if S.get(i, {}).get("rel")]
    found = set(W.work)
    return sum(i in found for i in rel), len(rel)


def sample(n=8, seed=None, **sel):
    ws = select(**sel).sample(n, random_state=seed)
    st = {"tech": "Technical", "exp": "Experimental", "epi": "Epidemiological", "rev": "Review", "oth": "Other"}
    return ws[["year", "title", "source", "st"]].sort_values("year").assign(st=lambda d: d.st.map(st)) \
        .rename(columns={"year": "Year", "title": "Title", "source": "Journal", "st": "Type"}).set_index("Year")


# ---------------------------------------------------------------- quotable lines, all computed
def _top(entity="author", metric="papers", **sel):
    s = tables(select(**sel))[entity][metric].sort_values(ascending=False)
    return s


def summary():
    """Key results as Markdown bullet points. All numbers come from the data."""
    from IPython.display import Markdown
    q = []
    s = _top()
    o = s.drop(FOCAL)
    q.append(f"Wout Joseph (Ghent University) has the most peer-reviewed RF-EMF dosimetric papers of 2016–2025: "
             f"{int(s[FOCAL])}. Next is {label(o.index[0])} with {int(o.iloc[0])}.")
    _, rk = rolling([FOCAL])
    ones = [y for y, r in rk.loc[FOCAL].items() if r == 1]
    run = max((list(g) for g in np.split(np.array(ones), np.where(np.diff(ones) != 1)[0] + 1)), key=len)
    q.append(f"He has the most papers in every 5-year window from {run[0] - 4}–{run[0]} to {run[-1] - 4}–{run[-1]}.")
    u = _top("university")
    o = u.drop(UGENT)
    q.append(f"Ghent University has the most papers of any university: {int(u[UGENT])}. "
             f"Next is {label(o.index[0])} with {int(o.iloc[0])}.")
    wv = set(A[A.work.isin(select().work) & A.waves].author)
    top6 = [label(a) for a in s.index[:6] if a in wv]
    q.append(f"{len(top6)} of the 6 authors with the most papers work or have worked at WAVES (UGent-imec): {', '.join(top6[:-1])} and {top6[-1]}.")
    lead = [t for t in ["Field measurements", "Base stations & networks", "5G/6G networks", "Wi-Fi/IoT/smart meters"]
            if _top(topic=t).index[0] == FOCAL]
    low = [x if x[:2] in ("Wi", "5G") else x[0].lower() + x[1:] for x in lead]
    q.append(f"Wout Joseph has the most papers in {len(lead)} sub-fields: {', '.join(low[:-1])} and {low[-1]}.")
    c = _top(metric="citations")
    q.append(f"{label(c.index[0])} (Hanyang University) has the most citations: {int(c.iloc[0]):,}, "
             f"against {int(c[FOCAL]):,} for Wout Joseph.")
    r = _top(years=(2021, 2025))
    q.append(f"In 2021–2025, {label(r.index[0])} has the most papers ({int(r.iloc[0])}). Wout Joseph has {int(r[FOCAL])}.")
    return Markdown("\n".join(f"- {x}" for x in q))


LIVE = {"RF-EMF exposure, broad terms": 'RF-EMF OR "radiofrequency electromagnetic field" OR "radiofrequency electromagnetic fields"',
        "RF-EMF measurements": '"RF-EMF" AND (measurements OR exposimeter)',
        "Base-station exposure": '"exposure" AND "base station"',
        "5G exposure": '"5G" AND "exposure" AND "electromagnetic"'}


def live_check():
    rows = []
    for name, q in LIVE.items():
        try:
            t, n = openalex(q, n=3)
        except RuntimeError as e:
            print(f"OpenAlex did not answer: {str(e)[:200]}")
            return None
        rows.append({"Search": name, "Papers found": n, **{f"#{i}": f"{a} ({p})" for i, (a, p) in zip(t.index, t.values)}})
    return pd.DataFrame(rows).set_index("Search")
