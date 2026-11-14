"""LLM screening of works via the headless `claude` CLI. Results are cached per work in data/labels/*.jsonl."""
import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "data" / "labels"

SYSTEM = """You screen scientific papers for a bibliometric study of non-ionising electromagnetic field (EMF) research, using the same scope as the EMF-Portal literature database (RWTH Aachen): static fields up to 300 GHz, and their exposure, dosimetry and biological or health effects.

For each paper return one JSON object with these keys:
- "i": the item number.
- "rel": true if the paper is about human, animal, plant or cell EXPOSURE to non-ionising EMF, its dosimetry, its measurement or modelling for exposure purposes, compliance with exposure limits, or biological/health effects of such fields. false for: ionising radiation, optics/lasers/UV, pure antenna or telecom design with no exposure analysis, MRI imaging physics with no RF safety/SAR analysis, therapeutic uses where EMF exposure is not studied (e.g. pure clinical TMS outcomes), materials science, and anything else.
- "st": study type, one of
  "tech" = technical/dosimetric: exposure assessment, field measurements (in situ, personal exposimetry), numerical or experimental dosimetry (SAR, absorbed/incident power density, induced fields, temperature rise), exposure modelling/prediction, compliance assessment, exposure-aware network design, phantoms, exposure systems design;
  "exp" = experimental biological/medical study (in vitro, in vivo, human provocation);
  "epi" = epidemiological or population survey of health outcomes;
  "rev" = review, meta-analysis, or position paper;
  "oth" = other relevant (risk perception, policy, commentary, mixed).
  Use "tech" also when a biological study's main contribution is the dosimetry/exposure set-up.
- "fb": list of frequency bands addressed, from "static" (0 Hz), "elf" (>0 to 300 Hz), "if" (300 Hz to 10 MHz), "rf" (10 MHz to 6 GHz), "mmw" (above 6 GHz incl. THz up to 300 GHz). Use [] if unclear.
- "tags": list of applicable tags from:
  "measure" (in situ / spot / drive-test field measurements),
  "exposimetry" (personal exposimeters, wearable measurement, personal exposure),
  "numerical" (computational dosimetry: FDTD, FEM, anatomical models),
  "sar" (SAR / absorbed or incident power density / temperature rise as an outcome),
  "5g" (5G, 6G, mmWave networks, massive MIMO, beamforming),
  "basestation" (base stations, networks, broadcast transmitters),
  "phone" (mobile phones, user devices, wearables),
  "wifi" (WLAN, Wi-Fi, IoT, Bluetooth, smart meters),
  "children" (children, foetus, pregnant women),
  "occupational" (workers, occupational exposure),
  "implant" (medical implants, in-body devices, implant safety, MRI RF safety),
  "wpt" (wireless power transfer, inductive charging, EV),
  "nonhuman" (insects, animals, plants, environment as exposed subjects),
  "model" (exposure prediction models, maps, machine learning, statistical exposure models),
  "compliance" (exposure limits, standards, guidelines, compliance distance),
  "lowexp" (exposure-aware / exposure-reducing network design, EMF-aware optimisation),
  "elf_source" (power lines, transformers, household appliances, ELF/IF sources).

Answer with ONLY a JSON array of objects, no prose, no code fences."""


def item_text(w, i):
    abstract = (w.get("abstract") or "")[:1100]
    src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
    return f"[{i}] {w.get('title') or ''} ({w.get('publication_year')}; {src})\n{abstract}"


def call(batch, model):
    prompt = "\n\n".join(item_text(w, i) for i, w in enumerate(batch))
    cmd = ["claude", "-p", "--model", model, "--system-prompt", SYSTEM, "--tools", "",
           "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence",
           "--output-format", "json"]
    for attempt in range(3):
        try:
            r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600, cwd="/tmp")
            out = json.loads(r.stdout)
            txt = out["result"].strip()
            if txt.startswith("```"):
                txt = txt.strip("`").split("\n", 1)[1]
            labs = json.loads(txt[txt.index("["): txt.rindex("]") + 1])
            byi = {int(x["i"]): x for x in labs}
            if len(byi) < len(batch) * 0.9:
                raise ValueError(f"only {len(byi)} of {len(batch)} labels")
            return [(w["id"], byi.get(i)) for i, w in enumerate(batch)], out.get("total_cost_usd", 0)
        except Exception as e:
            print(f"retry {attempt}: {e!r} {r.stderr[:200] if 'r' in locals() else ''}", file=sys.stderr, flush=True)
    return [], 0


def load_done(path):
    done = {}
    if path.exists():
        for line in path.open():
            d = json.loads(line)
            done[d["id"]] = d
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("works", help="jsonl of works to label")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    LABELS.mkdir(parents=True, exist_ok=True)
    out = Path(a.out) if a.out else LABELS / f"{a.model}.jsonl"
    done = load_done(out)
    todo = [json.loads(l) for l in open(a.works)]
    todo = [w for w in todo if w["id"] not in done]
    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    print(f"{len(done)} cached, {len(todo)} to label in {len(batches)} batches", flush=True)
    lock, cost, n = threading.Lock(), 0.0, 0
    with out.open("a") as f, cf.ThreadPoolExecutor(a.workers) as pool:
        for res, c in pool.map(lambda b: call(b, a.model), batches):
            with lock:
                cost += c
                for wid, lab in res:
                    if lab is not None:
                        lab = {k: v for k, v in lab.items() if k != "i"}
                        f.write(json.dumps({"id": wid, **lab}) + "\n")
                        n += 1
                f.flush()
                if n and n % 500 < a.batch:
                    print(f"labelled {n}, cost ${cost:.2f}", flush=True)
    print(f"done: {n} new labels, cost ${cost:.2f}", flush=True)


if __name__ == "__main__":
    main()
