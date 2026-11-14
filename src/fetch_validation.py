"""Fetch the complete OpenAlex record of the focal author (kept apart from the corpus, used only to measure recall)."""
import json
import random
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from fetch import SELECT, API, get, slim

WOUT_IDS = ["A5018341759", "A5151596005", "A5160593137", "A5014249593", "A5121054705",
            "A5053083828", "A5006156349", "A5008969605", "A5112307483"]
OUT = Path(__file__).resolve().parent.parent / "data" / "validation" / "wout_all.jsonl"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with httpx.Client() as c, OUT.open("w") as f:
        for lo, hi in [(1990, 2015), (2016, 2026)]:
            page = 1
            while True:
                d = get(c, {"filter": f"authorships.author.id:{'|'.join(WOUT_IDS)},publication_year:{lo}-{hi}",
                            "per_page": 200, "page": page, "select": SELECT})
                if not d["results"]:
                    break
                for w in d["results"]:
                    f.write(json.dumps(slim(w)) + "\n")
                n += len(d["results"])
                page += 1
    with OUT.with_name("wout_screen.jsonl").open("w") as f:
        for line in OUT.open():
            w = json.loads(line)
            if w["type"] in ("article", "review"):
                f.write(json.dumps(w | {"id": w["id"].rsplit("/", 1)[-1]}) + "\n")
    lines = (OUT.parents[1] / "screen_input.jsonl").open().readlines()
    random.seed(2026)
    OUT.with_name("agreement_sample.jsonl").write_text("".join(random.sample(lines, 2000)))
    print(n)


if __name__ == "__main__":
    main()
