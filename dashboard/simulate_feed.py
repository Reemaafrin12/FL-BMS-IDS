"""Replay real client_3 rows into the dashboard API as a simulated EVSE feed."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


FEATURE_COLUMNS = ["shunt_voltage", "bus_voltage_V", "current_mA", "power_mW", "State"]
LABEL_COLUMN = "Label"
SAMPLE_PER_CLASS = 15
RANDOM_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "clients" / "client_3.csv")
    parser.add_argument("--api-url", default="http://127.0.0.1:5000/api/readings")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between readings.")
    parser.add_argument("--limit", type=int, help="Optional maximum number of rows to replay.")
    parser.add_argument("--loop", action="store_true", help="Keep replaying the sample, reshuffling it each pass.")
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    rng = random.Random(RANDOM_SEED)
    benign = [row for row in rows if row[LABEL_COLUMN].strip().casefold() == "benign"]
    attacks = [row for row in rows if row[LABEL_COLUMN].strip().casefold() == "attack"]
    benign = rng.sample(benign, min(SAMPLE_PER_CLASS, len(benign)))
    attacks = rng.sample(attacks, min(SAMPLE_PER_CLASS, len(attacks)))
    sample = benign + attacks
    number = 0
    while True:
        replay_rows = sample.copy()
        rng.shuffle(replay_rows)
        if args.limit is not None:
            replay_rows = replay_rows[: max(0, args.limit)]

        for row in replay_rows:
            number += 1
            payload = {key: row[key] for key in FEATURE_COLUMNS}
            request = Request(
                args.api_url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            true_label = row[LABEL_COLUMN]
            try:
                with urlopen(request) as response:
                    result = json.loads(response.read())
                print(
                    f"{number}: true={true_label} prediction={result['prediction']} "
                    f"({result['confidence']:.3f})"
                )
            except URLError as error:
                print(f"{number}: true={true_label} request failed: {error}")
            time.sleep(args.interval)

        if not args.loop:
            break


if __name__ == "__main__":
    main()
