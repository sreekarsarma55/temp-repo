"""Make a fake live-session file from your training data, to practise the exam-day routine.

    python rehearse.py --seed 1
      -> rehearsal/air-ink-test-six.jsonl  (6 recordings, labels removed, like the real download)
      -> rehearsal/answer_key.json

Honest rehearsal (the model must not have seen these six):
    python train.py --exclude rehearsal/answer_key.json --out models/rehearsal.joblib
    python predict.py rehearsal/air-ink-test-six.jsonl --model models/rehearsal.joblib --key rehearsal/answer_key.json

Timing drill for the --watch mode: copy rehearsal/air-ink-test-six.jsonl into ~/Downloads while
`python predict.py --watch` is running.
"""
import argparse
import json
import os
import random

import airink as A


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="rehearsal")
    args = ap.parse_args()
    records = A.read_records(A.MASTER_JSONL) if os.path.exists(A.MASTER_JSONL) else A.sync_training_data()
    records = [r for r in records if A.is_labeled(r)]
    pick = random.Random(args.seed).sample(records, min(args.n, len(records)))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "air-ink-test-six.jsonl"), "w", encoding="utf-8") as f:
        for r in pick:
            f.write(json.dumps(A.strip_labels(r), separators=(",", ":")) + "\n")
    with open(os.path.join(args.out, "answer_key.json"), "w", encoding="utf-8") as f:
        json.dump([{"id": r["id"], "code": r["code"]} for r in pick], f, indent=1)
    A.log(f"Wrote {args.out}/air-ink-test-six.jsonl and {args.out}/answer_key.json ({len(pick)} recordings)")


if __name__ == "__main__":
    main()
