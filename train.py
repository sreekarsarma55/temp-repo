"""Train the Invisible Ink model on every pooled training recording and save it.

    python train.py                 # merge data/raw/*.jsonl -> master file, train, save models/airink_v15.joblib
    python train.py --fast          # quicker, slightly weaker model (good for a first try)
    python train.py --force         # retrain even if the saved model already matches the data
    python train.py --exclude rehearsal/answer_key.json --out models/rehearsal.joblib
"""
import argparse
import collections
import json
import os
import time

import numpy as np

import airink as A


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=A.DEFAULT_MODEL)
    ap.add_argument("--fast", action="store_true", help="1 Stage-1 model per fold, 3 folds, smaller OCR")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--group", choices=["canvas", "record"], default="canvas")
    ap.add_argument("--exclude", help="JSON answer key (from rehearse.py) whose ids must NOT be trained on")
    args = ap.parse_args()

    records = A.sync_training_data()
    if args.exclude:
        with open(args.exclude, encoding="utf-8") as f:
            drop = {row["id"] for row in json.load(f)}
        records = [r for r in records if r["id"] not in drop]
        A.log(f"Excluding {len(drop)} rehearsal recordings -> training on {len(records)}")
    if len(records) < 20:
        raise SystemExit("Need at least 20 labeled recordings. Put the downloaded training .jsonl files in data/raw/.")

    sha = A.data_fingerprint(records)
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    if not args.force and os.path.exists(args.out) and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("data_sha") == sha and meta.get("version") == A.VERSION and meta.get("fast") == args.fast:
            A.log(f"{args.out} already matches this data ({len(records)} recordings, sha {sha}). Use --force to retrain.")
            return

    letters = collections.Counter("".join(r["code"] for r in records))
    lengths = collections.Counter(r["length"] for r in records)
    A.log(f"Lengths: {dict(sorted(lengths.items()))} | rarest letters: "
          + ", ".join(f"{c}={letters.get(c, 0)}" for c in sorted(A.ALPHABET, key=lambda c: letters.get(c, 0))[:5]))
    # sanity check of undo/clear handling: surviving pen-down runs vs ground-truth stroke count
    match, gaps, durs, n_undo = 0, [], [], 0
    for r in records:
        df = A.trace_recording(r)
        _, alive = A.final_ink_mask(df, r)
        match += alive == len(r["strokes"])
        t = df["t"].values
        runs = A._runs((df["pen_down"].values == 1) & (df["after_clear"].values == 1))
        for ut in A.control_times(r)[1]:
            n_undo += 1
            before = [(a, b) for a, b in runs if t[a] < ut]
            if before:
                a, b = before[-1]
                gaps.append(ut - t[b])
                durs.append(t[b] - t[a])
    A.log(f"Undo/clear check: surviving pen-down runs == true stroke count in {match}/{len(records)} recordings")
    if gaps:
        q = lambda v: "/".join(f"{x / 1000:.1f}" for x in np.percentile(v, [10, 50, 90]))
        A.log(f"  {n_undo} undos: seconds from stroke end to undo (p10/p50/p90) {q(gaps)}, "
              f"undone stroke duration {q(durs)}")

    t0 = time.time()
    A.log("Training (this takes a few minutes)...")
    bundle = A.train_bundle(records, fast=args.fast, group_mode=args.group)
    A.save_bundle(bundle, args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    A.log(f"Saved {args.out} ({size_mb:.0f} MB) + {meta_path} in {time.time() - t0:.0f} s")
    A.log("Next: python evaluate.py   (honest held-out accuracy + picks the best decoder)")


if __name__ == "__main__":
    main()
