"""Honest accuracy estimate: end-to-end cross-validation on the pooled training data.

For each fold, the WHOLE pipeline (ink model, Stage 2, threshold, OCR) is trained only on the other
folds, then the held-out recordings are decoded exactly as in the exam. Every decoder in
airink.DECODERS is scored on the same held-out predictions, and the best one is written to
models/best_decoder.json (predict.py uses it by default).

    python evaluate.py                 # 5 folds, fast training per fold (~10-30 min on a laptop)
    python evaluate.py --folds 3       # quicker
    python evaluate.py --full          # full-size models per fold (slower, slightly higher scores)
    python evaluate.py --group record  # ignore the same-device grouping (optimistic)
"""
import argparse
import collections
import csv
import json
import os
import pickle
import time

import numpy as np
from sklearn.model_selection import GroupKFold

import airink as A


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--full", action="store_true", help="train full-size models in every fold")
    ap.add_argument("--group", choices=["canvas", "record"], default="canvas")
    ap.add_argument("--decoders", default=",".join(A.DECODERS), help="comma-separated names")
    ap.add_argument("--sheet", type=int, default=12, help="how many worst held-out rows to draw")
    ap.add_argument("--out", default="outputs/cv")
    ap.add_argument("--max-folds", type=int, default=0, help="stop after this fold (resume later)")
    args = ap.parse_args()
    names = [n.strip() for n in args.decoders.split(",") if n.strip()]
    for n in names:
        if n not in A.DECODERS:
            raise SystemExit(f"unknown decoder {n}; choose from {', '.join(A.DECODERS)}")

    t0 = time.time()
    records = A.sync_training_data()
    A.log("Building features once for all recordings...")
    feats = {r["id"]: A.build_features(r) for r in records}
    gmap = A.make_groups(records, args.group, verbose=True)
    groups = [gmap[r["id"]] for r in records]
    n_folds = max(2, min(args.folds, len(set(groups))))
    splits = list(GroupKFold(n_splits=n_folds).split(records, groups=groups))

    rows, keep = [], []
    os.makedirs(args.out, exist_ok=True)
    tag = f"{A.data_fingerprint(records)}-{args.group}-{n_folds}-{'full' if args.full else 'fast'}-{'_'.join(names)}"
    for fold, (tr, te) in enumerate(splits, 1):
        ckpt = os.path.join(args.out, f"fold{fold}.pkl")
        if os.path.exists(ckpt):  # resume: folds are saved as they finish
            with open(ckpt, "rb") as f:
                saved = pickle.load(f)
            if saved.get("tag") == tag:
                rows += saved["rows"]
                keep += saved["keep"]
                A.log(f"\nFold {fold}/{n_folds}: loaded from {ckpt}")
                continue
        A.log(f"\nFold {fold}/{n_folds}: train {len(tr)}, held out {len(te)}")
        fold_rows, fold_keep = [], []
        bundle = A.train_bundle([records[i] for i in tr], fast=not args.full, group_mode=args.group, feats=feats)
        for i in te:
            r = records[i]
            ctx = A.make_context(r, bundle, df_feat=feats[r["id"]])
            row = {"fold": fold, "id": r["id"], "length": r["length"], "code": r["code"]}
            for n in names:
                res = A.run_decoder(ctx, bundle, n)
                row[n] = res["code"]
                row[n + ":conf"] = round(res["conf"], 3)
                row[n + ":top3"] = "|".join("".join(c for c, _ in l["top"]) for l in res["letters"])
                row[n + ":fallback"] = sum(l["fallback"] for l in res["letters"])
                row[n + ":note"] = res.get("note", "")
            fold_rows.append(row)
            # keep only what the picture needs (contexts hold large frames)
            fold_keep.append((r, {"segs": ctx["segs"], "same": ctx["same"]},
                              {n: A.run_decoder(ctx, bundle, n) for n in names}))
        with open(ckpt, "wb") as f:
            pickle.dump({"tag": tag, "rows": fold_rows, "keep": fold_keep}, f)
        rows += fold_rows
        keep += fold_keep
        A.log(f"  fold {fold} done and saved ({time.time() - t0:.0f} s elapsed)")
        if args.max_folds and fold >= args.max_folds and fold < n_folds:
            A.log(f"Stopping after fold {fold} (--max-folds). Run the same command again to continue.")
            return

    # ---------------- metrics ----------------
    def letter_hits(row, n, topn=1):
        return [row["code"][i] in row[n + ":top3"].split("|")[i][:topn] for i in range(row["length"])]

    summary = {}
    for n in names:
        h1 = [x for row in rows for x in letter_hits(row, n, 1)]
        h2 = [x for row in rows for x in letter_hits(row, n, 2)]
        h3 = [x for row in rows for x in letter_hits(row, n, 3)]
        exact = [row[n] == row["code"] for row in rows]
        by_len = {L: float(np.mean([x for row in rows if row["length"] == L for x in letter_hits(row, n)]))
                  for L in (3, 4, 5) if any(row["length"] == L for row in rows)}
        summary[n] = {"letter_acc": float(np.mean(h1)), "top2_letter_acc": float(np.mean(h2)),
                      "top3_letter_acc": float(np.mean(h3)), "exact_code_acc": float(np.mean(exact)),
                      "letter_acc_by_length": by_len,
                      "fallback_letters": int(sum(row[n + ":fallback"] for row in rows)),
                      "notes": int(sum(bool(row[n + ":note"]) for row in rows))}
    best = max(names, key=lambda n: (round(summary[n]["letter_acc"], 4), n == A.DEFAULT_DECODER))

    # confidence calibration for the best decoder: accuracy of letters by OCR probability band
    bands = collections.defaultdict(list)
    confusions = collections.Counter()
    for (r, _, results) in keep:
        res = results[best]
        for i, l in enumerate(res["letters"]):
            p = l["top"][0][1]
            band = "<40%" if p < 0.4 else "40-60%" if p < 0.6 else "60-80%" if p < 0.8 else ">=80%"
            ok = l["top"][0][0] == r["code"][i]
            bands[band].append(ok)
            if not ok:
                confusions[f"{r['code'][i]}->{l['top'][0][0]}"] += 1

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    report = {"n_records": len(records), "folds": n_folds, "group": args.group, "full": args.full,
              "data_sha": A.data_fingerprint(records), "decoders": summary, "best": best,
              "best_calibration": {b: {"n": len(v), "acc": float(np.mean(v))} for b, v in bands.items()},
              "best_top_confusions": confusions.most_common(12), "seconds": round(time.time() - t0)}
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    os.makedirs(os.path.dirname(A.BEST_DECODER_FILE), exist_ok=True)
    with open(A.BEST_DECODER_FILE, "w", encoding="utf-8") as f:
        json.dump({"decoder": best, "letter_acc": summary[best]["letter_acc"], "from": "evaluate.py",
                   "n_records": len(records), "group": args.group}, f, indent=2)

    # held-out picture of the worst rows for the best decoder
    def badness(item):
        r, _, results = item
        res = results[best]
        return (sum(a != b for a, b in zip(res["code"], r["code"])), -res["conf"])
    worst = sorted(keep, key=badness, reverse=True)[:args.sheet]
    sheet_items = [(r, light, results[best], f" held-out") for r, light, results in worst]
    A.render_sheet(sheet_items, os.path.join(args.out, "worst_heldout.png"),
                   title=f"Worst held-out recordings, decoder={best} (models never saw these)")

    A.log("\n" + "=" * 78)
    A.log(f"HELD-OUT RESULTS  ({len(records)} recordings, {n_folds} folds, groups={args.group}, "
          f"{'full' if args.full else 'fast'} models)")
    A.log(f"{'decoder':<18} {'letter acc':>10} {'top-2':>7} {'top-3':>7} {'exact code':>11} {'3L/4L/5L':>18}")
    for n in names:
        s = summary[n]
        bl = "/".join(f"{s['letter_acc_by_length'].get(L, float('nan')) * 100:.0f}" for L in (3, 4, 5))
        A.log(f"{n:<18} {s['letter_acc'] * 100:>9.1f}% {s['top2_letter_acc'] * 100:>6.1f}% "
              f"{s['top3_letter_acc'] * 100:>6.1f}% {s['exact_code_acc'] * 100:>10.1f}% {bl:>18}"
              + ("   <- best" if n == best else ""))
    A.log("-" * 78)
    A.log("Best decoder, accuracy by OCR confidence: "
          + ", ".join(f"{b}: {v['acc'] * 100:.0f}% (n={v['n']})" for b, v in
                      sorted(report["best_calibration"].items())))
    A.log("Most common mistakes (true->pred): " + ", ".join(f"{k} x{v}" for k, v in confusions.most_common(8)))
    A.log(f"Saved {args.out}/report.json, predictions.csv, worst_heldout.png and {A.BEST_DECODER_FILE} "
          f"(decoder={best}) in {time.time() - t0:.0f} s")
    A.log("Letter accuracy is the closest number to your expected share of the Invisible Ink credit.")


if __name__ == "__main__":
    main()
