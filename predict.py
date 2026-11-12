"""Exam-day decoder for the six Invisible Ink test recordings. Never retrains.

    python predict.py --watch                 # load the model once, then decode every new
                                              # air-ink-test-six*.jsonl that lands in ~/Downloads
    python predict.py --latest                # decode the newest air-ink-test-six*.jsonl in ~/Downloads
    python predict.py path/to/air-ink-test-six.jsonl
    python predict.py rehearsal/air-ink-test-six.jsonl --model models/rehearsal.joblib --key rehearsal/answer_key.json

Answers are printed first (fast), then a picture is saved to outputs/live/<time>/decoded.png.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import airink as A

PATTERN = "air-ink-test-six*.jsonl"
LOW = 0.60  # letters below this OCR probability get a '*' (look at them yourself)


def decode_file(path, bundle, decoder, show_alts=True, plot=True, key=None, open_png=False):
    t0 = time.time()
    try:
        recs = A.read_records(path)
    except Exception as e:
        A.log(f"Could not read {path}: {e}")
        return None
    good = []
    for r in recs:
        try:
            good.append(A.check_recording(r))
        except ValueError as e:
            A.log(f"  skipping a record: {e}")
    if not good:
        A.log("No usable recordings in that file.")
        return None
    if any(A.is_labeled(r) for r in good) and key is None:
        A.log("NOTE: this file contains labels (a training file?). Decoding without looking at them.")
    alts = [n for n in ("v14", "v14-dip", "joint-time") if n != decoder] if show_alts else []

    results = []
    for r in good:
        r_test = A.strip_labels(r) if A.is_labeled(r) else r
        try:
            ctx = A.make_context(r_test, bundle)
            main = A.run_decoder(ctx, bundle, decoder)
            others = {n: A.run_decoder(ctx, bundle, n)["code"] for n in alts}
        except Exception as e:  # one broken recording must not hide the other five
            A.log(f"  ERROR decoding {r.get('id')}: {e}")
            ctx, main, others = None, None, {}
        results.append((r, ctx, main, others))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    A.log("")
    A.log("=" * 92)
    A.log(f" INVISIBLE INK  {os.path.basename(path)}   decoder={decoder}   {time.time() - t0:.1f} s")
    A.log("=" * 92)
    A.log(f" {'#':<2} {'recording':<13} {'len':>3}  {'ANSWER':<7} {'conf':>5}  per-letter top-3 (% OCR probability)")
    lines, n_ok, n_tot = [], 0, 0
    for i, (r, ctx, res, others) in enumerate(results, 1):
        if res is None:
            A.log(f" {i:<2} {r.get('id', '?'):<13} {r.get('length', '?'):>3}  FAILED - use the viewer / classmates")
            continue
        cells = []
        for l in res["letters"]:
            (c1, p1), *rest = l["top"]
            mark = "*" if p1 < LOW or l["fallback"] else " "
            cells.append(f"{c1}{int(round(p1 * 100)):>2}{mark}" + "".join(f" {c}{int(round(p * 100))}" for c, p in rest[:2]))
        alt = [f"{n}:{c}" for n, c in others.items() if c != res["code"]]
        A.log(f" {i:<2} {r['id']:<13} {r['length']:>3}  {res['code']:<7} {res['conf']:>5.2f}  " + " | ".join(cells))
        if alt:
            A.log(f" {'':<2} {'':<13} {'':>3}  other decoders disagree -> " + ", ".join(alt))
        if res.get("note"):
            A.log(f" {'':<2} {'':<13} {'':>3}  note: {res['note']}")
        lines.append(f"Drawing {i}  {r['id']}  {res['code']}  conf={res['conf']:.2f}")
        if key and r["id"] in key:
            truth = key[r["id"]]
            hits = sum(a == b for a, b in zip(res["code"], truth))
            n_ok, n_tot = n_ok + hits, n_tot + len(truth)
            A.log(f" {'':<2} {'':<13} {'':>3}  true: {truth}  ({hits}/{len(truth)} letters)")
    A.log("-" * 92)
    A.log(" Type each ANSWER into the box whose tile shows the same 'Recording <id>'. Letters with * are")
    A.log(" uncertain: compare with the 2nd/3rd choice using the picture and the game's replay.")
    if n_tot:
        A.log(f" Rehearsal score: {n_ok}/{n_tot} letters = {100 * n_ok / n_tot:.1f}%")

    out_dir = os.path.join("outputs", "live", stamp)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copyfile(path, os.path.join(out_dir, "input.jsonl"))
    with open(os.path.join(out_dir, "answers.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    A.log(f" Saved {out_dir}/answers.txt")
    if plot:
        items = [(A.strip_labels(r) if key is None else r, ctx, res, f" Drawing {i}")
                 for i, (r, ctx, res, _) in enumerate(results, 1) if res is not None]
        png = os.path.join(out_dir, "decoded.png")
        A.render_sheet(items, png, title=f"{os.path.basename(path)}  decoder={decoder}")
        A.log(f" Saved {png}")
        if open_png:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            try:
                subprocess.Popen([opener, png], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                pass
    A.log("=" * 92)
    return results


def newest(folder):
    files = glob.glob(os.path.join(os.path.expanduser(folder), PATTERN))
    return max(files, key=os.path.getmtime) if files else None


def watch(folder, bundle, args, key):
    folder = os.path.expanduser(folder)
    seen = {p: os.path.getmtime(p) for p in glob.glob(os.path.join(folder, PATTERN))}
    A.log(f"Model loaded. Watching {folder} for new {PATTERN} files (Ctrl+C to stop).")
    A.log("In the game: press Start, then 'Download all six motion files'.")
    while True:
        for p in sorted(glob.glob(os.path.join(folder, PATTERN)), key=os.path.getmtime):
            m = os.path.getmtime(p)
            if seen.get(p) == m:
                continue
            size = os.path.getsize(p)
            time.sleep(0.4)
            if os.path.getsize(p) != size:
                continue  # still downloading; try again next loop
            try:
                A.read_records(p)
            except Exception:
                continue
            seen[p] = m
            decode_file(p, bundle, args.decoder, plot=not args.no_plot, key=key, open_png=args.open)
            A.log(f"\nStill watching {folder} ...")
        time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help="test file (JSONL/JSON)")
    ap.add_argument("--model", default=A.DEFAULT_MODEL)
    ap.add_argument("--decoder", default=None, help=f"one of {', '.join(A.DECODERS)} (default: models/best_decoder.json)")
    ap.add_argument("--watch", action="store_true", help="wait for new downloads in --downloads")
    ap.add_argument("--latest", action="store_true", help=f"use the newest {PATTERN} in --downloads")
    ap.add_argument("--downloads", default="~/Downloads")
    ap.add_argument("--key", help="answer key from rehearse.py (rehearsal only)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--open", action="store_true", help="open the picture automatically")
    args = ap.parse_args()
    args.decoder = args.decoder or A.default_decoder()
    if args.decoder not in A.DECODERS:
        raise SystemExit(f"unknown decoder {args.decoder}")
    if not os.path.exists(args.model):
        raise SystemExit(f"No model at {args.model}. Run: python train.py")

    t0 = time.time()
    bundle = A.load_bundle(args.model)
    A.log(f"Loaded {args.model} ({bundle['n_records']} training recordings, {bundle['created']}) "
          f"in {time.time() - t0:.1f} s; decoder={args.decoder}")
    key = None
    if args.key:
        with open(args.key, encoding="utf-8") as f:
            key = {row["id"]: row["code"] for row in json.load(f)}
        overlap = set(key) & set(bundle.get("train_ids", []))
        if overlap:
            A.log(f"WARNING: the model was trained on {len(overlap)} of these recordings, so the rehearsal score is "
                  "inflated. Use train.py --exclude for an honest rehearsal.")

    if args.watch:
        try:
            watch(args.downloads, bundle, args, key)
        except KeyboardInterrupt:
            A.log("Stopped.")
        return
    path = args.file or (newest(args.downloads) if args.latest else None)
    if not path:
        raise SystemExit("Give a file, or use --latest / --watch.")
    decode_file(path, bundle, args.decoder, plot=not args.no_plot, key=key, open_png=args.open)


if __name__ == "__main__":
    main()
