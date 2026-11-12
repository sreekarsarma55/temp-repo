# Invisible Ink decoder (TDS 2026 Sep P1, Q9) — v15

Reads the six hidden-ink motion recordings in a live session and prints the letter codes, with
confidences and alternatives, in about 3 seconds.

## 1. How good is it, honestly?

Measured with `evaluate.py`: each recording is decoded by models trained **without it** (5 folds,
recordings from the same phone/canvas size kept in the same fold), on the 290 pooled recordings.

| decoder | letter accuracy | top-2 | top-3 | whole code right |
|---|---|---|---|---|
| v14 (your old pipeline) | 74.1% | 81.2% | 85.3% | 44.8% |
| rules-lite (v14 without the hand rules) | 74.9% | 82.6% | 86.4% | 46.6% |
| joint-time (new) | 76.7% | 83.3% | 85.9% | 49.3% |
| **auto (new, default)** | **77.4%** | 83.8% | 86.1% | 51.7% |

By length: 3 letters 85%, 4 letters 85%, 5 letters 69%. These folds used the quicker `fast`
models, so the full model you use in the exam should do slightly better.

**Why v14 showed 100%:** its benchmark rows were training recordings. The OCR had been trained on
those exact letters, and the v10–v14 geometric rules were written by looking at those same rows. On
recordings it had never seen, v14 gets about 74% of letters right.

**Confidence is meaningful** (held-out letters, best decoder):

| OCR probability of the top letter | how often it is right |
|---|---|
| ≥ 80% | 96% |
| 60–80% | 81% |
| 40–60% | 48% |
| < 40% | 35% |

So: trust letters at 80% or above. Spend your 18 minutes on letters marked `*` (below 60%). For
those, the right letter is usually the 2nd or 3rd choice printed next to it.

Most common mistakes: Y→V, R→P, V→N, N→K, R→K, Y→X, Y→M, M→W.

## 2. What the model is

1. **Trace.** `trace_recording` is an exact Python port of the game's `renderer.js trace()`: phone
   tilt (beta/gamma) → smoothed cursor position. Recenter events are handled the same way.
2. **Ink detector (is the pen down at this sample?).** Stage 1 is LightGBM on about 250 motion
   features per sample (speed, acceleration, turning, straightness, rolling windows, neighbours).
   Stage 2 is a second LightGBM that re-reads Stage 1's probabilities in context. The target is the
   4th sample column (pen state) in the training files. Held-out AUC is 0.97 and F1 0.89.
3. **Clear/undo.** Ink before the last `clear` is dropped. Each `undo` erases the predicted stroke
   just before it.
4. **Split into letters.** The game tells you the code length (3/4/5).
   - `v14`: the old x-position split plus hand rules.
   - `joint-*`: cuts the ink into small pieces, then tries the possible ways of grouping
     consecutive pieces into exactly N letters (or ignoring a piece as noise). It keeps the grouping
     whose letters the OCR is most confident about.
   - `auto`: runs v14 and joint, and keeps whichever reading is more confident.
5. **Letter OCR.** An ensemble of ExtraTrees, RandomForest, LightGBM, logistic regression and MLP
   over 17 letters. Each letter is described by 253 numbers: a 12×12 image, stroke directions and
   shape statistics. It is trained on the true ink, the traced path with the true pen state, and
   *predicted* ink from models that hadn't seen that recording, which is what it meets in the exam.

## 3. What changed from v14 (bugs fixed)

- An honest evaluation (`evaluate.py`) replaces the training-set benchmark. It saves
  `outputs/cv/worst_heldout.png` so you can see real failures.
- v14 could print `?` for an empty letter, and the game's input box rejects that. v15 always
  outputs valid letters.
- The empty-letter descriptor had 252 numbers instead of 253.
- The saved model was reused whenever the *number* of recordings matched. It is now keyed on a hash
  of the data.
- Stage-2 window features could cross from one recording into the next during training, and relied
  on the records being sorted by id.
- OCR training letters included strokes the student had undone.
- `predict.py` never retrains or touches the network, and finishes in about 3 seconds.

## 4. Setup (laptop)

```bash
git clone https://github.com/sreekarsarma55/temp-repo.git airink && cd airink
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw && unzip air-ink-training-60.zip -d data/raw   # the pooled training files
```

The repo already contains a trained model (`models/airink_v15.joblib`) and `models/best_decoder.json`.

Check it works (about 7 s):

```bash
python rehearse.py --seed 5
python predict.py rehearsal/air-ink-test-six.jsonl --key rehearsal/answer_key.json
```

The score printed here is inflated, because the shipped model has seen these recordings. It only
proves the setup works. For an honest rehearsal (about 3 minutes):

```bash
python train.py --fast --exclude rehearsal/answer_key.json --out models/rehearsal.joblib
python predict.py rehearsal/air-ink-test-six.jsonl --model models/rehearsal.joblib --key rehearsal/answer_key.json
```

## 5. Retraining with more data (optional, before the session)

Put any new `air-ink-training-60*.jsonl` (yours or classmates') in `data/raw/`, then run:

```bash
python train.py          # ~7 min; merges data/raw into master_training_pooled.jsonl, saves the model
python evaluate.py       # ~20 min; honest accuracy, rewrites models/best_decoder.json
```

`evaluate.py` saves each fold as it finishes. If it is interrupted, run the same command again and
it continues. To run it in parts: `python evaluate.py --max-folds 2`, then run it again.

## 6. Exam-day runbook

Sessions (IST): **Sat 14 Nov 4:00 PM**, **Sun 15 Nov 8:00 AM**, **Sun 15 Nov 8:00 PM**. You can
start up to 40 minutes after a session opens, and you can play once per session.

**10 minutes before**

1. Open P1, then open **Open my Invisible Ink game** (the link expires 20 minutes after you open P1).
2. In a terminal:
   ```bash
   cd airink && source venv/bin/activate
   python predict.py --watch --open
   ```
   This loads the model once and waits for the download in `~/Downloads`. If your browser saves
   somewhere else, add `--downloads /path/to/folder`.
3. Enter the waiting room (it opens 2 minutes early).

**Start → first 2 minutes (Drawing 1 only, worth 20%)**

1. Press **Start my 20 minutes**, then immediately **Download all six motion files**.
2. The terminal prints six rows within about 3 seconds. Find the row whose recording id matches
   **tile 1** ("Recording xxxxxxxxxxxx" under the tile). Type that ANSWER into the first-guess box.
   Wait for "Saved on the server".

**Minutes 2–20 (all six, worth 80%)**

1. Type all six answers. Match every row by **recording id**, not by row order.
2. Then fix the uncertain letters:
   - Letters marked `*` are below 60%. Look at that letter's crop in the picture
     (`outputs/live/<time>/decoded.png`, opened automatically with `--open`). Replay that drawing
     in the game at 1×, and pick between the 1st, 2nd and 3rd choice.
   - "other decoders disagree" lines show where to look first.
   - Classmates in the same session may have the same recordings, which is allowed. Compare
     answers by recording id.
3. Leave a few minutes spare. Every box must hold a complete code or be empty: a partly typed code
   is not saved.

**After 20:00**

Wait out the one-minute grace period, then copy the **signed record (JWT)**.

**Saving in P1 (careful: Save re-submits every answer on the page)**

1. In P1's "Recent saves" box, press **Reload** on the latest save first. This puts your other
   answers back into the form, so saving doesn't wipe them.
2. Paste the JWT into the Invisible Ink box, press **Check**, then **Save**.
3. Confirm the saved total did not drop (it was 6.58/13 before).

**Which session to keep:** you will not see scores. If you play more than one session, save the
record from the session where the model's average confidence was highest and you had the fewest `*`
letters left. Each run's answers are kept in `outputs/live/<time>/answers.txt`.

## 7. Sharing with classmates

- **Data:** `master_training_pooled.jsonl` holds every unique labeled recording (290 now, 11 MB).
  Classmates put it in their own `data/raw/` and run `train.py`; duplicates are removed by id. The
  question explicitly allows sharing examples and code.
- **Model:** `models/airink_v15.joblib` works only with the same package versions, so they should run
  `pip install -r requirements.txt`. Only load model files from people you trust: `.joblib`/`.pkl`
  files can run code when loaded. Data files are safe. When in doubt, share the `.jsonl` and let
  people retrain.
- **Never share** your P1 launch links or sign-in codes (anyone with them can act as you), or your
  signed record before you have saved it.

## 8. Troubleshooting

- `No model at models/airink_v15.joblib` → run `python train.py`.
- `WARNING: model was saved with different package versions` → `pip install -r requirements.txt`,
  or retrain.
- Watch mode prints nothing after the download → check the download folder and the file name
  (`air-ink-test-six*.jsonl`). As a fallback, run `python predict.py --latest`.
- To use a specific decoder instead of the chosen one: `python predict.py --watch --decoder v14`
  (choices: v14, v14-dip, rules-lite, joint-time, joint-time-loose, joint-x, auto).

## Files

| file | purpose |
|---|---|
| `airink.py` | library: trace, features, ink model, decoders, OCR, pictures |
| `train.py` | merge data, train, save `models/airink_v15.joblib` |
| `evaluate.py` | honest end-to-end cross-validation; picks `models/best_decoder.json` |
| `predict.py` | exam day: `--watch`, `--latest`, or a file path |
| `rehearse.py` | builds a fake six-recording file from training data |
| `tests/` | descriptor equivalence check, synthetic data generator |
| `outputs/cv/` | held-out report, per-recording predictions, worst cases picture |
