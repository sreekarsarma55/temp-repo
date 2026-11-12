# Invisible Ink helper — TDS 2026 Sep, Project 1, Q9

A ready-to-use predictor for the **Invisible Ink** live sessions. You download the six motion
recordings in the game, and this tool prints its best guess for every code within a few seconds,
with a confidence for each letter.

The question allows this: *"Make a predictor using any method: code, AI or your own eyes. You may
share examples, use AI and work with classmates."*

> **You still type and check the answers yourself.** The tool is right about 3 letters out of 4
> overall, and almost always right when it is confident. Use your 18 minutes on the letters it
> marks as uncertain (section 5).

---

## Contents

1. [What you need](#1-what-you-need)
2. [One-time setup](#2-one-time-setup-about-10-minutes)
3. [Practise once before the session](#3-practise-once-before-the-session-2-minutes)
4. [Exam day, step by step](#4-exam-day-step-by-step)
5. [How to read the output](#5-how-to-read-the-output)
6. [Saving your record in P1 (do not skip)](#6-saving-your-record-in-p1-do-not-skip)
7. [How accurate is it?](#7-how-accurate-is-it)
8. [How it works](#8-how-it-works)
9. [Optional: add more training data and retrain](#9-optional-add-more-training-data-and-retrain)
10. [Is the model file safe to open?](#10-is-the-model-file-safe-to-open)
11. [Troubleshooting](#11-troubleshooting)
12. [What is in this repo](#12-what-is-in-this-repo)

---

## 1. What you need

- A **laptop or desktop** (Windows, macOS or Linux). The game itself runs in the browser; this tool
  runs in a terminal next to it.
- **Python 3.10 or newer**. Check with `python3 --version` (macOS/Linux) or `python --version`
  (Windows). If you don't have it, install it from <https://www.python.org/downloads/>. On Windows,
  tick **"Add python.exe to PATH"** during install.
- **Git** (optional). You can also download the repo as a ZIP from GitHub
  (green **Code** button → **Download ZIP**).
- About **500 MB** of free disk space for the Python packages.

No GPU, no API keys and no internet are needed during the session.

---

## 2. One-time setup (about 10 minutes)

### 2.1 Get the code

```bash
git clone https://github.com/sreekarsarma55/temp-repo.git invisible-ink
cd invisible-ink
```

If you used **Download ZIP** instead, unzip it and open a terminal in that folder.

### 2.2 Create a virtual environment and install the packages

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell says *"running scripts is disabled"*, run this once and then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Windows (Command Prompt / cmd.exe)**: use `venv\Scripts\activate.bat` instead.

After activating, your prompt starts with `(venv)`. **Every time you open a new terminal**, `cd`
into the folder and activate again before running anything.

### 2.3 Check that it works

```bash
python tests/check_descriptor.py
```

You should see `OK` at the end.

That's all. The repo already contains a **trained model** (`models/airink_v15.joblib`) and the
**training data** (`master_training_pooled.jsonl`), so you do **not** need to train anything.

---

## 3. Practise once before the session (2 minutes)

This makes a fake "six recordings" file from the training data, just like the one the game gives
you, and runs the predictor on it.

```bash
python rehearse.py --seed 1
python predict.py rehearsal/air-ink-test-six.jsonl --key rehearsal/answer_key.json --open
```

You will see six rows of answers, a picture will open, and at the bottom a line like
`Rehearsal score: 22/24 letters`.

That score is **higher than real life**, because the shipped model has already seen these
recordings in training. This step only checks that everything runs. The honest numbers are in
section 7.

**Practise the watch mode too** (it is what you will use in the exam):

1. Run `python predict.py --watch --open` and leave it running.
2. Copy `rehearsal/air-ink-test-six.jsonl` into your **Downloads** folder.
3. Within a few seconds, the answers appear in the terminal.
4. Press **Ctrl+C** to stop.

---

## 4. Exam day, step by step

**Session times (IST):** Sat 14 Nov, 4:00–5:00 PM · Sun 15 Nov, 8:00–9:00 AM · Sun 15 Nov,
8:00–9:00 PM. You must press Start within the first 40 minutes of a session. You may play once in
each session.

### 10 minutes before the session

1. Open your terminal, `cd` into the folder and activate the venv (section 2.2).
2. Start the watcher:

   ```bash
   python predict.py --watch --open
   ```

   It loads the model and then waits. It shows `Watching ... for new air-ink-test-six*.jsonl files`.

   If your browser saves downloads somewhere other than your **Downloads** folder, tell it where:
   `python predict.py --watch --open --downloads "D:\My Downloads"`.
3. Open P1 → Invisible Ink question → **Open my Invisible Ink game**. The link expires 20 minutes
   after you load P1, so reload P1 if needed.
4. The waiting room opens 2 minutes before the session.

### Minute 0 to 2: Drawing 1 only (worth 20%)

1. Press **Start my 20 minutes**.
2. Immediately click **Download all six motion files**.
3. The answers print in the terminal within about 5 seconds, and the picture opens.
4. Look under **tile 1** in the game for `Recording xxxxxxxxxxxx`. Find the row with the **same
   recording id** in the terminal, and type its ANSWER into the first-guess box.
5. Wait until the game shows **"Saved on the server"**.

> Match answers by **recording id**, not by row number. The order may differ.

### Minute 2 to 20: all six drawings (worth 80%)

1. Type all six answers, matching each by recording id. Drawing 1's first guess is carried over.
2. Then improve the uncertain letters (section 5): check the picture, replay the drawing in the
   game at 1× speed, and pick between the 1st, 2nd and 3rd choice.
3. You may also compare with classmates in the same session, using the recording id.
4. Make sure every box has a **complete** code before 20:00. A partly typed code is **not saved**.

### After 20:00

1. Wait for the one-minute grace period to finish. Keep the page open.
2. Click **Copy signed record**.
3. Save it in P1 using section 6.

Every run is kept in `outputs/live/<date_time>/`: `answers.txt`, `decoded.png` and a copy of the
download.

---

## 5. How to read the output

```
 #  recording     len  ANSWER   conf  per-letter top-3 (% OCR probability)
 1  75d6909a464c    4  PHMR     0.71  P72  Y6 R5 | H90  N2 W2 | M52* N25 H4 | R68  K23 X2
 2  a0845a5e506d    4  FCNP     0.73  F93  E2 P1 | C98  E1 J0 | N61  M14 X5 | P39* M13 J11
 3  ecd0341c4808    3  FCU      0.77  F58* E12 M4 | C87  U3 E2 | U87  C5 V1
                       other decoders disagree -> v14:RCP, v14-dip:RCP
```

- **ANSWER** is the tool's best code. Type this unless you find a reason to change a letter.
- **conf** is the average confidence of the letters (0–1).
- Each `|`-separated block is one letter: the best guess and its probability, then the 2nd and
  3rd choices. In `M52* N25 H4`, the tool is 52% sure it's **M**, 25% **N**, 4% **H**.
- **`*` means uncertain** (below 60%). Check these first.
- **"other decoders disagree"** means a different way of splitting the ink into letters gives
  another reading. Above, letter 1 of drawing 3 may be **R** instead of **F**. Look at it.

**The picture** (`decoded.png`) has one row per drawing:

- column 1: the full cursor movement (orange = early, blue = late)
- column 2: the ink the tool thinks was drawn, coloured by letter
- then one box per letter, with the guesses written above it

Grey lines are movements the tool thinks had the pen up. If a letter's box looks wrong (half a
letter, or two letters merged), trust your eyes and the game's replay over the tool.

**How much to trust a letter:**

| confidence of the top guess | how often it was right in testing |
|---|---|
| 80% or more | 96% |
| 60–80% | 81% |
| 40–60% | 48% (often the 2nd choice is right) |
| below 40% | 35% (check the 2nd and 3rd choices) |

Letters it confuses most often: **Y↔V, R↔P, V↔N, N↔K, R↔K, Y↔X, M↔W**.

---

## 6. Saving your record in P1 (do not skip)

P1's **Save** button submits **every answer on the whole P1 page** again. If the other boxes are
empty in your browser (for example, on a new device), saving can **lower your marks on other
questions**.

1. Open P1. In the green **Recent saves** box, click **Reload** on your latest save. This fills in
   your earlier answers.
2. Paste the signed record (the long text starting with `eyJ`) into the **Invisible Ink** box.
3. Click **Check**, then **Save**.
4. Confirm the saved total did not go **down**.

If you played more than one session, you choose which session's record to save. Scores are not
shown, so a sensible rule is to save the session where most letters had high confidence and you
had the fewest `*` letters left. Only your last save in P1 counts. P1 closes **Tue 17 Nov,
11:59 PM IST**.

---

## 7. How accurate is it?

Tested honestly: the training recordings were split into 5 parts, and each part was decoded by a
model trained only on the other 4. Recordings from the same phone screen size were kept in the same
part, so the model is always tested on writers it has not seen.

| | result |
|---|---|
| letters right | **77%** |
| right letter within the top 2 guesses | 84% |
| right letter within the top 3 guesses | 86% |
| whole code exactly right | 52% |
| 3-letter and 4-letter codes | 85% of letters |
| 5-letter codes | 69% of letters |

The full report is in `outputs/cv/report.json`. The tool's biggest mistakes are in
`outputs/cv/worst_heldout.png`; it is worth a look to see what goes wrong.

So the realistic plan is: **the tool gets most letters, and you fix the uncertain ones.**

---

## 8. How it works

Each recording is the phone's tilt (two angles) over time, plus "undo", "clear" and "recenter"
button presses. The pen-down signal is hidden, so the tool must guess where the ink was.

1. **Rebuild the cursor path.** `trace_recording()` copies the game's own JavaScript
   (`renderer.js`) that turns tilt angles into the cursor position on the canvas.
2. **Guess where the pen was down.** Two LightGBM models look at about 250 motion features per
   sensor reading (speed, acceleration, turning, straightness, nearby readings). They give each
   reading a probability that it drew ink. These models were trained on the 60-example files,
   which do include the pen state.
3. **Apply clear and undo.** Ink before the last "clear" is dropped. Each "undo" removes the stroke
   just before it.
4. **Split the ink into letters.** The game tells you the number of letters. The default method
   (`auto`) tries two ways of splitting:
   - by horizontal position (`v14`);
   - by trying many groupings of strokes in writing order and keeping the one whose letters look
     most like real letters (`joint-time`).

   It keeps whichever result is more confident.
5. **Read each letter.** Five models vote together (ExtraTrees, RandomForest, LightGBM, logistic
   regression, small neural network). They look at a 12×12 image of the letter, its stroke
   directions and shape measurements. The 17 possible letters are `ACEFHJKMNPRTUVWXY`.

---

## 9. Optional: add more training data and retrain

More examples help. Each student's 60-example download includes 10 recordings chosen for them, so
your file may contain recordings that are not in this repo yet.

1. Open your Invisible Ink game link, then click **Download my 60 examples**.
2. Put the downloaded `air-ink-training-60*.jsonl` file into the `data/raw/` folder (create it if
   needed).
3. Retrain (5–10 minutes):

   ```bash
   python train.py
   ```

   It merges everything in `data/raw/` with `master_training_pooled.jsonl`, removes duplicates, and
   overwrites `models/airink_v15.joblib`. It prints how many unique recordings it found.
4. Optional, honest accuracy and best method (20–30 minutes; safe to stop and rerun, it resumes):

   ```bash
   python evaluate.py
   ```

   This updates `models/best_decoder.json`, which `predict.py` uses automatically.

To share new data with others, share your updated `master_training_pooled.jsonl`. It is a plain
text file, so it is always safe to share.

**Honest rehearsal** (the model has not seen the six test recordings):

```bash
python rehearse.py --seed 7
python train.py --fast --exclude rehearsal/answer_key.json --out models/rehearsal.joblib
python predict.py rehearsal/air-ink-test-six.jsonl --model models/rehearsal.joblib --key rehearsal/answer_key.json
```

---

## 10. Is the model file safe to open?

`models/airink_v15.joblib` is a saved Python object (a "pickle", compressed by `joblib`). When
Python loads a pickle, it rebuilds the objects using instructions stored inside the file. Those
instructions can include running any Python function. A normal model file only rebuilds the
trained models. But someone could craft a `.joblib` or `.pkl` file that, when loaded, deletes files
or installs malware.

What this means for you:

- The model in this repo was trained from the data in this repo with the code in this repo.
  Loading it is as safe as trusting this repo.
- **Don't load model files from people you don't know**, such as one sent on WhatsApp by a
  stranger.
- If you want zero risk, retrain the model yourself from the data with `python train.py`. Then you
  never load a file you didn't create. The `.jsonl` data files are plain text and can never run
  code.

The model also needs the **same package versions** to load correctly. That is why
`requirements.txt` pins exact versions. If you see a version warning, run
`pip install -r requirements.txt`, or retrain.

**Keep private:** your P1 game links and sign-in codes (anyone with them can act as you), and your
signed record until you have saved it.

---

## 11. Troubleshooting

| problem | fix |
|---|---|
| `python: command not found` | Use `python3` (macOS/Linux) or reinstall Python with "Add to PATH" (Windows). |
| `No module named numpy` (or lightgbm, …) | The venv is not active. Activate it (section 2.2), then run `pip install -r requirements.txt`. |
| `No model at models/airink_v15.joblib` | You are in the wrong folder, or the file is missing. Run `python train.py`. |
| `WARNING: model was saved with different package versions` | Run `pip install -r requirements.txt`, or retrain with `python train.py`. |
| Watch mode prints nothing after the download | Check where the browser saved the file and that it is named `air-ink-test-six…jsonl`. Use `--downloads <folder>`, or run `python predict.py --latest --downloads <folder>`, or give the path directly: `python predict.py path/to/file.jsonl`. |
| The picture doesn't open | Open `outputs/live/<date_time>/decoded.png` yourself. |
| `pip install` fails on lightgbm on macOS | `brew install libomp`, then try again. |
| A row says `FAILED` | That one recording couldn't be decoded. Use the game's replay and classmates for it; the other five are unaffected. |

To force a specific splitting method:
`python predict.py --watch --decoder v14` (choices: `auto`, `v14`, `v14-dip`, `rules-lite`,
`joint-time`, `joint-time-loose`, `joint-x`).

---

## 12. What is in this repo

| file | what it is |
|---|---|
| `predict.py` | **The one you use on exam day.** `--watch`, `--latest` or a file path. |
| `rehearse.py` | Makes a fake six-recording file for practice. |
| `train.py` | Trains the model from the data. |
| `evaluate.py` | Honest accuracy test; picks the best splitting method. |
| `airink.py` | The shared code: cursor trace, features, ink model, letter splitting, letter reader, pictures. |
| `models/airink_v15.joblib` | The trained model (290 recordings). |
| `models/best_decoder.json` | Which splitting method `predict.py` uses by default. |
| `master_training_pooled.jsonl` | All 290 unique labeled training recordings, pooled from many downloads. |
| `air-ink-training-60.zip` | The original downloads the pool was built from. |
| `outputs/cv/` | Accuracy report, per-recording test predictions, picture of the worst cases. |
| `tests/` | A self-check and a fake-data generator used for testing. |
| `requirements.txt` | Exact package versions. |

Good luck in the session.
