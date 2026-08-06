# Task 2 Submission Checklist (deadline: 2026-07-24 23:59 KST)

## 1. Internal validation results
- `internal_validation_results.md` — DONE. Headline: F1@0.5=0.9929
  (rad_dino + chexfound_vitl16 ensemble) on `Data/test.csv` (1,940 samples).
  Verified against the organizers' own official `evaluate_task2.py`
  (Precision=0.9989, Recall=0.9870, F1=0.9929) -- not just our own
  reimplementation.

## 2. Method paper (LNCS, <=4 pages)
- `paper/task2_paper.tex` — DRAFTED, written from scratch, title changed
  per your request, Figure 1 (shortcut-learning Grad-CAM) added.
- `paper/task2_paper.docx` — converted via pandoc for easier editing.
  Two small manual touch-ups needed in Word: (1) the title/author/institute
  block landed AFTER the Abstract instead of before it (pandoc quirk,
  cut-and-paste to fix), (2) a stray lone "8" appears just before the
  references list (artifact of `\begin{thebibliography}{8}`, just delete
  that line).
- Still needs a LaTeX compile (e.g. via Overleaf, which has the Springer
  LNCS template built in) to confirm page count/formatting — no LaTeX
  available on this Mac to verify directly.
- `paper/diagram_prompt.md` — a ready-to-paste ChatGPT prompt for an
  architecture diagram, with a fallback suggestion (ask for Mermaid/TikZ
  code instead of a raster image, since text-in-image generation is
  unreliable for exact labels).
- `paper/fig1_shortcut_learning.png` — the embedded figure; keep alongside
  the .tex/.docx if you re-upload to Overleaf.

## 3. Dockerfile — REVISED to match the organizers' actual template
- Fetched the real spec from
  github.com/mi2rl-challenge/treat-mmtb.miccai2026 (Task2/Dockerfile_task2,
  README_task2.md, predict_task2.py, evaluate_task2.py) -- this was NOT
  guessed, it's their literal repo.
- Fixed contract confirmed: `python predict_task2.py --input /input --output
  /output`, writing `/output/prediction.csv` with columns `filename,TB/Normal`
  (filename = the raw PNG's own basename, e.g. "abc.png").
- `Dockerfile.submission` — updated: entrypoint is now exactly
  `["python", "predict_task2.py", "--input", "/input", "--output", "/output"]`.
- `submission_predict_task2.py` (NEW) — copied into the image as
  `/workspace/predict_task2.py` (the literal file the fixed entrypoint runs).
  Wraps our real pipeline (lung-crop preprocessing + rad_dino/chexfound_vitl16
  ensemble via `Code/predict_task2.py`, which keeps its own different CLI
  unchanged) and translates our internal `new_id` back to the exact original
  filename the organizers require.
- Old `docker_entrypoint.sh` (bash version, wrong output filename/contract)
  deleted -- fully superseded.

## 4. Docker image
- `sbatch/build_submission_image.sbatch` — fixed two real bugs found while
  running it: (1) a `find | head` + `pipefail` SIGPIPE that silently killed
  the job right after the image built, before the smoke test ever ran
  (job 118018), (2) smoke test was checking for the wrong output filename
  (`submission.csv` instead of the now-correct `prediction.csv`).
- NEEDS RESUBMISSION now that the Dockerfile/entrypoint contract changed:
  ```
  sbatch sbatch/build_submission_image.sbatch
  ```
  Builds `lisa-task2-submission:v1`, smoke-tests it against 5 real images
  from `Data/test`, then `docker save | gzip` to
  `lisa-task2-submission_v1.tar.gz` for Google Drive upload.

## Still open / needs your input
- [ ] Run `sbatch/build_submission_image.sbatch` on the cluster (I can't run
      sbatch from this Mac) -- this is a fresh run against the corrected
      Dockerfile, not a resume of the earlier failed job.
- [ ] Compile `paper/task2_paper.tex` (e.g. via Overleaf) to verify it fits
      4 pages and looks right.
- [ ] Fix the two small manual issues in `paper/task2_paper.docx` (see above).
- [ ] Upload the resulting `.tar.gz` to Google Drive and get a shareable link.
- [ ] Send the email with: internal validation results, paper PDF,
      Dockerfile.submission, and the Drive link to the image.
