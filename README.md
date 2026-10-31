# sports-sci-paper-11 — Generalization limits of benchmark fall detection (SisFall)

BSPC (Elsevier) submission, double-anonymised, CPU-only (yunke pipeline).

## Result in one line
A conventional class-balanced detector reaches 0.998 ROC-AUC under
leave-one-subject-out on SisFall (154,137 windows, 38 subjects), but recognises
only a median 8.8% of entirely unseen fall types (LOFO), catches 10.9% of the
single elderly faller's falls, and doubles false alarms on elderly daily
activity (10.0% vs 5.0%) - benchmark saturation measures memorisation, not
deployment readiness.

## Pipeline
```
python -I scripts/01_build.py          # SisFall zip (nested) -> windows.parquet (audit trail)
python -I scripts/02_analysis.py       # P1 LOSO / P2 LOFO / P4 device (models.pkl cache)
python -I scripts/02b_age_transfer.py  # P3 cross-age deployment test
python -I scripts/04_figures_ejss.py
python -I manuscript/make_tables.py
cd manuscript && node build_ejss.js && soffice --headless --convert-to pdf ...
```
Desktop package: `投稿包_跌倒检测泛化\` (md5sums.txt included).

## Dataset facts that shaped the design
- SisFall falls were simulated ONLY by the 23 young subjects; the 15 elderly
  subjects perform ADLs (SE06 exception: 15 fall trials) -> "within-elderly"
  protocols are meaningless; the honest cross-age test is a false-alarm +
  single-faller recall test.
- Falls are labelled at the impact window only; pre/post-impact windows are
  kept as hard negatives (label positives = 2.3% of windows -> class-balanced
  training is mandatory; unbalanced HGB degenerates to an inverted model).
- Data channel: GitHub releases mirror BIng2325/SisFall (213 MB, nested zip);
  the authors' site is dead.
