# Generated OMR template calibration

The scanner templates are calibrated against the Manchester OMR PDFs rendered
to a canonical `1600 x 2263` pixel page.

## NEET / KCET

- Reference: `references/neet_kcet_generated.png`
- Four answer columns with exactly 60 physical rows each
- 240 physical answer rows in total
- Printed ranges: `1-60`, `61-120`, `121-180`, `181-240`
- Options are logically A, B, C, D in the scanner (the printed bubbles contain
  1, 2, 3, 4 in the corresponding A, B, C, D positions)
- Series selector: P, Q, R, S
- Canonical answer-row centres run from y=326 through y=1979

The print sheet, canonical reference, KCET template and NEET template now share
the same 60-row geometry.

## JEE

JEE remains on its existing independent template and reference. Do not replace
JEE geometry with the 240-question NEET/KCET grid.

- Template: `templates/jee.json`
- Reference: `references/jee_generated.png`
- MCQ questions: 1-20, 26-45, 51-70
- Numerical questions: 21-25, 46-50, 71-75
- Numerical responses support seven digit columns, one decimal selection, and
  the minus-sign bubble shown on the generated sheet
- Series selector: P, Q, R, S

## Answer keys

Scoring requires an answer-key JSON whose filename matches the detected series.
The P/Q/R/S files included in this patch are test-only dummy keys expanded to
240 questions for NEET/KCET. Replace them with official keys in production.
