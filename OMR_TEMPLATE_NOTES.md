# Generated OMR template calibration

The scanner templates are calibrated against the generated Manchester OMR
PDFs rendered to a canonical `1600 x 2263` pixel page.

## NEET / KCET

- Reference: `references/neet_kcet_generated.png`
- Four answer columns with 52 physical rows each
- 208 physical answer rows in total
- Options: A, B, C, D
- Series selector: P, Q, R, S

The supplied PDF does not contain 60 answer rows per column. Its printed
question labels also include repeated/skipped numbers, so the scanner maps the
physical rows sequentially from 1 through 208.

## JEE

- Reference: `references/jee_generated.png`
- MCQ questions: 1-20, 26-45, 51-70
- Numerical questions: 21-25, 46-50, 71-75
- Numerical responses support seven digit columns, one decimal selection, and
  the minus-sign bubble shown on the generated sheet
- Series selector: P, Q, R, S

## Answer keys

Scoring requires an answer-key JSON whose filename matches the detected
series, for example `answer_keys/jee/P.json`. The repository does not invent
or translate answer keys between older codes and the new P/Q/R/S selector.
