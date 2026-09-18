# Stability evaluation: profiles/kanadb.json

Enrollment order (by `created_at`): open_palm, peace_sign, fist, thumbs_up.

## Accuracy matrix (rows: gestures enrolled so far, blank: not yet enrolled)

| enrolled | open_palm | peace_sign | fist | thumbs_up |
|---|---|---|---|---|
| 1 | 1.000 |  |  |  |
| 2 | 1.000 | 1.000 |  |  |
| 3 | 1.000 | 1.000 | 1.000 |  |
| 4 | 1.000 | 1.000 | 1.000 | 0.000 |

## Prototype stability (O4)

Passed: all 4 gestures' prototypes were bit-identical to the fully enrolled profile at every earlier enrollment step.
