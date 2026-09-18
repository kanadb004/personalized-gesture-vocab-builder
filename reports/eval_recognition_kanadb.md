# Recognition evaluation: profiles/kanadb.json

Profile: `profiles/kanadb.json`. Sessions: `data/sessions/eval/kanadb`. 12 gesture clips, 1 non-gesture clips.

## Per-gesture clip accuracy

| gesture | clips | accuracy |
|---|---|---|
| open_palm | 3 | 1.000 |
| peace_sign | 3 | 1.000 |
| fist | 3 | 1.000 |
| thumbs_up | 3 | 0.000 |
| **overall** | 12 | 0.750 |

## Confusion matrix (rows: true gesture, columns: predicted label of first trigger)

| true \ predicted | open_palm | peace_sign | fist | thumbs_up | none |
|---|---|---|---|---|---|
| open_palm | 3 | 0 | 0 | 0 | 0 |
| peace_sign | 0 | 3 | 0 | 0 | 0 |
| fist | 0 | 0 | 3 | 0 | 0 |
| thumbs_up | 0 | 0 | 0 | 0 | 3 |

## Non-gesture clips

Total duration: 60.0 s. Total false triggers: 0. False triggers per minute: 0.000.

## Algorithmic latency

Median over 9 correctly triggered clips: 729.6 ms.

Full per-clip results in `reports/eval_recognition_kanadb.csv`.
