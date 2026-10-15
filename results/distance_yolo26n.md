# Distance error on nuScenes v1.0-mini CAM_FRONT - detector, 366/384 objects matched

## By class
| class | method | n | MAE_m | median_abs_m | AbsRel_% | RMSE_m |
|---|---|---|---|---|---|---|
| barrier | size | 126 | 1.73 | 0.93 | 5.7 | 4.44 |
| barrier | ground | 126 | 12.62 | 4.45 | 38.5 | 21.37 |
| barrier | fused | 126 | 2.42 | 1.01 | 7.5 | 4.95 |
| cone | size | 240 | 1.14 | 0.55 | 4.7 | 1.97 |
| cone | ground | 238 | 6.33 | 1.56 | 22.4 | 22.97 |
| cone | fused | 240 | 1.54 | 0.86 | 6.4 | 2.63 |

## By true distance
| bin | method | n | MAE_m | median_abs_m | AbsRel_% | RMSE_m |
|---|---|---|---|---|---|---|
| 0-10 m | size | 31 | 0.47 | 0.1 | 6.8 | 0.84 |
| 0-10 m | ground | 29 | 0.28 | 0.29 | 3.7 | 0.34 |
| 0-10 m | fused | 31 | 0.39 | 0.16 | 5.8 | 0.64 |
| 10-20 m | size | 117 | 0.58 | 0.42 | 3.9 | 0.87 |
| 10-20 m | ground | 117 | 1.91 | 1.2 | 12.0 | 2.88 |
| 10-20 m | fused | 117 | 0.9 | 0.68 | 5.8 | 1.24 |
| 20-30 m | size | 133 | 1.09 | 0.73 | 4.4 | 1.51 |
| 20-30 m | ground | 133 | 8.12 | 2.23 | 31.5 | 28.11 |
| 20-30 m | fused | 133 | 1.46 | 1.0 | 5.7 | 2.08 |
| 30-50 m | size | 70 | 2.6 | 1.92 | 7.0 | 3.55 |
| 30-50 m | ground | 70 | 21.85 | 18.2 | 57.5 | 30.76 |
| 30-50 m | fused | 70 | 3.63 | 2.25 | 9.7 | 5.24 |
| 50-inf m | size | 15 | 5.35 | 2.05 | 8.2 | 11.88 |
| 50-inf m | ground | 15 | 17.04 | 10.12 | 30.5 | 26.94 |
| 50-inf m | fused | 15 | 7.26 | 5.2 | 11.7 | 11.74 |
