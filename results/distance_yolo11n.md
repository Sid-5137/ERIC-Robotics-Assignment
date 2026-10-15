# Distance error on nuScenes v1.0-mini CAM_FRONT - detector, 362/384 objects matched

## By class
| class | method | n | MAE_m | median_abs_m | AbsRel_% | RMSE_m |
|---|---|---|---|---|---|---|
| barrier | size | 124 | 1.56 | 0.78 | 5.3 | 3.29 |
| barrier | ground | 124 | 12.99 | 4.92 | 40.4 | 21.87 |
| barrier | fused | 124 | 2.3 | 0.9 | 7.3 | 4.28 |
| cone | size | 238 | 1.27 | 0.6 | 5.3 | 2.25 |
| cone | ground | 235 | 6.08 | 1.62 | 22.1 | 24.29 |
| cone | fused | 238 | 1.57 | 0.82 | 6.6 | 2.66 |

## By true distance
| bin | method | n | MAE_m | median_abs_m | AbsRel_% | RMSE_m |
|---|---|---|---|---|---|---|
| 0-10 m | size | 31 | 0.51 | 0.15 | 7.4 | 0.91 |
| 0-10 m | ground | 28 | 0.3 | 0.32 | 3.9 | 0.35 |
| 0-10 m | fused | 31 | 0.46 | 0.23 | 6.7 | 0.85 |
| 10-20 m | size | 118 | 0.53 | 0.37 | 3.5 | 0.79 |
| 10-20 m | ground | 118 | 2.0 | 1.2 | 12.4 | 3.03 |
| 10-20 m | fused | 118 | 0.93 | 0.61 | 5.9 | 1.29 |
| 20-30 m | size | 134 | 1.23 | 0.92 | 4.9 | 1.67 |
| 20-30 m | ground | 134 | 8.79 | 2.37 | 34.2 | 31.12 |
| 20-30 m | fused | 134 | 1.41 | 0.92 | 5.5 | 2.06 |
| 30-50 m | size | 66 | 2.95 | 2.01 | 8.0 | 4.17 |
| 30-50 m | ground | 66 | 20.75 | 15.81 | 54.5 | 28.98 |
| 30-50 m | fused | 66 | 3.87 | 2.38 | 10.2 | 5.6 |
| 50-inf m | size | 13 | 4.39 | 2.01 | 7.1 | 8.44 |
| 50-inf m | ground | 13 | 19.04 | 12.61 | 34.9 | 30.01 |
| 50-inf m | fused | 13 | 6.91 | 5.95 | 11.6 | 9.2 |
