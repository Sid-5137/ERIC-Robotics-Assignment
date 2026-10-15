# Distance error on nuScenes v1.0-mini CAM_FRONT - oracle (projected ground-truth boxes)

## By class
| class | method | n | MAE_m | median_abs_m | AbsRel_% | RMSE_m |
|---|---|---|---|---|---|---|
| barrier | size | 135 | 2.64 | 2.22 | 10.1 | 3.28 |
| barrier | ground | 135 | 11.7 | 4.52 | 36.6 | 19.57 |
| barrier | fused | 135 | 2.24 | 1.76 | 8.3 | 3.06 |
| cone | size | 249 | 2.03 | 1.28 | 8.7 | 3.68 |
| cone | ground | 245 | 6.15 | 1.62 | 20.7 | 16.44 |
| cone | fused | 249 | 1.78 | 1.1 | 7.3 | 3.47 |

## By true distance
| bin | method | n | MAE_m | median_abs_m | AbsRel_% | RMSE_m |
|---|---|---|---|---|---|---|
| 0-10 m | size | 31 | 1.18 | 1.16 | 16.0 | 1.33 |
| 0-10 m | ground | 27 | 0.35 | 0.34 | 4.6 | 0.43 |
| 0-10 m | fused | 31 | 0.83 | 0.67 | 11.5 | 1.03 |
| 10-20 m | size | 120 | 1.32 | 1.26 | 8.8 | 1.62 |
| 10-20 m | ground | 120 | 1.78 | 1.13 | 10.8 | 2.66 |
| 10-20 m | fused | 120 | 0.99 | 0.7 | 6.3 | 1.34 |
| 20-30 m | size | 135 | 2.06 | 1.85 | 8.3 | 2.54 |
| 20-30 m | ground | 135 | 7.25 | 2.5 | 28.3 | 17.29 |
| 20-30 m | fused | 135 | 1.93 | 1.8 | 7.7 | 2.28 |
| 30-50 m | size | 77 | 2.64 | 2.16 | 7.2 | 3.17 |
| 30-50 m | ground | 77 | 21.53 | 16.41 | 56.4 | 30.19 |
| 30-50 m | fused | 77 | 2.69 | 1.91 | 7.2 | 3.62 |
| 50-inf m | size | 21 | 8.87 | 7.14 | 14.7 | 11.58 |
| 50-inf m | ground | 21 | 10.83 | 5.74 | 19.8 | 17.6 |
| 50-inf m | fused | 21 | 6.37 | 2.48 | 10.4 | 10.48 |
