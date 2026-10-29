# Integrated empirical results: exact-value writing interface

The main analysis contains56 dataset participants in5 evaluation cohorts. Kumar development6 and Dreyer source-extension85 are separate. All new intervals below are exploratory pointwise participant bootstrap10000, seed20261029. Native historical endpoint intervals and p-values remain in their separate source tables.

## Conventional full-support comparisons against frozen decoding

| Cohort | Method | BA % | C pp | I pp | D pp | G pp [new CI] |
|---|---|---:|---:|---:|---:|---|
| Kumar2024 | aligned_full_refit | 70.65681 | 24.16944 | 11.45002 | 35.61945 | +12.71942 [+6.61426, +19.11719] |
| Kumar2024 | classifier_only | 59.88717 | 10.70344 | 8.75365 | 19.45710 | +1.94979 [-0.96671, +5.08442] |
| Kumar2024 | frozen | 57.93739 | 0.00000 | 0.00000 | 0.00000 | +0.00000 [+0.00000, +0.00000] |
| Kumar2024 | full_refit | 65.35874 | 20.85118 | 13.42982 | 34.28100 | +7.42136 [+2.64064, +12.41522] |
| BNCI2014-001 | aligned_full_refit | 77.50000 | 14.72222 | 7.77778 | 22.50000 | +6.94444 [+3.24074, +11.75926] |
| BNCI2014-001 | classifier_only | 71.66667 | 3.24074 | 2.12963 | 5.37037 | +1.11111 [-0.46296, +2.96296] |
| BNCI2014-001 | frozen | 70.55556 | 0.00000 | 0.00000 | 0.00000 | +0.00000 [+0.00000, +0.00000] |
| BNCI2014-001 | full_refit | 71.85185 | 5.55556 | 4.25926 | 9.81481 | +1.29630 [-1.75926, +3.98148] |

## Matched conventional component contrasts

| Cohort | Operation | C pp | I pp | G pp [new CI] |
|---|---|---:|---:|---|
| Kumar2024 | alignment_vs_full | 14.61477 | 9.31670 | +5.29807 [+2.81570, +8.30855] |
| Kumar2024 | classifier_vs_frozen | 10.70344 | 8.75365 | +1.94979 [-0.96671, +5.08442] |
| Kumar2024 | spatial_vs_classifier | 16.04174 | 10.57018 | +5.47157 [+2.15870, +8.92197] |
| BNCI2014-001 | alignment_vs_full | 12.77778 | 7.12963 | +5.64815 [+3.14815, +8.70370] |
| BNCI2014-001 | classifier_vs_frozen | 3.24074 | 2.12963 | +1.11111 [-0.46296, +2.96296] |
| BNCI2014-001 | spatial_vs_classifier | 4.16667 | 3.98148 | +0.18519 [-1.94444, +2.12963] |

## Neural grid

Pan bias and stack both BA=63.35714286%; D=2.34523810pp and11.41666667pp; D ratio=4.86802030. Equal observed means do not establish equivalence.

| Reference | Metric | Median | Q25 | Q75 |
|---|---|---:|---:|---:|
| frozen | D_mean | 0.0952380952 | 0.0480300713 | 0.1279464286 |
| frozen | G_mean | 0.0067857143 | 0.0038620931 | 0.0095634921 |
| frozen | within_ratio | 0.8442211945 | 0.7614128032 | 0.8769554021 |
| frozen | between_ratio | 0.0706597573 | 0.0183429442 | 0.1334919150 |
| frozen | total_ratio | 0.9302342990 | 0.8779983698 | 0.9414432196 |
| support_ea | D_mean | 0.0808730159 | 0.0126205946 | 0.1220734127 |
| support_ea | G_mean | 0.0001190476 | -0.0036515626 | 0.0015178571 |
| support_ea | within_ratio | 0.8253787674 | 0.6747114943 | 0.8810505812 |
| support_ea | between_ratio | 0.1132381423 | 0.0790207433 | 0.1577793766 |
| support_ea | total_ratio | 0.9535257185 | 0.8411628106 | 0.9679142606 |

## Weibo alignment minus frozen across acquired positions

| Budget | BA candidate % | BA reference % | G pp [CI] | Worse/better |
|---|---:|---:|---|---|
| 4 | 66.92063 | 71.53968 | -4.61905 [-7.97619, -1.33333] | 8/2 |
| 8 | 69.99603 | 71.53968 | -1.54365 [-4.14286, +1.00000] | 5/5 |
| 12 | 70.90873 | 71.53968 | -0.63095 [-2.32143, +0.91667] | 5/4 |
| 16 | 71.46032 | 71.53968 | -0.07937 [-1.35714, +1.09524] | 4/5 |
| 20 | 71.91270 | 71.53968 | +0.37302 [-1.00794, +1.64286] | 4/6 |

## Separate source-model replacement extension

Dreyer85: C=18.86274510pp; I=13.00980392pp; D=31.87254902pp; G=+5.85294118pp; total cancellation=81.63641956%. This is source-model replacement with0target calibration labels; source-trained contrasts are not pooled with target recalibration effects.
