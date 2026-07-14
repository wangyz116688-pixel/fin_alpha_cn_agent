# Quant Agent Validation Report

Generated: 2026-07-14

## Data Used

- Training feature cache: `data/train_cache/trainset_20240101_20251231_s800_lb90.pkl`
  - 374,563 samples
  - 23 factors
  - 484 trading dates
- Historical K-line cache: `data/train_cache/*_20230824_20251231.pkl`
  - Used for 2025 H1 and 2025 Q4 replay tests
- Recent K-line cache: `data/prefetch_cache`
  - Used for 2026-05-15 to 2026-06-15 and 2026-06-17 to 2026-07-13 replay tests

## Factor Findings

Training-period IC from 2024-2025 shows that the strongest stable directions are not the same as the best short-window 2026 directions.

Top positive training-period factors:

- `rev_1d`
- `std_vol_10`
- `std_close_10`
- `decay_vol_5`

Strong negative training-period factors:

- `turn_5d_avg`
- `mom_20d`
- `vol_20d`
- `vwap_dev`
- `zscore_vol_5`

The short-window `ic_blend` worked well in 2026, but performed poorly in 2025 Q4, so it is not safe as the default strategy.

## Strategy Comparison

| Scenario | Period | Days | Return | Max Drawdown |
|---|---:|---:|---:|---:|
| XGBoost equal | 2025 H1 | 116 | +160.63% | 6.68% |
| Ensemble equal | 2025 H1 | 116 | +45.92% | 11.14% |
| XGBoost equal | 2025 Q4 | 59 | +19.44% | 5.82% |
| IC blend equal | 2025 Q4 | 59 | +0.37% | 23.88% |
| Train-IC blend equal | 2025 Q4 | 59 | +5.74% | 4.57% |
| Ensemble equal | 2025 Q4 | 59 | +7.53% | 7.31% |
| XGBoost equal, v2 cache | 2026-05-18 to 2026-06-15 | 21 | -2.42% | 6.77% |
| IC blend equal, v2 cache | 2026-05-18 to 2026-06-15 | 21 | +14.16% | 9.13% |
| Ensemble equal, v2 cache | 2026-05-18 to 2026-06-15 | 21 | +6.43% | 12.60% |
| XGBoost equal | 2026-06-17 to 2026-07-13, 500 stocks | 18 | +0.43% | 5.71% |
| IC blend equal | 2026-06-17 to 2026-07-13, 500 stocks | 18 | -3.58% | 14.68% |
| Train-IC blend equal | 2026-06-17 to 2026-07-13, 500 stocks | 18 | -0.25% | 4.73% |
| Ensemble equal | 2026-06-17 to 2026-07-13, 500 stocks | 18 | -2.74% | 16.10% |
| Adaptive blend, no risk stop | 2026-06-17 to 2026-07-13, 500 stocks | 18 | -13.38% | 21.59% |
| Adaptive blend + 8% risk stop | 2026-06-17 to 2026-07-13, 500 stocks | 18 | +0.49% | 9.04% |
| Adaptive blend + 8% risk stop | 2026-05-18 to 2026-06-15, v2 cache | 21 | +10.12% | 5.20% |
| Adaptive blend + 8% risk stop | 2025 Q4, 500 stocks | 59 | +16.39% | 5.41% |
| Ensemble blend + 8% risk stop | 2026-06-17 to 2026-07-13, 500 stocks | 18 | +5.45% | 9.04% |
| Ensemble blend + 8% risk stop | 2026-05-18 to 2026-06-15, v2 cache | 21 | -5.06% | 10.13% |
| IC blend + score allocation | 2026-05-18 to 2026-07-13, 500 stocks | 40 | +0.99% | 14.72% |
| Train-IC + score allocation, 5 positions | 2026-05-18 to 2026-07-13, 500 stocks | 40 | -0.23% | 6.80% |
| Train-IC + score allocation, 3 positions | 2026-05-18 to 2026-07-13, 500 stocks | 40 | +1.74% | 5.13% |
| Train-IC + score allocation, 3 positions | 2026-05-18 to 2026-06-15, 500 stocks | 21 | +0.28% | 4.80% |
| Train-IC + score allocation, 3 positions | 2026-06-17 to 2026-07-13, 500 stocks | 18 | +1.80% | 5.13% |
| Train-IC + score allocation, 3 positions | 2024 full year, 500 stocks | 241 | +17.77% | 21.93% |
| Train-IC + score allocation, 3 positions | 2025 full year, 500 stocks | 242 | +19.75% | 7.11% |
| Train-IC + score allocation, 3 positions | 2025 Q4, 500 stocks | 59 | +6.28% | 4.02% |

## Current Recommendation

Keep all ranking methods available, but use `train_ic_blend` with score-weighted allocation and 3 positions as the default for further testing.

The expanded 2026-05-18 to 2026-07-13 test exposed that the prior adaptive default was not robust. It over-switched between methods and produced -6.20% on the 40-day, 500-stock cache-covered sample.

The best cross-window configuration found in this round is:

- score method: `train_ic_blend`
- allocation: `score`
- positions: `3`
- drawdown stop: disabled by default (`0`)

This setting was positive across the tested 2026 long window, both 2026 sub-windows, 2024 full year, 2025 full year, and 2025 Q4. It has lower drawdowns than the high-return but unstable IC blend variants in most windows, but 2024 still shows a high 21.93% maximum drawdown and should be presented as the main remaining risk.

The current recommended production-style command is:


```powershell
.\venv\Scripts\python.exe -m src.backtest_stock_agent --start 20260515 --end 20260713 --offline --cache-dir data\prefetch_cache --factor-cache-dir data\factor_cache --no-plot --quiet-advice --max-stocks 500
```

The default method is now equivalent to:

```powershell
--score-method train_ic_blend --allocation-method score --position-count 3 --max-drawdown-stop 0
```

## Notes

- 2025 H1 and 2025 Q4 are inside or near the model training period, so XGBoost results there should be treated as in-sample diagnostics, not proof of live performance.
- The 2026 windows are better out-of-sample checks, but they are still short. More 2026 data should be fetched and tested before treating adaptive selection as final.
- Factor cache version was bumped to `v2` after fixing stale partial-cache behavior. Old `factors_*.pkl` files should not be used for final conclusions.
- The offline cache loader now handles non-trading start/end dates correctly. This fixed the 2024 full-year replay, where `2024-01-01` is not a trading day.
- The risk stop remains available, but it is no longer the default. On the expanded 2026 test it often stopped after the damage and missed the rebound.
- A `--max-stock-price` filter was added for experimentation, but the 200 CNY test worsened results and should not be enabled by default.
