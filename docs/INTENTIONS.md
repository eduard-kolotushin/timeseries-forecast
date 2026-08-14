# Project intentions

## Goal

Forecast future values of a univariate `timeseries.Series[float64]`. This package consumes the core timeseries library; it does not reimplement Series operations.

Implement models in an **optimized** way: one pass to fit, O(1) work per horizon step, pre-sized forecast output, no extra copies of the training series. Correctness stays first; computation cost is a standing design constraint.

## Locked choices

| Decision | Choice |
| --- | --- |
| Input/output | `timeseries.Series[float64]` |
| Module path | `github.com/eduard-kolotushin/timeseries-forecast` |
| Package name | `forecast` |
| Go version | 1.26+ |
| Core series library | sibling `github.com/eduard-kolotushin/timeseries` |
| Horizon clock | last timestamp + `k * step` |
| Calendars | Optional, file-embedded production calendars in this package (RU first); default off. Not a Series concern. |

## v1 must-have

- Fit/forecast API (`Fitted.Forecast(h)`)
- Models: naive, mean, drift, seasonal naive, seasonal baseline (hour / day / hour-of-week), simple exponential smoothing, Holt
- Optional production calendars (embedded files, RU first; default off) for workday / weekend / holiday classes
- Infer `step` from the last two observations
- Holdout evaluation
- Metrics: MAE, RMSE, MAPE

## v1 non-goals

Do not add these without first updating this document:

- ARIMA / SARIMA / ETS full Hyndman catalog
- Prophet-style decompositions
- Machine learning / neural forecasters
- Multivariate / panel models
- Prediction intervals / quantile forecasts
- CSV/JSON I/O for series, or plotting (visualization lives in sibling `timeseries-grafana`)
- Business calendars in the core `timeseries` library
- Duplicating Series ops that belong in `timeseries`

## Performance aims

- Prefer **O(n)** fit and **O(h)** forecast over nested scans
- Fitted state is the recurrence result, not the training series
- `Forecast(k)` arithmetic is **O(1)** per step (no re-walk of history)
- Copy `Times()` / `Values()` once at fit; compute on those slices
- Allocation should track **horizon size**, not hidden intermediates
- Add or update benchmarks when changing a fit or forecast hot path

## Quality bar

- Fit does not mutate the input series
- Empty or all-NaN series fail with a sentinel error
- Horizon must be positive
- Smoothing parameters in `(0, 1]`
- Table-driven tests for each model and for metrics
