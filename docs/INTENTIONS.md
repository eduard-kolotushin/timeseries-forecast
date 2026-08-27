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
| Core series library | `github.com/eduard-kolotushin/timeseries` (tagged module; no `replace`) |
| Horizon clock | last timestamp + `k * step` for `k ≥ 1`. `Forecast(h)` emits `k = 1..h`. `ForecastRange(from, to)` emits grid points with `from ≤ t ≤ to` (skip earlier `k`; no backcast before `last+step`; no off-grid times). |
| Calendars | Optional, file-embedded production calendars in this package (RU first); default off. Not a Series concern. |

## v1 must-have

- Fit/forecast API (`Fitted.Forecast(h)`)
- Models: naive, mean, drift, seasonal naive, seasonal baseline (hour / day / hour-of-week / minute-of-week), simple exponential smoothing, Holt
- Optional production calendars (embedded files, RU first; default off) for workday / weekend / holiday classes
- Infer `step` from the last two observations
- Holdout evaluation
- Metrics: MAE, RMSE, MAPE

## v2 must-have

- Prediction intervals for every v1 model (`Fitted.ForecastInterval(h, level)`)
- Gaussian two-sided bands from 1-step residual σ and Hyndman horizon widening
- σ is a scalar (or per-bucket for seasonal baseline); do not keep the residual vector
- Coverage `level` in `(0, 1)`; undefined bounds are `math.NaN()`

## v3 must-have

- `Fitted.ForecastRange(from, to)` and `ForecastIntervalRange(from, to, level)`: emit the `last+k×step` grid inside `[from, to]`
- Skip to the first `k ≥ 1` with `t ≥ from` in O(1); fill only window points (O(h_window), not a walk of skipped `k`)
- Empty window (inverted range, shorter than `step`, or entirely before `last+step`) fails with a sentinel error

## v4 must-have

- `SnapshotOf` / `Restore`: versioned envelope `{v, kind, last, step, data}` so a `Fitted` can be persisted and `ForecastRange` run without the training series
- Explicit fitted state (not closures). `at(k)` / `se(k)` stay O(1) for v1 models
- Calendar in baseline `data` by name (`""` / `"ru"`); `Restore` calls `CalendarByName`
- Unknown `v` or `kind` is an error

## v1/v2/v3/v4 non-goals

Do not add these without first updating this document:

- ARIMA / SARIMA / ETS full Hyndman catalog
- Prophet-style decompositions
- Machine learning / neural forecasters
- Multivariate / panel models
- Quantile forecasts, bootstrap, or conformal intervals
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
- Forecast range must contain at least one grid point (`last+k×step`)
- Interval coverage `level` in `(0, 1)`
- Smoothing parameters in `(0, 1]`
- Table-driven tests for each model, metrics, and interval golden paths
- GitHub Actions on `main` runs `gofmt` and `go test ./...`
