# Project intentions

## Goal

Forecast future values of a univariate `timeseries.Series[float64]`. This package consumes the core timeseries library; it does not reimplement Series operations.

Implement models in an optimized way: one pass to fit, pre-sized forecast output, no extra copies of the training series.

## Locked choices

| Decision | Choice |
| --- | --- |
| Input/output | `timeseries.Series[float64]` |
| Module path | `github.com/eduard-kolotushin/timeseries-forecast` |
| Package name | `forecast` |
| Go version | 1.26+ |
| Core series library | sibling `github.com/eduard-kolotushin/timeseries` |
| Horizon clock | last timestamp + `k * step` |

## v1 must-have

- Fit/forecast API (`Fitted.Forecast(h)`)
- Models: naive, mean, drift, seasonal naive, simple exponential smoothing, Holt
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
- CSV/JSON I/O or plotting
- Duplicating Series ops that belong in `timeseries`

## Quality bar

- Fit does not mutate the input series
- Empty or all-NaN series fail with a sentinel error
- Horizon must be positive
- Smoothing parameters in `(0, 1]`
- Table-driven tests for each model and for metrics
