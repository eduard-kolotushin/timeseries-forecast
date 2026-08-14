# Architecture

## Layout

Single package `forecast`:

| File | Responsibility |
| --- | --- |
| `fitted.go` | `Fitted` interface, horizon timestamps, prepare/DropNA |
| `naive.go` | Last-value and mean and drift |
| `seasonal.go` | Seasonal naive |
| `ses.go` | Simple exponential smoothing |
| `holt.go` | Holt linear trend |
| `metrics.go` | MAE, RMSE, MAPE |
| `evaluate.go` | Train/test split and holdout evaluate |
| `errors.go` | Sentinel errors |

## API

```go
type Fitted interface {
    Forecast(h int) (timeseries.Series[float64], error)
}

func FitNaive(s timeseries.Series[float64]) (Fitted, error)
func FitMean(s timeseries.Series[float64]) (Fitted, error)
func FitDrift(s timeseries.Series[float64]) (Fitted, error)
func FitSeasonalNaive(s timeseries.Series[float64], period int) (Fitted, error)
func FitSES(s timeseries.Series[float64], alpha float64) (Fitted, error)
func FitHolt(s timeseries.Series[float64], alpha, beta float64) (Fitted, error)
```

Callers use the public `timeseries` API only (`Times`, `Values`, `New`, `DropNA`, `SliceIndex`, `AlignFloat`). Do not reach into unexported Series fields.

## Horizon

Forecast `h` is the number of future points. Timestamps are `last.Add(k*step)` for `k=1..h`. `step` is `times[n-1]-times[n-2]` after DropNA.

## Performance

Optimize computation first: work per observation at fit, work per horizon step at forecast.

- Copy `Times()` / `Values()` once at fit; work on those slices
- Fit is one O(n) pass (SES, Holt, mean); drift and naive are O(1) after prepare
- Fitted models keep only forecast state (level/trend/season/last), not the training series
- `Forecast` is one O(h) loop; each step is O(1)
- Forecast allocates exactly `h` times and `h` values
- Do not clone the training series unless a public helper needs an independent copy

## Testing

Table-driven tests beside each model. Cover empty input, invalid parameters, and a numeric golden path per model.
