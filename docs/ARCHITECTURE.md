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

- Copy `Times()` / `Values()` once at fit; work on those slices
- Forecast allocates exactly `h` times and `h` values
- SES/Holt are a single O(n) pass
- Do not clone the training series unless a public helper needs an independent copy

## Testing

Table-driven tests beside each model. Cover empty input, invalid parameters, and a numeric golden path per model.
