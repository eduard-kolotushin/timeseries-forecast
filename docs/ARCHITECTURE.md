# Architecture

## Layout

Single package `forecast`:

| File | Responsibility |
| --- | --- |
| `fitted.go` | `Fitted` interface, horizon timestamps, prepare/DropNA |
| `naive.go` | Last-value and mean and drift |
| `seasonal.go` | Seasonal naive |
| `baseline.go` | Seasonal baseline (hour / day / hour-of-week means) |
| `calendar.go` | Optional production calendars (RU via `go:embed`) |
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
func FitSeasonalBaseline(s timeseries.Series[float64], season Seasonality, cal *Calendar) (Fitted, error)
func FitSES(s timeseries.Series[float64], alpha float64) (Fitted, error)
func FitHolt(s timeseries.Series[float64], alpha, beta float64) (Fitted, error)
```

`cal == nil` means calendar off (UTC, weekend = Sat/Sun). `CalendarByName("ru")` loads the embedded Russian production calendar (`Europe/Moscow`). A timestamp whose civil year is not in the file is classified as calendar off; holiday rules and the calendar timezone do not apply to other years.

Callers use the public `timeseries` API only (`Times`, `Values`, `New`, `DropNA`, `SliceIndex`, `AlignFloat`). Do not reach into unexported Series fields.

## Horizon

Forecast `h` is the number of future points. Timestamps are `last.Add(k*step)` for `k=1..h`. `step` is `times[n-1]-times[n-2]` after DropNA.

## Performance

Optimize computation first: work per observation at fit, work per horizon step at forecast.

- Copy `Times()` / `Values()` once at fit; work on those slices
- Fit is one O(n) pass (SES, Holt, mean, seasonal baseline); drift and naive are O(1) after prepare
- Seasonal baseline keeps a pre-sized means table filled at fit. Hour/day: holiday → weekend → workday → overall. Hour-of-week: (class, weekday, hour), else that class+weekday mean, else overall (Sunday does not copy Saturday; working Saturday is not weekend Saturday; holidays fall back by hour)
- Fitted models keep only forecast state (level/trend/season/last/means), not the training series
- `Forecast` is one O(h) loop; each step is O(1)
- Forecast allocates exactly `h` times and `h` values
- Do not clone the training series unless a public helper needs an independent copy

## Testing

Table-driven tests beside each model. Cover empty input, invalid parameters, and a numeric golden path per model.
