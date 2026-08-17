# Architecture

## Layout

Single package `forecast`:

| File | Responsibility |
| --- | --- |
| `fitted.go` | `Fitted` interface, horizon timestamps, prepare/DropNA |
| `naive.go` | Last-value and mean and drift |
| `seasonal.go` | Seasonal naive |
| `baseline.go` | Seasonal baseline (hour / day / hour-of-week / minute-of-week means) |
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
    ForecastInterval(h int, level float64) (timeseries.Series[float64], timeseries.Series[float64], error)
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

## Prediction intervals

`ForecastInterval(h, level)` returns Gaussian two-sided bounds `at(k) ± z·se(k)` with `z = √2 · erfinv(level)` and `level ∈ (0, 1)`. σ is the MLE 1-step residual scale `sqrt(SSE / n_resid)`. Bounds are `NaN` when `n_resid < 2` (or a baseline bucket with `n < 2`). `se(k)` is O(1):

- Naive: `σ √h` (σ from first differences)
- Mean: `σ √(1 + 1/n)`
- Drift: `σ √[h (1 + h/n)]` (σ from `y_t − y_{t−1} − b`)
- Seasonal naive: `σ √(k+1)` with `k = floor((h−1)/m)`
- SES: `σ √[1 + α²(h−1)]` (residual `y_t −` previous level)
- Holt: `σ √[1 + (h−1)(α² + αβh + h(h−1)β²/6)]` (residual `y_t −` previous level+trend)
- Seasonal baseline: per-bucket residual sd from one-pass `sum`/`sumsq`, `σ_b √(1 + 1/n_b)` at the future timestamp’s key, same fallback chain as the mean

Do not keep the residual vector. `Forecast(h)` stays the point series.

## Performance

Optimize computation first: work per observation at fit, work per horizon step at forecast.

- Copy `Times()` / `Values()` once at fit; work on those slices
- Fit is one O(n) pass (SES, Holt, mean, drift, naive σ, seasonal baseline including `sumsq`)
- Seasonal baseline keeps a pre-sized means table filled at fit. Hour/day: holiday → weekend → workday → overall. Hour-of-week: (class, weekday, hour), else that class+weekday mean, else overall. Minute-of-week: (class, weekday, minute of day), else that class+weekday mean, else overall. Sunday does not copy Saturday; working Saturday is not weekend Saturday; holidays fall back by hour or minute of day
- Fitted models keep only forecast state (level/trend/season/last/means/σ or per-bucket se), not the training series
- `Forecast` and `ForecastInterval` are each one O(h) loop; `at(k)` and `se(k)` are O(1)
- Forecast allocates exactly `h` times and `h` values; intervals allocate two series of length `h`
- Do not clone the training series unless a public helper needs an independent copy

## Testing

Table-driven tests beside each model. Cover empty input, invalid parameters, a numeric golden path per model, and interval bounds (`h=1`, a longer `h`, invalid `level`, NaN when σ is undefined).
