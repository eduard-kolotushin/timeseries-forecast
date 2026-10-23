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
| `snapshot.go` | `Snapshot` envelope, `SnapshotOf` / `Restore` |
| `errors.go` | Sentinel errors |
| `limits.go` | `MaxForecastPoints` and the shared emitted-window check |

## API

```go
type Fitted interface {
    Forecast(h int) (timeseries.Series[float64], error)
    ForecastInterval(h int, level float64) (timeseries.Series[float64], timeseries.Series[float64], error)
    ForecastRange(from, to time.Time) (timeseries.Series[float64], error)
    ForecastIntervalRange(from, to time.Time, level float64) (timeseries.Series[float64], timeseries.Series[float64], error)
}

func FitNaive(s timeseries.Series[float64]) (Fitted, error)
func FitMean(s timeseries.Series[float64]) (Fitted, error)
func FitDrift(s timeseries.Series[float64]) (Fitted, error)
func FitSeasonalNaive(s timeseries.Series[float64], period int) (Fitted, error)
func FitSeasonalBaseline(s timeseries.Series[float64], season Seasonality, cal *Calendar) (Fitted, error)
func FitSES(s timeseries.Series[float64], alpha float64) (Fitted, error)
func FitHolt(s timeseries.Series[float64], alpha, beta float64) (Fitted, error)

type Snapshot struct { V int; Kind string; Last, Step int64; Data json.RawMessage }
func SnapshotOf(f Fitted) (Snapshot, error)
func Restore(s Snapshot) (Fitted, error)
```

`cal == nil` means calendar off (UTC, weekend = Sat/Sun). `CalendarByName("ru")` loads the embedded Russian production calendar (`Europe/Moscow`). A timestamp whose civil year is not in the file is classified as calendar off; holiday rules and the calendar timezone do not apply to other years. The embedded table covers 2026 (one row per year in `calendars/ru.csv`, which is how the feature is maintained — a year missing from the file silently classifies as calendar off, and `TestEmbeddedRUCalendarCoversTheCurrentYear` fails once the newest row is older than the current year). Only the built-in calendars survive a snapshot: `SnapshotOf` refuses a baseline fitted with a caller-built table, because `Restore` can only map `CalendarByName` names back to a `*Calendar`.

Callers use the public `timeseries` API only (`Times`, `Values`, `New`, `DropNA`, `SliceIndex`, `AlignFloat`). Do not reach into unexported Series fields.

## Horizon

Forecast `h` is the number of future points. Timestamps are `last.Add(k*step)` for `k=1..h`. `step` is `times[n-1]-times[n-2]` after DropNA.

`ForecastRange(from, to)` uses the same grid for any integer `k`: `k ≥ 1` is the forecast horizon, `k ≤ 0` evaluates the final fitted state at or before the last training timestamp (a backcast). Values come from the fitted state alone (`last`, `mu`, `last+k·b`, the seasonal phase, the baseline's per-key mean, `level`, `level+k·trend`), never from the training series. Bands at `k ≤ 0` exist only for the seasonal baseline (its per-key SE is looked up by timestamp); every other model returns `NaN` there because its interval is a horizon widening.

The range finds the first `k` with `t_k ≥ from` and the last with `t_k ≤ to` in O(1): `k0 = ceil((from-last)/step)`, `k1 = floor((to-last)/step)`, both signed (Go's truncating division already ceils a negative numerator; the floor rounds a negative remainder down), and fills only that slice. `Forecast(h)` is the range `(last+step) .. (last+h×step)`. An inverted range returns `ErrRange`; a window with no grid point (e.g. entirely between two steps) returns `ErrEmptyRange`.

A window may emit at most `MaxForecastPoints` (1 000 000) points. A wider one — or a bound so distant that `Sub` saturates (about 292 years, on either side) — returns `ErrTooManyPoints` before any slice is allocated. All four entry points (`Forecast`, `ForecastInterval`, `ForecastRange`, `ForecastIntervalRange`) go through the same check, which bounds the point count and `|k|×step` for both signs, so no request can turn a range into an unbounded response.

## Prediction intervals

`ForecastInterval(h, level)` returns Gaussian two-sided bounds `at(k) ± z·se(k)` with `z = √2 · erfinv(level)` and `level ∈ (0, 1)`. σ is the MLE 1-step residual scale `sqrt(SSE / n_resid)`. Bounds are `NaN` when `n_resid < 2` (or a baseline bucket with `n < 2`). `se(k)` is O(1):

- Naive: `σ √h` (σ from first differences)
- Mean: `σ √(1 + 1/n)`
- Drift: `σ √[h (1 + h/n)]` (σ from `y_t − y_{t−1} − b`)
- Seasonal naive: `σ √(k+1)` with `k = floor((h−1)/m)`
- SES: `σ √[1 + α²(h−1)]` (residual `y_t −` previous level)
- Holt: `σ √[1 + α²(h−1)(1 + βh + β²·h(2h−1)/6)]` (trend moved by the level increment, `b_t = b_{t−1} + αβ·e_t`; residual `y_t −` previous level+trend). The trend is seeded from the first two points, so the first residual is zero by construction and is not counted: `n_resid` starts at the second one-step error
- Seasonal baseline: per-bucket residual sd from Welford state (`n`, running mean, `m2`), `σ_b √(1 + 1/n_b)` at the future timestamp’s key, same fallback chain as the mean. Welford, not `sum`/`sumsq`: a raw moment cancels away the whole variance when the values share a large offset (the minute-of-week buckets see few samples each)

Do not keep the residual vector. `Forecast(h)` stays the point series.

## Performance

Optimize computation first: work per observation at fit, work per horizon step at forecast.

- Copy `Times()` / `Values()` once at fit; work on those slices
- Fit is one O(n) pass (SES, Holt, mean, drift, naive σ, seasonal baseline including Welford `m2`)
- Seasonal baseline keeps a pre-sized means table filled at fit. Hour/day: holiday → weekend → workday → overall. Hour-of-week: (class, weekday, hour), else that class+weekday mean, else overall. Minute-of-week: (class, weekday, minute of day), else that class+weekday mean, else overall. Sunday does not copy Saturday; working Saturday is not weekend Saturday; holidays fall back by hour or minute of day
- Fitted models keep only forecast state (level/trend/season/last/means/σ or per-bucket se), not the training series. That state is what `SnapshotOf` encodes.
- `Forecast`, `ForecastInterval`, `ForecastRange`, and `ForecastIntervalRange` each fill one O(h) loop over emitted points; `at(k)` and `se(k)` are O(1)
- `Forecast` builds its grid with `timeseries.FromPoints` in one pass — one points slice, no second copy through `New`; intervals allocate two series of the emitted length
- Do not clone the training series unless a public helper needs an independent copy

## Testing

Table-driven tests beside each model. Cover empty input, invalid parameters, a numeric golden path per model, and interval bounds (`h=1`, a longer `h`, invalid `level`, NaN when σ is undefined).
