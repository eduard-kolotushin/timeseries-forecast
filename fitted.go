package forecast

import (
	"math"
	"time"

	"github.com/eduard-kolotushin/timeseries"
)

// Fitted is a model trained on a series. Forecast and ForecastInterval emit the
// horizon last+k*step for k = 1..h; the Range methods evaluate the fitted state
// at every grid point last+k*step inside [from, to] for any integer k, so
// k <= 0 is a backcast of the final state (the baseline also has bands there).
type Fitted interface {
	Forecast(h int) (timeseries.Series[float64], error)
	ForecastInterval(h int, level float64) (timeseries.Series[float64], timeseries.Series[float64], error)
	ForecastRange(from, to time.Time) (timeseries.Series[float64], error)
	ForecastIntervalRange(from, to time.Time, level float64) (timeseries.Series[float64], timeseries.Series[float64], error)
}

type pointForecast struct {
	lastTime time.Time
	step     time.Duration
	kind     string

	last, mu, b, level, trend, sigma, alpha, beta float64
	n, period                                     int

	season      []float64
	means, ses  []float64
	seasonality Seasonality
	cal         *Calendar
}

func (f pointForecast) Forecast(h int) (timeseries.Series[float64], error) {
	if h <= 0 {
		return timeseries.Series[float64]{}, ErrHorizon
	}
	if f.step <= 0 {
		return timeseries.Series[float64]{}, ErrNoFrequency
	}
	if err := f.checkWindow(1, h); err != nil {
		return timeseries.Series[float64]{}, err
	}
	return f.forecastK(1, h)
}

func (f pointForecast) ForecastInterval(h int, level float64) (timeseries.Series[float64], timeseries.Series[float64], error) {
	if h <= 0 {
		return timeseries.Series[float64]{}, timeseries.Series[float64]{}, ErrHorizon
	}
	if f.step <= 0 {
		return timeseries.Series[float64]{}, timeseries.Series[float64]{}, ErrNoFrequency
	}
	z, err := intervalZ(level)
	if err != nil {
		return timeseries.Series[float64]{}, timeseries.Series[float64]{}, err
	}
	if err := f.checkWindow(1, h); err != nil {
		return timeseries.Series[float64]{}, timeseries.Series[float64]{}, err
	}
	return f.intervalK(1, h, z)
}

func (f pointForecast) ForecastRange(from, to time.Time) (timeseries.Series[float64], error) {
	k0, k1, err := f.windowK(from, to)
	if err != nil {
		return timeseries.Series[float64]{}, err
	}
	return f.forecastK(k0, k1)
}

// forecastK emits the grid points k0..k1 (inclusive). The caller has validated
// the window, so the emitted length is bounded by MaxForecastPoints.
func (f pointForecast) forecastK(k0, k1 int) (timeseries.Series[float64], error) {
	n := k1 - k0 + 1
	points := make([]timeseries.Point[float64], n)
	t := f.lastTime.Add(time.Duration(k0) * f.step)
	for i := range n {
		points[i] = timeseries.Point[float64]{Time: t, Value: f.at(k0 + i)}
		t = t.Add(f.step)
	}
	// FromPoints builds the index it validates, so the grid is materialized once.
	// The timestamps ascend by step (checkWindow rejects step <= 0), so the error
	// branch is unreachable.
	s, err := timeseries.FromPoints(points)
	if err != nil {
		return timeseries.Series[float64]{}, err
	}
	return s, nil
}

func (f pointForecast) ForecastIntervalRange(from, to time.Time, level float64) (timeseries.Series[float64], timeseries.Series[float64], error) {
	z, err := intervalZ(level)
	if err != nil {
		return timeseries.Series[float64]{}, timeseries.Series[float64]{}, err
	}
	k0, k1, err := f.windowK(from, to)
	if err != nil {
		return timeseries.Series[float64]{}, timeseries.Series[float64]{}, err
	}
	return f.intervalK(k0, k1, z)
}

// intervalK emits the Gaussian band for the grid points k0..k1 (inclusive). The
// caller has validated the window and the level.
func (f pointForecast) intervalK(k0, k1 int, z float64) (timeseries.Series[float64], timeseries.Series[float64], error) {
	n := k1 - k0 + 1
	lo := make([]timeseries.Point[float64], n)
	hi := make([]timeseries.Point[float64], n)
	t := f.lastTime.Add(time.Duration(k0) * f.step)
	for i := range n {
		k := k0 + i
		pt := f.at(k)
		s := f.se(k)
		lo[i].Time, hi[i].Time = t, t
		if math.IsNaN(s) {
			lo[i].Value, hi[i].Value = math.NaN(), math.NaN()
		} else {
			lo[i].Value = pt - z*s
			hi[i].Value = pt + z*s
		}
		t = t.Add(f.step)
	}
	// One pass per bound, as in forecastK, with the same unreachable error.
	lower, err := timeseries.FromPoints(lo)
	if err != nil {
		return timeseries.Series[float64]{}, timeseries.Series[float64]{}, err
	}
	upper, err := timeseries.FromPoints(hi)
	if err != nil {
		return timeseries.Series[float64]{}, timeseries.Series[float64]{}, err
	}
	return lower, upper, nil
}

// WindowBounds returns the inclusive signed k bounds of the grid points
// lastTime + k*step inside [from, to]: k0 = ceil((from-last)/step),
// k1 = floor((to-last)/step). k >= 1 is the forecast horizon and k <= 0
// evaluates the final fitted state (a backcast); off-grid times are not
// emitted. It is the computation ForecastRange applies, exported so a caller
// that must refuse an oversized window before fitting or restoring has one
// implementation to call instead of mirroring the arithmetic.
//
// Errors: ErrNoFrequency for step <= 0, ErrRange for an inverted window,
// ErrEmptyRange when no grid point falls inside, and ErrTooManyPoints when the
// window would emit more than MaxForecastPoints points or a bound is so
// distant that time.Time.Sub saturates (about 292 years either side) or
// |k|*step leaves a time.Duration.
func WindowBounds(lastTime time.Time, step time.Duration, from, to time.Time) (int, int, error) {
	if step <= 0 {
		return 0, 0, ErrNoFrequency
	}
	if from.After(to) {
		return 0, 0, ErrRange
	}
	lo := from.Sub(lastTime)
	hi := to.Sub(lastTime)
	// Sub saturates about 292 years out on either side, so the real distance
	// is unknown: reject instead of emitting points from a clamped one.
	if hi == maxDuration || lo == minDuration {
		return 0, 0, ErrTooManyPoints
	}
	k0 := int(ceilDuration(lo, step))
	k1 := int(floorDuration(hi, step))
	if err := checkKBounds(step, k0, k1); err != nil {
		return 0, 0, err
	}
	return k0, k1, nil
}

// windowK applies the exported grid arithmetic to this fitted model.
func (f pointForecast) windowK(from, to time.Time) (int, int, error) {
	return WindowBounds(f.lastTime, f.step, from, to)
}

// ceilDuration returns ceil(num/den) for den > 0, exact for negative num too:
// Go's truncating division already ceils a negative num with a remainder, and a
// positive one rounds up. Divide before any arithmetic: int64(num)+int64(den)-1
// wraps in the step-wide band below maxDuration.
func ceilDuration(num, den time.Duration) int64 {
	q := int64(num) / int64(den)
	if int64(num) > 0 && int64(num)%int64(den) != 0 {
		q++
	}
	return q
}

// floorDuration returns floor(num/den) for den > 0: a negative num with a
// remainder rounds down instead of truncating toward zero.
func floorDuration(num, den time.Duration) int64 {
	q := int64(num) / int64(den)
	if int64(num) < 0 && int64(num)%int64(den) != 0 {
		q--
	}
	return q
}

func intervalZ(level float64) (float64, error) {
	if level <= 0 || level >= 1 || math.IsNaN(level) {
		return 0, ErrInvalidLevel
	}
	return math.Sqrt2 * math.Erfinv(level), nil
}

func mleSigma(sse float64, nResid int) float64 {
	if nResid < 2 {
		return math.NaN()
	}
	return math.Sqrt(sse / float64(nResid))
}

func (f pointForecast) at(k int) float64 {
	switch f.kind {
	case kindNaive:
		return f.last
	case kindMean:
		return f.mu
	case kindDrift:
		return f.last + float64(k)*f.b
	case kindSeasonal:
		if f.period <= 0 || len(f.season) == 0 {
			return math.NaN()
		}
		idx := (k - 1) % f.period
		if idx < 0 {
			idx += f.period
		}
		return f.season[idx]
	case kindBaseline:
		t := f.lastTime.Add(time.Duration(k) * f.step)
		local := t.In(zoneFor(f.cal, t))
		key := seasonKey(f.seasonality, f.cal.Classify(t), seasonSlot(f.seasonality, local), local.Weekday())
		if key < 0 || key >= len(f.means) {
			return math.NaN()
		}
		return f.means[key]
	case kindSES:
		return f.level
	case kindHolt:
		return f.level + float64(k)*f.trend
	default:
		return math.NaN()
	}
}

func (f pointForecast) se(k int) float64 {
	if f.kind == kindBaseline {
		t := f.lastTime.Add(time.Duration(k) * f.step)
		local := t.In(zoneFor(f.cal, t))
		key := seasonKey(f.seasonality, f.cal.Classify(t), seasonSlot(f.seasonality, local), local.Weekday())
		if key < 0 || key >= len(f.ses) {
			return math.NaN()
		}
		return f.ses[key]
	}
	if k <= 0 {
		// Backcast bands: only the baseline stores a per-key standard error.
		// Every other model's interval is a horizon widening undefined at k <= 0.
		return math.NaN()
	}
	if math.IsNaN(f.sigma) {
		return math.NaN()
	}
	switch f.kind {
	case kindNaive:
		return f.sigma * math.Sqrt(float64(k))
	case kindMean:
		if f.n <= 0 {
			return math.NaN()
		}
		return f.sigma * math.Sqrt(1+1/float64(f.n))
	case kindDrift:
		if f.n <= 0 {
			return math.NaN()
		}
		h := float64(k)
		return f.sigma * math.Sqrt(h*(1+h/float64(f.n)))
	case kindSeasonal:
		if f.period <= 0 || len(f.season) == 0 {
			// The same unrepresentable state at() reports as NaN; without this
			// guard the integer division below would panic.
			return math.NaN()
		}
		return f.sigma * math.Sqrt(float64((k-1)/f.period+1))
	case kindSES:
		return f.sigma * math.Sqrt(1+f.alpha*f.alpha*float64(k-1))
	case kindHolt:
		h := float64(k)
		// holt.go moves the trend by the same step's level increment:
		// b_t = b_{t−1} + αβ·e_t, so rolling the recursion out gives
		// ŷ_{t+h} = l_t + h·b_t + e_{t+h} + Σ_{j=1..h-1} α(1+βj)·e_{t+h-j}, hence
		// Var = σ²[1 + α² Σ(1+βj)²] = σ²[1 + α²(h−1)(1 + βh + β²·h(2h−1)/6)].
		return f.sigma * math.Sqrt(1+f.alpha*f.alpha*(h-1)*(1+f.beta*h+f.beta*f.beta*h*(2*h-1)/6))
	default:
		return math.NaN()
	}
}

type prepared struct {
	times  []time.Time
	values []float64
	step   time.Duration
}

func prepare(s timeseries.Series[float64]) (prepared, error) {
	clean := timeseries.DropNA(s)
	if clean.Empty() {
		return prepared{}, ErrEmpty
	}
	p := prepared{
		times:  clean.Times(),
		values: clean.Values(),
	}
	if len(p.times) >= 2 {
		p.step = p.times[len(p.times)-1].Sub(p.times[len(p.times)-2])
		if p.step <= 0 {
			return prepared{}, ErrNoFrequency
		}
	}
	return p, nil
}

func (p prepared) requireStep() error {
	if p.step <= 0 {
		return ErrNoFrequency
	}
	return nil
}

func (p prepared) last() time.Time {
	return p.times[len(p.times)-1]
}
