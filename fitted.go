package forecast

import (
	"math"
	"time"

	"github.com/eduard-kolotushin/timeseries"
)

// Fitted is a model trained on a series, ready to produce future points.
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

// windowK returns inclusive 1-based k bounds for grid points in [from, to].
// It rejects a window wider than MaxForecastPoints before any allocation.
func (f pointForecast) windowK(from, to time.Time) (int, int, error) {
	if f.step <= 0 {
		return 0, 0, ErrNoFrequency
	}
	if from.After(to) {
		return 0, 0, ErrRange
	}
	d := to.Sub(f.lastTime)
	if d == maxDuration {
		// Sub saturates about 292 years out, so the real distance is unknown:
		// reject the horizon instead of emitting points from a clamped one.
		return 0, 0, ErrTooManyPoints
	}
	k0 := int(ceilDuration(from.Sub(f.lastTime), f.step))
	if k0 < 1 {
		k0 = 1
	}
	k1 := int(d / f.step)
	if err := f.checkWindow(k0, k1); err != nil {
		return 0, 0, err
	}
	return k0, k1, nil
}

func ceilDuration(num, den time.Duration) int64 {
	if den <= 0 {
		return 0
	}
	if num <= 0 {
		return 0
	}
	// Divide before adding: int64(num)+int64(den)-1 wraps for a num in the
	// step-wide band below maxDuration, which is exactly where windowK may be
	// asked for a legitimate point (the saturated distance itself is rejected).
	q := int64(num) / int64(den)
	if int64(num)%int64(den) != 0 {
		q++
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
		return f.season[(k-1)%f.period]
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
