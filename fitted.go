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
	times := make([]time.Time, n)
	values := make([]float64, n)
	t := f.lastTime.Add(time.Duration(k0) * f.step)
	for i := range n {
		times[i] = t
		values[i] = f.at(k0 + i)
		t = t.Add(f.step)
	}
	// timeseries.New validates and copies both slices again: two extra
	// allocations the call cannot skip while Series' index fields are
	// unexported and no owned-slice constructor is exported.
	return timeseries.New(times, values)
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
	times := make([]time.Time, n)
	lo := make([]float64, n)
	hi := make([]float64, n)
	t := f.lastTime.Add(time.Duration(k0) * f.step)
	for i := range n {
		k := k0 + i
		times[i] = t
		pt := f.at(k)
		s := f.se(k)
		if math.IsNaN(s) {
			lo[i] = math.NaN()
			hi[i] = math.NaN()
		} else {
			lo[i] = pt - z*s
			hi[i] = pt + z*s
		}
		t = t.Add(f.step)
	}
	// Same re-validation and copy as forecastK, once per bound.
	lower, err := timeseries.New(times, lo)
	if err != nil {
		return timeseries.Series[float64]{}, timeseries.Series[float64]{}, err
	}
	upper, err := timeseries.New(times, hi)
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
	return (int64(num) + int64(den) - 1) / int64(den)
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
		return f.sigma * math.Sqrt(float64((k-1)/f.period+1))
	case kindSES:
		return f.sigma * math.Sqrt(1+f.alpha*f.alpha*float64(k-1))
	case kindHolt:
		h := float64(k)
		return f.sigma * math.Sqrt(1+(h-1)*(f.alpha*f.alpha+f.alpha*f.beta*h+h*(h-1)*f.beta*f.beta/6))
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
