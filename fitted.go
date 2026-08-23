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
	at       func(k int) float64 // k is 1-based horizon
	se       func(k int) float64 // O(1) standard error; NaN if undefined
}

func (f pointForecast) Forecast(h int) (timeseries.Series[float64], error) {
	if h <= 0 {
		return timeseries.Series[float64]{}, ErrHorizon
	}
	if f.step <= 0 {
		return timeseries.Series[float64]{}, ErrNoFrequency
	}
	from := f.lastTime.Add(f.step)
	to := f.lastTime.Add(time.Duration(h) * f.step)
	return f.ForecastRange(from, to)
}

func (f pointForecast) ForecastInterval(h int, level float64) (timeseries.Series[float64], timeseries.Series[float64], error) {
	if h <= 0 {
		return timeseries.Series[float64]{}, timeseries.Series[float64]{}, ErrHorizon
	}
	if f.step <= 0 {
		return timeseries.Series[float64]{}, timeseries.Series[float64]{}, ErrNoFrequency
	}
	from := f.lastTime.Add(f.step)
	to := f.lastTime.Add(time.Duration(h) * f.step)
	return f.ForecastIntervalRange(from, to, level)
}

func (f pointForecast) ForecastRange(from, to time.Time) (timeseries.Series[float64], error) {
	k0, k1, err := f.windowK(from, to)
	if err != nil {
		return timeseries.Series[float64]{}, err
	}
	n := k1 - k0 + 1
	times := make([]time.Time, n)
	values := make([]float64, n)
	for i := 0; i < n; i++ {
		k := k0 + i
		times[i] = f.lastTime.Add(time.Duration(k) * f.step)
		values[i] = f.at(k)
	}
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
	se := f.se
	if se == nil {
		se = nanSE
	}
	n := k1 - k0 + 1
	times := make([]time.Time, n)
	lo := make([]float64, n)
	hi := make([]float64, n)
	for i := 0; i < n; i++ {
		k := k0 + i
		times[i] = f.lastTime.Add(time.Duration(k) * f.step)
		pt := f.at(k)
		s := se(k)
		if math.IsNaN(s) {
			lo[i] = math.NaN()
			hi[i] = math.NaN()
			continue
		}
		lo[i] = pt - z*s
		hi[i] = pt + z*s
	}
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
func (f pointForecast) windowK(from, to time.Time) (int, int, error) {
	if f.step <= 0 {
		return 0, 0, ErrNoFrequency
	}
	if from.After(to) {
		return 0, 0, ErrRange
	}
	k0 := int(ceilDuration(from.Sub(f.lastTime), f.step))
	if k0 < 1 {
		k0 = 1
	}
	k1 := int(to.Sub(f.lastTime) / f.step)
	if k1 < k0 {
		return 0, 0, ErrEmptyRange
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

func nanSE(int) float64 { return math.NaN() }

func scaledSE(sigma float64, scale func(k int) float64) func(k int) float64 {
	if math.IsNaN(sigma) {
		return nanSE
	}
	return func(k int) float64 {
		return sigma * scale(k)
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
