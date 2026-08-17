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
	times := make([]time.Time, h)
	values := make([]float64, h)
	t := f.lastTime
	for k := 1; k <= h; k++ {
		t = t.Add(f.step)
		times[k-1] = t
		values[k-1] = f.at(k)
	}
	return timeseries.New(times, values)
}

func (f pointForecast) ForecastInterval(h int, level float64) (timeseries.Series[float64], timeseries.Series[float64], error) {
	z, err := intervalZ(level)
	if err != nil {
		return timeseries.Series[float64]{}, timeseries.Series[float64]{}, err
	}
	if h <= 0 {
		return timeseries.Series[float64]{}, timeseries.Series[float64]{}, ErrHorizon
	}
	if f.step <= 0 {
		return timeseries.Series[float64]{}, timeseries.Series[float64]{}, ErrNoFrequency
	}
	se := f.se
	if se == nil {
		se = nanSE
	}
	times := make([]time.Time, h)
	lo := make([]float64, h)
	hi := make([]float64, h)
	t := f.lastTime
	for k := 1; k <= h; k++ {
		t = t.Add(f.step)
		times[k-1] = t
		pt := f.at(k)
		s := se(k)
		if math.IsNaN(s) {
			lo[k-1] = math.NaN()
			hi[k-1] = math.NaN()
			continue
		}
		lo[k-1] = pt - z*s
		hi[k-1] = pt + z*s
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
