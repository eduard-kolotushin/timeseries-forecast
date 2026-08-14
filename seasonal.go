package forecast

import "github.com/eduard-kolotushin/timeseries"

// FitSeasonalNaive repeats the last `period` observations cyclically.
func FitSeasonalNaive(s timeseries.Series[float64], period int) (Fitted, error) {
	if period <= 0 {
		return nil, ErrInvalidPeriod
	}
	p, err := prepare(s)
	if err != nil {
		return nil, err
	}
	if err := p.requireStep(); err != nil {
		return nil, err
	}
	if len(p.values) < period {
		return nil, ErrTooShort
	}
	season := p.values[len(p.values)-period:]
	return pointForecast{
		lastTime: p.last(),
		step:     p.step,
		at: func(k int) float64 {
			return season[(k-1)%period]
		},
	}, nil
}
