package forecast

import (
	"github.com/eduard-kolotushin/timeseries"
)

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
	season := make([]float64, period)
	copy(season, p.values[len(p.values)-period:])
	var sse float64
	nResid := 0
	for i := period; i < len(p.values); i++ {
		r := p.values[i] - p.values[i-period]
		sse += r * r
		nResid++
	}
	sigma := mleSigma(sse, nResid)
	return pointForecast{
		lastTime: p.last(),
		step:     p.step,
		kind:     kindSeasonal,
		season:   season,
		period:   period,
		sigma:    sigma,
	}, nil
}
