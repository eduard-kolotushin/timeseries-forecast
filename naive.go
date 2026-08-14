package forecast

import "github.com/eduard-kolotushin/timeseries"

// FitNaive repeats the last non-NaN observation.
func FitNaive(s timeseries.Series[float64]) (Fitted, error) {
	p, err := prepare(s)
	if err != nil {
		return nil, err
	}
	if err := p.requireStep(); err != nil {
		return nil, err
	}
	last := p.values[len(p.values)-1]
	return pointForecast{lastTime: p.last(), step: p.step, at: func(int) float64 { return last }}, nil
}

// FitMean repeats the mean of non-NaN observations.
func FitMean(s timeseries.Series[float64]) (Fitted, error) {
	p, err := prepare(s)
	if err != nil {
		return nil, err
	}
	if err := p.requireStep(); err != nil {
		return nil, err
	}
	sum := 0.0
	for _, v := range p.values {
		sum += v
	}
	mu := sum / float64(len(p.values))
	return pointForecast{lastTime: p.last(), step: p.step, at: func(int) float64 { return mu }}, nil
}

// FitDrift extrapolates a line from the first to the last non-NaN point.
func FitDrift(s timeseries.Series[float64]) (Fitted, error) {
	p, err := prepare(s)
	if err != nil {
		return nil, err
	}
	if len(p.values) < 2 {
		return nil, ErrTooShort
	}
	n := len(p.values)
	b := (p.values[n-1] - p.values[0]) / float64(n-1)
	last := p.values[n-1]
	return pointForecast{
		lastTime: p.last(),
		step:     p.step,
		at:       func(k int) float64 { return last + float64(k)*b },
	}, nil
}
