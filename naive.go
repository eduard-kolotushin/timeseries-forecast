package forecast

import (
	"github.com/eduard-kolotushin/timeseries"
)

// FitNaive repeats the last non-NaN observation.
func FitNaive(s timeseries.Series[float64]) (Fitted, error) {
	p, err := prepare(s)
	if err != nil {
		return nil, err
	}
	if err := p.requireStep(); err != nil {
		return nil, err
	}
	n := len(p.values)
	last := p.values[n-1]
	var sse float64
	for i := 1; i < n; i++ {
		d := p.values[i] - p.values[i-1]
		sse += d * d
	}
	sigma := mleSigma(sse, n-1)
	return pointForecast{
		lastTime: p.last(),
		step:     p.step,
		kind:     kindNaive,
		last:     last,
		sigma:    sigma,
	}, nil
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
	n := len(p.values)
	var sum, sumsq float64
	for _, v := range p.values {
		sum += v
		sumsq += v * v
	}
	nf := float64(n)
	mu := sum / nf
	sse := sumsq - mu*sum
	if sse < 0 {
		sse = 0
	}
	sigma := mleSigma(sse, n)
	return pointForecast{
		lastTime: p.last(),
		step:     p.step,
		kind:     kindMean,
		mu:       mu,
		sigma:    sigma,
		n:        n,
	}, nil
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
	var sse float64
	for i := 1; i < n; i++ {
		r := p.values[i] - p.values[i-1] - b
		sse += r * r
	}
	sigma := mleSigma(sse, n-1)
	return pointForecast{
		lastTime: p.last(),
		step:     p.step,
		kind:     kindDrift,
		last:     last,
		b:        b,
		sigma:    sigma,
		n:        n,
	}, nil
}
