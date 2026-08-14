package forecast

import "github.com/eduard-kolotushin/timeseries"

// FitHolt fits Holt’s linear trend method with alpha, beta in (0, 1].
// Horizon k is forecast as level + k*trend.
func FitHolt(s timeseries.Series[float64], alpha, beta float64) (Fitted, error) {
	if !validSmooth(alpha) || !validSmooth(beta) {
		return nil, ErrInvalidAlpha
	}
	p, err := prepare(s)
	if err != nil {
		return nil, err
	}
	if len(p.values) < 2 {
		return nil, ErrTooShort
	}
	level := p.values[0]
	trend := p.values[1] - p.values[0]
	a1, b1 := 1-alpha, 1-beta
	for i := 1; i < len(p.values); i++ {
		prev := level
		level = alpha*p.values[i] + a1*(level+trend)
		trend = beta*(level-prev) + b1*trend
	}
	return pointForecast{
		lastTime: p.last(),
		step:     p.step,
		at:       func(k int) float64 { return level + float64(k)*trend },
	}, nil
}
