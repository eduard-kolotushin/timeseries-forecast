package forecast

import "github.com/eduard-kolotushin/timeseries"

func validSmooth(x float64) bool {
	return x > 0 && x <= 1
}

// FitSES fits simple exponential smoothing with the given alpha in (0, 1].
// The forecast is a constant equal to the final smoothed level.
func FitSES(s timeseries.Series[float64], alpha float64) (Fitted, error) {
	if !validSmooth(alpha) {
		return nil, ErrInvalidAlpha
	}
	p, err := prepare(s)
	if err != nil {
		return nil, err
	}
	if err := p.requireStep(); err != nil {
		return nil, err
	}
	level := p.values[0]
	oneMinus := 1 - alpha
	for i := 1; i < len(p.values); i++ {
		level = alpha*p.values[i] + oneMinus*level
	}
	return pointForecast{lastTime: p.last(), step: p.step, at: func(int) float64 { return level }}, nil
}
