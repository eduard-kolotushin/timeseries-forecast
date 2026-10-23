package forecast

import (
	"math"
	"time"
)

// MaxForecastPoints caps how many grid points one call may emit (Forecast,
// ForecastInterval, ForecastRange, ForecastIntervalRange). A window that needs
// more fails with ErrTooManyPoints before any slice is allocated, so a distant
// `to` or an out-of-range horizon cannot turn into an unbounded allocation.
const MaxForecastPoints = 1_000_000

// maxDuration is time.Time.Sub's saturation point (~292 years): a distance that
// reaches it is not a horizon this package can state as k×step.
const maxDuration = time.Duration(math.MaxInt64)

// minDuration is the negative saturation point of time.Time.Sub.
const minDuration time.Duration = -1 << 63

// checkWindow validates a horizon against this model's grid. It is the shared
// bound behind Forecast, ForecastInterval and WindowBounds.
func (f pointForecast) checkWindow(k0, k1 int) error {
	return checkKBounds(f.step, k0, k1)
}

// checkKBounds validates an inclusive signed k window before anything is
// allocated: k0 ≤ k1, at most MaxForecastPoints points, and |k|×step bounded by
// a time.Duration (at and se look a grid timestamp up as lastTime + k×step;
// k ≤ 0 is a backcast on the same grid).
func checkKBounds(step time.Duration, k0, k1 int) error {
	if step <= 0 {
		return ErrNoFrequency
	}
	if k1 < k0 {
		return ErrEmptyRange
	}
	// Unsigned subtraction: the signed difference of two extremes can overflow
	// int64, while the modulo-2^64 result is the exact span for k1 >= k0.
	if uint64(k1)-uint64(k0) >= MaxForecastPoints {
		return ErrTooManyPoints
	}
	maxK := int64(maxDuration) / int64(step)
	if int64(k1) > maxK || int64(k0) < -maxK {
		return ErrTooManyPoints
	}
	return nil
}
