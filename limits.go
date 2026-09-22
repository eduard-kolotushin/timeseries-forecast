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

// checkWindow validates an inclusive 1-based k window before anything is
// allocated: k0 ≤ k1, at most MaxForecastPoints points, and k1×step bounded by a
// time.Duration (at and se look a grid timestamp up as lastTime + k×step).
func (f pointForecast) checkWindow(k0, k1 int) error {
	if f.step <= 0 {
		return ErrNoFrequency
	}
	if k1 < k0 {
		return ErrEmptyRange
	}
	if k1-k0 >= MaxForecastPoints {
		return ErrTooManyPoints
	}
	if int64(k1) > int64(maxDuration)/int64(f.step) {
		return ErrTooManyPoints
	}
	return nil
}
