package forecast

import "errors"

var (
	// ErrEmpty is returned when the series has no usable (non-NaN) points.
	ErrEmpty = errors.New("forecast: series is empty")

	// ErrHorizon is returned when the forecast horizon is not positive.
	ErrHorizon = errors.New("forecast: horizon must be positive")

	// ErrNoFrequency is returned when a step cannot be inferred (fewer than two points).
	ErrNoFrequency = errors.New("forecast: need at least two points to infer time step")

	// ErrInvalidAlpha is returned when a smoothing parameter is outside (0, 1].
	ErrInvalidAlpha = errors.New("forecast: smoothing parameter must be in (0, 1]")

	// ErrInvalidPeriod is returned when a seasonal period is not positive.
	ErrInvalidPeriod = errors.New("forecast: seasonal period must be positive")

	// ErrInvalidSeason is returned when seasonal baseline seasonality is not hour, day, hour-of-week, or minute-of-week.
	ErrInvalidSeason = errors.New("forecast: invalid seasonality")

	// ErrUnknownCalendar is returned when a calendar name is not recognized.
	ErrUnknownCalendar = errors.New("forecast: unknown calendar")

	// ErrTooShort is returned when the series is shorter than the model requires.
	ErrTooShort = errors.New("forecast: series is too short for this model")

	// ErrSplit is returned when a train/test split size is invalid.
	ErrSplit = errors.New("forecast: invalid train/test split")
)
