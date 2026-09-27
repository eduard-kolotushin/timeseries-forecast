package forecast

import (
	"errors"
	"math"
	"testing"
	"time"

	"github.com/eduard-kolotushin/timeseries"
)

// meanModel fits the mean of {1, 2, 3} on the given step: its last timestamp is
// tAt(0)+2*step, so k=1 lands on tAt(0)+3*step.
func meanModel(t *testing.T, step time.Duration) Fitted {
	t.Helper()
	s := timeseries.MustNew(
		[]time.Time{tAt(0), tAt(0).Add(step), tAt(0).Add(2 * step)},
		[]float64{1, 2, 3},
	)
	m, err := FitMean(s)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestWindowKCap(t *testing.T) {
	t.Parallel()
	const day = 24 * time.Hour
	base := tAt(0)
	minute := func(n int) time.Time { return base.Add(time.Duration(2+n) * time.Minute) }
	daily := func(n int) time.Time { return base.Add(time.Duration(2+n) * day) }
	// 106751 days after the model's last point: a whole number of day steps, with
	// d+step past time.Duration's range. AddDate (not Add(n*day)) avoids the
	// overflow that multiplying the duration would hit.
	farLast := base.Add(2*day).AddDate(0, 0, 106751)

	tests := []struct {
		name    string
		step    time.Duration
		from    time.Time
		to      time.Time
		wantK0  int
		wantK1  int
		wantErr error
	}{
		{"exactly the cap", time.Minute, minute(1), minute(MaxForecastPoints), 1, MaxForecastPoints, nil},
		{"one over the cap", time.Minute, minute(1), minute(MaxForecastPoints + 1), 0, 0, ErrTooManyPoints},
		{"the cap starting off the first grid point", time.Minute, minute(MaxForecastPoints), minute(2*MaxForecastPoints - 1), MaxForecastPoints, 2*MaxForecastPoints - 1, nil},
		{"one over starting off the first grid point", time.Minute, minute(MaxForecastPoints), minute(2 * MaxForecastPoints), 0, 0, ErrTooManyPoints},
		{"single grid point", time.Minute, minute(3), minute(3), 3, 3, nil},
		// About 292 years out time.Time.Sub saturates, so the distance is unknown.
		{"saturated horizon", time.Minute, minute(1), minute(1).AddDate(300, 0, 0), 0, 0, ErrTooManyPoints},
		// A day step keeps k inside the point cap, so only the saturated distance rejects this.
		{"saturated horizon past the point cap", day, daily(1), daily(1).AddDate(300, 0, 0), 0, 0, ErrTooManyPoints},
		{"inverted", time.Minute, minute(2), minute(1), 0, 0, ErrRange},
		// The band where (num+den-1) wraps int64: the old ceiling went negative, k0
		// was clamped to 1, and a 106751-point grid ran from last+step for ~292 years
		// instead of the single point at the requested time.
		{"overflowing ceiling near the saturated horizon", day, farLast, farLast, 106751, 106751, nil},
		{"before the grid", time.Minute, minute(-1), minute(-1), 0, 0, ErrEmptyRange},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, ok := meanModel(t, tt.step).(pointForecast)
			if !ok {
				t.Fatal("FitMean did not return a pointForecast")
			}
			k0, k1, err := m.windowK(tt.from, tt.to)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("windowK(%v, %v) err = %v, want %v", tt.from, tt.to, err, tt.wantErr)
			}
			if tt.wantErr != nil {
				return
			}
			if k0 != tt.wantK0 || k1 != tt.wantK1 {
				t.Fatalf("k0, k1 = %d, %d, want %d, %d", k0, k1, tt.wantK0, tt.wantK1)
			}
		})
	}
}

func TestForecastAtTheCap(t *testing.T) {
	t.Parallel()
	m := meanModel(t, time.Minute)
	got, err := m.Forecast(MaxForecastPoints)
	if err != nil {
		t.Fatalf("Forecast(%d): %v", MaxForecastPoints, err)
	}
	if got.Len() != MaxForecastPoints {
		t.Fatalf("Len() = %d, want %d", got.Len(), MaxForecastPoints)
	}
	lastTime := tAt(0).Add(2 * time.Minute)
	wantEnd := lastTime.Add(MaxForecastPoints * time.Minute)
	end, err := got.Time(got.Len() - 1)
	if err != nil || !end.Equal(wantEnd) {
		t.Fatalf("last timestamp = %v (err %v), want %v", end, err, wantEnd)
	}
	vals := got.Values()
	if vals[0] != 2 || vals[len(vals)-1] != 2 {
		t.Fatalf("mean forecast = %v ... %v, want 2", vals[0], vals[len(vals)-1])
	}
}

func TestForecastRejectsAnUnboundedWindow(t *testing.T) {
	t.Parallel()
	const day = 24 * time.Hour
	minuteModel := meanModel(t, time.Minute)
	dayModel := meanModel(t, day)
	minuteLast := tAt(0).Add(2 * time.Minute)
	dayLast := tAt(0).Add(2 * day)

	tests := []struct {
		name string
		call func() error
	}{
		{"horizon one over the cap", func() error {
			_, err := minuteModel.Forecast(MaxForecastPoints + 1)
			return err
		}},
		{"absurd horizon", func() error {
			_, err := minuteModel.Forecast(math.MaxInt)
			return err
		}},
		{"interval horizon one over the cap", func() error {
			_, _, err := minuteModel.ForecastInterval(MaxForecastPoints+1, 0.95)
			return err
		}},
		{"interval absurd horizon", func() error {
			_, _, err := minuteModel.ForecastInterval(math.MaxInt, 0.95)
			return err
		}},
		{"range one over the cap", func() error {
			_, err := minuteModel.ForecastRange(minuteLast.Add(time.Minute), minuteLast.Add((MaxForecastPoints+1)*time.Minute))
			return err
		}},
		{"interval range one over the cap", func() error {
			_, _, err := minuteModel.ForecastIntervalRange(minuteLast.Add(time.Minute), minuteLast.Add((MaxForecastPoints+1)*time.Minute), 0.95)
			return err
		}},
		{"range with a saturated end", func() error {
			_, err := minuteModel.ForecastRange(minuteLast.Add(time.Minute), minuteLast.AddDate(300, 0, 0))
			return err
		}},
		{"interval range with a saturated end", func() error {
			_, _, err := minuteModel.ForecastIntervalRange(minuteLast.Add(time.Minute), minuteLast.AddDate(300, 0, 0), 0.95)
			return err
		}},
		// A day step keeps k×step inside a time.Duration for only ~106751 points.
		{"horizon past a Duration of k×step", func() error {
			_, err := dayModel.Forecast(200_000)
			return err
		}},
		{"interval horizon past a Duration of k×step", func() error {
			_, _, err := dayModel.ForecastInterval(200_000, 0.95)
			return err
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.call(); !errors.Is(err, ErrTooManyPoints) {
				t.Fatalf("err = %v, want ErrTooManyPoints", err)
			}
		})
	}

	// Control: 100000 day steps are 274 years of horizon but still inside a
	// time.Duration, so the guard must let them through.
	got, err := dayModel.Forecast(100_000)
	if err != nil {
		t.Fatalf("Forecast(100000) on a day step: %v", err)
	}
	if got.Len() != 100_000 {
		t.Fatalf("Len() = %d, want 100000", got.Len())
	}
	if end, err := got.Time(got.Len() - 1); err != nil || !end.Equal(dayLast.Add(100_000*day)) {
		t.Fatalf("last timestamp = %v (err %v), want %v", end, err, dayLast.Add(100_000*day))
	}
}
