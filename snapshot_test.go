package forecast

import (
	"encoding/json"
	"errors"
	"math"
	"testing"
	"time"

	"github.com/eduard-kolotushin/timeseries"
)

func seriesAlmostEqual(t *testing.T, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len %d want %d", len(got), len(want))
	}
	for i := range want {
		if !almostEqual(got[i], want[i]) {
			t.Fatalf("i=%d got %v want %v", i, got[i], want[i])
		}
	}
}

func roundTripJSON(t *testing.T, f Fitted) Fitted {
	t.Helper()
	snap, err := SnapshotOf(f)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(snap)
	if err != nil {
		t.Fatal(err)
	}
	var back Snapshot
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatal(err)
	}
	restored, err := Restore(back)
	if err != nil {
		t.Fatal(err)
	}
	return restored
}

func TestSnapshotRoundTrip(t *testing.T) {
	t.Parallel()
	s := series(1, 2, 3, 4)
	ru, err := CalendarRU()
	if err != nil {
		t.Fatal(err)
	}
	hourTimes := make([]time.Time, 48)
	hourVals := make([]float64, 48)
	start := time.Date(2026, 1, 12, 0, 0, 0, 0, time.UTC)
	for i := range hourTimes {
		hourTimes[i] = start.Add(time.Duration(i) * time.Hour)
		hourVals[i] = float64(i)
	}
	hourly := timeseriesMust(t, hourTimes, hourVals)

	cases := []struct {
		name string
		fit  Fitted
		h    int
	}{
		{"naive", mustFit(t, func() (Fitted, error) { return FitNaive(s) }), 3},
		{"mean", mustFit(t, func() (Fitted, error) { return FitMean(s) }), 2},
		{"drift", mustFit(t, func() (Fitted, error) { return FitDrift(s) }), 3},
		{"seasonal", mustFit(t, func() (Fitted, error) { return FitSeasonalNaive(s, 2) }), 4},
		{"ses", mustFit(t, func() (Fitted, error) { return FitSES(s, 0.8) }), 3},
		{"holt", mustFit(t, func() (Fitted, error) { return FitHolt(s, 0.8, 0.2) }), 3},
		{"baseline", mustFit(t, func() (Fitted, error) { return FitSeasonalBaseline(hourly, SeasonHour, nil) }), 5},
		{"baseline-ru", mustFit(t, func() (Fitted, error) { return FitSeasonalBaseline(hourly, SeasonHour, ru) }), 5},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := roundTripJSON(t, tc.fit)
			wantFC, err := tc.fit.Forecast(tc.h)
			if err != nil {
				t.Fatal(err)
			}
			gotFC, err := got.Forecast(tc.h)
			if err != nil {
				t.Fatal(err)
			}
			seriesAlmostEqual(t, gotFC.Values(), wantFC.Values())
			if !gotFC.Times()[0].Equal(wantFC.Times()[0]) {
				t.Fatalf("times %v want %v", gotFC.Times(), wantFC.Times())
			}
			wantLo, wantHi, err := tc.fit.ForecastInterval(tc.h, 0.95)
			if err != nil {
				t.Fatal(err)
			}
			gotLo, gotHi, err := got.ForecastInterval(tc.h, 0.95)
			if err != nil {
				t.Fatal(err)
			}
			seriesAlmostEqual(t, gotLo.Values(), wantLo.Values())
			seriesAlmostEqual(t, gotHi.Values(), wantHi.Values())
		})
	}
}

func TestSnapshotUnknown(t *testing.T) {
	t.Parallel()
	if _, err := Restore(Snapshot{V: 99, Kind: kindNaive, Step: int64(time.Second)}); !errors.Is(err, ErrUnknownSnapshot) {
		t.Fatalf("version: %v", err)
	}
	if _, err := Restore(Snapshot{V: 1, Kind: "arima", Step: int64(time.Second), Data: []byte(`{}`)}); !errors.Is(err, ErrUnknownSnapshot) {
		t.Fatalf("kind: %v", err)
	}
	if _, err := SnapshotOf(fakeFitted{}); !errors.Is(err, ErrInvalidSnapshot) {
		t.Fatalf("foreign Fitted: %v", err)
	}
}

func TestSnapshotSigmaNaN(t *testing.T) {
	t.Parallel()
	m, err := FitNaive(series(1, 1))
	if err != nil {
		t.Fatal(err)
	}
	got := roundTripJSON(t, m)
	_, hi, err := got.ForecastInterval(1, 0.95)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsNaN(hi.Values()[0]) {
		t.Fatalf("want NaN bounds after restore, got %v", hi.Values())
	}
}

func timeseriesMust(t *testing.T, times []time.Time, values []float64) timeseries.Series[float64] {
	t.Helper()
	s, err := timeseries.New(times, values)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func mustFit(t *testing.T, fn func() (Fitted, error)) Fitted {
	t.Helper()
	f, err := fn()
	if err != nil {
		t.Fatal(err)
	}
	return f
}

type fakeFitted struct{}

func (fakeFitted) Forecast(int) (timeseries.Series[float64], error) {
	return timeseries.Series[float64]{}, nil
}
func (fakeFitted) ForecastInterval(int, float64) (timeseries.Series[float64], timeseries.Series[float64], error) {
	return timeseries.Series[float64]{}, timeseries.Series[float64]{}, nil
}
func (fakeFitted) ForecastRange(time.Time, time.Time) (timeseries.Series[float64], error) {
	return timeseries.Series[float64]{}, nil
}
func (fakeFitted) ForecastIntervalRange(time.Time, time.Time, float64) (timeseries.Series[float64], timeseries.Series[float64], error) {
	return timeseries.Series[float64]{}, timeseries.Series[float64]{}, nil
}
