package forecast

import (
	"math"
	"testing"
	"time"

	"github.com/eduard-kolotushin/timeseries"
)

func tAt(sec int64) time.Time {
	return time.Unix(sec, 0).UTC()
}

func series(vals ...float64) timeseries.Series[float64] {
	times := make([]time.Time, len(vals))
	for i := range vals {
		times[i] = tAt(int64(i))
	}
	return timeseries.MustNew(times, vals)
}

func TestFitErrors(t *testing.T) {
	t.Parallel()
	empty := timeseries.MustNew([]time.Time{}, []float64{})
	if _, err := FitNaive(empty); err != ErrEmpty {
		t.Fatalf("empty: %v", err)
	}
	one := series(1)
	if _, err := FitNaive(one); err != ErrNoFrequency {
		t.Fatalf("one point: %v", err)
	}
	if _, err := FitSES(series(1, 2), 0); err != ErrInvalidAlpha {
		t.Fatalf("alpha: %v", err)
	}
	if _, err := FitSeasonalNaive(series(1, 2), 0); err != ErrInvalidPeriod {
		t.Fatalf("period: %v", err)
	}
	if _, err := FitSeasonalNaive(series(1, 2), 3); err != ErrTooShort {
		t.Fatalf("seasonal short: %v", err)
	}
	if _, err := FitHolt(series(1), 0.5, 0.5); err != ErrTooShort {
		t.Fatalf("holt short: %v", err)
	}
	if _, err := FitSeasonalBaseline(series(1, 2), 0, nil); err != ErrInvalidSeason {
		t.Fatalf("baseline season: %v", err)
	}
}

func TestNaiveMeanDrift(t *testing.T) {
	t.Parallel()
	s := series(1, 2, 3, 4)

	n, err := FitNaive(s)
	if err != nil {
		t.Fatal(err)
	}
	fc, err := n.Forecast(2)
	if err != nil {
		t.Fatal(err)
	}
	if got := fc.Values(); got[0] != 4 || got[1] != 4 {
		t.Fatalf("naive: %v", got)
	}
	if !fc.Times()[0].Equal(tAt(4)) || !fc.Times()[1].Equal(tAt(5)) {
		t.Fatalf("times: %v", fc.Times())
	}

	m, err := FitMean(s)
	if err != nil {
		t.Fatal(err)
	}
	fc, _ = m.Forecast(1)
	if fc.Values()[0] != 2.5 {
		t.Fatalf("mean: %v", fc.Values())
	}

	d, err := FitDrift(s)
	if err != nil {
		t.Fatal(err)
	}
	fc, _ = d.Forecast(2)
	if fc.Values()[0] != 5 || fc.Values()[1] != 6 {
		t.Fatalf("drift: %v", fc.Values())
	}
}

func TestSeasonalNaive(t *testing.T) {
	t.Parallel()
	s := series(1, 2, 3, 10, 20, 30)
	m, err := FitSeasonalNaive(s, 3)
	if err != nil {
		t.Fatal(err)
	}
	fc, err := m.Forecast(4)
	if err != nil {
		t.Fatal(err)
	}
	got := fc.Values()
	want := []float64{10, 20, 30, 10}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v want %v", got, want)
		}
	}
}

func TestSES(t *testing.T) {
	t.Parallel()
	s := series(1, 1, 1)
	m, err := FitSES(s, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	fc, _ := m.Forecast(2)
	if fc.Values()[0] != 1 || fc.Values()[1] != 1 {
		t.Fatalf("ses: %v", fc.Values())
	}
	naive, _ := FitSES(series(1, 2, 3), 1)
	fc, _ = naive.Forecast(1)
	if fc.Values()[0] != 3 {
		t.Fatalf("ses alpha=1: %v", fc.Values())
	}
}

func TestHolt(t *testing.T) {
	t.Parallel()
	s := series(1, 2, 3, 4)
	m, err := FitHolt(s, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	fc, _ := m.Forecast(2)
	if fc.Values()[0] != 5 || fc.Values()[1] != 6 {
		t.Fatalf("holt: %v", fc.Values())
	}
}

func TestForecastHorizon(t *testing.T) {
	t.Parallel()
	m, _ := FitNaive(series(1, 2, 3, 4))
	if _, err := m.Forecast(0); err != ErrHorizon {
		t.Fatalf("horizon: %v", err)
	}
	if _, err := m.Forecast(-1); err != ErrHorizon {
		t.Fatalf("negative horizon: %v", err)
	}
	fc, err := m.Forecast(3)
	if err != nil {
		t.Fatal(err)
	}
	if fc.Len() != 3 {
		t.Fatalf("len=%d want 3", fc.Len())
	}
	// Future timestamps are last + k*step, step inferred as 1s.
	want := []time.Time{tAt(4), tAt(5), tAt(6)}
	for i, w := range want {
		if !fc.Times()[i].Equal(w) {
			t.Fatalf("time %d = %v, want %v", i, fc.Times()[i], w)
		}
	}
}

func TestForecastRange(t *testing.T) {
	t.Parallel()
	naive, err := FitNaive(series(1, 2, 3, 4))
	if err != nil {
		t.Fatal(err)
	}
	full, err := naive.Forecast(3)
	if err != nil {
		t.Fatal(err)
	}

	got, err := naive.ForecastRange(tAt(4), tAt(6))
	if err != nil {
		t.Fatal(err)
	}
	if got.Len() != 3 || !got.Times()[0].Equal(tAt(4)) || !got.Times()[2].Equal(tAt(6)) {
		t.Fatalf("full window times: %v", got.Times())
	}
	if got.Values()[0] != full.Values()[0] || got.Values()[2] != full.Values()[2] {
		t.Fatalf("full window values: %v vs %v", got.Values(), full.Values())
	}

	skip, err := naive.ForecastRange(tAt(6), tAt(8))
	if err != nil {
		t.Fatal(err)
	}
	if skip.Len() != 3 || !skip.Times()[0].Equal(tAt(6)) || skip.Values()[0] != 4 {
		t.Fatalf("skip-ahead: times=%v values=%v", skip.Times(), skip.Values())
	}

	one, err := naive.ForecastRange(tAt(5), tAt(5))
	if err != nil {
		t.Fatal(err)
	}
	if one.Len() != 1 || !one.Times()[0].Equal(tAt(5)) {
		t.Fatalf("point window: %v", one.Times())
	}

	if _, err := naive.ForecastRange(tAt(6), tAt(5)); err != ErrRange {
		t.Fatalf("inverted: %v", err)
	}
	back, err := naive.ForecastRange(tAt(0), tAt(3))
	if err != nil {
		t.Fatalf("backcast: %v", err)
	}
	if back.Len() != 4 || !back.Times()[0].Equal(tAt(0)) || !back.Times()[3].Equal(tAt(3)) {
		t.Fatalf("backcast times: %v", back.Times())
	}
	for i, v := range back.Values() {
		if v != 4 {
			t.Fatalf("backcast value %d = %v, want 4 (the final observation)", i, v)
		}
	}
	if _, err := naive.ForecastRange(tAt(4).Add(500*time.Millisecond), tAt(4).Add(500*time.Millisecond)); err != ErrEmptyRange {
		t.Fatalf("off-grid: %v", err)
	}

	holt, err := FitHolt(series(1, 2, 3, 4), 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	hSkip, err := holt.ForecastRange(tAt(5), tAt(6))
	if err != nil {
		t.Fatal(err)
	}
	if hSkip.Len() != 2 || hSkip.Values()[0] != 6 || hSkip.Values()[1] != 7 {
		t.Fatalf("holt skip: %v", hSkip.Values())
	}
	if _, _, err := holt.ForecastIntervalRange(tAt(5), tAt(6), 0.95); err != nil {
		t.Fatal(err)
	}
}

func TestForecastRangeBackcast(t *testing.T) {
	t.Parallel()
	base := series(1, 2, 3, 4)
	last, step := tAt(3), time.Second
	naive := mustFit(t, func() (Fitted, error) { return FitNaive(base) })
	mean := mustFit(t, func() (Fitted, error) { return FitMean(base) })
	drift := mustFit(t, func() (Fitted, error) { return FitDrift(base) })
	ses := mustFit(t, func() (Fitted, error) { return FitSES(base, 0.5) })
	holt := mustFit(t, func() (Fitted, error) { return FitHolt(base, 0.5, 0.5) })

	tests := []struct {
		name string
		m    Fitted
		want []float64 // k = -2, -1, 0
	}{
		{"naive", naive, []float64{4, 4, 4}},
		{"mean", mean, []float64{2.5, 2.5, 2.5}},
		{"drift", drift, []float64{2, 3, 4}},
		{"ses", ses, []float64{3.125, 3.125, 3.125}},
		{"holt", holt, []float64{2, 3, 4}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.m.ForecastRange(last.Add(-2*step), last)
			if err != nil {
				t.Fatal(err)
			}
			if got.Len() != 3 {
				t.Fatalf("Len() = %d, want 3", got.Len())
			}
			if !got.Times()[0].Equal(last.Add(-2*step)) || !got.Times()[2].Equal(last) {
				t.Fatalf("times = %v", got.Times())
			}
			for i, want := range tt.want {
				if got.Values()[i] != want {
					t.Fatalf("k=%d value = %v, want %v", i-2, got.Values()[i], want)
				}
			}

			one, err := tt.m.ForecastRange(last, last)
			if err != nil || one.Len() != 1 || one.Values()[0] != tt.want[2] {
				t.Fatalf("k=0 window: len=%d values=%v err=%v", one.Len(), one.Values(), err)
			}

			straddle, err := tt.m.ForecastRange(last.Add(-2*step), last.Add(2*step))
			if err != nil {
				t.Fatal(err)
			}
			if straddle.Len() != 5 {
				t.Fatalf("straddling window len = %d, want 5", straddle.Len())
			}
			for i := range 5 {
				if !straddle.Times()[i].Equal(last.Add(time.Duration(i-2) * step)) {
					t.Fatalf("straddling timestamp %d = %v", i, straddle.Times()[i])
				}
			}
		})
	}

	// Seasonal naive: the season is the last period, so k <= 0 must keep walking
	// the phase backwards instead of indexing the slice negatively.
	seasonal := mustFit(t, func() (Fitted, error) { return FitSeasonalNaive(series(5, 6, 7, 8, 1, 2), 4) })
	back, err := seasonal.ForecastRange(tAt(0), tAt(3))
	if err != nil {
		t.Fatal(err)
	}
	wantSeasonal := []float64{1, 2, 7, 8} // k = -5..-2
	if back.Len() != len(wantSeasonal) {
		t.Fatalf("seasonal backcast len = %d, want %d", back.Len(), len(wantSeasonal))
	}
	for i, want := range wantSeasonal {
		if back.Values()[i] != want {
			t.Fatalf("seasonal k=%d value = %v, want %v", i-5, back.Values()[i], want)
		}
	}

	// Baseline: the stored per-key means answer any timestamp on the grid, even
	// one a week before the training range began.
	start := time.Date(2026, 1, 12, 10, 0, 0, 0, time.UTC) // Monday 10:00
	bs := mustSeries([]time.Time{
		start,
		start.Add(time.Hour),
		start.Add(24 * time.Hour),
		start.Add(25 * time.Hour),
	}, []float64{8, 1, 12, 1})
	baseline := mustFit(t, func() (Fitted, error) { return FitSeasonalBaseline(bs, SeasonHour, nil) })
	priorMonday := start.AddDate(0, 0, -7)
	b, err := baseline.ForecastRange(priorMonday, priorMonday)
	if err != nil {
		t.Fatal(err)
	}
	if b.Len() != 1 || b.Values()[0] != 10 || !b.Times()[0].Equal(priorMonday) {
		t.Fatalf("baseline backcast: times=%v values=%v, want one point at %v = 10", b.Times(), b.Values(), priorMonday)
	}
	atLast, err := baseline.ForecastRange(start.Add(25*time.Hour), start.Add(25*time.Hour))
	if err != nil || atLast.Len() != 1 || atLast.Values()[0] != 1 {
		t.Fatalf("baseline k=0: len=%d values=%v err=%v", atLast.Len(), atLast.Values(), err)
	}
}

func TestCompareAndEvaluate(t *testing.T) {
	t.Parallel()
	actual := series(1, 2, 3, 4)
	pred := series(1, 2, 3, 4)
	met, err := Compare(actual, pred)
	if err != nil || met.MAE != 0 || met.RMSE != 0 || met.N != 4 {
		t.Fatalf("perfect: %+v err=%v", met, err)
	}

	train, test, err := Split(actual, 2)
	if err != nil || train.Len() != 2 || test.Len() != 2 {
		t.Fatalf("split: %v %d %d", err, train.Len(), test.Len())
	}
	met, err = Evaluate(train, test, FitNaive)
	if err != nil {
		t.Fatal(err)
	}
	if met.N != 2 || met.MAE != 1.5 {
		t.Fatalf("evaluate naive: %+v", met)
	}

	if _, _, err := Split(actual, 0); err != ErrSplit {
		t.Fatalf("split err: %v", err)
	}
}

func TestCompareMAPENaN(t *testing.T) {
	t.Parallel()
	actual := series(0, 0)
	pred := series(1, 1)
	met, err := Compare(actual, pred)
	if err != nil || !math.IsNaN(met.MAPE) {
		t.Fatalf("mape: %+v err=%v", met, err)
	}
}

func TestDropNAOnFit(t *testing.T) {
	t.Parallel()
	s := timeseries.MustNew(
		[]time.Time{tAt(0), tAt(1), tAt(2)},
		[]float64{1, math.NaN(), 3},
	)
	m, err := FitNaive(s)
	if err != nil {
		t.Fatal(err)
	}
	fc, _ := m.Forecast(1)
	if fc.Values()[0] != 3 || !fc.Times()[0].Equal(tAt(4)) {
		t.Fatalf("dropna naive: %v %v", fc.Values(), fc.Times())
	}
}
