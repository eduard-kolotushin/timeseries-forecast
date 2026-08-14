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
	m, _ := FitNaive(series(1, 2))
	if _, err := m.Forecast(0); err != ErrHorizon {
		t.Fatalf("horizon: %v", err)
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
