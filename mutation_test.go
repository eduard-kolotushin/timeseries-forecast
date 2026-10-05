package forecast

import (
	"testing"

	"github.com/eduard-kolotushin/timeseries"
)

// TestOpsDoNotMutateInput asserts the package invariant from AGENTS.md:
// public ops do not mutate caller series. The fixtures are NaN-free so the
// library's own NaN-unequal Equal is a valid comparison.
func TestOpsDoNotMutateInput(t *testing.T) {
	t.Parallel()
	s := series(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	other := series(10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
	fitted, err := FitNaive(s)
	if err != nil {
		t.Fatal(err)
	}
	snap, err := SnapshotOf(fitted)
	if err != nil {
		t.Fatal(err)
	}

	before := s.Clone()
	otherBefore := other.Clone()

	cases := []struct {
		name string
		run  func()
	}{
		{"FitNaive", func() { _, _ = FitNaive(s) }},
		{"FitMean", func() { _, _ = FitMean(s) }},
		{"FitDrift", func() { _, _ = FitDrift(s) }},
		{"FitSeasonalNaive", func() { _, _ = FitSeasonalNaive(s, 4) }},
		{"FitSES", func() { _, _ = FitSES(s, 0.5) }},
		{"FitHolt", func() { _, _ = FitHolt(s, 0.5, 0.3) }},
		{"FitSeasonalBaseline", func() { _, _ = FitSeasonalBaseline(s, SeasonHour, nil) }},
		{"Compare", func() { _, _ = Compare(s, other) }},
		{"CompareReversed", func() { _, _ = Compare(other, s) }},
		{"Split", func() { _, _, _ = Split(s, 3) }},
		{"Evaluate", func() {
			_, _ = Evaluate(s, other, func(x timeseries.Series[float64]) (Fitted, error) {
				return FitNaive(x)
			})
		}},
		{"SnapshotOf", func() { _, _ = SnapshotOf(fitted) }},
		{"Restore", func() { _, _ = Restore(snap) }},
		{"Forecast", func() { _, _ = fitted.Forecast(3) }},
		{"ForecastInterval", func() { _, _, _ = fitted.ForecastInterval(3, 0.95) }},
		{"ForecastRange", func() { _, _ = fitted.ForecastRange(tAt(10), tAt(12)) }},
		{"ForecastIntervalRange", func() { _, _, _ = fitted.ForecastIntervalRange(tAt(10), tAt(12), 0.95) }},
	}
	for _, tc := range cases {
		tc.run()
		if !timeseries.Equal(before, s) {
			t.Fatalf("%s mutated its input: got %v %v, want %v %v",
				tc.name, s.Times(), s.Values(), before.Times(), before.Values())
		}
		if !timeseries.Equal(otherBefore, other) {
			t.Fatalf("%s mutated the second operand: got %v %v", tc.name, other.Times(), other.Values())
		}
	}
}
