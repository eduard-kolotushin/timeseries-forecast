package forecast

import (
	"math"
	"testing"
	"time"
)

func z95() float64 {
	return math.Sqrt2 * math.Erfinv(0.95)
}

func almostEqual(a, b float64) bool {
	if math.IsNaN(a) && math.IsNaN(b) {
		return true
	}
	return math.Abs(a-b) < 1e-9
}

func checkBounds(t *testing.T, m Fitted, h int, wantLo, wantHi []float64) {
	t.Helper()
	lo, hi, err := m.ForecastInterval(h, 0.95)
	if err != nil {
		t.Fatal(err)
	}
	gotLo, gotHi := lo.Values(), hi.Values()
	if len(gotLo) != h || len(gotHi) != h {
		t.Fatalf("len lo=%d hi=%d want %d", len(gotLo), len(gotHi), h)
	}
	for i := range wantLo {
		if !almostEqual(gotLo[i], wantLo[i]) || !almostEqual(gotHi[i], wantHi[i]) {
			t.Fatalf("k=%d lo=%v want %v hi=%v want %v", i+1, gotLo[i], wantLo[i], gotHi[i], wantHi[i])
		}
	}
	pt, err := m.Forecast(h)
	if err != nil {
		t.Fatal(err)
	}
	pv := pt.Values()
	for i := range pv {
		if math.IsNaN(gotLo[i]) {
			continue
		}
		if gotLo[i] > pv[i] || gotHi[i] < pv[i] {
			t.Fatalf("k=%d point %v not in [%v, %v]", i+1, pv[i], gotLo[i], gotHi[i])
		}
	}
}

func TestForecastIntervalInvalidLevel(t *testing.T) {
	t.Parallel()
	m, err := FitNaive(series(1, 2, 3, 4))
	if err != nil {
		t.Fatal(err)
	}
	for _, level := range []float64{0, 1, -0.1, 1.2, math.NaN()} {
		if _, _, err := m.ForecastInterval(1, level); err != ErrInvalidLevel {
			t.Fatalf("level %v: %v", level, err)
		}
	}
}

func TestForecastIntervalHorizon(t *testing.T) {
	t.Parallel()
	m, _ := FitNaive(series(1, 2, 3))
	if _, _, err := m.ForecastInterval(0, 0.95); err != ErrHorizon {
		t.Fatalf("horizon: %v", err)
	}
}

func TestNaiveInterval(t *testing.T) {
	t.Parallel()
	m, err := FitNaive(series(1, 2, 3, 4))
	if err != nil {
		t.Fatal(err)
	}
	// diffs 1,1,1; sse=3; n_resid=3; σ=1
	z := z95()
	checkBounds(t, m, 1, []float64{4 - z}, []float64{4 + z})
	s2 := math.Sqrt(2)
	checkBounds(t, m, 2, []float64{4 - z, 4 - z*s2}, []float64{4 + z, 4 + z*s2})
}

func TestNaiveIntervalTooFewResiduals(t *testing.T) {
	t.Parallel()
	m, err := FitNaive(series(1, 2))
	if err != nil {
		t.Fatal(err)
	}
	lo, hi, err := m.ForecastInterval(1, 0.95)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsNaN(lo.Values()[0]) || !math.IsNaN(hi.Values()[0]) {
		t.Fatalf("want NaN bounds, got %v %v", lo.Values(), hi.Values())
	}
}

func TestMeanInterval(t *testing.T) {
	t.Parallel()
	m, err := FitMean(series(1, 2, 3, 4))
	if err != nil {
		t.Fatal(err)
	}
	// mu=2.5; residuals -1.5,-0.5,0.5,1.5; sse=5; σ=sqrt(5/4)
	sigma := math.Sqrt(5.0 / 4.0)
	se := sigma * math.Sqrt(1+1.0/4.0)
	z := z95()
	checkBounds(t, m, 1, []float64{2.5 - z*se}, []float64{2.5 + z*se})
	checkBounds(t, m, 2, []float64{2.5 - z*se, 2.5 - z*se}, []float64{2.5 + z*se, 2.5 + z*se})
}

func TestDriftInterval(t *testing.T) {
	t.Parallel()
	m, err := FitDrift(series(1, 2, 3, 4))
	if err != nil {
		t.Fatal(err)
	}
	// b=1; residuals of y_t - y_{t-1} - b are 0; σ=0
	checkBounds(t, m, 2, []float64{5, 6}, []float64{5, 6})

	m, err = FitDrift(series(1, 3, 4, 8))
	if err != nil {
		t.Fatal(err)
	}
	// n=4, b=(8-1)/3=7/3
	// residuals: (3-1-7/3), (4-3-7/3), (8-4-7/3) = (-1/3), (-4/3), (5/3)
	// sse = 1/9 + 16/9 + 25/9 = 42/9 = 14/3
	// σ = sqrt((14/3)/3) = sqrt(14/9)
	sigma := math.Sqrt(14.0 / 9.0)
	n := 4.0
	se := func(k int) float64 {
		h := float64(k)
		return sigma * math.Sqrt(h*(1+h/n))
	}
	z := z95()
	pt1, pt2 := 8+7.0/3.0, 8+14.0/3.0
	checkBounds(t, m, 2,
		[]float64{pt1 - z*se(1), pt2 - z*se(2)},
		[]float64{pt1 + z*se(1), pt2 + z*se(2)},
	)
}

func TestSeasonalNaiveInterval(t *testing.T) {
	t.Parallel()
	m, err := FitSeasonalNaive(series(1, 2, 3, 10, 20, 30), 3)
	if err != nil {
		t.Fatal(err)
	}
	// residuals: 10-1, 20-2, 30-3 = 9,18,27; sse=9²+18²+27²=1134; n=3; σ=sqrt(378)
	sigma := math.Sqrt(1134.0 / 3.0)
	z := z95()
	// k=1,2,3: floor((k-1)/3)+1 = 1; k=4: 2
	se1 := sigma
	se4 := sigma * math.Sqrt(2)
	checkBounds(t, m, 4,
		[]float64{10 - z*se1, 20 - z*se1, 30 - z*se1, 10 - z*se4},
		[]float64{10 + z*se1, 20 + z*se1, 30 + z*se1, 10 + z*se4},
	)
}

func TestSeasonalNaiveIntervalTooFewResiduals(t *testing.T) {
	t.Parallel()
	m, err := FitSeasonalNaive(series(1, 2, 3, 4), 3)
	if err != nil {
		t.Fatal(err)
	}
	// n=4, period=3 → 1 residual
	lo, _, err := m.ForecastInterval(1, 0.95)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsNaN(lo.Values()[0]) {
		t.Fatalf("want NaN, got %v", lo.Values())
	}
}

func TestSESInterval(t *testing.T) {
	t.Parallel()
	m, err := FitSES(series(1, 2, 3), 1)
	if err != nil {
		t.Fatal(err)
	}
	// alpha=1: resid 2-1=1, 3-2=1; sse=2; σ=1; point=3
	// se(k)=sqrt(1+(k-1))
	z := z95()
	checkBounds(t, m, 2,
		[]float64{3 - z, 3 - z*math.Sqrt(2)},
		[]float64{3 + z, 3 + z*math.Sqrt(2)},
	)
}

// holtTrace rolls the recursion FitHolt runs out by impulse, so a test can pin the
// interval against the recursion itself rather than restating the variance formula.
// The state is zero at t and the h-step forecast is l_t + h·b_t = 0, so a unit
// innovation at step j contributes one realization of the h-step error; independent
// unit innovations make Var/σ² the sum of those contributions' squares.
func holtTrace(alpha, beta float64, h int) float64 {
	var trace float64
	for j := 1; j <= h; j++ {
		level, trend, y := 0.0, 0.0, 0.0
		for k := 1; k <= h; k++ {
			e := 0.0
			if k == j {
				e = 1
			}
			y = level + trend + e
			level, trend = level+trend+alpha*e, trend+alpha*beta*e
		}
		trace += y * y
	}
	return trace
}

func TestHoltInterval(t *testing.T) {
	t.Parallel()
	m, err := FitHolt(series(1, 2, 3, 4), 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	// Perfect linear; 1-step residuals are 0 after this init; σ=0; points 5,6
	checkBounds(t, m, 2, []float64{5, 6}, []float64{5, 6})

	m, err = FitHolt(series(1, 2, 2, 5), 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	// i=1: the seeded forecast is level+trend = 1+1 = 2 = y[1], so this residual
	// is structurally zero and is not counted; level=2, trend=1
	// i=2: pred=3, resid=2-3=-1; level=2, trend=0
	// i=3: pred=2, resid=5-2=3; level=5, trend=3
	// sse=1+9=10; n_resid=2; σ=sqrt(10/2)
	// point k: 5+3k
	sigma := math.Sqrt(10.0 / 2.0)
	holtSE := func(k int) float64 { return sigma * math.Sqrt(holtTrace(1, 1, k)) }
	z := z95()
	checkBounds(t, m, 2,
		[]float64{8 - z*holtSE(1), 11 - z*holtSE(2)},
		[]float64{8 + z*holtSE(1), 11 + z*holtSE(2)},
	)

	// α=β=1 makes the variance trace exact: Var(h)/σ² = 1, 5, 14 for h = 1..3.
	// The β² coefficient this test used to encode gave 11/3 and 11 there.
	lo, hi, err := m.ForecastInterval(3, 0.95)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := hi.Values()[2]-lo.Values()[2], 2*z*sigma*math.Sqrt(14); math.Abs(got-want) > 1e-9 {
		t.Fatalf("h=3 band width = %v, want %v (14σ²)", got, want)
	}

	// α=β=1 is the one point where every candidate variance formula coincides, so
	// the panel's own defaults (model holt at α=0.8, β=0.2) are pinned too. Widths
	// are compared as ratios between horizons, which cancels this fit's σ and
	// leaves only the α-dependence the interval has to reproduce.
	m2, err := FitHolt(series(1, 2, 2, 5), 0.8, 0.2)
	if err != nil {
		t.Fatal(err)
	}
	lo2, hi2, err := m2.ForecastInterval(6, 0.95)
	if err != nil {
		t.Fatal(err)
	}
	width := func(k int) float64 { return hi2.Values()[k-1] - lo2.Values()[k-1] }
	for k := 2; k <= 6; k++ {
		got := width(k) / width(k-1)
		want := math.Sqrt(holtTrace(0.8, 0.2, k) / holtTrace(0.8, 0.2, k-1))
		if math.Abs(got-want) > 1e-9 {
			t.Fatalf("α=0.8, β=0.2 width ratio h=%d/h=%d = %v, want %v", k, k-1, got, want)
		}
	}
}

func TestSeasonalBaselineInterval(t *testing.T) {
	t.Parallel()
	start := time.Date(2026, 1, 12, 10, 0, 0, 0, time.UTC) // Monday 10:00
	times := []time.Time{
		start,
		start.Add(time.Hour),
		start.Add(24 * time.Hour),
		start.Add(25 * time.Hour),
	}
	s := mustSeries(times, []float64{8, 1, 12, 1})
	m, err := FitSeasonalBaseline(s, SeasonHour, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Last is Tuesday 11:00. k=23 → Wednesday 10:00 workday.
	// Hour-10 workday values 8 and 12; mean=10; var=4; σ=2; se=2*sqrt(1.5)
	fc, err := m.Forecast(23)
	if err != nil {
		t.Fatal(err)
	}
	if fc.Values()[22] != 10 {
		t.Fatalf("point at Wed 10:00: %v", fc.Values()[22])
	}
	lo, hi, err := m.ForecastInterval(23, 0.95)
	if err != nil {
		t.Fatal(err)
	}
	se := 2 * math.Sqrt(1.5)
	z := z95()
	wantLo, wantHi := 10-z*se, 10+z*se
	if !almostEqual(lo.Values()[22], wantLo) || !almostEqual(hi.Values()[22], wantHi) {
		t.Fatalf("bounds %v %v want %v %v", lo.Values()[22], hi.Values()[22], wantLo, wantHi)
	}
}

func TestSeasonalBaselineIntervalSingleBucketNaN(t *testing.T) {
	t.Parallel()
	start := time.Date(2026, 1, 12, 10, 0, 0, 0, time.UTC)
	s := mustSeries(
		[]time.Time{start, start.Add(time.Hour)},
		[]float64{8, 1},
	)
	m, err := FitSeasonalBaseline(s, SeasonHour, nil)
	if err != nil {
		t.Fatal(err)
	}
	// k=1 is Monday 12:00 — empty hour, fallback to overall n=2 so defined.
	// The 10:00 bucket itself has n=1. k=23 is Tuesday 10:00, still only one 10:00 sample...
	// Last is Mon 11:00. k=23 → Tue 10:00. SeasonHour key is workday+10, count=1 → NaN se.
	lo, hi, err := m.ForecastInterval(23, 0.95)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsNaN(lo.Values()[22]) || !math.IsNaN(hi.Values()[22]) {
		t.Fatalf("want NaN for n=1 hour bucket, got %v %v", lo.Values()[22], hi.Values()[22])
	}
}

func TestForecastIntervalRangeBackcast(t *testing.T) {
	t.Parallel()
	z := z95()
	start := time.Date(2026, 1, 12, 10, 0, 0, 0, time.UTC) // Monday 10:00
	bs := mustSeries([]time.Time{
		start,
		start.Add(time.Hour),
		start.Add(24 * time.Hour),
		start.Add(25 * time.Hour),
	}, []float64{8, 1, 12, 1})
	baseline := mustFit(t, func() (Fitted, error) { return FitSeasonalBaseline(bs, SeasonHour, nil) })

	// The baseline looks its per-key SE up by grid timestamp, so a week before
	// the training range has the same band as that hour's horizon step.
	priorMonday := start.AddDate(0, 0, -7)
	lo, hi, err := baseline.ForecastIntervalRange(priorMonday, priorMonday, 0.95)
	if err != nil {
		t.Fatal(err)
	}
	se := 2 * math.Sqrt(1.5)
	wantLo, wantHi := 10-z*se, 10+z*se
	if !almostEqual(lo.Values()[0], wantLo) || !almostEqual(hi.Values()[0], wantHi) {
		t.Fatalf("baseline backcast bounds %v %v want %v %v", lo.Values()[0], hi.Values()[0], wantLo, wantHi)
	}

	// A bucket with one sample has no stored SE in-sample either.
	single := mustFit(t, func() (Fitted, error) {
		return FitSeasonalBaseline(mustSeries(
			[]time.Time{start, start.Add(time.Hour)},
			[]float64{8, 1},
		), SeasonHour, nil)
	})
	slo, shi, err := single.ForecastIntervalRange(start, start, 0.95)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsNaN(slo.Values()[0]) || !math.IsNaN(shi.Values()[0]) {
		t.Fatalf("single-sample backcast bounds = %v %v, want NaN NaN", slo.Values()[0], shi.Values()[0])
	}

	base := series(1, 2, 3, 4)
	tests := []struct {
		name string
		m    Fitted
		last time.Time
	}{
		{"naive", mustFit(t, func() (Fitted, error) { return FitNaive(base) }), tAt(3)},
		{"mean", mustFit(t, func() (Fitted, error) { return FitMean(base) }), tAt(3)},
		{"drift", mustFit(t, func() (Fitted, error) { return FitDrift(base) }), tAt(3)},
		{"seasonal", mustFit(t, func() (Fitted, error) { return FitSeasonalNaive(series(5, 6, 7, 8, 1, 2), 4) }), tAt(5)},
		{"ses", mustFit(t, func() (Fitted, error) { return FitSES(base, 0.5) }), tAt(3)},
		{"holt", mustFit(t, func() (Fitted, error) { return FitHolt(base, 0.5, 0.5) }), tAt(3)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lo, hi, err := tt.m.ForecastIntervalRange(tt.last.Add(-2*time.Second), tt.last, 0.95)
			if err != nil {
				t.Fatal(err)
			}
			if lo.Len() != 3 || hi.Len() != 3 {
				t.Fatalf("lens = %d, %d, want 3", lo.Len(), hi.Len())
			}
			for i := range 3 {
				if !math.IsNaN(lo.Values()[i]) || !math.IsNaN(hi.Values()[i]) {
					t.Fatalf("k=%d bounds = %v %v, want NaN NaN", i-2, lo.Values()[i], hi.Values()[i])
				}
			}
		})
	}

	// A straddling window keeps the horizon formula for k >= 1 and NaN for the
	// backcast half. This naive fit's three unit residuals make σ = 1.
	naive := mustFit(t, func() (Fitted, error) { return FitNaive(base) })
	slo, shi, err = naive.ForecastIntervalRange(tAt(3), tAt(4), 0.95)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsNaN(slo.Values()[0]) || !math.IsNaN(shi.Values()[0]) {
		t.Fatalf("k=0 bounds = %v %v, want NaN NaN", slo.Values()[0], shi.Values()[0])
	}
	if !almostEqual(slo.Values()[1], 4-z) || !almostEqual(shi.Values()[1], 4+z) {
		t.Fatalf("k=1 bounds = %v %v, want %v %v", slo.Values()[1], shi.Values()[1], 4-z, 4+z)
	}
}
