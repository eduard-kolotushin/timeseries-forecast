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
	// i=1: pred=1+1=2, resid=0; level=2, trend=1
	// i=2: pred=3, resid=2-3=-1; level=2, trend=0
	// i=3: pred=2, resid=5-2=3; level=5, trend=3
	// sse=0+1+9=10; n=3; σ=sqrt(10/3)
	// point k: 5+3k
	sigma := math.Sqrt(10.0 / 3.0)
	holtSE := func(k int) float64 {
		h := float64(k)
		return sigma * math.Sqrt(1+(h-1)*(1+h+h*(h-1)/6))
	}
	z := z95()
	checkBounds(t, m, 2,
		[]float64{8 - z*holtSE(1), 11 - z*holtSE(2)},
		[]float64{8 + z*holtSE(1), 11 + z*holtSE(2)},
	)
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
