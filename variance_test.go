package forecast

import (
	"math"
	"testing"
	"time"

	"github.com/eduard-kolotushin/timeseries"
)

// relDiff is the relative difference between two band widths.
func relDiff(a, b float64) float64 {
	return math.Abs(a-b) / math.Abs(a)
}

// offsetTol is the tolerance for "the same band at a different offset". Values
// near 1e9 are stored on a 1.2e-7 grid (ulp), so the sd of a ±0.5 spread can
// only be recovered to ~1e-7 absolute — about 5e-8 relative to a 2.1-wide band.
// The audited raw-moment formula does not fail this by a little: it collapses
// the width to exactly 0.
const offsetTol = 1e-6

func TestBucketSE(t *testing.T) {
	t.Parallel()
	// [0,1,0,1,0,1]: m2 = 1.5 over n = 6 → sd 0.5 with the sqrt(1+1/n) factor.
	want := 0.5 * math.Sqrt(1+1.0/6.0)
	if got := bucketSE(1.5, 6); math.Abs(got-want) > 1e-15 {
		t.Fatalf("bucketSE(1.5, 6) = %v, want %v", got, want)
	}
	if got := bucketSE(0, 3); got != 0 {
		t.Fatalf("flat bucket se = %v, want 0", got)
	}
	if got := bucketSE(0, 1); !math.IsNaN(got) {
		t.Fatalf("bucketSE(0, 1) = %v, want NaN", got)
	}
	// A rounding-level negative m2 must not turn into NaN.
	if got := bucketSE(-1e-18, 4); got != 0 {
		t.Fatalf("bucketSE(-1e-18, 4) = %v, want 0", got)
	}
}

func TestBucketStatOffsetInvariant(t *testing.T) {
	t.Parallel()
	// The same two-value noise pattern at two offsets must give one bucket sd.
	// A raw sum of squares loses it entirely at 1e9 (the variance cancels to 0).
	var small, large bucketStat
	for _, v := range []float64{0, 1, 0, 1, 0, 1} {
		small.add(v)
		large.add(1e9 + v)
	}
	if small.n != 6 || math.Abs(small.mean-0.5) > 1e-15 {
		t.Fatalf("mean over n=%d = %v, want 0.5", small.n, small.mean)
	}
	if !(small.se() > 0) {
		t.Fatalf("bucket se = %v, want > 0", small.se())
	}
	if rel := relDiff(small.se(), large.se()); rel > 1e-9 {
		t.Fatalf("se small=%v large=%v (rel %g)", small.se(), large.se(), rel)
	}
}

func TestMeanIntervalOffsetInvariant(t *testing.T) {
	t.Parallel()
	// Same noise shape, two offsets. The band is z·σ·sqrt(1+1/n) around the mean
	// 0.5, i.e. the edges -0.5585 and 1.5585, and the offset must not move it.
	small, err := FitMean(series(0, 1, 0, 1, 0, 1))
	if err != nil {
		t.Fatal(err)
	}
	large, err := FitMean(series(1e9, 1e9+1, 1e9, 1e9+1, 1e9, 1e9+1))
	if err != nil {
		t.Fatal(err)
	}
	half := z95() * 0.5 * math.Sqrt(1+1.0/6.0)
	smallLo, smallHi, err := small.ForecastInterval(1, 0.95)
	if err != nil {
		t.Fatal(err)
	}
	if got := smallLo.Values()[0]; math.Abs(got-(0.5-half)) > 1e-12 {
		t.Fatalf("lower = %v, want %v", got, 0.5-half)
	}
	if got := smallHi.Values()[0]; math.Abs(got-(0.5+half)) > 1e-12 {
		t.Fatalf("upper = %v, want %v", got, 0.5+half)
	}
	largeLo, largeHi, err := large.ForecastInterval(1, 0.95)
	if err != nil {
		t.Fatal(err)
	}
	smallWidth := smallHi.Values()[0] - smallLo.Values()[0]
	largeWidth := largeHi.Values()[0] - largeLo.Values()[0]
	if !(smallWidth > 0) {
		t.Fatalf("band width = %v, want > 0", smallWidth)
	}
	if rel := relDiff(smallWidth, largeWidth); rel > offsetTol {
		t.Fatalf("width small=%v large=%v (rel %g)", smallWidth, largeWidth, rel)
	}
}

func TestSeasonalBaselineIntervalOffsetInvariant(t *testing.T) {
	t.Parallel()
	// Two weeks of hourly values alternating by day: every (class, hour) bucket
	// holds both 0 and 1, so its sd is the same little spread at any offset. A raw
	// sum of squares collapses that bucket's se to 0 once an offset is present.
	start := time.Date(2026, 1, 12, 0, 0, 0, 0, time.UTC) // Monday
	n := 2 * 7 * 24
	times := make([]time.Time, n)
	small := make([]float64, n)
	large := make([]float64, n)
	for i := range n {
		times[i] = start.Add(time.Duration(i) * time.Hour)
		v := float64((i / 24) % 2)
		small[i] = v
		large[i] = 1e9 + v
	}
	width := func(vals []float64) float64 {
		m, err := FitSeasonalBaseline(timeseries.MustNew(times, vals), SeasonHour, nil)
		if err != nil {
			t.Fatal(err)
		}
		from := start.Add(time.Duration(n) * time.Hour)
		lo, hi, err := m.ForecastIntervalRange(from, from.Add(3*time.Hour), 0.95)
		if err != nil {
			t.Fatal(err)
		}
		return hi.Values()[0] - lo.Values()[0]
	}
	smallWidth := width(small)
	largeWidth := width(large)
	if !(smallWidth > 0) {
		t.Fatalf("band width = %v, want > 0", smallWidth)
	}
	if rel := relDiff(smallWidth, largeWidth); rel > offsetTol {
		t.Fatalf("width small=%v large=%v (rel %g)", smallWidth, largeWidth, rel)
	}
}
