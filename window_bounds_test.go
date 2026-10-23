package forecast

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/eduard-kolotushin/timeseries"
)

// naiveOnGrid fits a naive model whose last timestamp and step are exactly the
// ones the bounds are computed for, so ForecastRange can be compared with the
// brute-forced grid.
func naiveOnGrid(t *testing.T, last time.Time, step time.Duration) Fitted {
	t.Helper()
	s := timeseries.MustNew(
		[]time.Time{last.Add(-2 * step), last.Add(-step), last},
		[]float64{1, 2, 3},
	)
	m, err := FitNaive(s)
	if err != nil {
		t.Fatalf("fit: %v", err)
	}
	return m
}

// TestWindowBoundsExhaustiveGrid enumerates every signed window in a small
// domain, with jitter that lands inside and on both sides of grid steps, and
// checks WindowBounds against a brute-force scan of last+k*step. It also checks
// that ForecastRange emits exactly that grid — the bound is not merely close.
func TestWindowBoundsExhaustiveGrid(t *testing.T) {
	t.Parallel()
	last := time.Date(2026, 1, 12, 10, 0, 0, 0, time.UTC)
	steps := []time.Duration{time.Millisecond, time.Second, time.Minute, time.Hour, 24 * time.Hour}
	jitters := []time.Duration{-1, 0, 1} // below every tested step
	for _, step := range steps {
		m := naiveOnGrid(t, last, step)
		for fromK := -8; fromK <= 8; fromK++ {
			for toK := -8; toK <= 8; toK++ {
				for _, fj := range jitters {
					for _, tj := range jitters {
						from := last.Add(time.Duration(fromK)*step + fj)
						to := last.Add(time.Duration(toK)*step + tj)

						var want []int
						for k := -10; k <= 10; k++ {
							at := last.Add(time.Duration(k) * step)
							if !at.Before(from) && !at.After(to) {
								want = append(want, k)
							}
						}

						k0, k1, err := WindowBounds(last, step, from, to)
						switch {
						case from.After(to):
							if !errors.Is(err, ErrRange) {
								t.Fatalf("from=%v to=%v: err = %v, want ErrRange", from, to, err)
							}
							continue
						case len(want) == 0:
							if !errors.Is(err, ErrEmptyRange) {
								t.Fatalf("from=%v to=%v: err = %v, want ErrEmptyRange", from, to, err)
							}
							continue
						case err != nil:
							t.Fatalf("from=%v to=%v: err = %v", from, to, err)
						}
						if k0 != want[0] || k1 != want[len(want)-1] {
							t.Fatalf("from=%v to=%v: k = %d..%d, want %d..%d", from, to, k0, k1, want[0], want[len(want)-1])
						}

						got, err := m.ForecastRange(from, to)
						if err != nil {
							t.Fatalf("from=%v to=%v: ForecastRange: %v", from, to, err)
						}
						if got.Len() != len(want) {
							t.Fatalf("from=%v to=%v: ForecastRange emitted %d points, want %d", from, to, got.Len(), len(want))
						}
						for i, k := range want {
							if w, g := last.Add(time.Duration(k)*step), got.Times()[i]; !g.Equal(w) {
								t.Fatalf("point %d: got %v, want %v (k=%d)", i, g, w, k)
							}
						}
					}
				}
			}
		}
	}
}

// TestWindowBoundsCapAndSaturation pins the error classes at the cap and at
// time.Duration's edges, in both directions.
func TestWindowBoundsCapAndSaturation(t *testing.T) {
	t.Parallel()
	const day = 24 * time.Hour
	base := tAt(0)
	last := base.Add(2 * day)
	minute := func(n int) time.Time { return base.Add(time.Duration(2+n) * time.Minute) }
	minuteLast := base.Add(2 * time.Minute)
	farDay := last.AddDate(0, 0, 106751)
	// A step wider than half a time.Duration: one or two grid points are still
	// expressible, a third is not.
	hugeStep := time.Duration(1) << 62

	tests := []struct {
		name    string
		last    time.Time
		step    time.Duration
		from    time.Time
		to      time.Time
		wantK0  int
		wantK1  int
		wantErr error
	}{
		{"exactly the cap forward", minuteLast, time.Minute, minute(1), minute(MaxForecastPoints), 1, MaxForecastPoints, nil},
		{"one over the cap forward", minuteLast, time.Minute, minute(1), minute(MaxForecastPoints + 1), 0, 0, ErrTooManyPoints},
		{"exactly the cap backward", minuteLast, time.Minute, minute(-MaxForecastPoints + 1), minute(0), -MaxForecastPoints + 1, 0, nil},
		{"one over the cap backward", minuteLast, time.Minute, minute(-MaxForecastPoints), minute(0), 0, 0, ErrTooManyPoints},
		{"single point at the last timestamp", minuteLast, time.Minute, minute(0), minute(0), 0, 0, nil},
		{"one step before the last point", minuteLast, time.Minute, minute(-1), minute(-1), -1, -1, nil},
		{"straddling the last point", minuteLast, time.Minute, minute(-1), minute(1), -1, 1, nil},
		{"off-grid in the past", minuteLast, time.Minute, minute(-1).Add(30 * time.Second), minute(-1).Add(30 * time.Second), 0, 0, ErrEmptyRange},
		{"inverted", minuteLast, time.Minute, minute(1), minute(0), 0, 0, ErrRange},
		{"zero step", last, 0, last, last, 0, 0, ErrNoFrequency},
		{"negative step", last, -time.Second, last, last, 0, 0, ErrNoFrequency},
		{"saturated future", last, day, last.Add(day), last.AddDate(300, 0, 0), 0, 0, ErrTooManyPoints},
		{"saturated past", last, day, last.AddDate(-300, 0, 0), last.AddDate(-300, 0, 0), 0, 0, ErrTooManyPoints},
		{"overflowing ceiling near saturation", last, day, farDay, farDay, 106751, 106751, nil},
		{"huge step, one point", last, hugeStep, last, last, 0, 0, nil},
		{"huge step, two points", last, hugeStep, last, last.Add(hugeStep), 0, 1, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k0, k1, err := WindowBounds(tt.last, tt.step, tt.from, tt.to)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("err = %v, want %v", err, tt.wantErr)
			}
			if tt.wantErr != nil {
				return
			}
			if k0 != tt.wantK0 || k1 != tt.wantK1 {
				t.Fatalf("k = %d..%d, want %d..%d", k0, k1, tt.wantK0, tt.wantK1)
			}
		})
	}
}

// TestWindowBoundsRandomSweep sweeps reproducible pseudo-random grids across
// time.Duration's whole range (including saturated windows) and checks the
// invariants a caller relies on: the returned bounds are minimal and maximal,
// the cap and duration limits hold, and ForecastRange emits exactly that grid
// whenever the window is small enough to be worth materializing.
func TestWindowBoundsRandomSweep(t *testing.T) {
	t.Parallel()
	rng := rand.New(rand.NewPCG(7, 11))
	steps := []time.Duration{1, time.Microsecond, time.Millisecond, time.Second, time.Minute, time.Hour, 24 * time.Hour, 7 * 24 * time.Hour}
	for i := range 4000 {
		last := time.Unix(rng.Int64N(4_000_000_000)-2_000_000_000, 0).UTC()
		step := steps[rng.IntN(len(steps))]
		fromOff, toOff := randomOffset(rng), randomOffset(rng)
		// Deterministically walk the saturation branches too.
		switch {
		case i%499 == 0:
			fromOff = minDuration
		case i%499 == 1:
			toOff = maxDuration
		}
		from, to := last.Add(fromOff), last.Add(toOff)

		k0, k1, err := WindowBounds(last, step, from, to)
		lo, hi := from.Sub(last), to.Sub(last)
		saturated := lo == minDuration || hi == maxDuration

		switch {
		case errors.Is(err, ErrRange):
			if !from.After(to) {
				t.Fatalf("iter %d: ErrRange for %v..%v", i, from, to)
			}
			continue
		case errors.Is(err, ErrTooManyPoints):
			if saturated {
				continue
			}
			c0, c1 := int(ceilDuration(lo, step)), int(floorDuration(hi, step))
			maxK := int64(maxDuration) / int64(step)
			overCap := c1 >= c0 && uint64(c1)-uint64(c0) >= MaxForecastPoints
			overDuration := int64(c1) > maxK || int64(c0) < -maxK
			if !overCap && !overDuration {
				t.Fatalf("iter %d: ErrTooManyPoints but count=%d..%d fits", i, c0, c1)
			}
			continue
		case errors.Is(err, ErrEmptyRange):
			if saturated {
				t.Fatalf("iter %d: ErrEmptyRange on a saturated bound", i)
			}
			if ceilDuration(lo, step) <= floorDuration(hi, step) {
				t.Fatalf("iter %d: ErrEmptyRange but grid is non-empty", i)
			}
			continue
		case err != nil:
			t.Fatalf("iter %d: unexpected %v", i, err)
		}

		if from.After(to) {
			t.Fatalf("iter %d: nil error for an inverted window", i)
		}
		if k1 < k0 {
			t.Fatalf("iter %d: inverted bounds %d..%d", i, k0, k1)
		}
		if uint64(k1)-uint64(k0) >= MaxForecastPoints {
			t.Fatalf("iter %d: count over the cap", i)
		}
		first := last.Add(time.Duration(k0) * step)
		lastPt := last.Add(time.Duration(k1) * step)
		if first.Before(from) || lastPt.After(to) {
			t.Fatalf("iter %d: bounds %d..%d escape %v..%v", i, k0, k1, from, to)
		}
		// The neighbouring grid point exists only while its offset still fits a
		// time.Duration; the bound itself stops at that edge.
		maxK := int64(maxDuration) / int64(step)
		if int64(k0) > -maxK {
			if prev := last.Add(time.Duration(k0-1) * step); !prev.Before(from) {
				t.Fatalf("iter %d: k0=%d is not minimal (%v >= %v)", i, k0, prev, from)
			}
		}
		if int64(k1) < maxK {
			if next := last.Add(time.Duration(k1+1) * step); !next.After(to) {
				t.Fatalf("iter %d: k1=%d is not maximal (%v <= %v)", i, k1, next, to)
			}
		}

		if k1-k0+1 <= 64 {
			got, err := naiveOnGrid(t, last, step).ForecastRange(from, to)
			if err != nil {
				t.Fatalf("iter %d: ForecastRange: %v", i, err)
			}
			if got.Len() != k1-k0+1 || !got.Times()[0].Equal(first) || !got.Times()[got.Len()-1].Equal(lastPt) {
				t.Fatalf("iter %d: ForecastRange %d points (%v..%v), want %d (%v..%v)", i, got.Len(), got.Times()[0], got.Times()[got.Len()-1], k1-k0+1, first, lastPt)
			}
		}
	}
}

// randomOffset spans ordinary windows, windows near time.Duration's edges, and
// anything between, without overflowing the arithmetic that builds them.
func randomOffset(rng *rand.Rand) time.Duration {
	const day = 24 * time.Hour
	sign := time.Duration(1)
	if rng.IntN(2) == 0 {
		sign = -1
	}
	switch rng.IntN(3) {
	case 0: // near the saturation point
		return sign * (time.Duration(math.MaxInt64-1) - time.Duration(rng.Int64N(int64(48*time.Hour))))
	case 1: // ordinary years
		days := rng.Int64N(250 * 365)
		return sign * (time.Duration(days)*day + time.Duration(rng.Int64N(int64(day))))
	default: // minutes around the grid
		return sign * time.Duration(rng.Int64N(int64(10*time.Hour)))
	}
}
