package forecast

import (
	"math"
	"testing"
	"time"

	"github.com/eduard-kolotushin/timeseries"
)

func mustSeries(times []time.Time, values []float64) timeseries.Series[float64] {
	return timeseries.MustNew(times, values)
}

func TestFitSeasonalBaselineInvalidSeason(t *testing.T) {
	t.Parallel()
	s := series(1, 2, 3)
	if _, err := FitSeasonalBaseline(s, 0, nil); err != ErrInvalidSeason {
		t.Fatalf("season: %v", err)
	}
}

func TestSeasonalBaselineHourWeekdayVsWeekend(t *testing.T) {
	t.Parallel()
	// Hourly series: weekday 10:00 = 10, weekend 10:00 = 20, other hours = 1.
	start := time.Date(2026, 1, 12, 0, 0, 0, 0, time.UTC) // Monday
	n := 24 * 7
	times := make([]time.Time, n)
	values := make([]float64, n)
	for i := range n {
		t0 := start.Add(time.Duration(i) * time.Hour)
		times[i] = t0
		values[i] = 1
		if t0.Hour() == 10 {
			if t0.Weekday() == time.Saturday || t0.Weekday() == time.Sunday {
				values[i] = 20
			} else {
				values[i] = 10
			}
		}
	}
	s := mustSeries(times, values)
	m, err := FitSeasonalBaseline(s, SeasonHour, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Last point is Sunday 23:00. k=11 → Monday 10:00; k=11+24*5 = Saturday 10:00.
	fc, err := m.Forecast(24*6 + 11)
	if err != nil {
		t.Fatal(err)
	}
	monday10 := fc.Values()[10] // Monday 00:00 is k=1 (index 0); Monday 10:00 is k=11 (index 10)
	if monday10 != 10 {
		t.Fatalf("weekday 10:00 got %v", monday10)
	}
	sat10 := fc.Values()[10+24*5]
	if sat10 != 20 {
		t.Fatalf("weekend 10:00 got %v", sat10)
	}
}

func TestSeasonalBaselineHolidayNotWeekday(t *testing.T) {
	t.Parallel()
	cal, err := CalendarRU()
	if err != nil {
		t.Fatal(err)
	}
	// 2025 is missing from the RU file → Tuesday is a workday. Jan 6 2026 is a holiday Tuesday.
	s := mustSeries(
		[]time.Time{mskDate(2025, 1, 14, 12), mskDate(2026, 1, 5, 12), mskDate(2026, 1, 6, 12)},
		[]float64{10, 100, 100},
	)
	m, err := FitSeasonalBaseline(s, SeasonDay, cal)
	if err != nil {
		t.Fatal(err)
	}
	fc, err := m.Forecast(7)
	if err != nil {
		t.Fatal(err)
	}
	if fc.Values()[0] != 100 {
		t.Fatalf("holiday day got %v want 100", fc.Values()[0])
	}
	if fc.Values()[6] != 10 {
		t.Fatalf("workday Tuesday got %v want 10 (holiday must not leak)", fc.Values()[6])
	}
}

func TestSeasonalBaselineCalendarOffNoHoliday(t *testing.T) {
	t.Parallel()
	// Jan 1 2026 12:00 UTC is Thursday (New Year). Calendar off → workday, not holiday.
	times := []time.Time{
		time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC),
		time.Date(2026, 1, 2, 12, 0, 0, 0, time.UTC),
	}
	s := mustSeries(times, []float64{7, 8})
	m, err := FitSeasonalBaseline(s, SeasonDay, nil)
	if err != nil {
		t.Fatal(err)
	}
	fc, err := m.Forecast(1)
	if err != nil {
		t.Fatal(err)
	}
	// Jan 3 is Saturday → weekend bucket empty → fallback workday class mean (7+8)/2 = 7.5
	if fc.Values()[0] != 7.5 {
		t.Fatalf("calendar off saturday fallback got %v", fc.Values()[0])
	}
	if math.IsNaN(fc.Values()[0]) {
		t.Fatal("nan")
	}
}

func TestSeasonalBaselineHourOfWeek(t *testing.T) {
	t.Parallel()
	start := time.Date(2026, 1, 12, 0, 0, 0, 0, time.UTC)
	n := 24 * 7
	times := make([]time.Time, n)
	values := make([]float64, n)
	for i := range n {
		t0 := start.Add(time.Duration(i) * time.Hour)
		times[i] = t0
		values[i] = float64(int(t0.Weekday())*100 + t0.Hour())
	}
	m, err := FitSeasonalBaseline(mustSeries(times, values), SeasonHourOfWeek, nil)
	if err != nil {
		t.Fatal(err)
	}
	fc, err := m.Forecast(24)
	if err != nil {
		t.Fatal(err)
	}
	// last Sunday 23:00, k=1 Monday 00:00 → 1*100+0 = 100
	if fc.Values()[0] != 100 {
		t.Fatalf("hour-of-week monday 00 got %v", fc.Values()[0])
	}
	if fc.Values()[10] != 110 {
		t.Fatalf("hour-of-week monday 10 got %v", fc.Values()[10])
	}
}
