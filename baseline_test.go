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

func TestSeasonalBaselineHourOfWeekDoesNotCopySaturdayOntoSunday(t *testing.T) {
	t.Parallel()
	// Saturday hours 0..23 with distinct values. Sunday has no observations.
	start := time.Date(2026, 1, 10, 0, 0, 0, 0, time.UTC) // Saturday
	times := make([]time.Time, 24)
	values := make([]float64, 24)
	var sum float64
	for i := range 24 {
		times[i] = start.Add(time.Duration(i) * time.Hour)
		values[i] = float64(i)
		sum += float64(i)
	}
	overall := sum / 24
	m, err := FitSeasonalBaseline(mustSeries(times, values), SeasonHourOfWeek, nil)
	if err != nil {
		t.Fatal(err)
	}
	fc, err := m.Forecast(25)
	if err != nil {
		t.Fatal(err)
	}
	// Last is Saturday 23:00. k=1..24 are Sunday 00:00..23:00.
	for i := range 24 {
		if fc.Values()[i] != overall {
			t.Fatalf("Sunday hour %d got %v want overall %v (must not copy Saturday hour %d)", i, fc.Values()[i], overall, i)
		}
	}
	if fc.Values()[24] != overall {
		t.Fatalf("Monday 00:00 got %v want overall %v", fc.Values()[24], overall)
	}
}

func TestSeasonalBaselineHourOfWeekEmptyHourUsesSameWeekdayMean(t *testing.T) {
	t.Parallel()
	sat := time.Date(2026, 1, 10, 0, 0, 0, 0, time.UTC)
	times := make([]time.Time, 0, 27)
	values := make([]float64, 0, 27)
	for h := range 24 {
		times = append(times, sat.Add(time.Duration(h)*time.Hour))
		values = append(values, 100+float64(h))
	}
	sun := sat.Add(24 * time.Hour)
	times = append(times, sun, sun.Add(time.Hour), sun.Add(2*time.Hour))
	values = append(values, 5, 6, 7)
	sundayMean := 6.0
	m, err := FitSeasonalBaseline(mustSeries(times, values), SeasonHourOfWeek, nil)
	if err != nil {
		t.Fatal(err)
	}
	fc, err := m.Forecast(22)
	if err != nil {
		t.Fatal(err)
	}
	// Last is Sunday 02:00. k=1 is Sunday 03:00 — empty Sunday hour, not Saturday 03:00 (103).
	if fc.Values()[0] != sundayMean {
		t.Fatalf("Sunday 03:00 got %v want Sunday mean %v", fc.Values()[0], sundayMean)
	}
	if fc.Values()[20] != sundayMean {
		t.Fatalf("Sunday 23:00 got %v want Sunday mean %v", fc.Values()[20], sundayMean)
	}
	var overallSum float64
	for _, v := range values {
		overallSum += v
	}
	overall := overallSum / float64(len(values))
	if fc.Values()[21] != overall {
		t.Fatalf("Monday 00:00 got %v want overall %v", fc.Values()[21], overall)
	}
}

func TestSeasonalBaselineHourOfWeekCalendarSplitsWeekendAndHoliday(t *testing.T) {
	t.Parallel()
	// Only 10 January is listed → 10 Jan 2026 (Saturday) is weekend; 17 Jan is a working Saturday.
	cal, err := ParseProductionCalendar([]byte("2026;10;;;;;;;;;;;;\n"), time.UTC)
	if err != nil {
		t.Fatal(err)
	}
	sat := time.Date(2026, 1, 10, 12, 0, 0, 0, time.UTC)
	workSat := time.Date(2026, 1, 17, 12, 0, 0, 0, time.UTC)
	s := mustSeries([]time.Time{sat, workSat}, []float64{10, 50})

	off, err := FitSeasonalBaseline(s, SeasonHourOfWeek, nil)
	if err != nil {
		t.Fatal(err)
	}
	fcOff, err := off.Forecast(1)
	if err != nil {
		t.Fatal(err)
	}
	if fcOff.Values()[0] != 30 {
		t.Fatalf("calendar off Saturday mean got %v want 30", fcOff.Values()[0])
	}

	on, err := FitSeasonalBaseline(s, SeasonHourOfWeek, cal)
	if err != nil {
		t.Fatal(err)
	}
	fcOn, err := on.Forecast(1)
	if err != nil {
		t.Fatal(err)
	}
	// 24 Jan is unlisted Saturday → workday Saturday, not the weekend Saturday mean.
	if fcOn.Values()[0] != 50 {
		t.Fatalf("working Saturday got %v want 50 (must not mix weekend Saturday)", fcOn.Values()[0])
	}

	ru, err := CalendarRU()
	if err != nil {
		t.Fatal(err)
	}
	hol := mustSeries(
		[]time.Time{mskDate(2026, 1, 1, 12), mskDate(2026, 1, 8, 12), mskDate(2026, 1, 15, 12)},
		[]float64{100, 100, 20},
	)
	m, err := FitSeasonalBaseline(hol, SeasonHourOfWeek, ru)
	if err != nil {
		t.Fatal(err)
	}
	fc, err := m.Forecast(1)
	if err != nil {
		t.Fatal(err)
	}
	if fc.Values()[0] != 20 {
		t.Fatalf("workday Thursday got %v want 20 (holiday Thursday must not leak)", fc.Values()[0])
	}
}

func TestSeasonalBaselineUncoveredYearMatchesCalendarOff(t *testing.T) {
	t.Parallel()
	// 2015-09-12 is Saturday and is not in the RU file. RU must not shift hours to Moscow.
	start := time.Date(2015, 9, 12, 0, 0, 0, 0, time.UTC)
	times := make([]time.Time, 24)
	values := make([]float64, 24)
	for i := range 24 {
		times[i] = start.Add(time.Duration(i) * time.Hour)
		values[i] = float64(i)
	}
	s := mustSeries(times, values)
	off, err := FitSeasonalBaseline(s, SeasonHour, nil)
	if err != nil {
		t.Fatal(err)
	}
	ru, err := CalendarRU()
	if err != nil {
		t.Fatal(err)
	}
	on, err := FitSeasonalBaseline(s, SeasonHour, ru)
	if err != nil {
		t.Fatal(err)
	}
	fcOff, err := off.Forecast(3)
	if err != nil {
		t.Fatal(err)
	}
	fcOn, err := on.Forecast(3)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 3 {
		if fcOff.Values()[i] != fcOn.Values()[i] {
			t.Fatalf("k=%d off=%v ru=%v (2015 must ignore 2026 calendar)", i, fcOff.Values()[i], fcOn.Values()[i])
		}
	}
}
