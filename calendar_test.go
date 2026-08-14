package forecast

import (
	"errors"
	"testing"
	"time"
)

func mskDate(year int, month time.Month, day, hour int) time.Time {
	loc, err := time.LoadLocation("Europe/Moscow")
	if err != nil {
		panic(err)
	}
	return time.Date(year, month, day, hour, 0, 0, 0, loc)
}

func TestProductionCalendarRU2026(t *testing.T) {
	t.Parallel()
	cal, err := CalendarRU()
	if err != nil {
		t.Fatal(err)
	}
	if cal.Name() != "ru" {
		t.Fatalf("name: %q", cal.Name())
	}

	for _, tc := range []struct {
		name  string
		t     time.Time
		class DayClass
	}{
		{"jan1 holiday", mskDate(2026, 1, 1, 12), ClassHoliday},
		{"jan6 holiday tuesday", mskDate(2026, 1, 6, 12), ClassHoliday},
		{"jan9 plus holiday", mskDate(2026, 1, 9, 12), ClassHoliday},
		{"jan10 weekend", mskDate(2026, 1, 10, 12), ClassWeekend},
		{"jan12 workday", mskDate(2026, 1, 12, 12), ClassWorkday},
		{"may8 short workday", mskDate(2026, 5, 8, 12), ClassWorkday},
		{"missing year saturday", mskDate(2025, 1, 4, 12), ClassWeekend},
		{"missing year monday", mskDate(2025, 1, 6, 12), ClassWorkday},
		{"uncovered 2015 new year utc thursday", time.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC), ClassWorkday},
		{"uncovered 2015 saturday utc", time.Date(2015, 9, 12, 0, 30, 0, 0, time.UTC), ClassWeekend},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := cal.Classify(tc.t); got != tc.class {
				t.Fatalf("class=%d want %d", got, tc.class)
			}
		})
	}
}

func TestCalendarOffUTC(t *testing.T) {
	t.Parallel()
	var cal *Calendar
	sat := time.Date(2026, 1, 10, 12, 0, 0, 0, time.UTC) // Saturday
	mon := time.Date(2026, 1, 12, 12, 0, 0, 0, time.UTC) // Monday
	if cal.Classify(sat) != ClassWeekend {
		t.Fatalf("saturday: %d", cal.Classify(sat))
	}
	if cal.Classify(mon) != ClassWorkday {
		t.Fatalf("monday: %d", cal.Classify(mon))
	}
	holidayUTC := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC) // Thursday, New Year
	if cal.Classify(holidayUTC) != ClassWorkday {
		t.Fatalf("calendar off must not use holiday class: %d", cal.Classify(holidayUTC))
	}
}

func TestCalendarUncoveredYearUsesUTCNotMoscow(t *testing.T) {
	t.Parallel()
	cal, err := CalendarRU()
	if err != nil {
		t.Fatal(err)
	}
	// 2015 is not in ru.csv. 00:30 UTC must stay hour 0, not Moscow 03:30.
	t0 := time.Date(2015, 9, 12, 0, 30, 0, 0, time.UTC)
	if zoneFor(cal, t0) != time.UTC {
		t.Fatal("uncovered year must use UTC")
	}
	if cal.Classify(t0) != ClassWeekend {
		t.Fatalf("saturday utc: %d", cal.Classify(t0))
	}
	t2026 := time.Date(2026, 1, 6, 12, 0, 0, 0, time.UTC) // holiday Tuesday in Moscow
	if zoneFor(cal, t2026) == time.UTC {
		t.Fatal("covered year must use calendar location")
	}
}

func TestCalendarByName(t *testing.T) {
	t.Parallel()
	off, err := CalendarByName("off")
	if err != nil || off != nil {
		t.Fatalf("off: %v %v", off, err)
	}
	ru, err := CalendarByName("ru")
	if err != nil || ru == nil || ru.Name() != "ru" {
		t.Fatalf("ru: %v %v", ru, err)
	}
	_, err = CalendarByName("us")
	if !errors.Is(err, ErrUnknownCalendar) {
		t.Fatalf("unknown: %v", err)
	}
}

func TestParseProductionCalendarEmpty(t *testing.T) {
	t.Parallel()
	if _, err := ParseProductionCalendar([]byte("header only\n"), time.UTC); err == nil {
		t.Fatal("expected error")
	}
}
