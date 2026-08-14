package forecast

import (
	"bytes"
	_ "embed"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"
	_ "time/tzdata"
)

//go:embed calendars/ru.csv
var ruCalendarCSV []byte

// DayClass is workday, weekend, or holiday.
type DayClass int

const (
	ClassWorkday DayClass = iota
	ClassWeekend
	ClassHoliday
)

type civilDate struct {
	year  int
	month time.Month
	day   int
}

// Calendar is a production calendar used by seasonal baseline.
// Nil means calendar off: UTC, weekend = Saturday/Sunday, no holidays.
type Calendar struct {
	name  string
	loc   *time.Location
	years map[int]struct{}
	class map[civilDate]DayClass
}

// Name is the calendar id ("ru"), or empty.
func (c *Calendar) Name() string {
	if c == nil {
		return ""
	}
	return c.name
}

// Location is the civil timezone used to classify dates.
func (c *Calendar) Location() *time.Location {
	if c == nil {
		return time.UTC
	}
	return c.loc
}

// Classify returns the day class of t.
// Years not present in the calendar file are treated as calendar off
// (UTC, Saturday/Sunday weekend, no holidays), so a 2026 table cannot
// affect 2015 timestamps via timezone or holiday rules.
func (c *Calendar) Classify(t time.Time) DayClass {
	if c == nil || c.years == nil || c.loc == nil {
		return weekendOrWorkday(t.In(time.UTC).Weekday())
	}
	local := t.In(c.loc)
	y, m, d := local.Date()
	if _, ok := c.years[y]; !ok {
		return weekendOrWorkday(t.In(time.UTC).Weekday())
	}
	if cl, ok := c.class[civilDate{y, m, d}]; ok {
		return cl
	}
	return ClassWorkday
}

// zoneFor is UTC when cal is off or t's civil year (in cal.Location) is not in the file.
func zoneFor(cal *Calendar, t time.Time) *time.Location {
	if cal == nil || cal.loc == nil || cal.years == nil {
		return time.UTC
	}
	if _, ok := cal.years[t.In(cal.loc).Year()]; !ok {
		return time.UTC
	}
	return cal.loc
}

func weekendOrWorkday(wd time.Weekday) DayClass {
	if wd == time.Saturday || wd == time.Sunday {
		return ClassWeekend
	}
	return ClassWorkday
}

var (
	calendarRU     *Calendar
	calendarRUErr  error
	calendarRUOnce sync.Once
)

// CalendarRU returns the embedded Russian production calendar (Europe/Moscow).
func CalendarRU() (*Calendar, error) {
	calendarRUOnce.Do(func() {
		loc, err := time.LoadLocation("Europe/Moscow")
		if err != nil {
			calendarRUErr = err
			return
		}
		cal, err := ParseProductionCalendar(ruCalendarCSV, loc)
		if err != nil {
			calendarRUErr = err
			return
		}
		cal.name = "ru"
		calendarRU = cal
	})
	return calendarRU, calendarRUErr
}

// CalendarByName returns a built-in calendar.
// Empty, "off", and "none" mean calendar off (nil, nil).
func CalendarByName(name string) (*Calendar, error) {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "", "off", "none":
		return nil, nil
	case "ru":
		return CalendarRU()
	default:
		return nil, fmt.Errorf("%w: %s", ErrUnknownCalendar, name)
	}
}

// ParseProductionCalendar parses a Russian-style production calendar CSV.
// Month columns list special days; "*" is a shortened workday; "+" is extra rest.
// Weekday listed days (bare or "+") are holidays; listed Saturday/Sunday are weekends.
// Unlisted days in a known year, including working weekends, are workdays.
func ParseProductionCalendar(data []byte, loc *time.Location) (*Calendar, error) {
	if loc == nil {
		loc = time.UTC
	}
	cal := &Calendar{
		loc:   loc,
		years: make(map[int]struct{}),
		class: make(map[civilDate]DayClass),
	}
	lines := bytes.Split(data, []byte("\n"))
	parsed := 0
	for _, raw := range lines {
		line := strings.TrimSpace(string(bytes.TrimSuffix(raw, []byte("\r"))))
		if line == "" {
			continue
		}
		fields := strings.Split(line, ";")
		if len(fields) < 13 {
			continue
		}
		year, err := strconv.Atoi(strings.TrimSpace(fields[0]))
		if err != nil {
			continue
		}
		cal.years[year] = struct{}{}
		parsed++
		for month := 1; month <= 12; month++ {
			if err := parseMonthDays(cal, year, time.Month(month), fields[month]); err != nil {
				return nil, err
			}
		}
	}
	if parsed == 0 {
		return nil, fmt.Errorf("forecast: production calendar has no year rows")
	}
	return cal, nil
}

func parseMonthDays(cal *Calendar, year int, month time.Month, cell string) error {
	cell = strings.TrimSpace(cell)
	if cell == "" {
		return nil
	}
	for _, tok := range strings.Split(cell, ",") {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		short := strings.HasSuffix(tok, "*")
		extra := strings.HasSuffix(tok, "+")
		num := tok
		if short || extra {
			num = tok[:len(tok)-1]
		}
		day, err := strconv.Atoi(num)
		if err != nil || day < 1 || day > 31 {
			return fmt.Errorf("forecast: invalid calendar day %q", tok)
		}
		if short {
			continue
		}
		t := time.Date(year, month, day, 12, 0, 0, 0, cal.loc)
		if t.Month() != month || t.Day() != day {
			return fmt.Errorf("forecast: invalid calendar date %d-%02d-%02d", year, month, day)
		}
		key := civilDate{year, month, day}
		if t.Weekday() == time.Saturday || t.Weekday() == time.Sunday {
			cal.class[key] = ClassWeekend
		} else {
			cal.class[key] = ClassHoliday
		}
	}
	return nil
}
