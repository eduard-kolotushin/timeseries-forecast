package forecast

import (
	"math"
	"time"

	"github.com/eduard-kolotushin/timeseries"
)

// Seasonality selects the seasonal baseline bucket.
type Seasonality int

const (
	// SeasonHour buckets by (day class, hour of day).
	SeasonHour Seasonality = iota + 1
	// SeasonDay buckets holidays separately and other days by weekday.
	SeasonDay
	// SeasonHourOfWeek buckets by (day class, weekday, hour).
	// Weekend Saturday is not a working Saturday; Sunday does not copy Saturday.
	// Empty slots use that (class, weekday) mean, then overall; holidays also fall back by hour.
	SeasonHourOfWeek
	// SeasonMinuteOfWeek buckets by (day class, weekday, minute of day).
	// Same calendar rules as hour-of-week; empty slots use that (class, weekday)
	// mean, then overall; holidays also fall back by minute of day.
	SeasonMinuteOfWeek
)

const (
	nHour           = 24
	nHourMinutes    = 60
	nMinute         = nHour * nHourMinutes // 1440
	nClass          = 3
	nDOW            = 7
	nHourKeys       = nClass * nHour          // 72
	nDayKeys        = nDOW + 1                // 7 weekdays + holiday
	nWeekKeys       = nClass * nDOW * nHour   // 504
	nMinuteWeekKeys = nClass * nDOW * nMinute // 30240
)

func (s Seasonality) nKeys() int {
	switch s {
	case SeasonHour:
		return nHourKeys
	case SeasonDay:
		return nDayKeys
	case SeasonHourOfWeek:
		return nWeekKeys
	case SeasonMinuteOfWeek:
		return nMinuteWeekKeys
	default:
		return 0
	}
}

func seasonSlot(season Seasonality, local time.Time) int {
	if season == SeasonMinuteOfWeek {
		return local.Hour()*nHourMinutes + local.Minute()
	}
	return local.Hour()
}

func seasonKey(season Seasonality, class DayClass, slot int, dow time.Weekday) int {
	switch season {
	case SeasonHour:
		return int(class)*nHour + slot
	case SeasonDay:
		if class == ClassHoliday {
			return nDOW
		}
		return int(dow)
	case SeasonHourOfWeek:
		return int(class)*nDOW*nHour + int(dow)*nHour + slot
	case SeasonMinuteOfWeek:
		return int(class)*nDOW*nMinute + int(dow)*nMinute + slot
	default:
		return 0
	}
}

// FitSeasonalBaseline forecasts the mean of historical values that share a seasonal key.
// cal may be nil (calendar off: UTC, weekend = Sat/Sun, no holidays).
// A calendar applies only to timestamps whose civil year is in the file;
// other years use calendar-off rules (a 2026 table does not affect 2015).
// Hour-of-week keys (class, weekday, hour) and minute-of-week keys
// (class, weekday, minute of day) so the calendar's weekend/holiday
// classes are distinct from workdays on the same weekday. Empty slots use that
// (class, weekday) mean, then overall; they do not copy another weekday.
func FitSeasonalBaseline(s timeseries.Series[float64], season Seasonality, cal *Calendar) (Fitted, error) {
	nKeys := season.nKeys()
	if nKeys == 0 {
		return nil, ErrInvalidSeason
	}
	p, err := prepare(s)
	if err != nil {
		return nil, err
	}
	if err := p.requireStep(); err != nil {
		return nil, err
	}

	sum := make([]float64, nKeys)
	count := make([]int, nKeys)
	var classHourSum [nClass][nHour]float64
	var classHourN [nClass][nHour]int
	var classMinuteSum [nClass][nMinute]float64
	var classMinuteN [nClass][nMinute]int
	var classSum [nClass]float64
	var classN [nClass]int
	var classDowSum [nClass][nDOW]float64
	var classDowN [nClass][nDOW]int
	var overallSum float64
	var overallN int

	for i, t := range p.times {
		v := p.values[i]
		local := t.In(zoneFor(cal, t))
		class := cal.Classify(t)
		hour := local.Hour()
		slot := seasonSlot(season, local)
		dow := local.Weekday()
		key := seasonKey(season, class, slot, dow)
		sum[key] += v
		count[key]++
		classHourSum[class][hour] += v
		classHourN[class][hour]++
		if season == SeasonMinuteOfWeek {
			classMinuteSum[class][slot] += v
			classMinuteN[class][slot]++
		}
		classSum[class] += v
		classN[class]++
		classDowSum[class][dow] += v
		classDowN[class][dow]++
		overallSum += v
		overallN++
	}

	overall := overallSum / float64(overallN)
	means := fillBaselineMeans(season, sum, count, classHourSum, classHourN, classMinuteSum, classMinuteN, classSum, classN, classDowSum, classDowN, overall)

	last := p.last()
	step := p.step
	return pointForecast{
		lastTime: last,
		step:     step,
		at: func(k int) float64 {
			t := last.Add(time.Duration(k) * step)
			local := t.In(zoneFor(cal, t))
			key := seasonKey(season, cal.Classify(t), seasonSlot(season, local), local.Weekday())
			return means[key]
		},
	}, nil
}

func fillBaselineMeans(
	season Seasonality,
	sum []float64,
	count []int,
	classHourSum [nClass][nHour]float64,
	classHourN [nClass][nHour]int,
	classMinuteSum [nClass][nMinute]float64,
	classMinuteN [nClass][nMinute]int,
	classSum [nClass]float64,
	classN [nClass]int,
	classDowSum [nClass][nDOW]float64,
	classDowN [nClass][nDOW]int,
	overall float64,
) []float64 {
	means := make([]float64, len(count))
	hourMean := func(class DayClass, hour int) (float64, bool) {
		n := classHourN[class][hour]
		if n == 0 {
			return 0, false
		}
		return classHourSum[class][hour] / float64(n), true
	}
	minuteMean := func(class DayClass, minute int) (float64, bool) {
		n := classMinuteN[class][minute]
		if n == 0 {
			return 0, false
		}
		return classMinuteSum[class][minute] / float64(n), true
	}
	classMean := func(class DayClass) (float64, bool) {
		n := classN[class]
		if n == 0 {
			return 0, false
		}
		return classSum[class] / float64(n), true
	}
	fallbackBySlot := func(slotMean func(DayClass, int) (float64, bool), class DayClass, slot int) float64 {
		if class == ClassHoliday {
			if v, ok := slotMean(ClassWeekend, slot); ok {
				return v
			}
		}
		if class == ClassHoliday || class == ClassWeekend {
			if v, ok := slotMean(ClassWorkday, slot); ok {
				return v
			}
		}
		if v, ok := slotMean(class, slot); ok {
			return v
		}
		return overall
	}
	fallbackHour := func(class DayClass, hour int) float64 {
		return fallbackBySlot(hourMean, class, hour)
	}
	fallbackMinute := func(class DayClass, minute int) float64 {
		return fallbackBySlot(minuteMean, class, minute)
	}
	fallbackClass := func(class DayClass) float64 {
		if class == ClassHoliday {
			if v, ok := classMean(ClassWeekend); ok {
				return v
			}
		}
		if class == ClassHoliday || class == ClassWeekend {
			if v, ok := classMean(ClassWorkday); ok {
				return v
			}
		}
		if v, ok := classMean(class); ok {
			return v
		}
		return overall
	}

	switch season {
	case SeasonHour:
		for class := ClassWorkday; class <= ClassHoliday; class++ {
			for hour := 0; hour < nHour; hour++ {
				key := int(class)*nHour + hour
				if count[key] > 0 {
					means[key] = sum[key] / float64(count[key])
					continue
				}
				means[key] = fallbackHour(class, hour)
			}
		}
	case SeasonDay:
		for dow := 0; dow < nDOW; dow++ {
			if count[dow] > 0 {
				means[dow] = sum[dow] / float64(count[dow])
				continue
			}
			c := ClassWorkday
			if time.Weekday(dow) == time.Saturday || time.Weekday(dow) == time.Sunday {
				c = ClassWeekend
			}
			means[dow] = fallbackClass(c)
		}
		if count[nDOW] > 0 {
			means[nDOW] = sum[nDOW] / float64(count[nDOW])
		} else {
			means[nDOW] = fallbackClass(ClassHoliday)
		}
	case SeasonHourOfWeek:
		fillWeekSlots(means, count, sum, classDowSum, classDowN, nHour, fallbackHour, overall)
	case SeasonMinuteOfWeek:
		fillWeekSlots(means, count, sum, classDowSum, classDowN, nMinute, fallbackMinute, overall)
	}
	if math.IsNaN(overall) {
		for i := range means {
			means[i] = math.NaN()
		}
	}
	return means
}

func fillWeekSlots(
	means []float64,
	count []int,
	sum []float64,
	classDowSum [nClass][nDOW]float64,
	classDowN [nClass][nDOW]int,
	nSlot int,
	fallback func(DayClass, int) float64,
	overall float64,
) {
	for class := ClassWorkday; class <= ClassHoliday; class++ {
		for dow := 0; dow < nDOW; dow++ {
			for slot := 0; slot < nSlot; slot++ {
				key := int(class)*nDOW*nSlot + dow*nSlot + slot
				if count[key] > 0 {
					means[key] = sum[key] / float64(count[key])
					continue
				}
				if n := classDowN[class][dow]; n > 0 {
					means[key] = classDowSum[class][dow] / float64(n)
					continue
				}
				if class == ClassHoliday {
					means[key] = fallback(ClassHoliday, slot)
					continue
				}
				means[key] = overall
			}
		}
	}
}
