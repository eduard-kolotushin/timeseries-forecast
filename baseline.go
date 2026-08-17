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

	agg := newBaselineAgg(nKeys)
	for i, t := range p.times {
		v := p.values[i]
		local := t.In(zoneFor(cal, t))
		class := cal.Classify(t)
		hour := local.Hour()
		slot := seasonSlot(season, local)
		dow := local.Weekday()
		key := seasonKey(season, class, slot, dow)
		agg.add(key, class, hour, slot, dow, season, v)
	}

	overall := agg.overallSum / float64(agg.overallN)
	means, ses := fillBaselineMeans(season, agg, overall)

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
		se: func(k int) float64 {
			t := last.Add(time.Duration(k) * step)
			local := t.In(zoneFor(cal, t))
			key := seasonKey(season, cal.Classify(t), seasonSlot(season, local), local.Weekday())
			return ses[key]
		},
	}, nil
}

type baselineAgg struct {
	sum, sumsq                       []float64
	count                            []int
	classHourSum, classHourSumsq     [nClass][nHour]float64
	classHourN                       [nClass][nHour]int
	classMinuteSum, classMinuteSumsq [nClass][nMinute]float64
	classMinuteN                     [nClass][nMinute]int
	classSum, classSumsq             [nClass]float64
	classN                           [nClass]int
	classDowSum, classDowSumsq       [nClass][nDOW]float64
	classDowN                        [nClass][nDOW]int
	overallSum, overallSumsq         float64
	overallN                         int
}

func newBaselineAgg(nKeys int) *baselineAgg {
	return &baselineAgg{
		sum:   make([]float64, nKeys),
		sumsq: make([]float64, nKeys),
		count: make([]int, nKeys),
	}
}

func (a *baselineAgg) add(key int, class DayClass, hour, slot int, dow time.Weekday, season Seasonality, v float64) {
	vv := v * v
	a.sum[key] += v
	a.sumsq[key] += vv
	a.count[key]++
	a.classHourSum[class][hour] += v
	a.classHourSumsq[class][hour] += vv
	a.classHourN[class][hour]++
	if season == SeasonMinuteOfWeek {
		a.classMinuteSum[class][slot] += v
		a.classMinuteSumsq[class][slot] += vv
		a.classMinuteN[class][slot]++
	}
	a.classSum[class] += v
	a.classSumsq[class] += vv
	a.classN[class]++
	a.classDowSum[class][dow] += v
	a.classDowSumsq[class][dow] += vv
	a.classDowN[class][dow]++
	a.overallSum += v
	a.overallSumsq += vv
	a.overallN++
}

func bucketSE(sum, sumsq float64, n int) float64 {
	if n < 2 {
		return math.NaN()
	}
	nf := float64(n)
	mean := sum / nf
	v := sumsq/nf - mean*mean
	if v < 0 {
		v = 0
	}
	return math.Sqrt(v) * math.Sqrt(1+1/nf)
}

func fillBaselineMeans(season Seasonality, a *baselineAgg, overall float64) (means, ses []float64) {
	means = make([]float64, len(a.count))
	ses = make([]float64, len(a.count))
	overallSE := bucketSE(a.overallSum, a.overallSumsq, a.overallN)
	hourMean := func(class DayClass, hour int) (float64, bool) {
		n := a.classHourN[class][hour]
		if n == 0 {
			return 0, false
		}
		return a.classHourSum[class][hour] / float64(n), true
	}
	hourSE := func(class DayClass, hour int) (float64, bool) {
		n := a.classHourN[class][hour]
		if n == 0 {
			return 0, false
		}
		return bucketSE(a.classHourSum[class][hour], a.classHourSumsq[class][hour], n), true
	}
	minuteMean := func(class DayClass, minute int) (float64, bool) {
		n := a.classMinuteN[class][minute]
		if n == 0 {
			return 0, false
		}
		return a.classMinuteSum[class][minute] / float64(n), true
	}
	minuteSE := func(class DayClass, minute int) (float64, bool) {
		n := a.classMinuteN[class][minute]
		if n == 0 {
			return 0, false
		}
		return bucketSE(a.classMinuteSum[class][minute], a.classMinuteSumsq[class][minute], n), true
	}
	classMean := func(class DayClass) (float64, bool) {
		n := a.classN[class]
		if n == 0 {
			return 0, false
		}
		return a.classSum[class] / float64(n), true
	}
	classSE := func(class DayClass) (float64, bool) {
		n := a.classN[class]
		if n == 0 {
			return 0, false
		}
		return bucketSE(a.classSum[class], a.classSumsq[class], n), true
	}
	fallbackBySlot := func(slotMean, slotSEFn func(DayClass, int) (float64, bool), class DayClass, slot int) (float64, float64) {
		if class == ClassHoliday {
			if v, ok := slotMean(ClassWeekend, slot); ok {
				se, _ := slotSEFn(ClassWeekend, slot)
				return v, se
			}
		}
		if class == ClassHoliday || class == ClassWeekend {
			if v, ok := slotMean(ClassWorkday, slot); ok {
				se, _ := slotSEFn(ClassWorkday, slot)
				return v, se
			}
		}
		if v, ok := slotMean(class, slot); ok {
			se, _ := slotSEFn(class, slot)
			return v, se
		}
		return overall, overallSE
	}
	fallbackHour := func(class DayClass, hour int) (float64, float64) {
		return fallbackBySlot(hourMean, hourSE, class, hour)
	}
	fallbackMinute := func(class DayClass, minute int) (float64, float64) {
		return fallbackBySlot(minuteMean, minuteSE, class, minute)
	}
	fallbackClass := func(class DayClass) (float64, float64) {
		if class == ClassHoliday {
			if v, ok := classMean(ClassWeekend); ok {
				se, _ := classSE(ClassWeekend)
				return v, se
			}
		}
		if class == ClassHoliday || class == ClassWeekend {
			if v, ok := classMean(ClassWorkday); ok {
				se, _ := classSE(ClassWorkday)
				return v, se
			}
		}
		if v, ok := classMean(class); ok {
			se, _ := classSE(class)
			return v, se
		}
		return overall, overallSE
	}

	switch season {
	case SeasonHour:
		for class := ClassWorkday; class <= ClassHoliday; class++ {
			for hour := 0; hour < nHour; hour++ {
				key := int(class)*nHour + hour
				if a.count[key] > 0 {
					means[key] = a.sum[key] / float64(a.count[key])
					ses[key] = bucketSE(a.sum[key], a.sumsq[key], a.count[key])
					continue
				}
				means[key], ses[key] = fallbackHour(class, hour)
			}
		}
	case SeasonDay:
		for dow := 0; dow < nDOW; dow++ {
			if a.count[dow] > 0 {
				means[dow] = a.sum[dow] / float64(a.count[dow])
				ses[dow] = bucketSE(a.sum[dow], a.sumsq[dow], a.count[dow])
				continue
			}
			c := ClassWorkday
			if time.Weekday(dow) == time.Saturday || time.Weekday(dow) == time.Sunday {
				c = ClassWeekend
			}
			means[dow], ses[dow] = fallbackClass(c)
		}
		if a.count[nDOW] > 0 {
			means[nDOW] = a.sum[nDOW] / float64(a.count[nDOW])
			ses[nDOW] = bucketSE(a.sum[nDOW], a.sumsq[nDOW], a.count[nDOW])
		} else {
			means[nDOW], ses[nDOW] = fallbackClass(ClassHoliday)
		}
	case SeasonHourOfWeek:
		fillWeekSlots(means, ses, a, nHour, fallbackHour, overall, overallSE)
	case SeasonMinuteOfWeek:
		fillWeekSlots(means, ses, a, nMinute, fallbackMinute, overall, overallSE)
	}
	if math.IsNaN(overall) {
		for i := range means {
			means[i] = math.NaN()
			ses[i] = math.NaN()
		}
	}
	return means, ses
}

func fillWeekSlots(
	means, ses []float64,
	a *baselineAgg,
	nSlot int,
	fallback func(DayClass, int) (float64, float64),
	overall, overallSE float64,
) {
	for class := ClassWorkday; class <= ClassHoliday; class++ {
		for dow := 0; dow < nDOW; dow++ {
			for slot := 0; slot < nSlot; slot++ {
				key := int(class)*nDOW*nSlot + dow*nSlot + slot
				if a.count[key] > 0 {
					means[key] = a.sum[key] / float64(a.count[key])
					ses[key] = bucketSE(a.sum[key], a.sumsq[key], a.count[key])
					continue
				}
				if n := a.classDowN[class][dow]; n > 0 {
					means[key] = a.classDowSum[class][dow] / float64(n)
					ses[key] = bucketSE(a.classDowSum[class][dow], a.classDowSumsq[class][dow], n)
					continue
				}
				if class == ClassHoliday {
					means[key], ses[key] = fallback(ClassHoliday, slot)
					continue
				}
				means[key] = overall
				ses[key] = overallSE
			}
		}
	}
}
