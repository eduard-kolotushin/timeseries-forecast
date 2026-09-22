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

	overall := agg.overall.mean
	means, ses := fillBaselineMeans(season, agg, overall)

	return pointForecast{
		lastTime:    p.last(),
		step:        p.step,
		kind:        kindBaseline,
		means:       means,
		ses:         ses,
		seasonality: season,
		cal:         cal,
	}, nil
}

// bucketStat is Welford's streaming state for one seasonal baseline bucket: how
// many values it holds, their running mean, and the sum of squared deviations
// from that mean (m2). Accumulating the raw sum and sum of squares instead
// loses every significant digit of the variance for a series with a large
// offset, which the minute-of-week buckets hit (few samples per bucket).
type bucketStat struct {
	n    int
	mean float64
	m2   float64
}

// add folds v into the running mean and m2 (Welford's online update).
func (b *bucketStat) add(v float64) {
	b.n++
	d := v - b.mean
	b.mean += d / float64(b.n)
	b.m2 += d * (v - b.mean)
}

// se is the bucket's prediction standard error: its sample sd with the
// sqrt(1+1/n) factor, or NaN below two values.
func (b bucketStat) se() float64 {
	return bucketSE(b.m2, b.n)
}

// bucketSE turns a Welford m2 into the prediction standard error
// sqrt(m2/n)·sqrt(1+1/n). It is NaN when the bucket holds fewer than two values.
func bucketSE(m2 float64, n int) float64 {
	if n < 2 {
		return math.NaN()
	}
	if m2 < 0 {
		// m2 is a sum of non-negative products; this clamp only absorbs a
		// rounding-level negative so the square root stays real.
		m2 = 0
	}
	nf := float64(n)
	return math.Sqrt(m2/nf) * math.Sqrt(1+1/nf)
}

type baselineAgg struct {
	key         []bucketStat
	classHour   [nClass][nHour]bucketStat
	classMinute [nClass][nMinute]bucketStat
	class       [nClass]bucketStat
	classDow    [nClass][nDOW]bucketStat
	overall     bucketStat
}

func newBaselineAgg(nKeys int) *baselineAgg {
	return &baselineAgg{key: make([]bucketStat, nKeys)}
}

func (a *baselineAgg) add(key int, class DayClass, hour, slot int, dow time.Weekday, season Seasonality, v float64) {
	a.key[key].add(v)
	a.classHour[class][hour].add(v)
	if season == SeasonMinuteOfWeek {
		a.classMinute[class][slot].add(v)
	}
	a.class[class].add(v)
	a.classDow[class][dow].add(v)
	a.overall.add(v)
}

func fillBaselineMeans(season Seasonality, a *baselineAgg, overall float64) (means, ses []float64) {
	means = make([]float64, len(a.key))
	ses = make([]float64, len(a.key))
	overallSE := a.overall.se()
	hourMean := func(class DayClass, hour int) (float64, bool) {
		b := a.classHour[class][hour]
		if b.n == 0 {
			return 0, false
		}
		return b.mean, true
	}
	hourSE := func(class DayClass, hour int) (float64, bool) {
		b := a.classHour[class][hour]
		if b.n == 0 {
			return 0, false
		}
		return b.se(), true
	}
	minuteMean := func(class DayClass, minute int) (float64, bool) {
		b := a.classMinute[class][minute]
		if b.n == 0 {
			return 0, false
		}
		return b.mean, true
	}
	minuteSE := func(class DayClass, minute int) (float64, bool) {
		b := a.classMinute[class][minute]
		if b.n == 0 {
			return 0, false
		}
		return b.se(), true
	}
	classMean := func(class DayClass) (float64, bool) {
		b := a.class[class]
		if b.n == 0 {
			return 0, false
		}
		return b.mean, true
	}
	classSE := func(class DayClass) (float64, bool) {
		b := a.class[class]
		if b.n == 0 {
			return 0, false
		}
		return b.se(), true
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
				if b := a.key[key]; b.n > 0 {
					means[key] = b.mean
					ses[key] = b.se()
					continue
				}
				means[key], ses[key] = fallbackHour(class, hour)
			}
		}
	case SeasonDay:
		for dow := 0; dow < nDOW; dow++ {
			if b := a.key[dow]; b.n > 0 {
				means[dow] = b.mean
				ses[dow] = b.se()
				continue
			}
			c := ClassWorkday
			if time.Weekday(dow) == time.Saturday || time.Weekday(dow) == time.Sunday {
				c = ClassWeekend
			}
			means[dow], ses[dow] = fallbackClass(c)
		}
		if b := a.key[nDOW]; b.n > 0 {
			means[nDOW] = b.mean
			ses[nDOW] = b.se()
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
				if b := a.key[key]; b.n > 0 {
					means[key] = b.mean
					ses[key] = b.se()
					continue
				}
				if b := a.classDow[class][dow]; b.n > 0 {
					means[key] = b.mean
					ses[key] = b.se()
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
