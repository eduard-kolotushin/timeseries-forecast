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
	// SeasonHourOfWeek buckets by (weekday, hour), with holidays in a dedicated hour profile.
	// Empty weekday hours use that weekday's mean, then overall; they do not copy another weekday.
	SeasonHourOfWeek
)

const (
	nHour     = 24
	nClass    = 3
	nDOW      = 7
	nHourKeys = nClass * nHour // 72
	nDayKeys  = nDOW + 1       // 7 weekdays + holiday
	nWeekKeys = nDOW*nHour + nHour
)

func (s Seasonality) nKeys() int {
	switch s {
	case SeasonHour:
		return nHourKeys
	case SeasonDay:
		return nDayKeys
	case SeasonHourOfWeek:
		return nWeekKeys
	default:
		return 0
	}
}

func seasonKey(season Seasonality, class DayClass, hour int, dow time.Weekday) int {
	switch season {
	case SeasonHour:
		return int(class)*nHour + hour
	case SeasonDay:
		if class == ClassHoliday {
			return nDOW
		}
		return int(dow)
	case SeasonHourOfWeek:
		if class == ClassHoliday {
			return nDOW*nHour + hour
		}
		return int(dow)*nHour + hour
	default:
		return 0
	}
}

// FitSeasonalBaseline forecasts the mean of historical values that share a seasonal key.
// cal may be nil (calendar off: UTC, weekend = Sat/Sun, no holidays).
// Hour-of-week does not copy one weekday onto another: empty hours use that
// weekday's mean, then the overall mean. Holidays keep a separate hour profile.
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

	loc := time.UTC
	if cal != nil {
		loc = cal.Location()
	}

	sum := make([]float64, nKeys)
	count := make([]int, nKeys)
	var classHourSum [nClass][nHour]float64
	var classHourN [nClass][nHour]int
	var classSum [nClass]float64
	var classN [nClass]int
	var dowSum [nDOW]float64
	var dowN [nDOW]int
	var overallSum float64
	var overallN int

	for i, t := range p.times {
		v := p.values[i]
		local := t.In(loc)
		class := cal.Classify(t)
		hour := local.Hour()
		dow := local.Weekday()
		key := seasonKey(season, class, hour, dow)
		sum[key] += v
		count[key]++
		classHourSum[class][hour] += v
		classHourN[class][hour]++
		classSum[class] += v
		classN[class]++
		if class != ClassHoliday {
			dowSum[dow] += v
			dowN[dow]++
		}
		overallSum += v
		overallN++
	}

	overall := overallSum / float64(overallN)
	means := fillBaselineMeans(season, sum, count, classHourSum, classHourN, classSum, classN, dowSum, dowN, overall)

	last := p.last()
	step := p.step
	return pointForecast{
		lastTime: last,
		step:     step,
		at: func(k int) float64 {
			t := last.Add(time.Duration(k) * step)
			local := t.In(loc)
			key := seasonKey(season, cal.Classify(t), local.Hour(), local.Weekday())
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
	classSum [nClass]float64,
	classN [nClass]int,
	dowSum [nDOW]float64,
	dowN [nDOW]int,
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
	classMean := func(class DayClass) (float64, bool) {
		n := classN[class]
		if n == 0 {
			return 0, false
		}
		return classSum[class] / float64(n), true
	}
	fallbackHour := func(class DayClass, hour int) float64 {
		if class == ClassHoliday {
			if v, ok := hourMean(ClassWeekend, hour); ok {
				return v
			}
		}
		if class == ClassHoliday || class == ClassWeekend {
			if v, ok := hourMean(ClassWorkday, hour); ok {
				return v
			}
		}
		if v, ok := hourMean(class, hour); ok {
			return v
		}
		return overall
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
		for dow := 0; dow < nDOW; dow++ {
			for hour := 0; hour < nHour; hour++ {
				key := dow*nHour + hour
				if count[key] > 0 {
					means[key] = sum[key] / float64(count[key])
					continue
				}
				// Same weekday only: do not copy another day's hour profile
				// (Saturday must not fill Sunday).
				if n := dowN[dow]; n > 0 {
					means[key] = dowSum[dow] / float64(n)
					continue
				}
				means[key] = overall
			}
		}
		for hour := 0; hour < nHour; hour++ {
			key := nDOW*nHour + hour
			if count[key] > 0 {
				means[key] = sum[key] / float64(count[key])
				continue
			}
			means[key] = fallbackHour(ClassHoliday, hour)
		}
	}
	if math.IsNaN(overall) {
		for i := range means {
			means[i] = math.NaN()
		}
	}
	return means
}
