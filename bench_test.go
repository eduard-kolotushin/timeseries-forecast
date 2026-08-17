package forecast

import (
	"testing"
	"time"

	"github.com/eduard-kolotushin/timeseries"
)

func benchSeries(n int) timeseries.Series[float64] {
	times := make([]time.Time, n)
	values := make([]float64, n)
	for i := range n {
		times[i] = time.Unix(int64(i), 0).UTC()
		values[i] = float64(i)
	}
	return timeseries.MustNew(times, values)
}

func benchMinuteSeries(n int) timeseries.Series[float64] {
	times := make([]time.Time, n)
	values := make([]float64, n)
	start := time.Date(2026, 1, 12, 0, 0, 0, 0, time.UTC)
	for i := range n {
		times[i] = start.Add(time.Duration(i) * time.Minute)
		values[i] = float64(i)
	}
	return timeseries.MustNew(times, values)
}

func BenchmarkFitSES(b *testing.B) {
	s := benchSeries(10_000)
	b.ResetTimer()
	for b.Loop() {
		_, _ = FitSES(s, 0.3)
	}
}

func BenchmarkFitHolt(b *testing.B) {
	s := benchSeries(10_000)
	b.ResetTimer()
	for b.Loop() {
		_, _ = FitHolt(s, 0.8, 0.2)
	}
}

func BenchmarkForecastNaive(b *testing.B) {
	m, err := FitNaive(benchSeries(1_000))
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for b.Loop() {
		_, _ = m.Forecast(64)
	}
}

func BenchmarkForecastIntervalNaive(b *testing.B) {
	m, err := FitNaive(benchSeries(1_000))
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for b.Loop() {
		_, _, _ = m.ForecastInterval(64, 0.95)
	}
}

func BenchmarkFitSeasonalBaseline(b *testing.B) {
	s := benchSeries(10_000)
	b.ResetTimer()
	for b.Loop() {
		_, _ = FitSeasonalBaseline(s, SeasonHour, nil)
	}
}

func BenchmarkFitSeasonalBaselineMinuteOfWeek(b *testing.B) {
	s := benchMinuteSeries(10_000)
	b.ResetTimer()
	for b.Loop() {
		_, _ = FitSeasonalBaseline(s, SeasonMinuteOfWeek, nil)
	}
}
