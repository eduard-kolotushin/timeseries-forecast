package forecast

import (
	"encoding/json"
	"fmt"
	"math"
	"time"
)

// SnapshotVersion is the envelope version written by SnapshotOf.
const SnapshotVersion = 1

const (
	kindNaive    = "naive"
	kindMean     = "mean"
	kindDrift    = "drift"
	kindSeasonal = "seasonal"
	kindBaseline = "baseline"
	kindSES      = "ses"
	kindHolt     = "holt"
)

// Snapshot is a versioned envelope of fitted state. Restore rebuilds a Fitted
// that can ForecastRange without the training series.
type Snapshot struct {
	V    int             `json:"v"`
	Kind string          `json:"kind"`
	Last int64           `json:"last"` // unix milliseconds
	Step int64           `json:"step"` // nanoseconds
	Data json.RawMessage `json:"data"`
}

type jsonFloat float64

func (f jsonFloat) MarshalJSON() ([]byte, error) {
	x := float64(f)
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return []byte("null"), nil
	}
	return json.Marshal(x)
}

func (f *jsonFloat) UnmarshalJSON(b []byte) error {
	if string(b) == "null" {
		*f = jsonFloat(math.NaN())
		return nil
	}
	var v float64
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}
	*f = jsonFloat(v)
	return nil
}

func floatsToJSON(in []float64) []jsonFloat {
	out := make([]jsonFloat, len(in))
	for i, v := range in {
		out[i] = jsonFloat(v)
	}
	return out
}

func jsonToFloats(in []jsonFloat) []float64 {
	out := make([]float64, len(in))
	for i, v := range in {
		out[i] = float64(v)
	}
	return out
}

type naiveData struct {
	Last  jsonFloat `json:"last"`
	Sigma jsonFloat `json:"sigma"`
}

type meanData struct {
	Mu    jsonFloat `json:"mu"`
	Sigma jsonFloat `json:"sigma"`
	N     int       `json:"n"`
}

type driftData struct {
	Last  jsonFloat `json:"last"`
	B     jsonFloat `json:"b"`
	Sigma jsonFloat `json:"sigma"`
	N     int       `json:"n"`
}

type seasonalData struct {
	Season []jsonFloat `json:"season"`
	Period int         `json:"period"`
	Sigma  jsonFloat   `json:"sigma"`
}

type baselineData struct {
	Season   string      `json:"season"`
	Calendar string      `json:"calendar"`
	Means    []jsonFloat `json:"means"`
	SEs      []jsonFloat `json:"ses"`
}

type sesData struct {
	Level jsonFloat `json:"level"`
	Sigma jsonFloat `json:"sigma"`
	Alpha float64   `json:"alpha"`
}

type holtData struct {
	Level jsonFloat `json:"level"`
	Trend jsonFloat `json:"trend"`
	Sigma jsonFloat `json:"sigma"`
	Alpha float64   `json:"alpha"`
	Beta  float64   `json:"beta"`
}

// SnapshotOf encodes fitted state. f must come from a Fit* in this package.
func SnapshotOf(f Fitted) (Snapshot, error) {
	p, ok := f.(pointForecast)
	if !ok {
		return Snapshot{}, ErrInvalidSnapshot
	}
	data, err := marshalKindData(p)
	if err != nil {
		return Snapshot{}, err
	}
	return Snapshot{
		V:    SnapshotVersion,
		Kind: p.kind,
		Last: p.lastTime.UTC().UnixMilli(),
		Step: int64(p.step),
		Data: data,
	}, nil
}

func marshalKindData(p pointForecast) (json.RawMessage, error) {
	var (
		body any
		err  error
	)
	switch p.kind {
	case kindNaive:
		body = naiveData{Last: jsonFloat(p.last), Sigma: jsonFloat(p.sigma)}
	case kindMean:
		body = meanData{Mu: jsonFloat(p.mu), Sigma: jsonFloat(p.sigma), N: p.n}
	case kindDrift:
		body = driftData{Last: jsonFloat(p.last), B: jsonFloat(p.b), Sigma: jsonFloat(p.sigma), N: p.n}
	case kindSeasonal:
		body = seasonalData{Season: floatsToJSON(p.season), Period: p.period, Sigma: jsonFloat(p.sigma)}
	case kindBaseline:
		// Restore maps the name back through CalendarByName, which reads "" as
		// calendar off, so a table without a built-in name cannot be reproduced.
		if p.cal != nil && p.cal.Name() == "" {
			return nil, fmt.Errorf("%w: calendar has no built-in name", ErrInvalidSnapshot)
		}
		body = baselineData{
			Season:   seasonName(p.seasonality),
			Calendar: p.cal.Name(),
			Means:    floatsToJSON(p.means),
			SEs:      floatsToJSON(p.ses),
		}
	case kindSES:
		body = sesData{Level: jsonFloat(p.level), Sigma: jsonFloat(p.sigma), Alpha: p.alpha}
	case kindHolt:
		body = holtData{
			Level: jsonFloat(p.level),
			Trend: jsonFloat(p.trend),
			Sigma: jsonFloat(p.sigma),
			Alpha: p.alpha,
			Beta:  p.beta,
		}
	default:
		err = fmt.Errorf("%w: %s", ErrUnknownSnapshot, p.kind)
	}
	if err != nil {
		return nil, err
	}
	return json.Marshal(body)
}

// Restore rebuilds a Fitted from a snapshot. Unknown version or kind is an error.
func Restore(s Snapshot) (Fitted, error) {
	if s.V != SnapshotVersion {
		return nil, fmt.Errorf("%w: v=%d", ErrUnknownSnapshot, s.V)
	}
	if s.Step <= 0 {
		return nil, ErrInvalidSnapshot
	}
	p := pointForecast{
		lastTime: time.UnixMilli(s.Last).UTC(),
		step:     time.Duration(s.Step),
		kind:     s.Kind,
	}
	switch s.Kind {
	case kindNaive:
		var d naiveData
		if err := json.Unmarshal(s.Data, &d); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrInvalidSnapshot, err)
		}
		p.last = float64(d.Last)
		p.sigma = float64(d.Sigma)
	case kindMean:
		var d meanData
		if err := json.Unmarshal(s.Data, &d); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrInvalidSnapshot, err)
		}
		if d.N <= 0 {
			return nil, ErrInvalidSnapshot
		}
		p.mu = float64(d.Mu)
		p.sigma = float64(d.Sigma)
		p.n = d.N
	case kindDrift:
		var d driftData
		if err := json.Unmarshal(s.Data, &d); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrInvalidSnapshot, err)
		}
		if d.N <= 0 {
			return nil, ErrInvalidSnapshot
		}
		p.last = float64(d.Last)
		p.b = float64(d.B)
		p.sigma = float64(d.Sigma)
		p.n = d.N
	case kindSeasonal:
		var d seasonalData
		if err := json.Unmarshal(s.Data, &d); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrInvalidSnapshot, err)
		}
		if d.Period <= 0 || len(d.Season) != d.Period {
			return nil, ErrInvalidSnapshot
		}
		p.season = jsonToFloats(d.Season)
		p.period = d.Period
		p.sigma = float64(d.Sigma)
	case kindBaseline:
		var d baselineData
		if err := json.Unmarshal(s.Data, &d); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrInvalidSnapshot, err)
		}
		season, err := seasonFromName(d.Season)
		if err != nil {
			return nil, err
		}
		cal, err := CalendarByName(d.Calendar)
		if err != nil {
			return nil, err
		}
		nKeys := season.nKeys()
		if len(d.Means) != nKeys || len(d.SEs) != nKeys {
			return nil, ErrInvalidSnapshot
		}
		p.seasonality = season
		p.cal = cal
		p.means = jsonToFloats(d.Means)
		p.ses = jsonToFloats(d.SEs)
	case kindSES:
		var d sesData
		if err := json.Unmarshal(s.Data, &d); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrInvalidSnapshot, err)
		}
		if !validSmooth(d.Alpha) {
			return nil, ErrInvalidAlpha
		}
		p.level = float64(d.Level)
		p.sigma = float64(d.Sigma)
		p.alpha = d.Alpha
	case kindHolt:
		var d holtData
		if err := json.Unmarshal(s.Data, &d); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrInvalidSnapshot, err)
		}
		if !validSmooth(d.Alpha) || !validSmooth(d.Beta) {
			return nil, ErrInvalidAlpha
		}
		p.level = float64(d.Level)
		p.trend = float64(d.Trend)
		p.sigma = float64(d.Sigma)
		p.alpha = d.Alpha
		p.beta = d.Beta
	default:
		return nil, fmt.Errorf("%w: %s", ErrUnknownSnapshot, s.Kind)
	}
	return p, nil
}

func seasonName(s Seasonality) string {
	switch s {
	case SeasonHour:
		return "hour"
	case SeasonDay:
		return "day"
	case SeasonHourOfWeek:
		return "week"
	case SeasonMinuteOfWeek:
		return "minute-week"
	default:
		return ""
	}
}

func seasonFromName(name string) (Seasonality, error) {
	switch name {
	case "hour":
		return SeasonHour, nil
	case "day":
		return SeasonDay, nil
	case "week":
		return SeasonHourOfWeek, nil
	case "minute-week":
		return SeasonMinuteOfWeek, nil
	default:
		return 0, ErrInvalidSeason
	}
}
