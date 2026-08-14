package forecast

import (
	"math"

	"github.com/eduard-kolotushin/timeseries"
)

// Metrics holds point-forecast accuracy on aligned observations.
type Metrics struct {
	MAE  float64
	RMSE float64
	MAPE float64
	N    int
}

// Compare scores actual vs predicted on the inner time join.
// MAPE ignores actual values of zero. All metrics ignore NaN pairs.
func Compare(actual, pred timeseries.Series[float64]) (Metrics, error) {
	a, p := timeseries.AlignFloat(actual, pred, timeseries.JoinInner)
	if a.Empty() {
		return Metrics{}, ErrEmpty
	}
	av, pv := a.Values(), p.Values()
	var abs, sq, ape float64
	n := 0
	nApe := 0
	for i := range av {
		if math.IsNaN(av[i]) || math.IsNaN(pv[i]) {
			continue
		}
		d := av[i] - pv[i]
		if d < 0 {
			d = -d
		}
		abs += d
		sq += (av[i] - pv[i]) * (av[i] - pv[i])
		n++
		if av[i] != 0 {
			ape += d / math.Abs(av[i])
			nApe++
		}
	}
	if n == 0 {
		return Metrics{}, ErrEmpty
	}
	m := Metrics{
		MAE:  abs / float64(n),
		RMSE: math.Sqrt(sq / float64(n)),
		N:    n,
	}
	if nApe > 0 {
		m.MAPE = ape / float64(nApe)
	} else {
		m.MAPE = math.NaN()
	}
	return m, nil
}
