package forecast

import "github.com/eduard-kolotushin/timeseries"

// FitFunc trains a model on a series.
type FitFunc func(timeseries.Series[float64]) (Fitted, error)

// Split cuts s into train and test, with testSize points at the end.
func Split(s timeseries.Series[float64], testSize int) (train, test timeseries.Series[float64], err error) {
	n := s.Len()
	if testSize <= 0 || testSize >= n {
		return timeseries.Series[float64]{}, timeseries.Series[float64]{}, ErrSplit
	}
	train, err = s.SliceIndex(0, n-testSize)
	if err != nil {
		return timeseries.Series[float64]{}, timeseries.Series[float64]{}, err
	}
	test, err = s.SliceIndex(n-testSize, n)
	if err != nil {
		return timeseries.Series[float64]{}, timeseries.Series[float64]{}, err
	}
	return train, test, nil
}

// Evaluate fits on train and scores an h-step forecast against test, where h = test.Len().
func Evaluate(train, test timeseries.Series[float64], fit FitFunc) (Metrics, error) {
	if test.Empty() {
		return Metrics{}, ErrEmpty
	}
	m, err := fit(train)
	if err != nil {
		return Metrics{}, err
	}
	pred, err := m.Forecast(test.Len())
	if err != nil {
		return Metrics{}, err
	}
	return Compare(test, pred)
}
