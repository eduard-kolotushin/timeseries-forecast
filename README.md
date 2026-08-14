# timeseries-forecast

Univariate forecasting for [`timeseries`](https://github.com/eduard-kolotushin/timeseries) series.

**Module:** `github.com/eduard-kolotushin/timeseries-forecast`  
**Package:** `forecast`  
**Go:** 1.26+

This repo is a sibling of the core timeseries library. Grafana visualization lives in [`timeseries-grafana`](../timeseries-grafana). Open all three with [`../timeseries-workspace.code-workspace`](../timeseries-workspace.code-workspace).

See [docs/INTENTIONS.md](docs/INTENTIONS.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Local sibling

`go.mod` replaces the core module with `../timeseries` while both live side by side.

```bash
go test ./...
```

## Quick start

```go
package main

import (
	"fmt"
	"time"

	"github.com/eduard-kolotushin/timeseries"
	forecast "github.com/eduard-kolotushin/timeseries-forecast"
)

func main() {
	s := timeseries.MustNew(
		[]time.Time{
			time.Unix(0, 0).UTC(),
			time.Unix(1, 0).UTC(),
			time.Unix(2, 0).UTC(),
			time.Unix(3, 0).UTC(),
		},
		[]float64{1, 2, 3, 4},
	)

	model, err := forecast.FitHolt(s, 0.8, 0.2)
	if err != nil {
		panic(err)
	}
	fc, err := model.Forecast(3)
	if err != nil {
		panic(err)
	}
	fmt.Println(fc.Values())
}
```

## Models

| Fit | Forecast |
| --- | --- |
| `FitNaive` | last value |
| `FitMean` | training mean |
| `FitDrift` | last + k × slope |
| `FitSeasonalNaive` | last season, cycling |
| `FitSES` | constant smoothed level |
| `FitHolt` | level + k × trend |

Future times are `last + k*step` for `k = 1..h`, with `step` taken from the last two observations.

## Agents

Contributors and coding agents: start with [AGENTS.md](AGENTS.md).
