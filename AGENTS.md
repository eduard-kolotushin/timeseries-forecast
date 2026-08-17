# AGENTS.md

Operating manual for agents working in this repository.

## Project

Univariate forecasting on top of `github.com/eduard-kolotushin/timeseries`.

- **Module:** `github.com/eduard-kolotushin/timeseries-forecast`
- **Package:** `forecast`
- **Go:** 1.26+
- **Series library:** `github.com/eduard-kolotushin/timeseries` (tagged module; no `replace`)

## Read first

1. [docs/INTENTIONS.md](docs/INTENTIONS.md) — product scope and non-goals
2. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — API, models, evaluation

This repo is one root of the Cursor workspace `timeseries-workspace.code-workspace` (sibling of `timeseries` and `timeseries-grafana`). Do not fold forecasting into the core timeseries package.

## Hard constraints

- Depend on `timeseries.Series[float64]` for input and output; do not fork Series
- Public ops do not mutate caller series
- Future timestamps: last time + `k*step` for horizon `k=1..h` (step inferred from the last interval unless given)
- Missing values: DropNA before fit; `math.NaN()` in outputs where undefined
- Stay within v1/v2 scope unless `docs/INTENTIONS.md` is updated first
- Implement fits in linear time; O(1) work per horizon step; pre-size forecast slices

## v1 in scope

Naive, mean, drift, seasonal naive, seasonal baseline (hour/day/hour-of-week/minute-of-week), SES, Holt; optional RU production calendar (default off); holdout evaluate; MAE/RMSE/MAPE.

## v2 in scope

Gaussian prediction intervals (`ForecastInterval`) for every v1 model.

## v1/v2 out of scope

ARIMA/SARIMA, Prophet, ML models, multivariate, quantile/bootstrap intervals, series I/O, plotting (see sibling `timeseries-grafana`). Business calendars belong here, not in `timeseries`.

## Workflow

- Table-driven tests next to the code under test
- Depend on a tagged `timeseries` module; do not add a `replace` directive
- Do not copy Series internals; use the public timeseries API only
- GitHub Actions on `main`: `gofmt` and `go test ./...`
