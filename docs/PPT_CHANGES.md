# PPT changes for the final deck

The deck was written before the build. These edits make every claim match what the code does and what we measured. Every number below comes from `data/lake/*.json` on the real MSR traces, scored on unseen hours 120–167.

## Rule for every slide

Quote a measured number only with the conditions it was measured under: real MSR data, unseen hours, median of 5 replays.

---

## Slide 3: solution and datasets

| Change | If the slide says | Change it to |
|---|---|---|
| Alibaba claim | "Validated on MSR Cambridge and Alibaba traces" | "Validated on MSR Cambridge (8 volumes). Alibaba block-trace adapter built and tested (streams the 181 GB archive); cross-dataset run pending trace access." |
| Generalization | "Generalizes to unseen workloads" | "Hotness model transfers to unseen volumes (F1 0.665 vs rule 0.625); the placement plan does not yet transfer (SLA 92.6%). This is our next step." |

## Slide 4: architecture

| Change | If the slide says | Change it to |
|---|---|---|
| Forecasting | "LSTM / Prophet forecaster" | "Prophet forecaster (seasonal EWMA fallback). LSTM: stretch goal" |
| Policy | "English → policy" | "Policy as code (OPA / Rego, cross-checked by a Python guard). English → policy: stretch goal" |
| RL | "RL agent decides placement" | "LinUCB contextual bandit tunes the exact solver; it is promoted only if it beats static settings (on our data it stays in shadow mode)" |

## Slide 5: tech stack

| Remove | Why |
|---|---|
| React | The dashboard is a single-page web dashboard (HTML + Chart.js) served by FastAPI. Say "web dashboard". |
| OpenTelemetry | Not used. Metrics go out as Prometheus text on `/metrics`. |
| DuckDB | Not used. The data lake is Parquet files read with pandas / PyArrow. |

These are now **real and running**; keep them, and add "✔ built" next to each:
Prometheus · Grafana · Redis · OPA · OR-Tools (CP-SAT) / HiGHS · LightGBM · XGBoost · Prophet · Isolation Forest · MinIO · FastAPI · Docker Compose.

## Slide 6: KPIs

| KPI | If the slide says | Change it to |
|---|---|---|
| Cost | "≥ 15% lower cost than rule-based tiering" | "14.5% lower cost than the best policy-compliant rule (median of 5 replays, range 10.0–15.8%); 61.8% lower than keeping everything hot. Target ≥ 15%: close, not yet met." |
| SLA | "≥ 99% SLA" | "100% of I/Os within their latency target (median; worst replay 99.5%)" |
| Forecast | "MAPE < 15% at 7 days" | "Capacity MAPE 3.1% at 24 h ahead (the trace is one week long, so 7 days ahead cannot be scored)" |
| Hotness | "macro-F1 ≥ 0.85" | "macro-F1 0.724 vs rule 0.596 (target 0.85 not met)" |
| Failover | "≤ 10 s" | "8.5 s measured from the dead leader's last heartbeat" |
| Availability | "99.9% availability" | "Chaos drill: 12 cycles with injected faults, 100% completed safely, 0 unsafe actions" |
| Data movement | "≤ 5% of I/O" | "8.5% of I/O (not met)" |
| Hit ratio | "Higher hot-tier hit ratio than LRU / LFU" | "Not a goal: Ayojna serves busy data from warm when the SLA allows, which is cheaper (34% vs LRU 69%)" |

Scorecard line for the slide: **"5 of 9 targets met on real data, and all 9 measured. The misses are shown, not hidden."**

## Results / closing slide

Replace any "68% cheaper" headline with the real-data median. 68.5% is the **synthetic** result; if you show it, label it "synthetic sanity check".

| Use | Number |
|---|---|
| Headline | 61.8% cheaper than all-hot · 14.5% cheaper than the best compliant rule · 100% SLA · 100% compliance |
| Reliability | Failover 8.5 s · chaos drill 0 unsafe actions |
| Honest limitation | Cross-volume plan: SLA 92.6% → pricing the risk of re-promotion is the next step |