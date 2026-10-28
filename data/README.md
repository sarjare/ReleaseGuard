# ReleaseGuard sample datasets

IMPORTANT: These are synthetic, illustrative datasets created for a hackathon prototype. They are not real production data and must not be presented as real observations.

## Files
- `slo_definitions.csv`: service names, SLO targets (percent), SLI type, descriptions.
- `service_metrics.csv`: daily request counts, failed requests, and average latency per service.
- `release_history.csv`: synthetic deployments and metadata such as files changed, affected services, deployment hour, and author tenure.
- `incidents.csv`: synthetic incidents linked to some release IDs.
- `traffic_calendar.csv`: synthetic peak windows and freeze periods.

## Suggested relationships
- Join `service_metrics.service` to `slo_definitions.service`.
- Join `release_history.release_id` to `incidents.related_release`.
- Join calendar records using the date column.

## Important calculation notes
- `slo_target` is a percentage, e.g. 99.9 means 99.9%, not 0.999.
- Daily service metrics are daily observations; sum total and failed requests over the chosen measurement window before calculating aggregate availability.
- Allowed error rate = `1 - (slo_target / 100)`.
- Allowed errors over a window = total requests × allowed error rate.
- Budget consumed (%) = observed failed requests / allowed errors × 100.
- Burn rate = observed error rate / allowed error rate.
- These simplified sample records demonstrate a prototype. A real system should define its SLO measurement window, handle low traffic, data gaps, latency SLOs, and incident attribution carefully.
- Do not use author tenure to rank or penalize individual developers. Aggregate or anonymize it if used, and document bias and limitations.
- Risk score weights should be labelled as heuristic unless trained and validated on a held-out dataset.
