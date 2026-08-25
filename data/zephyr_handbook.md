# Zephyr Analytics — Product Handbook

> This is a FICTIONAL product. Nothing here is real. That is intentional:
> because a language model has never seen these facts during training, any
> correct answer it gives MUST come from retrieval — which is exactly what
> makes this corpus useful for testing a RAG system honestly.

## About Zephyr

Zephyr Analytics is a product-analytics platform founded in 2021 in Waterloo,
Ontario. Zephyr helps teams track how users move through their apps. The
company runs its infrastructure from two data centres: one in Toronto and one
in Frankfurt. Data never leaves the region it was collected in.

## Plans and Limits

Zephyr has three plans: Free, Pro, and Enterprise.

The Free plan includes 3 dashboards, 10,000 tracked events per month, and
7-day data retention. The Free plan does not include the Pulse feature.

The Pro plan includes 25 dashboards, 1,000,000 tracked events per month, and
90-day data retention. Pro costs 49 dollars per seat per month.

The Enterprise plan includes unlimited dashboards, custom event volume, and
365-day data retention. Enterprise pricing is custom and requires contacting
the sales team.

## Retention Add-On

Any paid plan can extend data retention beyond its default using the Retention
Add-On. The add-on costs 15 dollars per month for each additional 30 days of
retention. Retention cannot be extended on the Free plan.

## The Pulse Feature

Pulse is Zephyr's real-time alerting engine. Pulse watches event streams and
fires an alert when a metric crosses a threshold you define. Pulse is available
on the Pro and Enterprise plans only. Pulse evaluates rules every 60 seconds.

## API Access

Every plan includes API access. The API rate limit is 100 requests per minute
on the Free plan, 1,000 requests per minute on Pro, and 10,000 requests per
minute on Enterprise. API keys are created in the Settings page under
"Developer". A key that has not been used for 90 days is automatically revoked.

## Support

Free plan support is community-only, through the public forum. Pro plan support
responds to tickets within 24 hours. Enterprise support responds within 4 hours
and includes a named account manager. Support does not operate on statutory
holidays in Ontario.

## Data Export

All plans can export raw event data as CSV. Pro and Enterprise can also export
to Parquet and can schedule automatic daily exports to an S3 bucket. Exports on
the Free plan are capped at 10,000 rows per file.
