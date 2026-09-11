# pricesanity
Machine-learning Transformer program intended to describe market behavior based upon price-action in the E-mini S&amp;P 500 futures market.

## Databento data

PriceSanity requests the volume-ranked continuous contract `ES.v.0`. Every
request includes its start date and excludes its end date.

Estimate a request before downloading:

```zsh
pricesanity-estimate --start 2026-09-08 --end 2026-09-09
```

Download only after supplying an approved maximum cost:

```zsh
pricesanity-download \
  --start 2026-09-08 \
  --end 2026-09-09 \
  --max-cost-usd 0.01
```

Raw CME data is written beneath `data/raw/`, which is excluded from Git. Never
force-add raw data, processed training data, annotations, or model artifacts to
the repository.
