# Propose {{n}} strategy genome(s)

## The market

{{market}}

## Genomes already in the population

These are the **shapes** already being explored. Propose something structurally
different — a different indicator family, a different comparison, a different
holding period — rather than a variation on one of them.

{{population_signatures}}

You are not being told how any of them performed, and you should not assume the
ones listed first are better. They are in an arbitrary order.

## Output

Return a JSON object with these fields and no others:

```json
{
  "hypothesis": "why an edge should exist here, in at least twenty characters",
  "genome": { ... a genome per the schema in the system message ... },
  "expected_trade_frequency": "hours" | "days" | "weeks",
  "notes": ""
}
```
