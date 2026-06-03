"""Commercial product layer: customer-facing artifacts and the data flywheel.

Three modules planned (see ``docs/COORDINATION.md``):

  * ``daily_brief`` — generator that turns a fitted MultiAssetReport
    into a structured JSON document conforming to
    ``docs/daily_brief_schema.md``.
  * ``brief_renderer`` — JSON to human-readable email / PDF body.
  * ``outcome_log`` (not yet implemented) — append-only writer for
    every prediction the brief generator makes, joined nightly with
    resolutions for the audit / flywheel.
"""
