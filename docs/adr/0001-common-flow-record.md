# A common flow record instead of full CICFlowMeter compatibility

NetGen reads two inputs, packet captures and CICFlowMeter flow exports. They must produce the same flow record so that artefacts fitted on one can transform the other. We chose a common record of about 30 fields that both inputs fill exactly, in NetGen's own names and units (seconds, payload bytes). The alternative was to make the capture path reproduce all of CICFlowMeter's roughly 78 features. That would have made NetGen's capture output a drop-in for CICIDS, but it means re-implementing bulk, subflow, active/idle and window features, quirks included, which is large, fragile and hard to test for a portfolio project. CICFlowMeter-only columns are dropped on read. The record can grow later by adding a field only when both inputs can fill it exactly.

## Considered options

- Full CICFlowMeter feature set computed from captures: rejected for cost and fragility.
- Common core plus CICFlowMeter extras kept when present: rejected because the two inputs would then produce datasets of different shapes, and artefacts fitted on one could not always transform the other.
