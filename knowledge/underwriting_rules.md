# Underwriting & Triage Rules (SYNTHETIC)

Each rule has a human-readable body (what the Coverage agent retrieves and reasons over)
and a machine-readable annotation in an HTML comment (what the deterministic guardrail
evaluates). The rule text lives here once; nothing below is copied into any agent prompt.

Recommended actions, in increasing order of severity:
`auto_approve` < `request_more_documentation` < `route_to_investigator`.

## UW-01 Early-tenure claims
Claims whose loss date falls within 7 days of policy inception (the policy start date,
`active_from`) must be flagged for review regardless of amount. Early-tenure losses are a known pattern for pre-existing damage or
policies purchased in anticipation of a loss. Route to an investigator.
<!-- rule: {"id": "UW-01", "when": [["days_since_inception", "<=", 7]], "action": "route_to_investigator"} -->

## UW-02 Near-limit claim amounts
Claim amounts at or above 98% of the policy limit (i.e. within 2% of the limit) warrant an
investigator review. Amounts clustered at the limit suggest the figure was fitted to the
policy rather than to the loss.
<!-- rule: {"id": "UW-02", "when": [["pct_of_limit", ">=", 98]], "action": "route_to_investigator"} -->

## UW-03 Amount exceeds policy limit
If the claim amount exceeds the policy limit, the payable amount is capped at the limit.
The claim cannot be auto-approved; request itemised documentation so the adjuster can
settle within limit.
<!-- rule: {"id": "UW-03", "when": [["exceeds_limit", "==", true]], "action": "request_more_documentation"} -->

## UW-04 Loss outside the active policy period
A loss dated before `active_from` or after `active_to` is not covered. Any denial on this
basis must be reviewed by an investigator before the policyholder is notified, because
reinstatements and backdated endorsements occasionally change the effective period.
<!-- rule: {"id": "UW-04", "when": [["in_policy_period", "==", false]], "action": "route_to_investigator"} -->

## UW-05 Late reporting
Claims reported more than 30 days after the loss date require supporting documentation
(photos dated at the time of loss, contractor estimates, weather reports) explaining the
delay. Late reporting alone is not evidence of fraud.
<!-- rule: {"id": "UW-05", "when": [["days_to_report", ">", 30]], "action": "request_more_documentation"} -->

## UW-06 Repeat same-peril claims
A second or subsequent claim for the same peril on the same policy within 12 months must be
routed to an investigator. For water damage specifically, repeat claims frequently indicate
gradual seepage or deferred maintenance, which is excluded under standard homeowners forms.
<!-- rule: {"id": "UW-06", "when": [["prior_same_peril_12m", ">=", 1]], "action": "route_to_investigator"} -->

## UW-07 Auto-approval ceiling
Only claims of 10,000 or less may be auto-approved, and only when no other rule is
triggered. Covered, in-period claims above 10,000 with no other flags need standard
documentation (estimate, invoices) before settlement.
<!-- rule: {"id": "UW-07", "when": [["claim_amount", ">", 10000]], "action": "request_more_documentation"} -->

## UW-08 Peril not covered
If the claim type is not in the policy's covered perils, the claim is not covered.
Homeowners forms in this book exclude `flood` (surface water) - it is distinct from
`water_damage` (sudden and accidental discharge from plumbing). Coverage denials are routed
to an investigator for sign-off.
<!-- rule: {"id": "UW-08", "when": [["peril_covered", "==", false]], "action": "route_to_investigator"} -->
