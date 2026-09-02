# PRODUCT.md — Cerberus

## Register

`product` for the four analyst pages (Review Queue, Ring Network, Adversarial Hardening,
System Health) — data must dominate, design serves the work.
`brand` for the explainer surface at `/about` — that page *is* the argument, design carries it.

## Product purpose

A transaction-risk engine that red-teams itself. It scores payments for fraud, detects
coordinated fraud rings through shared devices and cards, and then attacks its own
detector to measure how far detection falls, retrains on what the attack found, and
reports the recovery — including the attack it could not defend against.

Built solo in ten days for Razorpay's Track 2 hackathon. Defensive-only research.

## Users

**Primary — the hackathon judge.** Reviewing dozens of submissions, deciding in about
thirty seconds whether this one is different from the other fraud classifiers. Technical,
skeptical, has seen a hundred dashboards with impressive numbers and no substance. Lands
cold with no context. Will not read a paragraph before deciding to care.

**Secondary — the fraud analyst.** The person the dashboard is designed *for*, in-fiction.
Works a review queue during business hours on a wide monitor. Needs density, needs to know
why a transaction was flagged, needs to act on it.

**Tertiary — a future engineer or interviewer** reading the repo without the author present.

## Tone

Forensic. Precise. Unhurried. The voice of someone reporting findings, including the
inconvenient ones, rather than selling a result.

Numbers are stated plainly and never rounded in the flattering direction. Limitations are
volunteered before they are asked for. No superlatives, no "powerful", no "cutting-edge".
If a result is a null result, it says so.

## Strategic principles

1. **Honesty is the product.** The 22.3% false-positive rate, the evasion that does not
   recover, the GNN whose perfect score a one-line control matches — these are the
   differentiators, not the blemishes. Design must give them the same weight as the wins,
   not bury them in a footnote.
2. **Show the mechanism, not the claim.** A before/attack/after chart beats the sentence
   "we tested robustness". A live degradation demo beats "graceful degradation" in a list.
3. **Every decision carries its reasons and its cost.** Never a bare score.
4. **Density is respect** on the analyst pages. An analyst working a queue does not want
   generous whitespace; they want to see forty rows without scrolling.
5. **The explainer must work for someone who knows nothing.** A judge who cannot tell what
   this is within one screen will score it as another classifier.

## Anti-references

- **The generic admin dashboard.** Zinc/slate neutrals, one blue accent, evenly spaced
  stat cards in a four-across grid. This is what the project currently looks like and it is
  the first thing a judge's eye discards.
- **The dark "security console" / SOC aesthetic.** Neon on near-black, monospace
  everything, terminal green. The second-order reflex: having avoided navy-and-gold, this
  is the trap one tier down. The scene here is daylight review, not a 2am incident.
- **SaaS hero-metric pages.** Enormous gradient number, three supporting stats, a CTA.
- **Fintech navy and gold.** Trust-signalling by cliché.

## Non-negotiables

- The risk ramp (approve / review / block) stays semantic and stays green / amber / red,
  always paired with an icon and a text label. Never color alone. It is the one place
  convention beats novelty, because misreading it has a cost.
- Every number rendered must trace to `reports/*.json`. No decorative or illustrative data.
- The defensive-scope statement stays visible.
- Mono, tabular figures for every score, amount, and identifier.
