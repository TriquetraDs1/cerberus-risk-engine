# DESIGN.md — Cerberus

Tokens live in `dashboard/app/globals.css` and are the single source of truth. This file
records the reasoning, so a later change doesn't quietly undo a decision that was made
for a reason.

## The scene

A reviewer at a desk in daylight, reading an argument and deciding whether to believe it.
Not an SRE watching an incident wall at 2am.

That sentence settles the theme. **Light ground, dark type.** The dark security-console
look is the reflex this category falls into, and reaching for it here would be costume:
nothing about this product is consumed in a dark room under time pressure.

Dark mode exists and is properly built, but it is the accommodation, not the design.

## Colour

OKLCH throughout. No `#000`, no `#fff`. Every neutral is tinted toward hue 60–70 so the
greys read as warm paper rather than cold zinc — the zinc-plus-blue combination is the
first thing that made the old console look like every other admin dashboard.

**Three vocabularies, held strictly apart.** This is the rule most worth protecting:

| Vocabulary | Tokens | Means |
|---|---|---|
| Chrome | `--paper` `--surface` `--surface-sunken` `--rule` `--ink*` | Nothing. Structure only. |
| Adversarial | `--rust` `--rust-surface` `--rust-soft` `--rust-rule` | Attack, damage, hardening. |
| Routing | `--risk-approve` `--risk-review` `--risk-block` | The approve / review / block decision. |

Green, amber and red mean a **routing decision** and nothing else. The adversarial chart
originally used green for "unattacked" and red for "under attack", which overloaded both
signals on the one page where precision matters most; it now uses neutral / rust / ink.
If a future chart needs a third meaning, give it a fourth vocabulary rather than
borrowing the ramp.

Rust sits at hue 42 and `block` at hue 25 so they are not mistaken for the same signal,
and rust is kept off any surface that displays a routing decision.

**Strategy differs by register.** `/about` is Committed: rust fills the whole first fold
because on that page colour carries the argument. The console is Restrained: rust appears
only in selection, focus, ring identifiers and the attack series.

`--rust-surface` is deliberately separate from `--rust`. In dark mode the accent has to
go light to stay legible as text on a dark ground, but a hero filled with that light rust
becomes a bright orange warning panel. The fill stays deep in both themes.

## Type

**Archivo** (Google Fonts, 400–800), one family, with `Geist Mono` for figures.

Picked by the font-selection procedure, not by reflex. The three brand-voice words were
*forensic, unhurried, unflinching*; the physical object was a technical incident report
printed on warm stock. The reflex picks — Inter, IBM Plex, Space Grotesk — were rejected
as training-data defaults. Archivo is a grotesque drawn for signage and technical
printing; its slightly condensed, institutional cut suits a document that reports
findings, and its weight range is wide enough to carry both a 12px table label and a
5.75rem headline in one family.

No display serif anywhere. "Forensic report" tempts toward the editorial-typographic lane
(display serif + italic + mono labels + ruled separators + monochrome restraint), which
is currently saturated and would read as the second-order reflex. Committed colour and a
single grotesque avoid it.

Mono is used for every score, amount, threshold and identifier — tabular, so columns of
figures align on the decimal and a changing value never reflows its row. Mono here is
data alignment, not "technical" costume.

Console type is a fixed rem scale (`--step--1` … `--step-4`, ~1.2 ratio). `/about` opts
into fluid `clamp()` display sizes; the console does not, because users sit at a
consistent DPI and a heading that shrinks inside a panel looks worse, not better.

## Layout

- `.kicker` is the one section-label treatment. Used for column headings and panel
  titles. **Not** stacked above every heading — repeated tiny uppercase labels as section
  grammar is scaffolding, not voice.
- Stat strips separate with a 1px grid gap over `--rule` rather than four cards each
  drawing their own edge. `StatRow` derives its column count from the child count;
  hardcoding four leaves empty cells that render as grey blocks.
- Prose caps at 64–75ch. Tables run dense and wide, which is correct for the register.
- Density is deliberate on the console. An analyst working a queue wants forty rows
  visible, not generous whitespace.

## Motion

- Console: 150ms transitions on state only. No entrance choreography — it loads into a
  task and nobody should watch it arrive.
- `/about`: one staggered `.rise` on the hero, `ease-out-quart`. Nothing else.
- Everything collapses under `prefers-reduced-motion`.

## Things not to undo

1. **Don't reuse the routing ramp for a non-decision meaning.** See the table above.
2. **Don't add a display serif** to make `/about` feel more "editorial". That is the lane
   this design is deliberately outside of.
3. **Don't hardcode `StatRow` to four columns.**
4. **Don't collapse `--rust-surface` back into `--rust`.** Dark mode needs them separate.
5. **Don't seed the ring-graph force layout at a single point.** Coincident starts give
   the charge force nothing to separate, and the graph settles as one blob; the
   phyllotaxis seed plus `forceX`/`forceY` (not `forceCenter`, which only translates) is
   what makes the 25 rings legible as 25 clusters.
