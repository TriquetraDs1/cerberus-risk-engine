import Link from "next/link";
import { ArrowRight, ArrowSquareOut } from "@phosphor-icons/react/dist/ssr";
import { AttackLoop } from "@/components/AttackLoop";
import { getAdversarialHardening, getSystemHealth } from "@/lib/data";

// Reads the same pipeline output every other page reads. If the model changes, the
// argument on this page changes with it — there are no hand-typed figures here.
export const dynamic = "force-dynamic";

const REPO = "https://github.com/TriquetraDs1/cerberus-risk-engine";

const STRATEGY_COPY: Record<string, { name: string; how: string }> = {
  structuring: { name: "Structuring", how: "Split one payment into many small ones." },
  identity_rotation: { name: "Identity rotation", how: "Spread the ring across more devices." },
  slow_ramp: { name: "Slow ramp", how: "Stretch the burst over a longer window." },
};

function pct(n: number) {
  return `${(n * 100).toFixed(0)}%`;
}

export default async function AboutPage() {
  const [adversarial, health] = await Promise.all([getAdversarialHardening(), getSystemHealth()]);

  const metrics = health?.point_risk_model;
  const ring = health?.ring_detection;
  const savings = health?.decision_layer?.overall_savings_pct_vs_global_threshold;
  const strategies = adversarial ? Object.entries(adversarial.strategies) : [];

  return (
    <div className="min-h-full">
      {/* Committed rust. This is the one surface where the colour carries the argument
          rather than decorating it, so it takes the whole fold instead of a stripe. */}
      <section
        className="px-6 sm:px-10 lg:px-16 pt-16 pb-14 lg:pt-24 lg:pb-20"
        style={{ background: "var(--rust-surface)", color: "var(--on-rust)" }}
      >
        <div className="max-w-[68rem]">
          <p
            className="mono-figure text-xs tracking-widest uppercase rise"
            style={{ opacity: 0.75, animationDelay: "40ms" }}
          >
            Razorpay Track 2 · defensive research prototype
          </p>
          <h1
            className="rise mt-5 font-extrabold tracking-[-0.035em] leading-[0.94]"
            style={{ fontSize: "clamp(2.75rem, 7.5vw, 5.75rem)", animationDelay: "90ms" }}
          >
            A fraud detector
            <br />
            that attacks itself.
          </h1>
          <p
            className="rise mt-8 text-lg sm:text-xl leading-relaxed max-w-[46ch]"
            style={{ opacity: 0.92, animationDelay: "150ms" }}
          >
            Most fraud systems report how well they scored on last year&rsquo;s data. This one
            hires a burglar, records how they got in, fixes what it can, and publishes the
            lock it couldn&rsquo;t fix.
          </p>

          <div
            className="rise mt-12 flex flex-wrap items-stretch gap-px"
            style={{ background: "color-mix(in oklch, var(--on-rust) 22%, transparent)", animationDelay: "210ms" }}
          >
            {[
              { v: metrics ? metrics.roc_auc.toFixed(4) : "—", l: "ROC-AUC, held out" },
              { v: ring ? `${ring.n_perfectly_recovered}/${ring.n_rings}` : "—", l: "fraud rings recovered" },
              { v: ring ? pct(ring.household_false_positive_rate) : "—", l: "false positives, reported" },
              { v: savings != null ? `${savings.toFixed(1)}%` : "—", l: "cheaper than one threshold" },
            ].map((s) => (
              <div key={s.l} className="px-5 py-4 flex-1 min-w-[9.5rem]" style={{ background: "var(--rust-surface)" }}>
                <div className="mono-figure text-2xl sm:text-3xl font-semibold">{s.v}</div>
                <div className="text-[11.5px] mt-1.5 leading-snug" style={{ opacity: 0.8 }}>
                  {s.l}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* The problem, stated as the thing a judge already half-believes. */}
      <section className="px-6 sm:px-10 lg:px-16 py-16 lg:py-20 border-b" style={{ borderColor: "var(--rule)" }}>
        <div className="grid lg:grid-cols-[19rem_1fr] gap-10 lg:gap-16 max-w-[72rem]">
          <div>
            <h2 className="text-2xl font-bold tracking-tight leading-tight">
              Static thresholds get
              <br />
              reverse-engineered
            </h2>
          </div>
          <div className="max-w-[64ch] space-y-5 text-[15px] leading-[1.7]" style={{ color: "var(--ink-secondary)" }}>
            <p>
              Block everything over ₹2,000 and a fraud ring sends ₹1,900. The rule still
              reports excellent accuracy on the data it was measured against, while quietly
              failing on the fraud arriving this month. A model that has only ever been
              tested against a fixed dataset has never been tested against an opponent.
            </p>
            <p style={{ color: "var(--ink)" }}>
              So the question this project asks is not{" "}
              <em className="not-italic font-semibold">how accurate is it</em>, but{" "}
              <em className="not-italic font-semibold">how much accuracy survives contact with someone trying to beat it</em>.
            </p>
          </div>
        </div>
      </section>

      {/* Mechanism. */}
      <section className="px-6 sm:px-10 lg:px-16 py-16 lg:py-20 border-b" style={{ borderColor: "var(--rule)" }}>
        <div className="max-w-[72rem]">
          <p className="kicker">How it works</p>
          <h2 className="mt-3 text-2xl font-bold tracking-tight max-w-[24ch] leading-tight">
            Build a detector, break it, measure the damage, fix it, prove the fix
          </h2>
          <div className="mt-10">
            <AttackLoop />
          </div>
        </div>
      </section>

      {/* The attack results — the actual differentiator, with real numbers. */}
      <section
        className="px-6 sm:px-10 lg:px-16 py-16 lg:py-20 border-b"
        style={{ borderColor: "var(--rule)", background: "var(--surface-sunken)" }}
      >
        <div className="max-w-[72rem]">
          <p className="kicker">What the attacks did</p>
          <h2 className="mt-3 text-2xl font-bold tracking-tight max-w-[30ch] leading-tight">
            Three ways to evade the detector, and how much of each one survived hardening
          </h2>

          {strategies.length > 0 ? (
            <div className="mt-10 divide-y" style={{ borderColor: "var(--rule)" }}>
              {strategies.map(([key, s]) => {
                const copy = STRATEGY_COPY[key] ?? { name: key, how: "" };
                const base = s.baseline_detection.combined_score;
                const attacked = s.evaded_original_model.combined_score;
                const hardened = s.evaded_hardened_model.combined_score;
                const recovered = hardened >= base * 0.9;
                return (
                  <div key={key} className="py-7 grid md:grid-cols-[16rem_1fr] gap-5 md:gap-10 items-start">
                    <div>
                      <h3 className="text-base font-semibold tracking-tight">{copy.name}</h3>
                      <p className="text-[13px] mt-1 leading-snug" style={{ color: "var(--ink-secondary)" }}>
                        {copy.how}
                      </p>
                    </div>

                    <div>
                      <div className="flex flex-col gap-2.5">
                        {[
                          { l: "Unattacked", v: base, c: "var(--ink-tertiary)" },
                          { l: "Under attack", v: attacked, c: "var(--rust)" },
                          { l: "After hardening", v: hardened, c: "var(--ink)" },
                        ].map((bar) => (
                          <div key={bar.l} className="flex items-center gap-3">
                            <span
                              className="text-[11.5px] w-[6.75rem] shrink-0 text-right"
                              style={{ color: "var(--ink-secondary)" }}
                            >
                              {bar.l}
                            </span>
                            <span
                              className="h-[9px] rounded-[1px] transition-[width] duration-500"
                              style={{
                                width: `max(2px, ${bar.v * 100}%)`,
                                background: bar.c,
                                maxWidth: "calc(100% - 3.5rem)",
                              }}
                            />
                            <span className="mono-figure text-[12.5px]" style={{ color: "var(--ink-secondary)" }}>
                              {bar.v.toFixed(2)}
                            </span>
                          </div>
                        ))}
                      </div>
                      {!recovered && (
                        <p
                          className="text-[13px] mt-4 leading-relaxed max-w-[58ch] pl-[7.5rem]"
                          style={{ color: "var(--ink-secondary)" }}
                        >
                          <span className="font-semibold" style={{ color: "var(--rust)" }}>
                            Does not recover.
                          </span>{" "}
                          This one attacks the graph detector, and that detector is
                          unsupervised — retraining a classifier cannot teach a clustering
                          algorithm anything. Reported rather than patched.
                        </p>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="mt-8 text-sm" style={{ color: "var(--ink-secondary)" }}>
              No hardening report yet — run <code className="mono-figure">scripts/run_adversarial_harness.py</code>.
            </p>
          )}
        </div>
      </section>

      {/* The honesty section. Given equal weight to the results, per PRODUCT.md. */}
      <section className="px-6 sm:px-10 lg:px-16 py-16 lg:py-20 border-b" style={{ borderColor: "var(--rule)" }}>
        <div className="grid lg:grid-cols-[19rem_1fr] gap-10 lg:gap-16 max-w-[72rem]">
          <div>
            <p className="kicker">Read this before the panel does</p>
            <h2 className="mt-3 text-2xl font-bold tracking-tight leading-tight">
              What this does not claim
            </h2>
          </div>
          <ul className="max-w-[64ch] space-y-6 text-[15px] leading-[1.7]">
            {[
              {
                t: "The rings are synthetic and the attacker is code I wrote.",
                d: "This validates robustness against known evasion classes, not a real-world guarantee. A production version needs real labelled fraud and a human red team.",
              },
              {
                t: "One evasion got worse, not better.",
                d: "Adding graph features raised ROC-AUC to the highest of any version I tried, and simultaneously collapsed identity-rotation robustness, because the model learned to lean on structure that attack destroys. Aggregate metrics scored it a clean win. Only the harness caught it.",
              },
              {
                t: "A learned graph detector scored 1.0000, and that means nothing.",
                d: "A one-line control that flags any account with two or more connections matches it exactly. The synthetic graph is separable without learning, so the number measures the dataset, not the model. The control runs on every invocation so the score can never be quoted alone.",
              },
              {
                t: "Chargebacks arrive 30 to 90 days late in reality.",
                d: "This does not model that lag, and the label-timing problem it creates is genuinely hard.",
              },
            ].map((item) => (
              <li key={item.t}>
                <p className="font-semibold" style={{ color: "var(--ink)" }}>
                  {item.t}
                </p>
                <p className="mt-1.5" style={{ color: "var(--ink-secondary)" }}>
                  {item.d}
                </p>
              </li>
            ))}
          </ul>
        </div>
      </section>

      {/* Where to go next — the console this page introduces. */}
      <section className="px-6 sm:px-10 lg:px-16 py-16 lg:py-20">
        <div className="max-w-[72rem]">
          <p className="kicker">The console</p>
          <h2 className="mt-3 text-2xl font-bold tracking-tight">Four views, all reading real pipeline output</h2>
          <div className="mt-9 grid sm:grid-cols-2 lg:grid-cols-4 gap-px" style={{ background: "var(--rule)" }}>
            {[
              { href: "/adversarial", t: "Adversarial Hardening", d: "The attack results above, in full, per strategy." },
              { href: "/", t: "Review Queue", d: "Every transaction, its decision, and why." },
              { href: "/rings", t: "Ring Network", d: "Accounts linked by shared devices and cards." },
              { href: "/health", t: "System Health", d: "Calibration and per-segment routing." },
            ].map((c) => (
              <Link
                key={c.href}
                href={c.href}
                className="group px-5 py-6 transition-colors"
                style={{ background: "var(--surface)" }}
              >
                <div className="flex items-start justify-between gap-3">
                  <h3 className="text-[15px] font-semibold tracking-tight leading-snug">{c.t}</h3>
                  <ArrowRight
                    size={15}
                    className="mt-0.5 shrink-0 transition-transform duration-200 group-hover:translate-x-0.5"
                    style={{ color: "var(--rust)" }}
                    aria-hidden
                  />
                </div>
                <p className="text-[13px] mt-2 leading-snug" style={{ color: "var(--ink-secondary)" }}>
                  {c.d}
                </p>
              </Link>
            ))}
          </div>

          <div
            className="mt-10 pt-8 border-t flex flex-wrap items-baseline gap-x-8 gap-y-3 text-[13px]"
            style={{ borderColor: "var(--rule)", color: "var(--ink-secondary)" }}
          >
            <a
              href={REPO}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 font-medium hover:underline"
              style={{ color: "var(--rust)" }}
            >
              Source, model card, and the full experiment log
              <ArrowSquareOut size={13} aria-hidden />
            </a>
            <span>
              Defensive research only. The harness attacks a sandboxed copy of this
              project&rsquo;s own model; nothing here targets a real system.
            </span>
          </div>
        </div>
      </section>
    </div>
  );
}
