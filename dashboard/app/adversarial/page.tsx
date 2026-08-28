import { EmptyState } from "@/components/EmptyState";
import { PageHeader } from "@/components/PageHeader";
import { RecallDecayChart } from "@/components/RecallDecayChart";
import { StatRow, StatTile } from "@/components/StatTile";
import { getAdversarialHardening } from "@/lib/data";
import type { EvasionStrategy } from "@/lib/types";

const STRATEGY_LABELS: Record<EvasionStrategy, string> = {
  structuring: "Structuring",
  identity_rotation: "Identity rotation",
  slow_ramp: "Slow ramp",
};

const STRATEGY_DESCRIPTIONS: Record<EvasionStrategy, string> = {
  structuring: "Splits each transaction into many smaller ones, betting the model only looks per-transaction.",
  identity_rotation: "Spreads the ring across more distinct devices, betting the graph layer needs one shared entity to form a community.",
  slow_ramp: "Stretches the coordinated burst over a much longer window, betting velocity features only catch fast bursts.",
};

export default async function AdversarialPage() {
  const report = await getAdversarialHardening();

  if (!report) {
    return (
      <div className="flex flex-col h-full">
        <PageHeader title="Adversarial Hardening" subtitle="Attack the detector, measure the damage, retrain, prove the recovery." />
        <EmptyState
          title="No hardening report yet."
          command="python scripts/run_adversarial_harness.py && python scripts/export_dashboard_data.py"
        />
      </div>
    );
  }

  const strategies = Object.entries(report.strategies) as [EvasionStrategy, (typeof report.strategies)[EvasionStrategy]][];

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <PageHeader
        title="Adversarial Hardening"
        subtitle={`${report.n_restarts} restarts × ${report.n_steps} steps per strategy · ${report.n_adversarial_examples.toLocaleString("en-US")} adversarial examples used to retrain`}
      />

      <section className="px-6 py-5">
        <RecallDecayChart strategies={report.strategies} />
      </section>

      <section className="px-6 pb-2">
        <h2 className="text-xs font-medium uppercase tracking-wide mb-3" style={{ color: "var(--text-tertiary)" }}>
          Per-strategy detail
        </h2>
        <div className="flex flex-col gap-3">
          {strategies.map(([name, r]) => {
            const graphEvaded = r.evaded_original_model.ring_recovered_fraction < 0.5;
            const graphStillEvaded = r.evaded_hardened_model.ring_recovered_fraction < 0.5;
            return (
              <div key={name} className="rounded-lg border p-4" style={{ borderColor: "var(--border)" }}>
                <div className="flex items-baseline justify-between flex-wrap gap-2">
                  <h3 className="text-sm font-medium">{STRATEGY_LABELS[name]}</h3>
                  <span className="text-xs mono-figure" style={{ color: "var(--text-tertiary)" }}>
                    best params: {Object.entries(r.best_evasion_params).map(([k, v]) => `${k}=${v.toFixed(2)}`).join(", ")}
                  </span>
                </div>
                <p className="text-xs mt-1" style={{ color: "var(--text-secondary)" }}>
                  {STRATEGY_DESCRIPTIONS[name]}
                </p>
                <StatRow>
                  <StatTile label="Recall decay under attack" value={`-${(r.recall_decay_original * 100).toFixed(0)}%`} tone="block" />
                  <StatTile
                    label="Recovered after hardening"
                    value={`+${(r.recall_recovered_after_hardening * 100).toFixed(0)}%`}
                    tone={r.recall_recovered_after_hardening > 0.3 ? "approve" : "review"}
                  />
                  <StatTile
                    label="Point-risk model, hardened"
                    value={r.evaded_hardened_model.point_risk_caught_fraction.toFixed(2)}
                    hint={`was ${r.evaded_original_model.point_risk_caught_fraction.toFixed(2)} before hardening`}
                    tone="approve"
                  />
                  <StatTile
                    label="Ring detector, hardened"
                    value={r.evaded_hardened_model.ring_recovered_fraction.toFixed(2)}
                    hint={graphStillEvaded ? "still evades the graph layer" : "graph layer catches it"}
                    tone={graphStillEvaded ? "block" : "approve"}
                  />
                </StatRow>
                {graphEvaded && graphStillEvaded && (
                  <p className="text-xs mt-3 px-3 py-2 rounded-md" style={{ background: "var(--risk-block-bg)", color: "var(--text-primary)", border: "1px solid var(--risk-block-border)" }}>
                    Honest limitation: this strategy evades the Louvain ring detector both
                    before and after hardening. Retraining the point-risk model doesn&apos;t
                    fix an unsupervised graph algorithm — the graph layer&apos;s vulnerability
                    to {STRATEGY_LABELS[name].toLowerCase()} is real and unresolved.
                  </p>
                )}
              </div>
            );
          })}
        </div>
      </section>

      <section className="px-6 py-5 flex flex-col gap-3">
        <h2 className="text-xs font-medium uppercase tracking-wide" style={{ color: "var(--text-tertiary)" }}>
          Known limitations
        </h2>
        <ul className="text-sm flex flex-col gap-1.5 list-disc pl-5" style={{ color: "var(--text-secondary)" }}>
          {report.limitations.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}
