import { CheckCircle } from "@phosphor-icons/react/dist/ssr";
import { PageHeader } from "@/components/PageHeader";
import { EmptyState } from "@/components/EmptyState";
import { StatRow, StatTile } from "@/components/StatTile";
import { getSystemHealth } from "@/lib/data";

export default async function SystemHealthPage() {
  const health = await getSystemHealth();

  if (!health) {
    return (
      <div className="flex flex-col h-full">
        <PageHeader title="System Health" subtitle="Model metrics, drift, and pipeline status." />
        <EmptyState
          title="No system health data yet."
          command="python scripts/train_baseline.py && python scripts/export_dashboard_data.py"
        />
      </div>
    );
  }

  const m = health.point_risk_model;
  const savings = m.cost_at_default_threshold - m.cost_at_optimal_threshold;
  const savingsPct = (savings / m.cost_at_default_threshold) * 100;

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <PageHeader title="System Health" subtitle={`Last updated ${new Date(health.generated_at).toLocaleString("en-US")}`}>
        <span
          className="inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full border"
          style={{ color: "var(--risk-approve)", background: "var(--risk-approve-bg)", borderColor: "var(--risk-approve-border)" }}
        >
          <CheckCircle size={13} weight="bold" aria-hidden />
          Graph cache: {health.graph_cache_status}
        </span>
      </PageHeader>

      <section>
        <h2 className="px-6 pt-5 pb-1 text-xs font-medium uppercase tracking-wide" style={{ color: "var(--text-tertiary)" }}>
          Point-risk model
        </h2>
        <StatRow>
          <StatTile label="ROC-AUC" value={m.roc_auc.toFixed(3)} />
          <StatTile label="PR-AUC" value={m.pr_auc.toFixed(3)} hint="average precision, held-out set" />
          <StatTile label="Test set size" value={m.n_test.toLocaleString("en-US")} hint={`trained on ${m.n_train.toLocaleString("en-US")}`} />
          <StatTile
            label="Cost saved vs. naive 0.5"
            value={`${savingsPct.toFixed(0)}%`}
            hint={`${savings.toLocaleString("en-US")} cost units at threshold ${m.cost_optimal_threshold.toFixed(3)}`}
            tone="approve"
          />
        </StatRow>
      </section>

      {health.ring_detection && (
        <section>
          <h2 className="px-6 pt-5 pb-1 text-xs font-medium uppercase tracking-wide" style={{ color: "var(--text-tertiary)" }}>
            Ring detector
          </h2>
          <StatRow>
            <StatTile
              label="Rings recovered"
              value={`${health.ring_detection.n_perfectly_recovered} / ${health.ring_detection.n_rings}`}
            />
            <StatTile label="Mean recovery" value={`${(health.ring_detection.mean_ring_recovery * 100).toFixed(0)}%`} tone="approve" />
            <StatTile
              label="Household FP rate"
              value={`${(health.ring_detection.household_false_positive_rate * 100).toFixed(1)}%`}
              tone="review"
            />
            <StatTile label="Communities" value={`${health.ring_detection.n_flagged_communities} flagged`} hint={`of ${health.ring_detection.n_total_communities} total`} />
          </StatRow>
        </section>
      )}

      <section className="px-6 py-5 flex flex-col gap-4">
        <h2 className="text-xs font-medium uppercase tracking-wide" style={{ color: "var(--text-tertiary)" }}>
          Routing preview (Day 4 pending)
        </h2>
        <div className="rounded-lg border p-4 text-sm" style={{ borderColor: "var(--risk-review-border)", background: "var(--risk-review-bg)" }}>
          <p style={{ color: "var(--text-primary)" }}>
            Block ≥ <span className="mono-figure font-medium">{health.routing_preview.block_threshold}</span> · Review ≥{" "}
            <span className="mono-figure font-medium">{health.routing_preview.review_threshold}</span>
          </p>
          <p className="mt-1.5 text-xs" style={{ color: "var(--text-secondary)" }}>
            {health.routing_preview.note}
          </p>
        </div>

        <h2 className="text-xs font-medium uppercase tracking-wide mt-2" style={{ color: "var(--text-tertiary)" }}>
          Known limitations
        </h2>
        <ul className="text-sm flex flex-col gap-1.5 list-disc pl-5" style={{ color: "var(--text-secondary)" }}>
          <li>Cost ratio (FP:{m.fp_cost} / FN:{m.fn_cost}) is a placeholder, not a calibrated merchant cost study.</li>
          <li>All data is synthetic — fraud rings are injected by this repo&apos;s own generator, not real fraud.</li>
          <li>No adversarial hardening yet — this model&apos;s robustness under an adaptive fraud ring is unmeasured. See MODEL_CARD.md.</li>
        </ul>
      </section>
    </div>
  );
}
