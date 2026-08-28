import { CheckCircle } from "@phosphor-icons/react/dist/ssr";
import { CalibrationChart } from "@/components/CalibrationChart";
import { PageHeader } from "@/components/PageHeader";
import { EmptyState } from "@/components/EmptyState";
import { StatRow, StatTile } from "@/components/StatTile";
import { formatSegment } from "@/lib/format";
import { getSystemHealth } from "@/lib/data";
import type { Segment } from "@/lib/types";

export default async function SystemHealthPage() {
  const health = await getSystemHealth();

  if (!health) {
    return (
      <div className="flex flex-col h-full">
        <PageHeader title="System Health" subtitle="Model metrics, drift, and pipeline status." />
        <EmptyState
          title="No system health data yet."
          command="python scripts/train_baseline.py && python scripts/build_decision_layer.py && python scripts/export_dashboard_data.py"
        />
      </div>
    );
  }

  const m = health.point_risk_model;
  const savings = m.cost_at_default_threshold - m.cost_at_optimal_threshold;
  const savingsPct = (savings / m.cost_at_default_threshold) * 100;
  const cal = m.calibration;
  const decisionLayer = health.decision_layer;
  const segments = Object.entries(decisionLayer.segments) as [Segment, (typeof decisionLayer.segments)[Segment]][];

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
          <StatTile
            label="Split"
            value={`${m.n_train.toLocaleString("en-US")} / ${m.n_calib.toLocaleString("en-US")} / ${m.n_test.toLocaleString("en-US")}`}
            hint="train / calibration / test"
          />
          <StatTile
            label="Cost saved vs. naive 0.5"
            value={`${savingsPct.toFixed(0)}%`}
            hint={`${savings.toLocaleString("en-US")} cost units at global threshold ${m.cost_optimal_threshold.toFixed(3)}`}
            tone="approve"
          />
        </StatRow>
      </section>

      <section>
        <h2 className="px-6 pt-5 pb-1 text-xs font-medium uppercase tracking-wide" style={{ color: "var(--text-tertiary)" }}>
          Calibration — is 0.79 actually a 79% chance?
        </h2>
        <StatRow>
          <StatTile
            label="Brier score"
            value={cal.brier_after.toFixed(4)}
            hint={`raw model: ${cal.brier_before.toFixed(4)} (lower is better)`}
            tone="approve"
          />
          <StatTile
            label="Expected calibration error"
            value={cal.expected_calibration_error_after.toFixed(4)}
            hint={`raw model: ${cal.expected_calibration_error_before.toFixed(4)}`}
            tone="approve"
          />
        </StatRow>
        <div className="px-6 py-5">
          <CalibrationChart curve={cal.reliability_curve} />
          <p className="text-xs mt-3 max-w-prose" style={{ color: "var(--text-secondary)" }}>
            Fit as isotonic regression on a held-out calibration split (never the test
            set) — see <span className="mono-figure">src/cerberus/detection/calibration.py</span>.
            Points near the dashed diagonal mean a predicted score of X actually
            corresponds to an X likelihood of fraud in this held-out data.
          </p>
        </div>
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

      <section className="px-6 py-5 flex flex-col gap-3">
        <div className="flex items-baseline justify-between">
          <h2 className="text-xs font-medium uppercase tracking-wide" style={{ color: "var(--text-tertiary)" }}>
            Decision layer — per-segment cost-optimal routing
          </h2>
          <span className="text-xs font-medium" style={{ color: "var(--risk-approve)" }}>
            {decisionLayer.overall_savings_pct_vs_global_threshold.toFixed(1)}% cheaper than one global threshold
          </span>
        </div>

        <div className="overflow-x-auto rounded-lg border" style={{ borderColor: "var(--border)" }}>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-[11px] uppercase tracking-wide" style={{ color: "var(--text-tertiary)", background: "var(--surface-raised)" }}>
                <th className="font-medium px-4 py-2">Segment</th>
                <th className="font-medium px-3 py-2 text-right">Mean amount</th>
                <th className="font-medium px-3 py-2 text-right">FP cost</th>
                <th className="font-medium px-3 py-2 text-right">FN cost</th>
                <th className="font-medium px-3 py-2 text-right">Block ≥</th>
                <th className="font-medium px-3 py-2 text-right">Review ≥</th>
                <th className="font-medium px-3 py-2 text-right">Split (B/R/A)</th>
                <th className="font-medium px-3 py-2 text-right">Savings</th>
              </tr>
            </thead>
            <tbody className="divide-y" style={{ borderColor: "var(--border)" }}>
              {segments.map(([seg, r]) => (
                <tr key={seg}>
                  <td className="px-4 py-2.5">{formatSegment(seg)}</td>
                  <td className="px-3 py-2.5 text-right mono-figure">₹{r.cost_matrix.mean_amount.toFixed(0)}</td>
                  <td className="px-3 py-2.5 text-right mono-figure">₹{r.cost_matrix.fp_cost.toFixed(2)}</td>
                  <td className="px-3 py-2.5 text-right mono-figure">₹{r.cost_matrix.fn_cost.toFixed(2)}</td>
                  <td className="px-3 py-2.5 text-right mono-figure">{r.block_threshold.toFixed(3)}</td>
                  <td className="px-3 py-2.5 text-right mono-figure">{r.review_threshold.toFixed(3)}</td>
                  <td className="px-3 py-2.5 text-right mono-figure text-xs" style={{ color: "var(--text-secondary)" }}>
                    {r.n_block}/{r.n_review}/{r.n_approve}
                  </td>
                  <td className="px-3 py-2.5 text-right mono-figure" style={{ color: "var(--risk-approve)" }}>
                    {r.cost_savings_pct.toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
          Each segment&apos;s cost matrix is derived from that segment&apos;s own mean
          transaction amount, not one global assumption — see{" "}
          <span className="mono-figure">src/cerberus/decision/cost_matrix.py</span>.
        </p>
      </section>

      <section className="px-6 py-5 flex flex-col gap-4">
        <h2 className="text-xs font-medium uppercase tracking-wide" style={{ color: "var(--text-tertiary)" }}>
          Known limitations
        </h2>
        <ul className="text-sm flex flex-col gap-1.5 list-disc pl-5" style={{ color: "var(--text-secondary)" }}>
          {decisionLayer.limitations.map((l) => (
            <li key={l}>{l}</li>
          ))}
          <li>All data is synthetic — fraud rings are injected by this repo&apos;s own generator, not real fraud.</li>
          <li>No adversarial hardening yet — this model&apos;s robustness under an adaptive fraud ring is unmeasured. See MODEL_CARD.md.</li>
        </ul>
      </section>
    </div>
  );
}
