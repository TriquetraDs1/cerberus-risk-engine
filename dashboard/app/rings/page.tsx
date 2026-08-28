import { PageHeader } from "@/components/PageHeader";
import { RingGraph } from "@/components/RingGraph";
import { EmptyState } from "@/components/EmptyState";
import { StatRow, StatTile } from "@/components/StatTile";
import { latestActionByTarget } from "@/lib/caseActions";
import { getRingGraph, getSystemHealth } from "@/lib/data";
import type { CaseAction } from "@/lib/types";

// See app/page.tsx for why this is required: this page reads case_actions.json,
// mutable state a plain fs.readFile gives Next.js no signal about — without this it
// would be prerendered once at build time and never show a case action again.
export const dynamic = "force-dynamic";

export default async function RingNetworkPage() {
  const [graph, health, actionsByTarget] = await Promise.all([getRingGraph(), getSystemHealth(), latestActionByTarget()]);
  const report = health?.ring_detection;

  // Re-key from "ring:detected_8" to "detected_8" — what RingGraph looks up by.
  const ringActions: Record<string, CaseAction> = {};
  for (const action of Object.values(actionsByTarget)) {
    if (action.target_type === "ring") ringActions[action.target_id] = action;
  }

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        title="Ring Network"
        subtitle="Entity-link graph (shared device / card / IP) with Louvain-detected coordinated clusters."
      />

      {report && (
        <StatRow>
          <StatTile
            label="Rings recovered"
            value={`${report.n_perfectly_recovered} / ${report.n_rings}`}
            hint="Perfectly recovered injected rings"
          />
          <StatTile
            label="Mean recovery"
            value={`${(report.mean_ring_recovery * 100).toFixed(0)}%`}
            tone="approve"
          />
          <StatTile
            label="Household FP rate"
            value={`${(report.household_false_positive_rate * 100).toFixed(1)}%`}
            hint={`${report.n_household_false_positives} / ${report.n_household_pairs} innocent pairs wrongly co-flagged`}
            tone="review"
          />
          <StatTile
            label="Flagged communities"
            value={`${report.n_flagged_communities}`}
            hint={`of ${report.n_total_communities} total communities`}
          />
        </StatRow>
      )}

      <div className="p-6 flex-1 min-h-0 overflow-auto">
        {graph ? (
          <RingGraph graph={graph} ringActions={ringActions} />
        ) : (
          <EmptyState
            title="No ring graph yet."
            command="python scripts/detect_rings.py && python scripts/export_dashboard_data.py"
          />
        )}
      </div>
    </div>
  );
}
