import { PageHeader } from "@/components/PageHeader";
import { RingGraph } from "@/components/RingGraph";
import { EmptyState } from "@/components/EmptyState";
import { StatRow, StatTile } from "@/components/StatTile";
import { getRingGraph, getSystemHealth } from "@/lib/data";

export default async function RingNetworkPage() {
  const [graph, health] = await Promise.all([getRingGraph(), getSystemHealth()]);
  const report = health?.ring_detection;

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
          <RingGraph graph={graph} />
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
