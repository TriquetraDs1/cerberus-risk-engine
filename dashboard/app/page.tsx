import { PageHeader } from "@/components/PageHeader";
import { QueueTable } from "@/components/QueueTable";
import { EmptyState } from "@/components/EmptyState";
import { latestActionByTarget } from "@/lib/caseActions";
import { getQueue } from "@/lib/data";
import type { CaseAction } from "@/lib/types";

// This page reads case_actions.json on every render — a plain `fs.readFile` gives
// Next.js no signal that the data is request-varying, so without this it gets
// prerendered once at build time and a `next build && next start` deploy would never
// show a case action recorded after that build. Case actions are the whole point of
// this page's workflow, so it must render per-request, not serve a stale snapshot.
export const dynamic = "force-dynamic";

export default async function ReviewQueuePage() {
  const [queue, actionsByTarget] = await Promise.all([getQueue(), latestActionByTarget()]);

  const transactionActions: Record<string, CaseAction> = {};
  for (const action of Object.values(actionsByTarget)) {
    if (action.target_type === "transaction") transactionActions[action.target_id] = action;
  }

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        title="Review Queue"
        subtitle="Point-risk score + ring membership for every held-out transaction, sampled across all three routing tiers."
      />
      {queue ? (
        <div className="flex-1 min-h-0">
          <QueueTable transactions={queue} transactionActions={transactionActions} />
        </div>
      ) : (
        <EmptyState
          title="No queue data yet."
          command="python scripts/generate_data.py && python scripts/detect_rings.py && python scripts/train_baseline.py && python scripts/export_dashboard_data.py"
        />
      )}
    </div>
  );
}
