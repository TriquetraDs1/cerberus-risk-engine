import { PageHeader } from "@/components/PageHeader";
import { QueueTable } from "@/components/QueueTable";
import { EmptyState } from "@/components/EmptyState";
import { getQueue } from "@/lib/data";

export default async function ReviewQueuePage() {
  const queue = await getQueue();

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        title="Review Queue"
        subtitle="Point-risk score + ring membership for every held-out transaction, sampled across all three routing tiers."
      />
      {queue ? (
        <div className="flex-1 min-h-0">
          <QueueTable transactions={queue} />
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
