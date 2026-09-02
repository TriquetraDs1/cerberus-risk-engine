"use client";

import { useState } from "react";
import { ArrowCounterClockwise, CheckCircle, Prohibit, WarningOctagon } from "@phosphor-icons/react/dist/ssr";
import type { CaseAction, CaseActionType, CaseTargetType } from "@/lib/types";
import { Button } from "./Button";

const ACTION_LABELS: Record<CaseActionType, string> = {
  escalate: "Escalated",
  dismiss: "Dismissed as false positive",
  mark_reviewed: "Reviewed",
  clear: "Cleared",
};

/**
 * Turns a detected ring or a queued transaction from something an analyst only
 * *looks at* into something they can *act on* — the action persists (POST
 * /api/case-actions) and is visible to the next viewer of the page (case-management
 * workflow, not just a read-only queue).
 */
export function CaseActionControls({
  targetType,
  targetId,
  currentAction,
  compact = false,
}: {
  targetType: CaseTargetType;
  targetId: string;
  currentAction?: CaseAction;
  compact?: boolean;
}) {
  const [action, setAction] = useState<CaseAction | undefined>(currentAction);
  const [pending, setPending] = useState<CaseActionType | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function record(actionType: CaseActionType) {
    setPending(actionType);
    setError(null);
    try {
      const res = await fetch("/api/case-actions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_type: targetType, target_id: targetId, action: actionType }),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({})))?.error ?? "Request failed");
      const created: CaseAction = await res.json();
      setAction(created);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setPending(null);
    }
  }

  if (action && action.action !== "clear") {
    const tone = action.action === "escalate" ? "block" : action.action === "dismiss" ? "approve" : "review";
    return (
      <div className="flex items-center gap-2 flex-wrap">
        <span
          className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium border"
          style={{
            color: `var(--risk-${tone})`,
            background: `var(--risk-${tone}-bg)`,
            borderColor: `var(--risk-${tone}-border)`,
          }}
        >
          {action.action === "escalate" ? <WarningOctagon size={13} weight="bold" aria-hidden /> : <CheckCircle size={13} weight="bold" aria-hidden />}
          {ACTION_LABELS[action.action]}
        </span>
        <Button
          tone="ghost"
          compact
          loading={pending === "clear"}
          disabled={pending !== null}
          onClick={() => record("clear")}
          icon={<ArrowCounterClockwise size={12} aria-hidden />}
        >
          {pending === "clear" ? "Clearing…" : "Clear"}
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2 flex-wrap">
        <Button
          tone="danger"
          compact={compact}
          loading={pending === "escalate"}
          disabled={pending !== null}
          onClick={() => record("escalate")}
          icon={<WarningOctagon size={13} weight="bold" aria-hidden />}
        >
          {pending === "escalate" ? "Escalating…" : "Escalate"}
        </Button>
        <Button
          tone="neutral"
          compact={compact}
          loading={pending === "dismiss"}
          disabled={pending !== null}
          onClick={() => record("dismiss")}
          icon={<Prohibit size={13} aria-hidden />}
        >
          {pending === "dismiss" ? "Dismissing…" : compact ? "Dismiss" : "Dismiss as false positive"}
        </Button>
      </div>
      {error && (
        <p className="text-xs" style={{ color: "var(--risk-block)" }} role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
