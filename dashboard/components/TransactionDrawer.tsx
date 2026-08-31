"use client";

import { useEffect, useRef } from "react";
import { WarningCircle, X } from "@phosphor-icons/react/dist/ssr";
import { formatReasonCode, formatSegment } from "@/lib/format";
import type { CaseAction, QueueTransaction } from "@/lib/types";
import { CaseActionControls } from "./CaseActionControls";
import { RiskBadge } from "./RiskBadge";

export function TransactionDrawer({
  transaction,
  currentAction,
  onClose,
}: {
  transaction: QueueTransaction | null;
  currentAction?: CaseAction;
  onClose: () => void;
}) {
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!transaction) return;
    closeButtonRef.current?.focus();
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [transaction, onClose]);

  if (!transaction) return null;
  const t = transaction;

  return (
    <div className="fixed inset-0 z-40 flex justify-end" role="dialog" aria-modal="true" aria-label="Transaction detail">
      {/* Scrim — strong enough to isolate the foreground panel, per modal-legibility guidance */}
      <button
        aria-label="Close transaction detail"
        onClick={onClose}
        className="absolute inset-0"
        style={{ background: "oklch(0.20 0.014 60 / 0.42)" }}
      />
      <div
        className="relative w-full sm:w-[420px] h-full overflow-y-auto border-l shadow-xl"
        style={{ background: "var(--surface)", borderColor: "var(--rule)" }}
      >
        <div className="flex items-center justify-between px-5 py-4 border-b sticky top-0" style={{ borderColor: "var(--rule)", background: "var(--surface)" }}>
          <div>
            <p className="text-xs" style={{ color: "var(--ink-tertiary)" }}>Transaction</p>
            <p className="mono-figure text-sm font-medium">{t.transaction_id}</p>
          </div>
          <button
            ref={closeButtonRef}
            onClick={onClose}
            aria-label="Close"
            className="p-2 rounded-md hover:bg-[var(--rust-soft)]"
          >
            <X size={16} aria-hidden />
          </button>
        </div>

        <div className="px-5 py-5 flex flex-col gap-6">
          <div className="flex items-center justify-between">
            <RiskBadge decision={t.decision} />
            <span className="mono-figure text-2xl font-semibold">{t.risk_score.toFixed(4)}</span>
          </div>

          <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
            <div>
              <dt className="text-xs" style={{ color: "var(--ink-tertiary)" }}>Account</dt>
              <dd className="mono-figure">{t.account_id}</dd>
            </div>
            <div>
              <dt className="text-xs" style={{ color: "var(--ink-tertiary)" }}>Amount</dt>
              <dd className="mono-figure">₹{t.amount.toLocaleString("en-US")}</dd>
            </div>
            <div>
              <dt className="text-xs" style={{ color: "var(--ink-tertiary)" }}>Segment</dt>
              <dd>{formatSegment(t.segment)}</dd>
            </div>
            <div>
              <dt className="text-xs" style={{ color: "var(--ink-tertiary)" }}>Timestamp</dt>
              <dd className="mono-figure text-xs">{new Date(t.timestamp).toLocaleString("en-US")}</dd>
            </div>
            <div>
              <dt className="text-xs" style={{ color: "var(--ink-tertiary)" }}>Ring</dt>
              <dd className="mono-figure" style={{ color: t.ring_id ? "var(--rust)" : "var(--ink-secondary)" }}>
                {t.ring_id ?? "none detected"}
              </dd>
            </div>
          </dl>

          {t.explanation ? (
            <div>
              <h3 className="kicker block mb-2" style={{ color: "var(--ink-tertiary)" }}>
                Summary
              </h3>
              <p className="text-sm leading-relaxed" style={{ color: "var(--ink-secondary)" }}>
                {t.explanation}
              </p>
              <p className="text-[11px] mt-1.5" style={{ color: "var(--ink-tertiary)" }}>
                Generated from this transaction&apos;s reason codes and cost basis — it does not re-score.
              </p>
            </div>
          ) : null}

          <div>
            <h3 className="kicker block mb-2" style={{ color: "var(--ink-tertiary)" }}>
              Reason codes
            </h3>
            <ul className="flex flex-col gap-1.5">
              {t.reason_codes.map((code) => (
                <li key={code} className="flex items-center gap-2 text-sm">
                  <WarningCircle size={14} style={{ color: "var(--risk-review)" }} aria-hidden />
                  {formatReasonCode(code)}
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className="kicker block mb-2" style={{ color: "var(--ink-tertiary)" }}>
              Cost basis
            </h3>
            <div
              className="grid grid-cols-2 divide-x rounded-lg border text-center"
              style={{ borderColor: "var(--rule)" }}
            >
              <div className="px-3 py-2.5" style={{ borderColor: "var(--rule)" }}>
                <p className="mono-figure text-base font-medium">{t.cost_basis.fp_cost}</p>
                <p className="text-[11px]" style={{ color: "var(--ink-tertiary)" }}>FP cost</p>
              </div>
              <div className="px-3 py-2.5">
                <p className="mono-figure text-base font-medium">{t.cost_basis.fn_cost}</p>
                <p className="text-[11px]" style={{ color: "var(--ink-tertiary)" }}>FN cost</p>
              </div>
            </div>
            <p className="text-[11px] mt-2" style={{ color: "var(--ink-tertiary)" }}>
              Block ≥ {t.cost_basis.block_threshold.toFixed(3)} · Review ≥{" "}
              {t.cost_basis.review_threshold.toFixed(3)} — this segment&apos;s cost-optimal
              thresholds, see System Health for all four segments.
            </p>
          </div>

          <div>
            <h3 className="kicker block mb-2" style={{ color: "var(--ink-tertiary)" }}>
              Case action
            </h3>
            <CaseActionControls
              key={t.transaction_id}
              targetType="transaction"
              targetId={t.transaction_id}
              currentAction={currentAction}
            />
          </div>

          <div className="pt-3 border-t text-xs" style={{ borderColor: "var(--rule)", color: "var(--ink-tertiary)" }}>
            Synthetic ground truth for this demo transaction:{" "}
            <span className="font-medium" style={{ color: "var(--ink-secondary)" }}>
              {t.actual_label === 1 ? "labeled fraud" : "labeled legitimate"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
