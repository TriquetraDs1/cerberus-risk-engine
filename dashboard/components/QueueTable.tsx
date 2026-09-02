"use client";

import { useMemo, useState } from "react";
import { MagnifyingGlass } from "@phosphor-icons/react/dist/ssr";
import { formatReasonCode, formatSegment } from "@/lib/format";
import type { CaseAction, Decision, QueueTransaction } from "@/lib/types";
import { RiskBadge } from "./RiskBadge";
import { TransactionDrawer } from "./TransactionDrawer";

const FILTERS: { key: Decision | "all"; label: string }[] = [
  { key: "all", label: "All" },
  { key: "block", label: "Block" },
  { key: "review", label: "Review" },
  { key: "approve", label: "Approve" },
];

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

// Formatted by hand, not Intl: this component SSRs then hydrates, and Node's ICU can
// disagree with the browser's ICU on a locale's default hour cycle even with identical
// options (observed: server "Feb 22, 03:33 PM" vs. client "22 Feb, 03:33 pm" for the
// same `en-GB` call) — a hydration mismatch neither environment is "wrong" about.
// Fixed-format, UTC-based arithmetic has no locale to disagree on.
function formatTime(iso: string) {
  const d = new Date(iso);
  const hour24 = d.getUTCHours();
  const hour12 = hour24 % 12 || 12;
  const minute = String(d.getUTCMinutes()).padStart(2, "0");
  const ampm = hour24 < 12 ? "AM" : "PM";
  return `${MONTHS[d.getUTCMonth()]} ${String(d.getUTCDate()).padStart(2, "0")}, ${hour12}:${minute} ${ampm}`;
}

export function QueueTable({
  transactions,
  transactionActions = {},
}: {
  transactions: QueueTransaction[];
  transactionActions?: Record<string, CaseAction>;
}) {
  const [filter, setFilter] = useState<Decision | "all">("all");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<QueueTransaction | null>(null);

  const counts = useMemo(() => {
    const c: Record<string, number> = { all: transactions.length, block: 0, review: 0, approve: 0 };
    for (const t of transactions) c[t.decision]++;
    return c;
  }, [transactions]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return transactions.filter((t) => {
      if (filter !== "all" && t.decision !== filter) return false;
      if (!q) return true;
      return (
        t.account_id.toLowerCase().includes(q) ||
        t.transaction_id.toLowerCase().includes(q) ||
        t.segment.toLowerCase().includes(q) ||
        (t.ring_id ?? "").toLowerCase().includes(q)
      );
    });
  }, [transactions, filter, query]);

  return (
    <div className="flex flex-col h-full">
      <div className="flex flex-col sm:flex-row sm:items-center gap-3 px-6 py-3 border-b" style={{ borderColor: "var(--rule)" }}>
        <div
          role="tablist"
          aria-label="Filter by decision"
          className="flex gap-1 rounded-lg p-1 w-fit"
          style={{ background: "var(--surface-sunken)", border: "1px solid var(--rule)" }}
        >
          {FILTERS.map((f) => (
            <button
              key={f.key}
              role="tab"
              aria-selected={filter === f.key}
              onClick={() => setFilter(f.key)}
              className="px-3 py-1.5 text-xs font-medium rounded-md transition-colors"
              style={
                filter === f.key
                  ? { background: "var(--surface)", color: "var(--ink)", boxShadow: "0 1px 1px oklch(0 0 0 / 0.05)" }
                  : { color: "var(--ink-secondary)" }
              }
            >
              {f.label}
              <span className="mono-figure ml-1.5 text-[10px]" style={{ color: "var(--ink-tertiary)" }}>
                {counts[f.key]}
              </span>
            </button>
          ))}
        </div>

        <div className="relative sm:ml-auto w-full sm:w-64">
          <MagnifyingGlass
            size={14}
            className="absolute left-2.5 top-1/2 -translate-y-1/2"
            style={{ color: "var(--ink-tertiary)" }}
            aria-hidden
          />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search account, txn, segment, or ring…"
            aria-label="Search transactions"
            className="w-full pl-8 pr-3 py-1.5 text-xs rounded-md border outline-none"
            style={{ borderColor: "var(--rule)", background: "var(--surface)" }}
          />
        </div>
      </div>

      {filtered.length === 0 ? (
        <p className="px-6 py-12 text-sm text-center" style={{ color: "var(--ink-secondary)" }}>
          No transactions match this filter.
        </p>
      ) : (
        <div className="flex-1 overflow-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr
                className="sticky top-0 text-left kicker z-10 border-b"
                style={{ background: "var(--surface-sunken)", color: "var(--ink-tertiary)" }}
              >
                <th className="font-semibold px-6 py-2.5">Time</th>
                <th className="font-semibold px-3 py-2.5">Account</th>
                <th className="font-semibold px-3 py-2.5">Segment</th>
                <th className="font-semibold px-3 py-2.5 text-right">Amount</th>
                <th className="font-semibold px-3 py-2.5 text-right">Risk score</th>
                <th className="font-semibold px-3 py-2.5">Decision</th>
                <th className="font-semibold px-3 py-2.5">Top reason</th>
                <th className="font-semibold px-3 py-2.5">Ring</th>
              </tr>
            </thead>
            <tbody className="divide-y" style={{ borderColor: "var(--rule)" }}>
              {filtered.map((t) => (
                <tr
                  key={t.transaction_id}
                  tabIndex={0}
                  role="button"
                  aria-label={`Open transaction ${t.transaction_id}`}
                  onClick={() => setSelected(t)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setSelected(t);
                    }
                  }}
                  className="cerb-row hover:bg-[var(--rust-soft)] focus-visible:bg-[var(--rust-soft)]"
                  style={{ borderColor: "var(--rule)" }}
                >
                  <td className="px-6 py-2 whitespace-nowrap mono-figure text-[12px]" style={{ color: "var(--ink-secondary)" }}>
                    {formatTime(t.timestamp)}
                  </td>
                  <td className="px-3 py-2 mono-figure text-[12px]">{t.account_id}</td>
                  <td className="px-3 py-2 text-[12.5px]" style={{ color: "var(--ink-secondary)" }}>
                    {formatSegment(t.segment)}
                  </td>
                  <td className="px-3 py-2 text-right mono-figure text-[13px]" style={{ color: "var(--ink-secondary)" }}>₹{t.amount.toLocaleString("en-US")}</td>
                  <td className="px-3 py-2 text-right mono-figure font-semibold">{t.risk_score.toFixed(3)}</td>
                  <td className="px-3 py-2">
                    <RiskBadge decision={t.decision} />
                  </td>
                  <td className="px-3 py-2 text-[12.5px]" style={{ color: "var(--ink-secondary)" }}>
                    {t.reason_codes[0] ? formatReasonCode(t.reason_codes[0]) : "—"}
                  </td>
                  <td className="px-3 py-2 mono-figure text-[12px]" style={{ color: t.ring_id ? "var(--rust)" : "var(--ink-tertiary)" }}>
                    {t.ring_id ?? "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <TransactionDrawer
        transaction={selected}
        currentAction={selected ? transactionActions[selected.transaction_id] : undefined}
        onClose={() => setSelected(null)}
      />
    </div>
  );
}
