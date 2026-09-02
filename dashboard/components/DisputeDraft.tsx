"use client";

import { useState } from "react";
import { CheckCircle, Copy, FileText, Warning } from "@phosphor-icons/react/dist/ssr";
import { ApiError, apiConfigured, draftDispute } from "@/lib/api";
import type { QueueTransaction } from "@/lib/types";
import { Button } from "./Button";

/**
 * A2 in the drawer. Deliberately not fetched on open: drafting is a generative call, and
 * spending one every time an analyst glances at a transaction would be wasteful and slow.
 * It runs on request, and the button says what it will do before it does it.
 */
export function DisputeDraft({ transaction }: { transaction: QueueTransaction }) {
  const [state, setState] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [draft, setDraft] = useState("");
  const [source, setSource] = useState<"llm" | "template" | null>(null);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);

  async function run() {
    setState("loading");
    setError("");
    try {
      // The decision travels with the request. These queue rows were scored offline by
      // the export script, so the live service has no record of them — sending what the
      // dashboard already holds is what makes the feature work from this surface.
      const res = await draftDispute(transaction.transaction_id, {
        decision: transaction.decision,
        risk_score: transaction.risk_score,
        segment: transaction.segment,
        amount: transaction.amount,
        account_id: transaction.account_id,
        reason_codes: transaction.reason_codes,
        ring_id: transaction.ring_id,
        timestamp: transaction.timestamp,
      });
      setDraft(res.draft);
      setSource(res.source);
      setState("done");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong.");
      setState("error");
    }
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(draft);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard is permission-gated and can simply refuse. The draft is selectable in
      // the panel either way, so a failed copy needs no error state of its own.
    }
  }

  if (!apiConfigured) {
    return (
      <p className="text-[12px] leading-relaxed" style={{ color: "var(--ink-tertiary)" }}>
        Dispute drafting runs on the live API, which isn&rsquo;t connected to this
        deployment. Available at <code className="mono-figure">POST /dispute/{"{id}"}</code>.
      </p>
    );
  }

  return (
    <div>
      {state !== "done" && (
        <Button
          tone="neutral"
          loading={state === "loading"}
          onClick={run}
          icon={<FileText size={14} aria-hidden />}
        >
          {state === "loading" ? "Drafting…" : "Draft dispute evidence"}
        </Button>
      )}

      {state === "loading" && (
        <p className="text-[11.5px] mt-2 leading-snug" style={{ color: "var(--ink-tertiary)" }}>
          If the API has been idle it may take up to a minute to wake.
        </p>
      )}

      {state === "error" && (
        <div className="mt-3 flex gap-2 items-start">
          <Warning size={14} className="mt-0.5 shrink-0" style={{ color: "var(--risk-review)" }} aria-hidden />
          <div>
            <p className="text-[12px] leading-relaxed" style={{ color: "var(--ink-secondary)" }}>
              {error}
            </p>
            <Button tone="ghost" compact onClick={run} style={{ marginTop: 6, marginLeft: -12 }}>
              Try again
            </Button>
          </div>
        </div>
      )}

      {state === "done" && (
        <div>
          <div className="flex items-center justify-between gap-3 mb-2">
            <span className="text-[11px]" style={{ color: "var(--ink-tertiary)" }}>
              {source === "llm" ? "Model-written draft" : "Templated draft (no API key set)"}
            </span>
            <Button
              tone="ghost"
              compact
              onClick={copy}
              aria-label={copied ? "Draft copied to clipboard" : "Copy draft to clipboard"}
              style={{ color: copied ? "var(--risk-approve)" : "var(--rust)", marginRight: -12 }}
              icon={copied ? <CheckCircle size={13} weight="bold" aria-hidden /> : <Copy size={13} aria-hidden />}
            >
              {copied ? "Copied" : "Copy"}
            </Button>
          </div>
          <pre
            className="text-[12px] leading-[1.65] whitespace-pre-wrap font-sans p-3 max-h-[22rem] overflow-y-auto border"
            style={{ background: "var(--surface-sunken)", borderColor: "var(--rule)", borderRadius: 3, color: "var(--ink-secondary)" }}
          >
            {draft}
          </pre>
          <p className="text-[11px] mt-2 leading-snug" style={{ color: "var(--ink-tertiary)" }}>
            A draft for a human to check and file, not a submission. It states only what the
            pipeline recorded.
          </p>
        </div>
      )}
    </div>
  );
}
