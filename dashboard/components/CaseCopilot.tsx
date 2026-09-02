"use client";

import { useRef, useState } from "react";
import { ArrowUp, Warning } from "@phosphor-icons/react/dist/ssr";
import { ApiError, apiConfigured, askCopilot } from "@/lib/api";
import { Button } from "./Button";

type Turn = { role: "user" | "assistant"; content: string };

// Openers, so the first interaction isn't a blank box. Analysts ask the same three
// questions of a new case; offering them is faster than typing and teaches what the
// copilot is grounded in.
const SUGGESTIONS = [
  "Why was this ring flagged?",
  "Which account is the hub?",
  "Could this be innocent device sharing?",
];

export function CaseCopilot({ ringId }: { ringId: string }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  async function send(question: string) {
    const trimmed = question.trim();
    if (!trimmed || busy) return;

    const next: Turn[] = [...turns, { role: "user", content: trimmed }];
    setTurns(next);
    setInput("");
    setBusy(true);
    setError("");

    try {
      const res = await askCopilot(ringId, next);
      setTurns([...next, { role: "assistant", content: res.answer }]);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong.");
      // Drop the unanswered question rather than leaving it stranded above an error —
      // resending would otherwise duplicate it in the transcript.
      setTurns(turns);
    } finally {
      setBusy(false);
      requestAnimationFrame(() => scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight }));
    }
  }

  if (!apiConfigured) {
    return (
      <div className="p-4 border" style={{ borderColor: "var(--rule)", background: "var(--surface)", borderRadius: 3 }}>
        <p className="kicker">Case copilot</p>
        <p className="text-[12px] mt-2 leading-relaxed" style={{ color: "var(--ink-tertiary)" }}>
          Runs on the live API, which isn&rsquo;t connected to this deployment. Available at{" "}
          <code className="mono-figure">POST /copilot/{"{ring_id}"}</code>.
        </p>
      </div>
    );
  }

  return (
    <div className="border flex flex-col" style={{ borderColor: "var(--rule)", background: "var(--surface)", borderRadius: 3 }}>
      <div className="px-4 py-3 border-b" style={{ borderColor: "var(--rule)" }}>
        <p className="kicker">Case copilot</p>
        <p className="text-[11.5px] mt-1 leading-snug" style={{ color: "var(--ink-tertiary)" }}>
          Answers from this case only. It cannot change anything.
        </p>
      </div>

      <div ref={scrollRef} className="px-4 py-3 flex flex-col gap-3 max-h-[19rem] overflow-y-auto">
        {turns.length === 0 && !busy && (
          <div className="flex flex-col gap-1.5 items-start">
            {SUGGESTIONS.map((q) => (
              <Button
                key={q}
                tone="ghost"
                compact
                onClick={() => send(q)}
                style={{ color: "var(--rust)", marginLeft: -12, justifyContent: "flex-start" }}
              >
                {q}
              </Button>
            ))}
          </div>
        )}

        {turns.map((t, i) => (
          <div key={i} className={t.role === "user" ? "self-end max-w-[85%]" : "max-w-[92%]"}>
            <p
              className="text-[12.5px] leading-relaxed px-3 py-2"
              style={{
                background: t.role === "user" ? "var(--rust-soft)" : "var(--surface-sunken)",
                color: t.role === "user" ? "var(--ink)" : "var(--ink-secondary)",
                borderRadius: 3,
                whiteSpace: "pre-wrap",
              }}
            >
              {t.content}
            </p>
          </div>
        ))}

        {busy && (
          <p className="text-[12px]" style={{ color: "var(--ink-tertiary)" }}>
            Thinking… (a sleeping API can take up to a minute)
          </p>
        )}

        {error && (
          <div className="flex gap-2 items-start">
            <Warning size={13} className="mt-0.5 shrink-0" style={{ color: "var(--risk-review)" }} aria-hidden />
            <p className="text-[12px] leading-relaxed" style={{ color: "var(--ink-secondary)" }}>
              {error}
            </p>
          </div>
        )}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
        className="flex items-center gap-2 px-3 py-2.5 border-t"
        style={{ borderColor: "var(--rule)" }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about this case…"
          aria-label="Ask about this case"
          disabled={busy}
          maxLength={800}
          className="flex-1 min-w-0 text-[12.5px] bg-transparent outline-none disabled:opacity-60"
        />
        <Button
          type="submit"
          tone="primary"
          compact
          loading={busy}
          disabled={busy || !input.trim()}
          aria-label="Send question"
          style={{ minWidth: 34, paddingLeft: 0, paddingRight: 0 }}
          icon={busy ? undefined : <ArrowUp size={13} weight="bold" aria-hidden />}
        />
      </form>
    </div>
  );
}
