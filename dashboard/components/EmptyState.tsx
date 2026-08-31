import { Terminal } from "@phosphor-icons/react/dist/ssr";

export function EmptyState({ title, command }: { title: string; command: string }) {
  return (
    <div className="flex flex-col items-center justify-center text-center gap-3 py-24 px-6">
      <Terminal size={28} style={{ color: "var(--ink-tertiary)" }} aria-hidden />
      <p className="text-sm font-medium" style={{ color: "var(--ink)" }}>
        {title}
      </p>
      <p className="text-xs" style={{ color: "var(--ink-secondary)" }}>
        Run the pipeline first, then reload:
      </p>
      <code
        className="mono-figure text-xs rounded px-3 py-2 border"
        style={{ background: "var(--surface-raised)", borderColor: "var(--rule)" }}
      >
        {command}
      </code>
    </div>
  );
}
