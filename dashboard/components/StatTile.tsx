export function StatTile({
  label,
  value,
  hint,
  tone = "default",
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "default" | "approve" | "review" | "block";
}) {
  const toneColor =
    tone === "approve"
      ? "var(--risk-approve)"
      : tone === "review"
      ? "var(--risk-review)"
      : tone === "block"
      ? "var(--risk-block)"
      : "var(--text-primary)";

  return (
    <div className="px-4 py-3">
      <p className="text-[11px] uppercase tracking-wide" style={{ color: "var(--text-tertiary)" }}>
        {label}
      </p>
      <p className="mono-figure text-2xl font-semibold mt-1" style={{ color: toneColor }}>
        {value}
      </p>
      {hint && (
        <p className="text-xs mt-1" style={{ color: "var(--text-secondary)" }}>
          {hint}
        </p>
      )}
    </div>
  );
}

export function StatRow({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="grid grid-cols-2 sm:grid-cols-4 divide-x divide-y sm:divide-y-0 border-b"
      style={{ borderColor: "var(--border)" }}
    >
      {children}
    </div>
  );
}
