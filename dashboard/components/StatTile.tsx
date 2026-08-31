import { Children } from "react";

export function StatTile({
  label,
  value,
  hint,
  tone = "default",
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "default" | "approve" | "review" | "block" | "attack";
}) {
  const toneColor =
    tone === "approve"
      ? "var(--risk-approve)"
      : tone === "review"
      ? "var(--risk-review)"
      : tone === "block"
      ? "var(--risk-block)"
      : tone === "attack"
      ? "var(--rust)"
      : "var(--ink)";

  return (
    <div className="px-5 py-4 h-full" style={{ background: "var(--surface)" }}>
      <p className="kicker">{label}</p>
      {/* Figure first, at a size that reads across a room, then the qualifier under it.
          The qualifier is not decoration: a bare number with no cost or denominator
          attached is exactly what this project argues against. */}
      <p
        className="mono-figure text-[26px] font-semibold mt-2 leading-none tracking-[-0.02em]"
        style={{ color: toneColor }}
      >
        {value}
      </p>
      {hint && (
        <p className="text-[12px] mt-2 leading-snug max-w-[36ch]" style={{ color: "var(--ink-secondary)" }}>
          {hint}
        </p>
      )}
    </div>
  );
}

/**
 * Hairline-separated stat strip. One rule colour showing through a 1px grid gap reads
 * cleaner than four cards each drawing their own edge, and never doubles up on a shared
 * boundary.
 *
 * Columns are derived from the child count rather than hardcoded to four: a row of two
 * on a four-column grid leaves empty cells, and with the rule colour behind the grid
 * those empties render as grey blocks. Deriving the count keeps any number of tiles
 * flush.
 */
export function StatRow({ children }: { children: React.ReactNode }) {
  const count = Children.toArray(children).length;
  const columns = Math.min(count, 4);

  return (
    <div
      className="grid gap-px border-b"
      style={{
        background: "var(--rule)",
        borderColor: "var(--rule)",
        gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
      }}
    >
      {children}
    </div>
  );
}
