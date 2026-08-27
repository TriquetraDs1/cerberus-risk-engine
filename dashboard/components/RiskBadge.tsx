import { CheckCircle, Eye, Prohibit } from "@phosphor-icons/react/dist/ssr";
import type { Decision } from "@/lib/types";

/**
 * Decision is never conveyed by color alone: icon + text + color together, so the
 * dashboard stays legible for colorblind analysts and readable in a black-and-white
 * printout of an incident report.
 */
const DECISION_CONFIG: Record<
  Decision,
  { label: string; Icon: typeof CheckCircle; fg: string; bg: string; border: string }
> = {
  approve: {
    label: "Approve",
    Icon: CheckCircle,
    fg: "var(--risk-approve)",
    bg: "var(--risk-approve-bg)",
    border: "var(--risk-approve-border)",
  },
  review: {
    label: "Review",
    Icon: Eye,
    fg: "var(--risk-review)",
    bg: "var(--risk-review-bg)",
    border: "var(--risk-review-border)",
  },
  block: {
    label: "Block",
    Icon: Prohibit,
    fg: "var(--risk-block)",
    bg: "var(--risk-block-bg)",
    border: "var(--risk-block-border)",
  },
};

export function RiskBadge({ decision }: { decision: Decision }) {
  const cfg = DECISION_CONFIG[decision];
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium border"
      style={{ color: cfg.fg, background: cfg.bg, borderColor: cfg.border }}
    >
      <cfg.Icon size={13} weight="bold" aria-hidden />
      {cfg.label}
    </span>
  );
}
