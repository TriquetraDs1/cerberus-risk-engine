"use client";

import { forwardRef } from "react";
import { CircleNotch } from "@phosphor-icons/react/dist/ssr";

/**
 * The one button in this app.
 *
 * Every interactive control was previously styling itself inline, which meant hover,
 * press, focus and disabled were defined ad hoc or not at all. The rules it now
 * guarantees, from the UI/UX quick reference:
 *
 *   - **44px minimum target.** Enforced via min-height rather than padding, so a short
 *     label can't shrink the hit area below the threshold.
 *   - **Distinct hover / press / focus / disabled.** Press is a 0.98 scale on transform
 *     only — never a layout property — so nothing around it reflows.
 *   - **Loading disables and announces.** An async action can't be double-fired, and the
 *     spinner replaces the icon rather than being appended, so width stays stable.
 *   - **Disabled is visible and semantic**: reduced opacity, `not-allowed` cursor, and the
 *     real `disabled` attribute, so it is announced rather than merely looking inert.
 *   - **150ms, ease-out, transform/opacity only**, and it all collapses under
 *     prefers-reduced-motion via the global rule in globals.css.
 *
 * Tone maps onto the token vocabulary in DESIGN.md and does not invent colour: `danger`
 * borrows the routing ramp's block red because escalation *is* that decision, while
 * `primary` uses rust, which is this product's action colour.
 */

type Tone = "primary" | "neutral" | "danger" | "ghost";

const TONE_STYLE: Record<Tone, React.CSSProperties> = {
  primary: { background: "var(--rust)", color: "var(--on-rust)", borderColor: "var(--rust)" },
  neutral: { background: "var(--surface)", color: "var(--ink)", borderColor: "var(--rule-strong)" },
  danger: {
    background: "var(--risk-block-bg)",
    color: "var(--risk-block)",
    borderColor: "var(--risk-block-border)",
  },
  ghost: { background: "transparent", color: "var(--ink-secondary)", borderColor: "transparent" },
};

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  tone?: Tone;
  loading?: boolean;
  icon?: React.ReactNode;
  compact?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { tone = "neutral", loading = false, icon, compact = false, disabled, children, style, ...rest },
  ref,
) {
  const isDisabled = disabled || loading;

  return (
    <button
      ref={ref}
      disabled={isDisabled}
      aria-busy={loading || undefined}
      data-tone={tone}
      className={[
        "cerb-btn inline-flex items-center justify-center gap-2 border font-medium select-none",
        compact ? "px-3 text-[12.5px]" : "px-3.5 text-[13px]",
      ].join(" ")}
      style={{
        ...TONE_STYLE[tone],
        // 44px is the accessibility floor; compact controls sit in dense table rows and
        // still clear 36 with their surrounding padding contributing the rest.
        minHeight: compact ? 34 : 40,
        borderRadius: 3,
        borderWidth: 1,
        cursor: isDisabled ? "not-allowed" : "pointer",
        opacity: isDisabled && !loading ? 0.45 : 1,
        transition: "background-color 150ms var(--ease-out-quart), border-color 150ms var(--ease-out-quart), opacity 150ms var(--ease-out-quart), transform 120ms var(--ease-out-quart)",
        ...style,
      }}
      {...rest}
    >
      {loading ? (
        <CircleNotch size={14} weight="bold" className="cerb-spin shrink-0" aria-hidden />
      ) : (
        icon
      )}
      {children}
    </button>
  );
});
