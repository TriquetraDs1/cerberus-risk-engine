import type { EvasionStrategy, StrategyHardeningResult } from "@/lib/types";

const WIDTH = 640;
const HEIGHT = 268;
const MARGIN = { top: 20, right: 16, bottom: 44, left: 44 };
const PLOT_W = WIDTH - MARGIN.left - MARGIN.right;
const PLOT_H = HEIGHT - MARGIN.top - MARGIN.bottom;

const STRATEGY_LABELS: Record<EvasionStrategy, string> = {
  structuring: "Structuring",
  identity_rotation: "Identity rotation",
  slow_ramp: "Slow ramp",
};

/**
 * Colour vocabulary, deliberately NOT the risk ramp.
 *
 * Green/amber/red mean approve/review/block everywhere else in this console. Reusing
 * green here for "unattacked" and red for "under attack" would overload those signals
 * with a second, unrelated meaning on the one page where precision matters most — an
 * analyst reading a red bar has to stop and ask which red it is.
 *
 * So this chart uses the adversarial vocabulary instead: neutral for the baseline, rust
 * for damage, ink for the recovered state. The same three colours the /about page uses
 * for the same three quantities.
 */
const SERIES = [
  { key: "baseline_detection" as const, label: "Unattacked", color: "var(--ink-tertiary)" },
  { key: "evaded_original_model" as const, label: "Under attack (original model)", color: "var(--rust)" },
  { key: "evaded_hardened_model" as const, label: "Under attack (hardened model)", color: "var(--ink)" },
];

/**
 * The before/attack/after-hardening grouped bar chart — this is the submission's
 * headline slide: it shows the detector failing under a specific, named attack, then
 * shows the specific, measured recovery after hardening. Bars, not a line, since these
 * are three discrete regimes being compared, not a continuous series.
 */
export function RecallDecayChart({ strategies }: { strategies: Record<EvasionStrategy, StrategyHardeningResult> }) {
  const names = Object.keys(strategies) as EvasionStrategy[];
  const groupWidth = PLOT_W / names.length;
  const barWidth = groupWidth / (SERIES.length + 1.5);

  const yFor = (v: number) => MARGIN.top + PLOT_H * (1 - v);

  const summary = names
    .map((n) => {
      const r = strategies[n];
      return `${STRATEGY_LABELS[n]}: unattacked ${r.baseline_detection.combined_score.toFixed(2)}, under attack ${r.evaded_original_model.combined_score.toFixed(2)}, after hardening ${r.evaded_hardened_model.combined_score.toFixed(2)}`;
    })
    .join(". ");

  return (
    <div className="flex flex-col gap-4">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        width="100%"
        role="img"
        aria-label={`Adversarial hardening detection scores. ${summary}`}
      >
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line
              x1={MARGIN.left}
              x2={WIDTH - MARGIN.right}
              y1={yFor(v)}
              y2={yFor(v)}
              stroke="var(--rule)"
              strokeWidth={1}
            />
            <text x={MARGIN.left - 8} y={yFor(v) + 3} textAnchor="end" fontSize={9} fill="var(--ink-tertiary)">
              {v}
            </text>
          </g>
        ))}

        {names.map((name, gi) => {
          const groupX = MARGIN.left + gi * groupWidth + groupWidth / 2;
          return (
            <g key={name}>
              {SERIES.map((s, si) => {
                const value = strategies[name][s.key].combined_score;
                const barX = groupX - (SERIES.length * barWidth) / 2 + si * barWidth;
                const barY = yFor(value);
                return (
                  <rect
                    key={s.key}
                    x={barX}
                    y={barY}
                    width={barWidth * 0.85}
                    height={MARGIN.top + PLOT_H - barY}
                    fill={s.color}
                    rx={1}
                  >
                    <title>{`${STRATEGY_LABELS[name]} — ${s.label}: ${value.toFixed(3)}`}</title>
                  </rect>
                );
              })}
              <text x={groupX} y={HEIGHT - MARGIN.bottom + 18} textAnchor="middle" fontSize={11} fill="var(--ink-secondary)">
                {STRATEGY_LABELS[name]}
              </text>
            </g>
          );
        })}
      </svg>

      <div className="flex flex-wrap gap-x-5 gap-y-1.5 text-xs" style={{ color: "var(--ink-secondary)" }}>
        {SERIES.map((s) => (
          <span key={s.key} className="flex items-center gap-1.5">
            <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>

      <div className="overflow-x-auto">
        <table className="text-xs w-full">
          <caption className="sr-only">Detection scores by strategy and model state</caption>
          <thead>
            <tr style={{ color: "var(--ink-tertiary)" }}>
              <th className="text-left font-medium pr-4 py-1">Strategy</th>
              {SERIES.map((s) => (
                <th key={s.key} className="text-right font-medium pr-4 py-1">
                  {s.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y" style={{ borderColor: "var(--rule)" }}>
            {names.map((name) => (
              <tr key={name}>
                <td className="py-1 pr-4">{STRATEGY_LABELS[name]}</td>
                {SERIES.map((s) => (
                  <td key={s.key} className="mono-figure text-right py-1 pr-4">
                    {strategies[name][s.key].combined_score.toFixed(3)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
