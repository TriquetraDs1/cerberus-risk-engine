import type { ReliabilityBin } from "@/lib/types";

const SIZE = 280;
const MARGIN = 32;
const PLOT = SIZE - MARGIN * 2;

function toPx(v: number) {
  return MARGIN + v * PLOT;
}

/**
 * A reliability diagram: predicted probability (x) vs. observed fraud rate (y),
 * bucketed into deciles. A perfectly calibrated model's points sit on the diagonal.
 * Bubble size encodes bin population — a bin with 3 transactions and a bin with 3,000
 * both being drawn as identical dots would overstate the confidence of sparse bins.
 *
 * Accessible by construction, not as an afterthought: axis labels, a text summary for
 * screen readers, and a data table underneath — a chart alone is not screen-reader
 * friendly, and this metric belongs to an auditor as much as a chart-reader.
 */
export function CalibrationChart({ curve }: { curve: ReliabilityBin[] }) {
  if (curve.length === 0) {
    return <p className="text-sm" style={{ color: "var(--ink-secondary)" }}>No calibration data yet.</p>;
  }

  const maxCount = Math.max(...curve.map((b) => b.count));
  const radius = (count: number) => 3 + (count / maxCount) * 7;

  const summary = curve
    .map((b) => `at predicted ${b.predicted_mean.toFixed(2)}, observed rate was ${b.observed_rate.toFixed(2)}`)
    .join("; ");

  return (
    <div className="flex flex-col sm:flex-row gap-6 items-start">
      <figure className="shrink-0">
        <svg
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          width={SIZE}
          height={SIZE}
          role="img"
          aria-label={`Calibration reliability diagram. ${summary}`}
        >
          {/* gridlines — low contrast so they don't compete with the data */}
          {[0, 0.25, 0.5, 0.75, 1].map((v) => (
            <g key={v}>
              <line x1={toPx(v)} y1={MARGIN} x2={toPx(v)} y2={SIZE - MARGIN} stroke="var(--rule)" strokeWidth={1} />
              <line x1={MARGIN} y1={toPx(1 - v)} x2={SIZE - MARGIN} y2={toPx(1 - v)} stroke="var(--rule)" strokeWidth={1} />
              <text x={toPx(v)} y={SIZE - MARGIN + 16} textAnchor="middle" fontSize={9} fill="var(--ink-tertiary)">
                {v}
              </text>
              <text x={MARGIN - 8} y={toPx(1 - v) + 3} textAnchor="end" fontSize={9} fill="var(--ink-tertiary)">
                {v}
              </text>
            </g>
          ))}

          {/* perfect-calibration reference */}
          <line
            x1={toPx(0)}
            y1={toPx(1)}
            x2={toPx(1)}
            y2={toPx(0)}
            stroke="var(--ink-tertiary)"
            strokeWidth={1.5}
            strokeDasharray="4,3"
          />

          {/* actual reliability curve */}
          <polyline
            points={curve.map((b) => `${toPx(b.predicted_mean)},${toPx(1 - b.observed_rate)}`).join(" ")}
            fill="none"
            stroke="var(--rust)"
            strokeWidth={2}
          />
          {curve.map((b) => (
            <circle
              key={b.bin_center}
              cx={toPx(b.predicted_mean)}
              cy={toPx(1 - b.observed_rate)}
              r={radius(b.count)}
              fill="var(--rust)"
              fillOpacity={0.85}
            >
              {/* SVG <title> (and HTML <title>) must receive a single string child, not
                  interleaved text/expression nodes — React treats the tag name "title"
                  specially regardless of namespace, warns on an array of children, and
                  the resulting server/client serialization mismatch causes a hydration
                  error, not just a console warning. */}
              <title>{`predicted ${b.predicted_mean.toFixed(3)} · observed ${b.observed_rate.toFixed(3)} · n=${b.count}`}</title>
            </circle>
          ))}

          <text x={SIZE / 2} y={SIZE - 4} textAnchor="middle" fontSize={10} fill="var(--ink-secondary)">
            Predicted probability
          </text>
          <text
            x={10}
            y={SIZE / 2}
            textAnchor="middle"
            fontSize={10}
            fill="var(--ink-secondary)"
            transform={`rotate(-90, 10, ${SIZE / 2})`}
          >
            Observed rate
          </text>
        </svg>
      </figure>

      <div className="overflow-x-auto w-full">
        <table className="text-xs w-full">
          <caption className="sr-only">Calibration bins: predicted vs. observed fraud rate</caption>
          <thead>
            <tr style={{ color: "var(--ink-tertiary)" }}>
              <th className="text-left font-medium pr-4 py-1">Predicted</th>
              <th className="text-left font-medium pr-4 py-1">Observed</th>
              <th className="text-left font-medium py-1">n</th>
            </tr>
          </thead>
          <tbody className="divide-y" style={{ borderColor: "var(--rule)" }}>
            {curve.map((b) => (
              <tr key={b.bin_center}>
                <td className="mono-figure pr-4 py-1">{b.predicted_mean.toFixed(3)}</td>
                <td className="mono-figure pr-4 py-1">{b.observed_rate.toFixed(3)}</td>
                <td className="mono-figure py-1">{b.count.toLocaleString("en-US")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
