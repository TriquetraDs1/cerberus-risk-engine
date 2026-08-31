/**
 * The five-step loop the whole project is built around, drawn rather than listed.
 *
 * A bulleted list of the same five words would say the steps happen; the diagram shows
 * that step five feeds back into step one, which is the only part that distinguishes this
 * from "we trained a classifier and tested it". The return arrow is the argument.
 */

const STEPS = [
  { n: "01", label: "Build", detail: "score transactions, map rings" },
  { n: "02", label: "Attack", detail: "search for an evasion" },
  { n: "03", label: "Measure", detail: "how far detection falls" },
  { n: "04", label: "Retrain", detail: "on what the search found" },
  { n: "05", label: "Prove", detail: "re-attack, fresh search" },
];

export function AttackLoop() {
  return (
    <figure className="m-0">
      <svg
        viewBox="0 0 900 232"
        className="w-full h-auto"
        role="img"
        aria-label="A five-step loop: build, attack, measure, retrain, prove — with the fifth step feeding back into the first."
      >
        {STEPS.map((step, i) => {
          const x = 8 + i * 178;
          return (
            <g key={step.n}>
              <rect
                x={x}
                y={26}
                width={158}
                height={92}
                rx={3}
                fill="var(--surface)"
                stroke="var(--rust-rule)"
                strokeWidth={1}
              />
              <text
                x={x + 18}
                y={54}
                fill="var(--rust)"
                fontSize={11}
                fontWeight={700}
                letterSpacing="0.08em"
                fontFamily="var(--font-geist-mono), monospace"
              >
                {step.n}
              </text>
              <text x={x + 18} y={80} fill="var(--ink)" fontSize={19} fontWeight={700}>
                {step.label}
              </text>
              <text x={x + 18} y={101} fill="var(--ink-secondary)" fontSize={11.5}>
                {step.detail}
              </text>
              {i < STEPS.length - 1 && (
                <path
                  d={`M ${x + 162} 72 L ${x + 174} 72`}
                  stroke="var(--rule-strong)"
                  strokeWidth={1.5}
                  markerEnd="url(#cerb-arrow)"
                />
              )}
            </g>
          );
        })}

        {/* The feedback edge. Heavier and rust-coloured because it is the thesis: the
            output of proving becomes the input to building, continuously. */}
        <path
          d="M 858 122 L 858 176 Q 858 190 844 190 L 100 190 Q 86 190 86 176 L 86 126"
          fill="none"
          stroke="var(--rust)"
          strokeWidth={2}
          markerEnd="url(#cerb-arrow-rust)"
        />
        <text
          x={472}
          y={214}
          textAnchor="middle"
          fill="var(--rust)"
          fontSize={12}
          fontWeight={600}
        >
          every change re-enters the loop, and the gate runs in CI
        </text>

        <defs>
          <marker id="cerb-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="var(--rule-strong)" />
          </marker>
          <marker id="cerb-arrow-rust" markerWidth="8" markerHeight="8" refX="6" refY="3.5" orient="auto">
            <path d="M0,0 L7,3.5 L0,7 Z" fill="var(--rust)" />
          </marker>
        </defs>
      </svg>
    </figure>
  );
}
