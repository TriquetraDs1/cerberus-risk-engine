"use client";

import { forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation } from "d3-force";
import { useEffect, useMemo, useState } from "react";
import type { RingGraph as RingGraphData } from "@/lib/types";

interface SimNode {
  id: string;
  detected_ring_id: string | null;
  ground_truth_ring_id: string | null;
  x: number;
  y: number;
}

interface SimLink {
  source: string;
  target: string;
  weight: number;
}

// After forceLink().id(...) resolves and simulation.tick() runs, d3-force mutates
// link.source/target in place from string ids into references to the actual node
// objects (with settled x/y) — so post-simulation, a link's source/target IS a node.
interface ResolvedSimLink {
  source: SimNode;
  target: SimNode;
  weight: number;
}

const PALETTE = [
  "#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed",
  "#0891b2", "#c026d3", "#65a30d", "#e11d48", "#0d9488",
  "#9333ea", "#ca8a04", "#4f46e5", "#16a34a", "#db2777",
];

const WIDTH = 900;
const HEIGHT = 600;

interface Layout {
  nodes: SimNode[];
  links: SimLink[];
}

export function RingGraph({ graph }: { graph: RingGraphData }) {
  const [selected, setSelected] = useState<string | null>(null);

  // d3-force's tie-breaking jitter for coincident starting positions is not
  // guaranteed to match between Node's SSR pass and the browser's hydration pass —
  // observed as a whole-tree "server rendered HTML didn't match" hydration error on
  // this SVG. This is a client-only visualization anyway (no SEO value in physics-sim
  // coordinates), so it renders a stable skeleton until mounted, then computes the
  // real layout — the standard fix for a randomized/browser-only render tree.
  // Documented "client-only mount" exception (react.dev/learn/you-might-not-need-an-effect):
  // the effect body has nothing to synchronize with, it only detects that hydration
  // has completed.
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMounted(true);
  }, []);

  const colorByRing = useMemo(() => {
    const rings = Array.from(new Set(graph.nodes.map((n) => n.detected_ring_id).filter(Boolean))) as string[];
    const map = new Map<string, string>();
    rings.forEach((ring, i) => map.set(ring, PALETTE[i % PALETTE.length]));
    return map;
  }, [graph]);

  // Settling a force layout over plain data is a pure, synchronous computation — no
  // DOM or browser API involved — so it belongs in useMemo, not an effect. This is a
  // static triage read (taste-skill MOTION_INTENSITY=4: settle once, don't animate),
  // so 300 ticks up front replaces a live physics loop entirely.
  const layout = useMemo<Layout>(() => {
    const nodes: SimNode[] = graph.nodes.map((n) => ({ ...n, x: WIDTH / 2, y: HEIGHT / 2 }));
    const links: SimLink[] = graph.edges.map((e) => ({ ...e }));

    const simulation = forceSimulation(nodes as unknown as { x: number; y: number }[])
      .force(
        "link",
        forceLink(links as unknown as { source: string; target: string }[])
          .id((d: unknown) => (d as SimNode).id)
          .distance((l: unknown) => 40 / Math.max(1, (l as SimLink).weight))
          .strength(0.5),
      )
      .force("charge", forceManyBody().strength(-45))
      .force("center", forceCenter(WIDTH / 2, HEIGHT / 2))
      .force("collide", forceCollide(7))
      .stop();

    for (let i = 0; i < 300; i++) simulation.tick();

    return { nodes, links };
  }, [graph]);

  if (!mounted) {
    return <div className="skeleton w-full h-[600px] rounded-lg" aria-label="Loading ring graph" />;
  }

  const positions = layout.nodes;

  const posById = new Map(positions.map((n) => [n.id, n]));
  const selectedNode = selected ? posById.get(selected) : null;

  return (
    <div className="flex flex-col lg:flex-row gap-4">
      <div className="flex-1 min-w-0 rounded-lg border overflow-hidden" style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full h-auto" role="img" aria-label="Fraud ring entity-link network">
          <g opacity={0.35}>
            {(layout.links as unknown as ResolvedSimLink[]).map((l, i) => {
              const s = l.source;
              const t = l.target;
              if (!s || typeof s.x !== "number" || !t || typeof t.x !== "number") return null;
              return (
                <line
                  key={i}
                  x1={s.x}
                  y1={s.y}
                  x2={t.x}
                  y2={t.y}
                  stroke="var(--border-strong)"
                  strokeWidth={Math.min(l.weight, 2)}
                />
              );
            })}
          </g>
          <g>
            {positions.map((n) => {
              const isFalsePositive = n.detected_ring_id !== null && n.ground_truth_ring_id === null;
              const color = n.detected_ring_id ? colorByRing.get(n.detected_ring_id) : "var(--text-tertiary)";
              const isSelected = n.id === selected;
              return (
                <circle
                  key={n.id}
                  cx={n.x}
                  cy={n.y}
                  r={isSelected ? 7 : 5}
                  fill={color}
                  stroke={isFalsePositive ? "var(--risk-block)" : isSelected ? "var(--text-primary)" : "none"}
                  strokeWidth={isFalsePositive ? 2 : 1.5}
                  strokeDasharray={isFalsePositive ? "2,1.5" : undefined}
                  className="cursor-pointer"
                  onClick={() => setSelected(n.id)}
                  tabIndex={0}
                  role="button"
                  aria-label={`Account ${n.id}${n.detected_ring_id ? `, in ring ${n.detected_ring_id}` : ""}`}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") setSelected(n.id);
                  }}
                >
                  {/* single string child, not interleaved text/expression nodes — see
                      the note in CalibrationChart.tsx on why this is a hydration bug,
                      not just a console warning, for the tag name "title". */}
                  <title>{`${n.id} — detected: ${n.detected_ring_id ?? "none"} · ground truth: ${n.ground_truth_ring_id ?? "none (household or clean)"}`}</title>
                </circle>
              );
            })}
          </g>
        </svg>
      </div>

      <aside className="w-full lg:w-64 shrink-0 flex flex-col gap-4">
        <div>
          <h3 className="text-xs font-medium uppercase tracking-wide mb-2" style={{ color: "var(--text-tertiary)" }}>
            Legend
          </h3>
          <ul className="flex flex-col gap-2 text-xs" style={{ color: "var(--text-secondary)" }}>
            <li className="flex items-center gap-2">
              <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: "var(--accent)" }} />
              Node color = detected community
            </li>
            <li className="flex items-center gap-2">
              <span
                className="inline-block w-2.5 h-2.5 rounded-full border-2"
                style={{ borderColor: "var(--risk-block)", borderStyle: "dashed" }}
              />
              Flagged, but not an injected ring member (honest false positive — e.g. household device sharing)
            </li>
          </ul>
        </div>

        <div className="rounded-lg border p-3" style={{ borderColor: "var(--border)" }}>
          <h3 className="text-xs font-medium uppercase tracking-wide mb-1" style={{ color: "var(--text-tertiary)" }}>
            {selectedNode ? "Selected account" : "Select a node"}
          </h3>
          {selectedNode ? (
            <dl className="text-xs flex flex-col gap-1">
              <div className="flex justify-between">
                <dt style={{ color: "var(--text-tertiary)" }}>Account</dt>
                <dd className="mono-figure">{selectedNode.id}</dd>
              </div>
              <div className="flex justify-between">
                <dt style={{ color: "var(--text-tertiary)" }}>Detected ring</dt>
                <dd className="mono-figure">{selectedNode.detected_ring_id ?? "—"}</dd>
              </div>
              <div className="flex justify-between">
                <dt style={{ color: "var(--text-tertiary)" }}>Ground truth ring</dt>
                <dd className="mono-figure">{selectedNode.ground_truth_ring_id ?? "—"}</dd>
              </div>
            </dl>
          ) : (
            <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
              Click any node to inspect it.
            </p>
          )}
        </div>
      </aside>
    </div>
  );
}
