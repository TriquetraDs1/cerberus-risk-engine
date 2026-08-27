import { ChartLine, Graph, Queue as QueueIcon, ShieldWarning } from "@phosphor-icons/react/dist/ssr";
import Link from "next/link";

const NAV = [
  { href: "/", label: "Review Queue", Icon: QueueIcon },
  { href: "/rings", label: "Ring Network", Icon: Graph },
  { href: "/health", label: "System Health", Icon: ChartLine },
];

export function Sidebar() {
  return (
    <nav
      aria-label="Primary"
      className="flex md:w-56 w-full md:h-full md:flex-col shrink-0 border-b md:border-b-0 md:border-r"
      style={{ borderColor: "var(--border)", background: "var(--surface)" }}
    >
      <div className="flex items-center gap-2 px-4 py-4 md:border-b" style={{ borderColor: "var(--border)" }}>
        <ShieldWarning size={20} weight="fill" style={{ color: "var(--accent)" }} aria-hidden />
        <span className="font-semibold tracking-tight text-[15px]">Cerberus</span>
      </div>
      <ul className="flex md:flex-col md:py-2 overflow-x-auto md:overflow-visible">
        {NAV.map(({ href, label, Icon }) => (
          <li key={href} className="shrink-0">
            <Link
              href={href}
              className="flex items-center gap-2.5 px-4 py-2.5 text-sm hover:bg-[var(--accent-soft)] transition-colors"
              style={{ color: "var(--text-secondary)" }}
            >
              <Icon size={16} aria-hidden />
              {label}
            </Link>
          </li>
        ))}
      </ul>
      <div
        className="hidden md:block mt-auto px-4 py-3 text-[11px] leading-snug border-t"
        style={{ color: "var(--text-tertiary)", borderColor: "var(--border)" }}
      >
        Defensive research prototype. Adversarial harness attacks only its own
        sandboxed model — no offense-capable code ships here.
      </div>
    </nav>
  );
}
