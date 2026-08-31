"use client";

import { BookOpen, ChartLine, Graph, Queue as QueueIcon, Sword } from "@phosphor-icons/react/dist/ssr";
import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/about", label: "What this is", Icon: BookOpen },
  { href: "/adversarial", label: "Adversarial Hardening", Icon: Sword },
  { href: "/", label: "Review Queue", Icon: QueueIcon },
  { href: "/rings", label: "Ring Network", Icon: Graph },
  { href: "/health", label: "System Health", Icon: ChartLine },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <nav
      aria-label="Primary"
      className="flex md:w-[15rem] w-full md:h-full md:flex-col shrink-0 border-b md:border-b-0 md:border-r"
      style={{ borderColor: "var(--rule)", background: "var(--surface)" }}
    >
      <Link
        href="/about"
        className="flex items-center gap-2.5 px-5 py-4 md:border-b shrink-0"
        style={{ borderColor: "var(--rule)" }}
      >
        {/* Three marks for the three detection layers the project is named after. */}
        <span className="flex gap-[3px] items-end" aria-hidden>
          <span className="w-[3px] h-[11px] rounded-[1px]" style={{ background: "var(--rust)" }} />
          <span className="w-[3px] h-[15px] rounded-[1px]" style={{ background: "var(--rust)" }} />
          <span className="w-[3px] h-[8px] rounded-[1px]" style={{ background: "var(--rust)", opacity: 0.45 }} />
        </span>
        <span className="font-bold tracking-[-0.02em] text-[15px]">Cerberus</span>
      </Link>

      <ul className="flex md:flex-col md:py-2.5 overflow-x-auto md:overflow-visible">
        {NAV.map(({ href, label, Icon }) => {
          const active = pathname === href;
          return (
            <li key={href} className="shrink-0 md:px-2">
              <Link
                href={href}
                aria-current={active ? "page" : undefined}
                className="flex items-center gap-2.5 px-3 py-2 md:rounded-[3px] text-[13.5px] transition-colors duration-150"
                style={{
                  color: active ? "var(--ink)" : "var(--ink-secondary)",
                  background: active ? "var(--rust-soft)" : "transparent",
                  fontWeight: active ? 600 : 450,
                }}
              >
                <Icon
                  size={15}
                  weight={active ? "fill" : "regular"}
                  style={{ color: active ? "var(--rust)" : "var(--ink-tertiary)" }}
                  aria-hidden
                />
                {label}
              </Link>
            </li>
          );
        })}
      </ul>

      <p
        className="hidden md:block mt-auto px-5 py-4 text-[11px] leading-[1.55] border-t"
        style={{ color: "var(--ink-tertiary)", borderColor: "var(--rule)" }}
      >
        Defensive research prototype. The harness attacks only its own sandboxed model.
      </p>
    </nav>
  );
}
