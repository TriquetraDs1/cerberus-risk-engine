export function PageHeader({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children?: React.ReactNode;
}) {
  return (
    <header
      className="flex flex-wrap items-end justify-between gap-x-6 gap-y-3 px-6 sm:px-7 pt-6 pb-5 border-b"
      style={{ borderColor: "var(--rule)", background: "var(--surface)" }}
    >
      <div className="min-w-0">
        <h1 className="text-[19px] font-bold tracking-[-0.025em] leading-none">{title}</h1>
        {/* Capped near 75ch: this is prose, and a subtitle running the full width of a
            wide monitor is a paragraph nobody finishes. */}
        <p
          className="text-[13.5px] mt-2 leading-relaxed max-w-[72ch]"
          style={{ color: "var(--ink-secondary)" }}
        >
          {subtitle}
        </p>
      </div>
      {children}
    </header>
  );
}
