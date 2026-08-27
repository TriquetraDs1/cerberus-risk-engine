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
      className="flex items-start justify-between gap-4 px-6 py-5 border-b"
      style={{ borderColor: "var(--border)" }}
    >
      <div>
        <h1 className="text-lg font-semibold tracking-tight">{title}</h1>
        <p className="text-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>
          {subtitle}
        </p>
      </div>
      {children}
    </header>
  );
}
