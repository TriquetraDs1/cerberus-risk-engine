import type { Metadata } from "next";
import { Archivo, Geist_Mono } from "next/font/google";
import { Sidebar } from "@/components/Sidebar";
import "./globals.css";

// Archivo: a grotesque drawn for signage and technical printing, with a weight range wide
// enough to carry both a 12px table label and a display headline in one family. Chosen
// over the reflex picks (Inter, Plex, Space Grotesk) because those are the fonts every
// dashboard already uses — and because Archivo's slightly condensed, institutional cut
// suits a document that reports findings.
const archivo = Archivo({
  variable: "--font-archivo",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
});

// Kept for figures only. Mono here is not costume: every score, threshold and identifier
// is data that needs tabular alignment.
const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Cerberus — Risk Ops",
  description:
    "A transaction-risk engine that attacks its own detector, measures the damage, and reports what it could not fix.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${archivo.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col md:flex-row" style={{ background: "var(--paper)" }}>
        <Sidebar />
        <main className="flex-1 min-w-0 md:h-screen md:overflow-y-auto">{children}</main>
      </body>
    </html>
  );
}
