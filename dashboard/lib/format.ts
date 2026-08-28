import type { Segment } from "./types";

/**
 * Reason codes look like "high_transaction_velocity" or, for a ring hit,
 * "shared_device_with_flagged_ring:detected_8" — only the label before the colon is
 * underscore_case; the ring id after it is a real identifier and must render verbatim,
 * or "detected_8" reads as the nonsensical "detected 8".
 */
export function formatReasonCode(code: string): string {
  const [label, ...rest] = code.split(":");
  const prettyLabel = label.replaceAll("_", " ");
  return rest.length ? `${prettyLabel}: ${rest.join(":")}` : prettyLabel;
}

const SEGMENT_LABELS: Record<Segment, string> = {
  grocery_essentials: "Grocery",
  electronics_highvalue: "Electronics",
  digital_subscription: "Digital",
  travel_luxury: "Travel",
};

/** Short display label for the cockpit-density table — the full segment name lives in
 * the transaction drawer and System Health, where there's room for it. */
export function formatSegment(segment: Segment): string {
  return SEGMENT_LABELS[segment] ?? segment;
}
