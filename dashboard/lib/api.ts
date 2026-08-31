/**
 * Client for the live Cerberus API (the FastAPI service, not this app's own routes).
 *
 * The four analytical pages read static pipeline output and need no backend. These calls
 * are different: dispute drafting and the case copilot are generative and only exist on
 * the running service. So every surface that uses them has to handle three states
 * honestly rather than assuming a healthy API:
 *
 *   unconfigured — NEXT_PUBLIC_API_BASE_URL isn't set. Not an error; the dashboard is
 *                  deliberately deployable on its own.
 *   cold         — the free tier sleeps after 15 minutes and takes ~50s to wake. The UI
 *                  must say that rather than looking hung.
 *   unreachable  — down, blocked by CORS, or refused.
 */

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "";

export const apiConfigured = API_BASE_URL.length > 0;

// Longer than a normal fetch timeout on purpose: a sleeping free-tier instance genuinely
// needs this long, and aborting at 10s would report "unreachable" for a service that is
// merely waking up.
const COLD_START_TIMEOUT_MS = 75_000;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly kind: "unconfigured" | "timeout" | "http" | "network",
    readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  if (!apiConfigured) {
    throw new ApiError(
      "The live API isn't configured for this deployment. Set NEXT_PUBLIC_API_BASE_URL to enable it.",
      "unconfigured",
    );
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), COLD_START_TIMEOUT_MS);

  try {
    const res = await fetch(`${API_BASE_URL}${path}`, {
      method: "POST",
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });

    if (!res.ok) {
      const detail =
        res.status === 404
          ? "The API couldn't find or reconstruct that decision."
          : `The API returned ${res.status}.`;
      throw new ApiError(detail, "http", res.status);
    }
    return (await res.json()) as T;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError(
        "The API didn't respond in 75 seconds. A free-tier instance sleeps after 15 minutes idle; try once more to wake it.",
        "timeout",
      );
    }
    throw new ApiError(
      "Couldn't reach the API. It may be asleep, down, or not allowing requests from this origin.",
      "network",
    );
  } finally {
    clearTimeout(timer);
  }
}

export interface DisputeResponse {
  transaction_id: string;
  draft: string;
  reason_codes: string[];
  source: "llm" | "template";
  // "audit_log" when the API scored this transaction itself, "supplied" when the
  // dashboard passed the decision. Queue rows are produced offline by the export
  // script, so they are the second case, and the UI says so.
  facts_from: "audit_log" | "supplied";
}

export interface DisputeFacts {
  decision: string;
  risk_score: number;
  segment: string;
  amount: number;
  account_id: string;
  reason_codes: string[];
  ring_id: string | null;
  timestamp?: string;
}

export interface CopilotResponse {
  ring_id: string;
  answer: string;
  n_members: number;
  source: "llm" | "template";
}

export const draftDispute = (transactionId: string, facts?: DisputeFacts) =>
  post<DisputeResponse>(`/dispute/${encodeURIComponent(transactionId)}`, facts);

export const askCopilot = (ringId: string, messages: { role: "user" | "assistant"; content: string }[]) =>
  post<CopilotResponse>(`/copilot/${encodeURIComponent(ringId)}`, { messages });
