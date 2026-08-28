import { mkdir, readFile, writeFile } from "fs/promises";
import path from "path";
import type { CaseAction, CaseActionInput } from "./types";

/**
 * A minimal, file-backed case-action log — turns the Ring Network and Review Queue
 * from read-only viewers into a workflow: an analyst can escalate a detected ring or
 * mark a transaction reviewed, and that action persists and is visible to the next
 * viewer of the page.
 *
 * Deliberately not the Day 7 SQLite audit log (that's the *scoring* decision trail,
 * written by the Python service). This is the *analyst's* action trail, written by
 * the dashboard itself. The two get unified into one real audit system in a later
 * pass — noted here rather than silently pretending they're already the same thing.
 *
 * File-backed, not a database: this is a local demo app, not a deployed multi-user
 * service. Good enough to demonstrate the workflow; the production path is a real
 * table (or reusing the Day 7 audit log schema) once there's a real backend to own it.
 */

const STORE_PATH = path.join(process.cwd(), "data", "case_actions.json");

async function readStore(): Promise<CaseAction[]> {
  try {
    const raw = await readFile(STORE_PATH, "utf-8");
    return JSON.parse(raw) as CaseAction[];
  } catch {
    return [];
  }
}

async function writeStore(actions: CaseAction[]): Promise<void> {
  await mkdir(path.dirname(STORE_PATH), { recursive: true });
  await writeFile(STORE_PATH, JSON.stringify(actions, null, 2), "utf-8");
}

export async function listCaseActions(): Promise<CaseAction[]> {
  return readStore();
}

/** Latest action per target, keyed as "type:id" — what the UI actually renders. */
export async function latestActionByTarget(): Promise<Record<string, CaseAction>> {
  const actions = await readStore();
  const latest: Record<string, CaseAction> = {};
  for (const a of actions) {
    const key = `${a.target_type}:${a.target_id}`;
    if (!latest[key] || a.timestamp > latest[key].timestamp) {
      latest[key] = a;
    }
  }
  return latest;
}

export async function recordCaseAction(input: CaseActionInput): Promise<CaseAction> {
  const actions = await readStore();
  const action: CaseAction = {
    ...input,
    id: `case_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    timestamp: new Date().toISOString(),
  };
  actions.push(action);
  await writeStore(actions);
  return action;
}
