import { readFile } from "fs/promises";
import path from "path";
import type { AdversarialHardeningReport, QueueTransaction, RingGraph, SystemHealth } from "./types";

/**
 * All dashboard data is read directly off disk from public/data/*.json, which is
 * populated by scripts/export_dashboard_data.py from the real trained model, real
 * SHAP attributions, and the real Day 3 Louvain output — never hand-written mocks.
 *
 * Reading with fs here (rather than fetch) works in Server Components without needing
 * a running HTTP server, and returns `null` on a missing file so pages can render an
 * honest "pipeline hasn't been run yet" empty state instead of crashing.
 */

const DATA_DIR = path.join(process.cwd(), "public", "data");

async function readJson<T>(filename: string): Promise<T | null> {
  try {
    const raw = await readFile(path.join(DATA_DIR, filename), "utf-8");
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

export function getQueue(): Promise<QueueTransaction[] | null> {
  return readJson<QueueTransaction[]>("queue.json");
}

export function getRingGraph(): Promise<RingGraph | null> {
  return readJson<RingGraph>("ring_graph.json");
}

export function getSystemHealth(): Promise<SystemHealth | null> {
  return readJson<SystemHealth>("system_health.json");
}

export function getAdversarialHardening(): Promise<AdversarialHardeningReport | null> {
  return readJson<AdversarialHardeningReport>("adversarial_hardening.json");
}
