import { NextResponse } from "next/server";
import { listCaseActions, recordCaseAction } from "@/lib/caseActions";
import type { CaseActionInput, CaseActionType, CaseTargetType } from "@/lib/types";

const VALID_TARGET_TYPES: CaseTargetType[] = ["ring", "transaction"];
const VALID_ACTIONS: CaseActionType[] = ["escalate", "dismiss", "mark_reviewed", "clear"];

export async function GET() {
  const actions = await listCaseActions();
  return NextResponse.json(actions);
}

export async function POST(request: Request) {
  let body: Partial<CaseActionInput>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  if (!body.target_type || !VALID_TARGET_TYPES.includes(body.target_type)) {
    return NextResponse.json({ error: `target_type must be one of ${VALID_TARGET_TYPES.join(", ")}` }, { status: 422 });
  }
  if (!body.target_id || typeof body.target_id !== "string") {
    return NextResponse.json({ error: "target_id is required" }, { status: 422 });
  }
  if (!body.action || !VALID_ACTIONS.includes(body.action)) {
    return NextResponse.json({ error: `action must be one of ${VALID_ACTIONS.join(", ")}` }, { status: 422 });
  }

  const action = await recordCaseAction({
    target_type: body.target_type,
    target_id: body.target_id,
    action: body.action,
    analyst: body.analyst?.trim() || "demo-analyst",
    note: body.note?.trim() || undefined,
  });

  return NextResponse.json(action, { status: 201 });
}
