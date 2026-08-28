export type Decision = "approve" | "review" | "block";

export type Segment =
  | "grocery_essentials"
  | "electronics_highvalue"
  | "digital_subscription"
  | "travel_luxury";

export interface CostBasis {
  fp_cost: number;
  fn_cost: number;
  block_threshold: number;
  review_threshold: number;
}

export interface QueueTransaction {
  transaction_id: string;
  account_id: string;
  segment: Segment;
  timestamp: string;
  amount: number;
  risk_score: number;
  decision: Decision;
  reason_codes: string[];
  ring_id: string | null;
  actual_label: number;
  cost_basis: CostBasis;
}

export interface RingGraphNode {
  id: string;
  detected_ring_id: string | null;
  ground_truth_ring_id: string | null;
}

export interface RingGraphEdge {
  source: string;
  target: string;
  weight: number;
}

export interface RingGraph {
  nodes: RingGraphNode[];
  edges: RingGraphEdge[];
}

export interface ReliabilityBin {
  bin_center: number;
  predicted_mean: number;
  observed_rate: number;
  count: number;
}

export interface CalibrationMetrics {
  brier_before: number;
  brier_after: number;
  expected_calibration_error_before: number;
  expected_calibration_error_after: number;
  reliability_curve: ReliabilityBin[];
}

export interface PointRiskMetrics {
  generated_at: string;
  n_train: number;
  n_calib: number;
  n_test: number;
  roc_auc: number;
  pr_auc: number;
  fp_cost: number;
  fn_cost: number;
  cost_optimal_threshold: number;
  cost_at_optimal_threshold: number;
  cost_at_default_threshold: number;
  calibration: CalibrationMetrics;
}

export interface RingDetectionReport {
  n_rings: number;
  n_perfectly_recovered: number;
  mean_ring_recovery: number;
  per_ring_recovery: Record<string, number>;
  n_household_pairs: number;
  n_household_false_positives: number;
  household_false_positive_rate: number;
  n_flagged_communities: number;
  n_total_communities: number;
  generated_at: string;
}

export interface SegmentCostMatrix {
  segment: Segment;
  mean_amount: number;
  fp_cost: number;
  fn_cost: number;
}

export interface SegmentRouting {
  cost_matrix: SegmentCostMatrix;
  block_threshold: number;
  review_threshold: number;
  n_transactions: number;
  n_block: number;
  n_review: number;
  n_approve: number;
  cost_at_optimal_threshold: number;
  cost_at_global_default_threshold: number;
  cost_savings_pct: number;
}

export interface DecisionLayer {
  generated_at: string;
  global_default_threshold: number;
  overall_savings_pct_vs_global_threshold: number;
  segments: Record<Segment, SegmentRouting>;
  limitations: string[];
}

export interface SystemHealth {
  generated_at: string;
  point_risk_model: PointRiskMetrics;
  ring_detection: RingDetectionReport | null;
  decision_layer: DecisionLayer;
  graph_cache_status: string;
}

export type EvasionStrategy = "structuring" | "identity_rotation" | "slow_ramp";

export interface DetectionScoreBreakdown {
  point_risk_caught_fraction: number;
  ring_recovered_fraction: number;
  combined_score: number;
}

export interface StrategyHardeningResult {
  baseline_detection: DetectionScoreBreakdown;
  evaded_original_model: DetectionScoreBreakdown;
  evaded_hardened_model: DetectionScoreBreakdown;
  best_evasion_params: Record<string, number>;
  recall_decay_original: number;
  recall_recovered_after_hardening: number;
}

export interface AdversarialHardeningReport {
  generated_at: string;
  n_restarts: number;
  n_steps: number;
  n_adversarial_examples: number;
  strategies: Record<EvasionStrategy, StrategyHardeningResult>;
  limitations: string[];
}

export type CaseTargetType = "ring" | "transaction";
export type CaseActionType = "escalate" | "dismiss" | "mark_reviewed" | "clear";

export interface CaseActionInput {
  target_type: CaseTargetType;
  target_id: string;
  action: CaseActionType;
  analyst: string;
  note?: string;
}

export interface CaseAction extends CaseActionInput {
  id: string;
  timestamp: string;
}
