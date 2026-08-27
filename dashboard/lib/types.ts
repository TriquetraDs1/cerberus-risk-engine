export type Decision = "approve" | "review" | "block";

export interface CostBasis {
  fp_cost: number;
  fn_cost: number;
  block_threshold: number;
  review_threshold: number;
}

export interface QueueTransaction {
  transaction_id: string;
  account_id: string;
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

export interface PointRiskMetrics {
  generated_at: string;
  n_train: number;
  n_test: number;
  roc_auc: number;
  pr_auc: number;
  fp_cost: number;
  fn_cost: number;
  cost_optimal_threshold: number;
  cost_at_optimal_threshold: number;
  cost_at_default_threshold: number;
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

export interface SystemHealth {
  generated_at: string;
  point_risk_model: PointRiskMetrics;
  ring_detection: RingDetectionReport | null;
  routing_preview: {
    block_threshold: number;
    review_threshold: number;
    note: string;
  };
  graph_cache_status: string;
}
