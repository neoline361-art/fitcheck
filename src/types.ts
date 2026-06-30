export interface DataIssue {
  column: string;
  type: "missing_values" | "duplicate_rows" | "constant_column" | "outliers" | "class_imbalance";
  severity: "critical" | "warning" | "info";
  message: string;
  suggestion: string;
}

export interface CheckSummary {
  total_rows: number;
  total_columns: number;
  issues_count: number;
  passed: boolean;
}

export interface CheckReport {
  summary: CheckSummary;
  issues: DataIssue[];
  passed: boolean;
  fix_script: string;
  headers: string[];
}

export interface DriftResult {
  feature: string;
  test_used: string;
  statistic: number;
  p_value: number;
  drifted: boolean;
  severity: "critical" | "warning" | "none";
}

export interface ModelMetrics {
  accuracy?: number;
  f1_macro?: number;
  precision_macro?: number;
  recall_macro?: number;
  mse?: number;
  rmse?: number;
  mae?: number;
  r2_score?: number;
}

export interface ModelReport {
  task_type: "classification" | "regression";
  metrics: ModelMetrics;
  confusion_matrix?: {
    classes: string[];
    matrix: number[][];
  };
  roc_curve?: { fpr: number; tpr: number }[];
  residuals?: { predicted: number; actual: number; residual: number }[];
  importances: Record<string, number>;
}
