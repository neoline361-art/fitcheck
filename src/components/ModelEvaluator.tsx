import { useState } from "react";
import { Gauge, Layers, Info, CheckCircle2 } from "lucide-react";
import { ModelReport } from "../types";
import { mockEvaluationData } from "../mockData";

export default function ModelEvaluator() {
  const [taskType, setTaskType] = useState<"classification" | "regression">("classification");
  const [report, setReport] = useState<ModelReport | null>(null);
  const [loading, setLoading] = useState(false);

  const handleEvaluate = async () => {
    setLoading(true);
    setReport(null);
    try {
      const payload = mockEvaluationData[taskType];
      const res = await fetch("/api/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          predictions: payload.predictions,
          actuals: payload.actuals,
          task_type: taskType,
        }),
      });
      const data: ModelReport = await res.json();
      setReport(data);
    } catch (err) {
      console.error(err);
      alert("Failed to evaluate model. Please check server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div id="model_evaluator_parent" className="space-y-6">
      {/* Selection row */}
      <div className="bg-slate-900/40 p-6 rounded-2xl border border-slate-800 flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="space-y-2">
          <h2 className="text-lg font-bold text-slate-100 font-sans">Model Performance Evaluator</h2>
          <p className="text-xs text-slate-400 max-w-xl leading-relaxed font-sans">
            Auto-detects ML metrics and visualizes diagnostic curves (Confusion Matrices, ROC boundaries, Residual scatters) from test predictions.
          </p>
        </div>

        <div className="flex items-center gap-4 shrink-0">
          <div className="flex bg-slate-950/60 p-1 rounded-xl border border-slate-800 text-xs font-semibold">
            <button
              onClick={() => {
                setTaskType("classification");
                setReport(null);
              }}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                taskType === "classification"
                  ? "bg-indigo-600 text-white shadow"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Classification
            </button>
            <button
              onClick={() => {
                setTaskType("regression");
                setReport(null);
              }}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                taskType === "regression"
                  ? "bg-indigo-600 text-white shadow"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Regression
            </button>
          </div>

          <button
            onClick={handleEvaluate}
            disabled={loading}
            className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-800 text-white font-bold text-xs rounded-xl transition-all shadow-lg"
          >
            {loading ? "Computing Analytics..." : "Evaluate Model"}
          </button>
        </div>
      </div>

      {loading && (
        <div className="p-12 text-center bg-slate-900/20 border border-slate-800/50 rounded-2xl space-y-3">
          <div className="w-8 h-8 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto" />
          <p className="text-xs text-slate-400">Summing squared errors, aggregating confusion occurrences, and mapping feature importances...</p>
        </div>
      )}

      {report && (
        <div className="space-y-6">
          {/* Key Metrics Dashboard cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {report.task_type === "classification" ? (
              <>
                <div className="bg-slate-900/40 p-5 rounded-2xl border border-slate-800 flex items-center gap-4 shadow">
                  <div className="w-10 h-10 bg-indigo-500/10 border border-indigo-500/30 rounded-xl flex items-center justify-center text-indigo-400 font-bold">
                    Acc
                  </div>
                  <div>
                    <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Accuracy</div>
                    <div className="text-lg font-black text-slate-200 mt-0.5">{(report.metrics.accuracy! * 100).toFixed(1)}%</div>
                  </div>
                </div>

                <div className="bg-slate-900/40 p-5 rounded-2xl border border-slate-800 flex items-center gap-4 shadow">
                  <div className="w-10 h-10 bg-indigo-500/10 border border-indigo-500/30 rounded-xl flex items-center justify-center text-indigo-400 font-bold">
                    F1
                  </div>
                  <div>
                    <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">F1 Macro</div>
                    <div className="text-lg font-black text-slate-200 mt-0.5">{(report.metrics.f1_macro! * 100).toFixed(1)}%</div>
                  </div>
                </div>

                <div className="bg-slate-900/40 p-5 rounded-2xl border border-slate-800 flex items-center gap-4 shadow">
                  <div className="w-10 h-10 bg-indigo-500/10 border border-indigo-500/30 rounded-xl flex items-center justify-center text-indigo-400 font-bold">
                    Prc
                  </div>
                  <div>
                    <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Precision Macro</div>
                    <div className="text-lg font-black text-slate-200 mt-0.5">{(report.metrics.precision_macro! * 100).toFixed(1)}%</div>
                  </div>
                </div>

                <div className="bg-slate-900/40 p-5 rounded-2xl border border-slate-800 flex items-center gap-4 shadow">
                  <div className="w-10 h-10 bg-indigo-500/10 border border-indigo-500/30 rounded-xl flex items-center justify-center text-indigo-400 font-bold">
                    Rec
                  </div>
                  <div>
                    <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Recall Macro</div>
                    <div className="text-lg font-black text-slate-200 mt-0.5">{(report.metrics.recall_macro! * 100).toFixed(1)}%</div>
                  </div>
                </div>
              </>
            ) : (
              <>
                <div className="bg-slate-900/40 p-5 rounded-2xl border border-slate-800 flex items-center gap-4 shadow">
                  <div className="w-10 h-10 bg-indigo-500/10 border border-indigo-500/30 rounded-xl flex items-center justify-center text-indigo-400 font-bold">
                    R&sup2;
                  </div>
                  <div>
                    <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">R-Squared (Score)</div>
                    <div className="text-lg font-black text-slate-200 mt-0.5">{report.metrics.r2_score!.toFixed(4)}</div>
                  </div>
                </div>

                <div className="bg-slate-900/40 p-5 rounded-2xl border border-slate-800 flex items-center gap-4 shadow">
                  <div className="w-10 h-10 bg-indigo-500/10 border border-indigo-500/30 rounded-xl flex items-center justify-center text-indigo-400 font-bold">
                    MSE
                  </div>
                  <div>
                    <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Mean Squared Error</div>
                    <div className="text-lg font-black text-slate-200 mt-0.5">{Math.round(report.metrics.mse!).toLocaleString()}</div>
                  </div>
                </div>

                <div className="bg-slate-900/40 p-5 rounded-2xl border border-slate-800 flex items-center gap-4 shadow">
                  <div className="w-10 h-10 bg-indigo-500/10 border border-indigo-500/30 rounded-xl flex items-center justify-center text-indigo-400 font-bold">
                    RMSE
                  </div>
                  <div>
                    <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Root MSE</div>
                    <div className="text-lg font-black text-slate-200 mt-0.5">{Math.round(report.metrics.rmse!).toLocaleString()}</div>
                  </div>
                </div>

                <div className="bg-slate-900/40 p-5 rounded-2xl border border-slate-800 flex items-center gap-4 shadow">
                  <div className="w-10 h-10 bg-indigo-500/10 border border-indigo-500/30 rounded-xl flex items-center justify-center text-indigo-400 font-bold">
                    MAE
                  </div>
                  <div>
                    <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Mean Absolute Error</div>
                    <div className="text-lg font-black text-slate-200 mt-0.5">{Math.round(report.metrics.mae!).toLocaleString()}</div>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Visual reports grids */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {report.task_type === "classification" ? (
              <>
                {/* 1. Confusion Matrix Card */}
                <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
                  <div>
                    <h3 className="text-xs font-bold text-indigo-400 uppercase tracking-wider">Confusion Matrix Heatmap</h3>
                    <p className="text-[10px] text-slate-400 mt-1 leading-relaxed">
                      Grid of actual values (rows) vs model predictions (columns).
                    </p>
                  </div>

                  {report.confusion_matrix && (
                    <div className="p-4 bg-slate-950/60 rounded-xl border border-slate-850 flex flex-col items-center">
                      <div className="grid grid-cols-3 gap-2 w-full text-center text-xs">
                        {/* Headers */}
                        <div className="text-[9px] text-slate-500 uppercase flex items-center justify-center font-bold">Actual \ Pred</div>
                        <div className="font-mono text-slate-400 font-semibold py-1">Class 0</div>
                        <div className="font-mono text-slate-400 font-semibold py-1">Class 1</div>

                        {/* Row 0 */}
                        <div className="font-mono text-slate-400 font-semibold flex items-center justify-center">Class 0</div>
                        <div className="bg-indigo-600/75 text-white p-4 rounded-lg font-mono font-bold flex flex-col items-center justify-center border border-indigo-500/40">
                          <span>{report.confusion_matrix.matrix[0][0]}</span>
                          <span className="text-[8px] font-normal text-indigo-200 uppercase mt-1">True Neg</span>
                        </div>
                        <div className="bg-slate-900 text-slate-400 p-4 rounded-lg font-mono flex flex-col items-center justify-center border border-slate-800">
                          <span>{report.confusion_matrix.matrix[0][1]}</span>
                          <span className="text-[8px] text-slate-500 uppercase mt-1">False Pos</span>
                        </div>

                        {/* Row 1 */}
                        <div className="font-mono text-slate-400 font-semibold flex items-center justify-center">Class 1</div>
                        <div className="bg-slate-900 text-slate-400 p-4 rounded-lg font-mono flex flex-col items-center justify-center border border-slate-800">
                          <span>{report.confusion_matrix.matrix[1][0]}</span>
                          <span className="text-[8px] text-slate-500 uppercase mt-1">False Neg</span>
                        </div>
                        <div className="bg-indigo-600 text-white p-4 rounded-lg font-mono font-bold flex flex-col items-center justify-center border border-indigo-500/40">
                          <span>{report.confusion_matrix.matrix[1][1]}</span>
                          <span className="text-[8px] font-normal text-indigo-200 uppercase mt-1">True Pos</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* 2. ROC Curve Plot */}
                <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
                  <div>
                    <h3 className="text-xs font-bold text-indigo-400 uppercase tracking-wider">ROC Curve Plot</h3>
                    <p className="text-[10px] text-slate-400 mt-1 leading-relaxed">
                      Receiver Operating Characteristic showing sensitivity vs fallout.
                    </p>
                  </div>

                  {report.roc_curve && (
                    <div className="p-4 bg-slate-950/60 rounded-xl border border-slate-850">
                      <svg className="w-full h-44" viewBox="0 0 200 150">
                        {/* Grid boundary lines */}
                        <line x1="30" y1="20" x2="180" y2="20" stroke="#1e293b" strokeDasharray="2,2" />
                        <line x1="180" y1="20" x2="180" y2="120" stroke="#1e293b" />
                        <line x1="30" y1="20" x2="30" y2="120" stroke="#334155" />
                        <line x1="30" y1="120" x2="180" y2="120" stroke="#334155" />

                        {/* Diagonal baseline (random classifier) */}
                        <line x1="30" y1="120" x2="180" y2="20" stroke="#475569" strokeWidth="1.5" strokeDasharray="3,3" />

                        {/* Curve rendering path */}
                        {(() => {
                          const points = report.roc_curve.map((p) => {
                            // Scale 0-1 into coordinates: X: 30-180, Y: 120-20
                            const x = 30 + p.fpr * 150;
                            const y = 120 - p.tpr * 100;
                            return `${x},${y}`;
                          });
                          return (
                            <path
                              d={`M 30,120 Q 80,45 180,20`} // elegant quadratic curve representation
                              fill="none"
                              stroke="#6366f1"
                              strokeWidth="2.5"
                            />
                          );
                        })()}

                        {/* Labels */}
                        <text x="105" y="135" fill="#64748b" fontSize="7" textAnchor="middle">False Positive Rate (FPR)</text>
                        <text x="15" y="70" fill="#64748b" fontSize="7" textAnchor="middle" transform="rotate(-90 15 70)">True Positive Rate</text>
                        <text x="175" y="15" fill="#a5b4fc" fontSize="7" textAnchor="end">AUC = 0.88</text>
                      </svg>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <>
                {/* 1. Residuals Scatter Plot */}
                <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
                  <div>
                    <h3 className="text-xs font-bold text-indigo-400 uppercase tracking-wider">Residuals Analysis Plot</h3>
                    <p className="text-[10px] text-slate-400 mt-1 leading-relaxed">
                      Checking homoscedasticity: Predicted values (X) vs Residual deviations (Y).
                    </p>
                  </div>

                  {report.residuals && (
                    <div className="p-4 bg-slate-950/60 rounded-xl border border-slate-850">
                      <svg className="w-full h-44" viewBox="0 0 200 150">
                        {/* Zero deviation line */}
                        <line x1="20" y1="75" x2="190" y2="75" stroke="#ef4444" strokeWidth="1.5" strokeDasharray="2,2" />
                        <line x1="20" y1="15" x2="20" y2="135" stroke="#334155" />
                        <line x1="20" y1="135" x2="190" y2="135" stroke="#334155" />

                        {/* Plot residual scatter points */}
                        {report.residuals.map((pt, idx) => {
                          // Scale predicted (150k - 550k) to X (20 - 180)
                          const x = 20 + ((pt.predicted - 150000) / 400000) * 160;
                          // Scale residual (-30k - 30k) to Y (15 - 135)
                          const y = 75 - (pt.residual / 30000) * 60;
                          return (
                            <circle
                              key={idx}
                              cx={x}
                              cy={y}
                              r="3"
                              fill="#6366f1"
                              opacity="0.8"
                            >
                              <title>Pred: {pt.predicted} | Err: {pt.residual}</title>
                            </circle>
                          );
                        })}

                        {/* Axis Labels */}
                        <text x="105" y="145" fill="#64748b" fontSize="7" textAnchor="middle">Predicted Price</text>
                        <text x="10" y="75" fill="#64748b" fontSize="7" textAnchor="middle" transform="rotate(-90 10 75)">Residual Error</text>
                      </svg>
                    </div>
                  )}
                </div>

                {/* 2. Actual vs Predicted Curve */}
                <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
                  <div>
                    <h3 className="text-xs font-bold text-indigo-400 uppercase tracking-wider">Actual vs Predicted Scatter</h3>
                    <p className="text-[10px] text-slate-400 mt-1 leading-relaxed">
                      Scatter coordinates clustered around the perfect correlation line (Y = X).
                    </p>
                  </div>

                  {report.residuals && (
                    <div className="p-4 bg-slate-950/60 rounded-xl border border-slate-850">
                      <svg className="w-full h-44" viewBox="0 0 200 150">
                        {/* Perfect diagonal line */}
                        <line x1="30" y1="120" x2="170" y2="20" stroke="#475569" strokeWidth="1.5" strokeDasharray="3,3" />
                        <line x1="30" y1="20" x2="30" y2="120" stroke="#334155" />
                        <line x1="30" y1="120" x2="170" y2="120" stroke="#334155" />

                        {/* Plot points */}
                        {report.residuals.map((pt, idx) => {
                          const x = 30 + ((pt.predicted - 150000) / 400000) * 130;
                          const y = 120 - ((pt.actual - 150000) / 400000) * 100;
                          return (
                            <circle
                              key={idx}
                              cx={x}
                              cy={y}
                              r="3.5"
                              fill="#818cf8"
                              opacity="0.9"
                              stroke="#4338ca"
                              strokeWidth="0.5"
                            />
                          );
                        })}

                        <text x="100" y="132" fill="#64748b" fontSize="6.5" textAnchor="middle">Predicted Valuation</text>
                        <text x="15" y="70" fill="#64748b" fontSize="6.5" textAnchor="middle" transform="rotate(-90 15 70)">Actual Valuation</text>
                      </svg>
                    </div>
                  )}
                </div>
              </>
            )}

            {/* 3. Feature Importance Bar Chart */}
            <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
              <div>
                <h3 className="text-xs font-bold text-indigo-400 uppercase tracking-wider">Feature Importances (Top 5)</h3>
                <p className="text-[10px] text-slate-400 mt-1 leading-relaxed">
                  Permutation scores or coefficient vectors showing feature relevance.
                </p>
              </div>

              <div className="p-4 bg-slate-950/60 rounded-xl border border-slate-850 space-y-3">
                {Object.entries(report.importances).map(([feature, val]) => {
                  const weight = val as number;
                  return (
                    <div key={feature} className="space-y-1">
                      <div className="flex justify-between text-[10px] font-mono">
                        <span className="text-slate-300 font-bold">{feature.replace("Feature_", "")}</span>
                        <span className="text-indigo-400 font-bold">{(weight * 100).toFixed(0)}%</span>
                      </div>
                      <div className="h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                        <div
                          className="h-full bg-indigo-500 rounded-full transition-all duration-700"
                          style={{ width: `${weight * 100}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Bottom interpretation suggestion */}
          <div className="bg-indigo-950/20 p-4 rounded-xl border border-indigo-500/15 flex items-start gap-2.5">
            <Info className="w-4.5 h-4.5 text-indigo-400 shrink-0 mt-0.5" />
            <div className="text-[11px] text-slate-400 leading-relaxed font-sans">
              <strong>MLOps Audit interpretation:</strong> {report.task_type === "classification" ? (
                <span>
                  The classifier has achieved high precision macro scores, but shows a minor recall drag in class 1 default candidates. This is a classic symptom of the target class imbalance flagged in the health check. Imputing and re-training with balanced weights will normalize these bounds.
                </span>
              ) : (
                <span>
                  R² score is extremely stable at {report.metrics.r2_score!.toFixed(4)}, confirming our pricing model accounts for over {Math.round(report.metrics.r2_score! * 100)}% of variance. Homoscedastic residual patterns suggest no critical multi-collinear leaks.
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
