import { useState } from "react";
import { Info, AlertTriangle, ShieldCheck, BarChart3 } from "lucide-react";
import { DriftResult } from "../types";
import { mockDriftData } from "../mockData";

export default function FeatureDrift() {
  const [driftResults, setDriftResults] = useState<DriftResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedFeature, setSelectedFeature] = useState<string>("income");

  const handleDriftAnalysis = async () => {
    setLoading(true);
    setDriftResults(null);
    try {
      const res = await fetch("/api/drift", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          reference: mockDriftData.reference,
          production: mockDriftData.production,
          threshold: 0.05,
        }),
      });
      const data = await res.json();
      setDriftResults(data.results);
      if (data.results.length > 0) {
        setSelectedFeature(data.results[0].feature);
      }
    } catch (err) {
      console.error(err);
      alert("Failed to analyze drift. Check backend server.");
    } finally {
      setLoading(false);
    }
  };

  // Helper to compute distributions for drawing SVG charts
  const getDistributionData = (feature: string) => {
    const refVals = mockDriftData.reference.map((r: any) => r[feature]);
    const prodVals = mockDriftData.production.map((p: any) => p[feature]);

    const isNumeric = refVals.every((v) => !isNaN(Number(v)));

    if (isNumeric) {
      const refNum = refVals.map(Number);
      const prodNum = prodVals.map(Number);
      const allVals = [...refNum, ...prodNum];
      const min = Math.min(...allVals);
      const max = Math.max(...allVals);
      const range = max - min;
      const binCount = 5;
      const binSize = range / binCount;

      const refBins = Array(binCount).fill(0);
      const prodBins = Array(binCount).fill(0);

      refNum.forEach((v) => {
        const binIdx = Math.min(binCount - 1, Math.floor((v - min) / binSize));
        refBins[binIdx]++;
      });

      prodNum.forEach((v) => {
        const binIdx = Math.min(binCount - 1, Math.floor((v - min) / binSize));
        prodBins[binIdx]++;
      });

      // Normalize to percentages
      const refPct = refBins.map((c) => (c / refNum.length) * 100);
      const prodPct = prodBins.map((c) => (c / prodNum.length) * 100);

      const labels = Array(binCount)
        .fill(0)
        .map((_, idx) => {
          const start = min + idx * binSize;
          const end = start + binSize;
          return `${Math.round(start / 1000)}k-${Math.round(end / 1000)}k`;
        });

      return { labels, refPct, prodPct };
    } else {
      // Categorical
      const refStr = refVals.map(String);
      const prodStr = prodVals.map(String);
      const categories = Array.from(new Set([...refStr, ...prodStr])).sort();

      const refPct = categories.map((cat) => {
        const count = refStr.filter((c) => c === cat).length;
        return (count / refStr.length) * 100;
      });

      const prodPct = categories.map((cat) => {
        const count = prodStr.filter((c) => c === cat).length;
        return (count / prodStr.length) * 100;
      });

      return { labels: categories, refPct, prodPct };
    }
  };

  const chartData = selectedFeature ? getDistributionData(selectedFeature) : null;

  return (
    <div id="feature_drift_container" className="space-y-6">
      {/* Control instructions */}
      <div className="bg-slate-900/40 p-6 rounded-2xl border border-slate-800 flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="space-y-2">
          <h2 className="text-lg font-bold text-slate-100 font-sans">Covariate Feature Drift Validation</h2>
          <p className="text-xs text-slate-400 max-w-xl leading-relaxed">
            Compares production samples against baseline reference tables to alert on data distribution shifts (covariate drift). Checks statistical deviations dynamically.
          </p>
        </div>
        <button
          onClick={handleDriftAnalysis}
          disabled={loading}
          className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-800 text-white font-bold text-xs rounded-xl transition-all shadow-lg shrink-0"
        >
          {loading ? "Calculating Matrices..." : "Compare Baseline vs Production"}
        </button>
      </div>

      {loading && (
        <div className="p-12 text-center bg-slate-900/20 border border-slate-800/50 rounded-2xl space-y-3">
          <div className="w-8 h-8 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto" />
          <p className="text-xs text-slate-400">Performing Kolmogorov-Smirnov cumulative distance integrals and categorical relative frequency evaluations...</p>
        </div>
      )}

      {driftResults && (
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Drift table results list */}
          <div className="lg:col-span-3 bg-slate-900/40 border border-slate-800 rounded-2xl overflow-hidden shadow-lg">
            <div className="border-b border-slate-800 bg-slate-950/40 p-4">
              <h3 className="text-sm font-bold text-slate-200 font-sans">Distribution Drift Matrix</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 bg-slate-950/20 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                    <th className="p-4">Feature</th>
                    <th className="p-4">Statistical Test</th>
                    <th className="p-4">KS/Chi2 Stat</th>
                    <th className="p-4">P-Value</th>
                    <th className="p-4">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 text-xs">
                  {driftResults.map((res) => (
                    <tr
                      key={res.feature}
                      onClick={() => setSelectedFeature(res.feature)}
                      className={`cursor-pointer hover:bg-slate-800/40 transition-colors ${
                        selectedFeature === res.feature ? "bg-indigo-500/5" : ""
                      }`}
                    >
                      <td className="p-4 font-mono font-bold text-slate-300">{res.feature}</td>
                      <td className="p-4 text-slate-400">{res.test_used}</td>
                      <td className="p-4 font-mono text-slate-400">{res.statistic.toFixed(4)}</td>
                      <td className="p-4 font-mono text-slate-400">
                        {res.p_value < 0.001 ? "< 0.001" : res.p_value.toFixed(4)}
                      </td>
                      <td className="p-4">
                        {res.drifted ? (
                          <span className="flex items-center gap-1 text-rose-400 font-bold">
                            <AlertTriangle className="w-3.5 h-3.5" />
                            Drifted
                          </span>
                        ) : (
                          <span className="flex items-center gap-1 text-emerald-400 font-bold">
                            <ShieldCheck className="w-3.5 h-3.5" />
                            Stable
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="p-4 bg-slate-950/30 text-[10px] text-slate-500 flex items-center gap-1 border-t border-slate-800">
              <Info className="w-3.5 h-3.5" />
              Significance threshold configured at &alpha; = 0.05. A p-value less than &alpha; rejects the null hypothesis (stable distribution).
            </div>
          </div>

          {/* SVG comparison visualization graph */}
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-5">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold text-slate-200 flex items-center gap-1.5 font-sans">
                    <BarChart3 className="w-4 h-4 text-indigo-400" />
                    Feature Profile: <span className="font-mono text-indigo-300">{selectedFeature}</span>
                  </h3>
                  <p className="text-[10px] text-slate-400 mt-1 leading-relaxed">
                    Visual side-by-side distribution percentage comparator.
                  </p>
                </div>
                <div className="flex gap-2 text-[10px] font-bold">
                  <span className="flex items-center gap-1 text-slate-400">
                    <span className="w-2.5 h-2.5 bg-slate-600 rounded-sm inline-block" />
                    Reference
                  </span>
                  <span className="flex items-center gap-1 text-indigo-400">
                    <span className="w-2.5 h-2.5 bg-indigo-500 rounded-sm inline-block" />
                    Production
                  </span>
                </div>
              </div>

              {chartData && (
                <div className="bg-slate-950/40 p-4 rounded-xl border border-slate-850">
                  <svg className="w-full h-56" viewBox="0 0 400 220" xmlns="http://www.w3.org/2000/svg">
                    {/* Y Axis helper grids */}
                    <line x1="40" y1="30" x2="380" y2="30" stroke="#1e293b" strokeDasharray="3,3" />
                    <line x1="40" y1="75" x2="380" y2="75" stroke="#1e293b" strokeDasharray="3,3" />
                    <line x1="40" y1="120" x2="380" y2="120" stroke="#1e293b" strokeDasharray="3,3" />
                    <line x1="40" y1="165" x2="380" y2="165" stroke="#1e293b" strokeDasharray="3,3" />
                    <line x1="40" y1="180" x2="380" y2="180" stroke="#334155" strokeWidth="1.5" />

                    {/* Plot columns */}
                    {chartData.labels.map((lbl, idx) => {
                      const colWidth = 60;
                      const xBase = 60 + idx * colWidth;

                      // Map percentages to height (max height = 140px, mapping 0-100%)
                      const rVal = chartData.refPct[idx] || 0;
                      const pVal = chartData.prodPct[idx] || 0;

                      const rHeight = (rVal / 100) * 140;
                      const pHeight = (pVal / 100) * 140;

                      const rY = 180 - rHeight;
                      const pY = 180 - pHeight;

                      return (
                        <g key={idx} className="transition-all hover:opacity-90">
                          {/* Reference bar (grey-slate) */}
                          <rect
                            x={xBase}
                            y={rY}
                            width="20"
                            height={rHeight}
                            fill="#475569"
                            rx="2"
                            className="transition-all duration-500"
                          >
                            <title>Reference: {rVal.toFixed(1)}%</title>
                          </rect>

                          {/* Production bar (indigo-violet) */}
                          <rect
                            x={xBase + 24}
                            y={pY}
                            width="20"
                            height={pHeight}
                            fill="#6366f1"
                            rx="2"
                            className="transition-all duration-500"
                          >
                            <title>Production: {pVal.toFixed(1)}%</title>
                          </rect>

                          {/* Column X Labels */}
                          <text
                            x={xBase + 22}
                            y="198"
                            fill="#94a3b8"
                            fontSize="9"
                            fontFamily="monospace"
                            textAnchor="middle"
                          >
                            {lbl}
                          </text>
                        </g>
                      );
                    })}

                    {/* Y Labels */}
                    <text x="32" y="34" fill="#64748b" fontSize="8" fontFamily="monospace" textAnchor="end">100%</text>
                    <text x="32" y="79" fill="#64748b" fontSize="8" fontFamily="monospace" textAnchor="end">50%</text>
                    <text x="32" y="124" fill="#64748b" fontSize="8" fontFamily="monospace" textAnchor="end">25%</text>
                    <text x="32" y="169" fill="#64748b" fontSize="8" fontFamily="monospace" textAnchor="end">5%</text>
                    <text x="32" y="184" fill="#64748b" fontSize="8" fontFamily="monospace" textAnchor="end">0%</text>
                  </svg>
                </div>
              )}

              {/* Statistical insights on the drift */}
              <div className="bg-indigo-950/20 p-4 rounded-xl border border-indigo-500/15">
                <h4 className="text-xs font-bold text-indigo-300 font-sans">MLOps Safeguard Suggestion</h4>
                <p className="text-[11px] text-slate-400 mt-1.5 leading-relaxed font-sans">
                  {selectedFeature === "income" ? (
                    <span>
                      🚨 <strong>Income covariate drift is extremely critical!</strong> The production dataset has shifted towards higher incomes. Feeding this raw production data straight into the baseline model will trigger catastrophic silent failures. Consider retraining your model with a sample mixture or applying density ratios.
                    </span>
                  ) : (
                    <span>
                      🟢 <strong>Feature profile is stable!</strong> The distribution profiles between train references and production samples align correctly. Covariances match historical bounds.
                    </span>
                  )}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
