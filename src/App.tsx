import { useState } from "react";
import { Shield, Sparkles, Database, BarChart4, FolderCode, HelpCircle, Activity } from "lucide-react";
import DatasetScanner from "./components/DatasetScanner";
import FeatureDrift from "./components/FeatureDrift";
import ModelEvaluator from "./components/ModelEvaluator";
import CodeExplorer from "./components/CodeExplorer";

export default function App() {
  const [activeTab, setActiveTab] = useState<"check" | "drift" | "report" | "explorer">("check");

  return (
    <div id="fitcheck_dashboard_root" className="min-h-screen bg-[#090a0f] text-slate-100 flex flex-col font-sans selection:bg-indigo-500/30 selection:text-white">
      
      {/* Decorative top ambient aura glowing bar */}
      <div className="h-1 bg-gradient-to-r from-indigo-500 via-purple-500 to-emerald-500" />

      {/* Main Container */}
      <div className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 flex flex-col gap-6">
        
        {/* Navigation & Header section */}
        <header className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 bg-slate-900/20 p-6 rounded-2xl border border-slate-800/80 backdrop-blur-md">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="p-1.5 bg-indigo-600/15 rounded-lg border border-indigo-500/30 text-indigo-400">
                <Shield className="w-6 h-6" />
              </span>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-2xl font-black text-slate-100 tracking-tight font-sans">
                    FitCheck <span className="text-indigo-400 font-medium">v2.0</span>
                  </h1>
                  <span className="text-[10px] font-mono bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 font-bold px-2 py-0.5 rounded-full">
                    Apache-2.0
                  </span>
                </div>
                <p className="text-xs text-slate-400 mt-0.5">
                  Zero-boilerplate ML data validation, distribution drift profiling, and model evaluation.
                </p>
              </div>
            </div>
          </div>

          {/* Social Stats/Badges Row */}
          <div className="flex flex-wrap items-center gap-3 self-start md:self-auto">
            <span className="flex items-center gap-1 text-[11px] font-semibold text-slate-400 bg-slate-950/60 p-1.5 px-3 rounded-lg border border-slate-800">
              <Activity className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
              Coverage: <strong className="text-emerald-400 font-bold">92%</strong>
            </span>
            <span className="flex items-center gap-1 text-[11px] font-semibold text-slate-400 bg-slate-950/60 p-1.5 px-3 rounded-lg border border-slate-800">
              <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
              Philosophy: <strong className="text-indigo-300 font-bold">Immutability</strong>
            </span>
          </div>
        </header>

        {/* Tab Controls Bar */}
        <nav className="flex flex-wrap gap-2 border-b border-slate-800 pb-2">
          <button
            onClick={() => setActiveTab("check")}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-xl border transition-all ${
              activeTab === "check"
                ? "bg-indigo-600/15 border-indigo-500/50 text-indigo-300 shadow"
                : "bg-transparent border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/40"
            }`}
          >
            <Database className="w-4 h-4" />
            Dataset Health (Check)
          </button>
          
          <button
            onClick={() => setActiveTab("drift")}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-xl border transition-all ${
              activeTab === "drift"
                ? "bg-indigo-600/15 border-indigo-500/50 text-indigo-300 shadow"
                : "bg-transparent border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/40"
            }`}
          >
            <Activity className="w-4 h-4" />
            Feature Drift (KS Test)
          </button>
          
          <button
            onClick={() => setActiveTab("report")}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-xl border transition-all ${
              activeTab === "report"
                ? "bg-indigo-600/15 border-indigo-500/50 text-indigo-300 shadow"
                : "bg-transparent border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/40"
            }`}
          >
            <BarChart4 className="w-4 h-4" />
            Model Evaluation (Report)
          </button>
          
          <button
            onClick={() => setActiveTab("explorer")}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-xl border transition-all ${
              activeTab === "explorer"
                ? "bg-indigo-600/15 border-indigo-500/50 text-indigo-300 shadow"
                : "bg-transparent border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/40"
            }`}
          >
            <FolderCode className="w-4 h-4" />
            Python Package Explorer
          </button>
        </nav>

        {/* Dynamic Panel Content Router */}
        <main className="flex-1 min-h-[500px]">
          {activeTab === "check" && <DatasetScanner />}
          {activeTab === "drift" && <FeatureDrift />}
          {activeTab === "report" && <ModelEvaluator />}
          {activeTab === "explorer" && <CodeExplorer />}
        </main>

        {/* Cohesive design footer */}
        <footer className="border-t border-slate-900/80 pt-6 pb-2 text-center text-[10px] text-slate-500 font-mono flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            &copy; 2026 FitCheck Open Source Contributors. Published under the Apache License, Version 2.0.
          </div>
          <div className="flex justify-center gap-4 text-slate-400">
            <span className="flex items-center gap-1">
              <Shield className="w-3.5 h-3.5 text-indigo-500" />
              Diagnose, don&apos;t operate.
            </span>
          </div>
        </footer>

      </div>
    </div>
  );
}
