import React, { useState, useRef } from "react";
import { AlertCircle, CheckCircle, Brain, Play, Code, Copy, Check, Upload, HelpCircle, FileSpreadsheet } from "lucide-react";
import { CheckReport } from "../types";
import { mockDatasets } from "../mockData";

export default function DatasetScanner() {
  const [selectedPreset, setSelectedPreset] = useState<keyof typeof mockDatasets | "custom">("creditRisk");
  const [report, setReport] = useState<CheckReport | null>(null);
  const [scanning, setScanning] = useState(false);
  const [geminiResponse, setGeminiResponse] = useState<string>("");
  const [geminiLoading, setGeminiLoading] = useState(false);
  const [scriptCopied, setScriptCopied] = useState(false);
  const [customData, setCustomData] = useState<any[] | null>(null);
  const [customFilename, setCustomFilename] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Parse CSV string into an array of row objects
  const parseCSV = (text: string): any[] => {
    const lines = text.split(/\r?\n/).filter((line) => line.trim() !== "");
    if (lines.length < 2) return [];
    const headers = lines[0].split(",").map((h) => h.trim().replace(/^["']|["']$/g, ""));
    const data: any[] = [];
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(",").map((v) => v.trim().replace(/^["']|["']$/g, ""));
      if (values.length === headers.length) {
        const obj: any = {};
        headers.forEach((h, idx) => {
          const val = values[idx];
          if (val === "" || val.toLowerCase() === "null" || val.toLowerCase() === "nan") {
            obj[h] = null;
          } else if (!isNaN(Number(val))) {
            obj[h] = Number(val);
          } else {
            obj[h] = val;
          }
        });
        data.push(obj);
      }
    }
    return data;
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setCustomFilename(file.name);
    setSelectedPreset("custom");
    setReport(null);
    setGeminiResponse("");

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      const parsed = parseCSV(text);
      setCustomData(parsed);
    };
    reader.readAsText(file);
  };

  const handleScan = async () => {
    setScanning(true);
    setReport(null);
    setGeminiResponse("");

    let payload: any[] = [];
    let target = "";
    let fname = "";

    if (selectedPreset === "custom") {
      if (!customData || customData.length === 0) {
        alert("Please upload a valid CSV first.");
        setScanning(false);
        return;
      }
      payload = customData;
      target = Object.keys(customData[0])[Object.keys(customData[0]).length - 1]; // default to last column
      fname = customFilename;
    } else {
      payload = mockDatasets[selectedPreset].data;
      target = mockDatasets[selectedPreset].target;
      fname = mockDatasets[selectedPreset].filename;
    }

    try {
      const res = await fetch("/api/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: payload, target, filename: fname }),
      });
      const reportData: CheckReport = await res.json();
      setReport(reportData);
    } catch (err) {
      console.error(err);
      alert("Validation failed. Please check backend connection.");
    } finally {
      setScanning(false);
    }
  };

  const handleAskGemini = async () => {
    if (!report) return;
    setGeminiLoading(true);
    setGeminiResponse("");

    try {
      const res = await fetch("/api/gemini/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ issues: report.issues, summary: report.summary }),
      });
      const data = await res.json();
      setGeminiResponse(data.analysis);
    } catch (err) {
      console.error(err);
      setGeminiResponse("### ❌ Gemini Integration Error\nFailed to pull live analysis. Is the server online?");
    } finally {
      setGeminiLoading(false);
    }
  };

  const handleCopyScript = () => {
    if (!report) return;
    navigator.clipboard.writeText(report.fix_script);
    setScriptCopied(true);
    setTimeout(() => setScriptCopied(false), 2000);
  };

  const handleDownloadScript = () => {
    if (!report) return;
    const element = document.createElement("a");
    const file = new Blob([report.fix_script], { type: "text/plain" });
    element.href = URL.createObjectURL(file);
    element.download = "fitcheck_fix_script.py";
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  // Safe manual markdown parser for beautiful dark text presentation
  const formatMarkdown = (text: string) => {
    const lines = text.split("\n");
    return lines.map((line, i) => {
      if (line.startsWith("### ")) {
        return <h4 key={i} className="text-sm font-bold text-indigo-300 mt-4 mb-2 font-sans">{line.replace("### ", "")}</h4>;
      }
      if (line.startsWith("## ")) {
        return <h3 key={i} className="text-base font-bold text-slate-100 mt-5 mb-2 font-sans">{line.replace("## ", "")}</h3>;
      }
      if (line.startsWith("# ")) {
        return <h2 key={i} className="text-lg font-bold text-slate-100 mt-6 mb-3 font-sans">{line.replace("# ", "")}</h2>;
      }
      if (line.startsWith("- ") || line.startsWith("* ")) {
        return (
          <li key={i} className="ml-4 list-disc text-xs text-slate-300 my-1 font-sans">
            {line.replace(/^[-*]\s+/, "")}
          </li>
        );
      }
      if (line.trim() === "") {
        return <div key={i} className="h-1" />;
      }
      const parts = line.split(/\*\*(.*?)\*\*/g);
      return (
        <p key={i} className="text-xs leading-relaxed text-slate-300 my-1.5 font-sans">
          {parts.map((part, idx) => (idx % 2 === 1 ? <strong key={idx} className="text-indigo-200 font-semibold">{part}</strong> : part))}
        </p>
      );
    });
  };

  const currentDatasetDetails = selectedPreset === "custom" 
    ? { name: customFilename || "Custom CSV", description: `Loaded custom file with ${customData?.length || 0} rows.` }
    : mockDatasets[selectedPreset];

  return (
    <div id="dataset_scanner_parent" className="space-y-6">
      {/* Selector and Controller header */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 bg-slate-900/40 p-6 rounded-2xl border border-slate-800">
        <div className="md:col-span-2 space-y-4">
          <div>
            <label className="block text-xs font-bold text-indigo-400 uppercase tracking-wider mb-2">
              Select ML Dataset Source
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {(Object.keys(mockDatasets) as Array<keyof typeof mockDatasets>).map((key) => (
                <button
                  key={key}
                  onClick={() => {
                    setSelectedPreset(key);
                    setReport(null);
                    setGeminiResponse("");
                  }}
                  className={`p-3 text-left rounded-xl border transition-all ${
                    selectedPreset === key
                      ? "bg-indigo-500/10 border-indigo-500/50 text-slate-200 shadow"
                      : "bg-slate-950/40 border-slate-800 hover:border-slate-700 text-slate-400"
                  }`}
                >
                  <div className="text-xs font-bold truncate">{mockDatasets[key].name}</div>
                  <div className="text-[10px] text-slate-500 mt-1 truncate">{mockDatasets[key].filename}</div>
                </button>
              ))}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-4 pt-1">
            <span className="text-xs text-slate-400">Or drag and drop standard CSV outputs:</span>
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              ref={fileInputRef}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg border transition-all ${
                selectedPreset === "custom"
                  ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300"
                  : "bg-slate-800 border-slate-700 hover:bg-slate-700 text-slate-300"
              }`}
            >
              <Upload className="w-3.5 h-3.5" />
              {selectedPreset === "custom" ? `CSV Loaded: ${customFilename}` : "Upload Custom CSV"}
            </button>
          </div>
        </div>

        {/* Action scanner */}
        <div className="bg-slate-950/60 p-5 rounded-xl border border-slate-800 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-slate-100 flex items-center gap-1.5">
              <FileSpreadsheet className="w-4 h-4 text-indigo-400" />
              Dataset Spec
            </h3>
            <p className="text-[11px] text-slate-400 mt-2 leading-relaxed">
              <strong>Source:</strong> {currentDatasetDetails.name}<br />
              <strong>Descr:</strong> {currentDatasetDetails.description}
            </p>
          </div>

          <button
            onClick={handleScan}
            disabled={scanning}
            className="w-full mt-4 flex items-center justify-center gap-2 px-4 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-800 disabled:text-indigo-400 text-white font-bold text-xs rounded-xl transition-all shadow-lg"
          >
            <Play className="w-4 h-4 fill-white" />
            {scanning ? "Processing Quality Audits..." : "Scan Dataset for Issues"}
          </button>
        </div>
      </div>

      {scanning && (
        <div className="p-12 text-center bg-slate-900/20 border border-slate-800/50 rounded-2xl space-y-3">
          <div className="w-8 h-8 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto" />
          <p className="text-xs text-slate-400">Computing Interquartile IQR anomalies, null distribution matrices, and deduplicating rows...</p>
        </div>
      )}

      {/* Scanned Report Layout */}
      {report && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main audit metrics and tables */}
          <div className="lg:col-span-2 space-y-6">
            {/* Header Status card */}
            <div className={`p-5 rounded-2xl border flex items-center justify-between shadow-xl ${
              report.passed 
                ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300" 
                : "bg-rose-500/10 border-rose-500/30 text-rose-300"
            }`}>
              <div className="flex items-center gap-3">
                {report.passed ? (
                  <CheckCircle className="w-8 h-8 text-emerald-400" />
                ) : (
                  <AlertCircle className="w-8 h-8 text-rose-400" />
                )}
                <div>
                  <h3 className="text-base font-bold text-slate-100 font-sans">
                    {report.passed ? "Data Quality Integrity Control Passed" : "Data Anomaly Alarms Detected"}
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    {report.passed 
                      ? "Zero structural issues detected. Safe to export into modeling pipeline." 
                      : `Found ${report.summary.issues_count} critical/warning anomalies in raw columns.`}
                  </p>
                </div>
              </div>
              <div className="text-right">
                <div className="text-[10px] uppercase tracking-wider text-slate-400 font-bold">Health Score</div>
                <div className={`text-2xl font-black ${report.passed ? "text-emerald-400" : "text-rose-400"}`}>
                  {report.passed ? "100%" : `${Math.max(20, 100 - report.summary.issues_count * 15)}%`}
                </div>
              </div>
            </div>

            {/* Total columns rows details card */}
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-slate-900/40 p-4 rounded-xl border border-slate-800 text-center">
                <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Total Rows</div>
                <div className="text-lg font-black text-slate-200 mt-1">{report.summary.total_rows}</div>
              </div>
              <div className="bg-slate-900/40 p-4 rounded-xl border border-slate-800 text-center">
                <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Total Columns</div>
                <div className="text-lg font-black text-slate-200 mt-1">{report.summary.total_columns}</div>
              </div>
              <div className="bg-slate-900/40 p-4 rounded-xl border border-slate-800 text-center">
                <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Anomalies</div>
                <div className={`text-lg font-black mt-1 ${report.summary.issues_count > 0 ? "text-amber-400" : "text-emerald-400"}`}>
                  {report.summary.issues_count}
                </div>
              </div>
            </div>

            {/* List of anomalies in a tabular view */}
            <div className="bg-slate-900/40 border border-slate-800 rounded-2xl overflow-hidden shadow-lg">
              <div className="border-b border-slate-800 bg-slate-950/40 p-4">
                <h3 className="text-sm font-bold text-slate-200 font-sans">Active Validation Rules Diagnostics</h3>
              </div>
              <div className="p-4 space-y-3">
                {report.issues.length === 0 ? (
                  <div className="text-center py-8 text-slate-500 text-xs">
                    No diagnostics triggered. Dataset passed clean!
                  </div>
                ) : (
                  report.issues.map((issue, idx) => {
                    const sevStyles = 
                      issue.severity === "critical"
                        ? "bg-rose-500/10 border-rose-500/30 text-rose-400"
                        : issue.severity === "warning"
                        ? "bg-amber-500/10 border-amber-500/30 text-amber-400"
                        : "bg-blue-500/10 border-blue-500/30 text-blue-400";
                    return (
                      <div key={idx} className="bg-slate-950/50 p-4 rounded-xl border border-slate-850 space-y-3">
                        <div className="flex items-center justify-between gap-4">
                          <span className="text-xs font-mono font-bold text-indigo-300">
                            {issue.column === "all_columns" ? "Global Scope" : `col: ${issue.column}`}
                          </span>
                          <span className={`text-[9px] uppercase tracking-wider font-bold px-2 py-0.5 border rounded-full ${sevStyles}`}>
                            {issue.severity}
                          </span>
                        </div>
                        <p className="text-xs text-slate-300 leading-relaxed font-sans">{issue.message}</p>
                        <div className="bg-slate-900/80 p-2.5 rounded-lg border border-slate-850 text-[11px] text-slate-400 font-mono">
                          <span className="text-indigo-400 font-bold">💡 Solution:</span> {issue.suggestion}
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>

          {/* Right column: Auto-Fix script & Gemini Copilot */}
          <div className="space-y-6">
            {/* Gemini Analysis copilot */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl overflow-hidden shadow-xl flex flex-col min-h-[250px]">
              <div className="border-b border-slate-800 bg-indigo-950/20 p-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Brain className="w-5 h-5 text-indigo-400" />
                  <h3 className="text-sm font-bold text-slate-100 font-sans">Gemini MLOps Copilot</h3>
                </div>
                {!geminiResponse && !geminiLoading && (
                  <button
                    onClick={handleAskGemini}
                    className="flex items-center gap-1 px-3 py-1 bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-[10px] rounded-lg transition-all"
                  >
                    Generate Report
                  </button>
                )}
              </div>

              <div className="flex-1 p-5 overflow-y-auto max-h-[300px]">
                {geminiLoading ? (
                  <div className="flex flex-col items-center justify-center h-full space-y-3 py-8">
                    <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                    <p className="text-[10px] text-slate-400 font-mono">Querying @google/genai (3.5-flash)...</p>
                  </div>
                ) : geminiResponse ? (
                  <div className="space-y-2 text-xs">
                    {formatMarkdown(geminiResponse)}
                  </div>
                ) : (
                  <div className="text-center py-8 text-slate-500 text-xs flex flex-col items-center justify-center h-full">
                    <Brain className="w-8 h-8 text-slate-700 mb-2" />
                    Click &quot;Generate Report&quot; to prompt Gemini for a deep-dive down-stream modeling impact analysis.
                  </div>
                )}
              </div>
            </div>

            {/* Custom generated Python script code panel */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl overflow-hidden shadow-xl flex flex-col">
              <div className="border-b border-slate-800 bg-slate-950/40 p-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Code className="w-5 h-5 text-emerald-400" />
                  <h3 className="text-sm font-bold text-slate-100 font-sans">fitcheck_fix_script.py</h3>
                </div>
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={handleCopyScript}
                    className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 transition-all"
                    title="Copy Script"
                  >
                    {scriptCopied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  </button>
                  <button
                    onClick={handleDownloadScript}
                    className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 transition-all"
                    title="Download Script"
                  >
                    <Upload className="w-3.5 h-3.5 transform rotate-180" />
                  </button>
                </div>
              </div>

              <div className="p-4 bg-slate-950/80">
                <div className="text-[10px] text-slate-400 mb-2 leading-relaxed">
                  💡 This script performs median imputation, clips outliers, and filters duplicate records non-mutatively.
                </div>
                <pre className="text-[10px] font-mono text-emerald-300 overflow-x-auto leading-relaxed max-h-[180px] p-2 bg-slate-950 rounded border border-slate-900">
                  {report.fix_script}
                </pre>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
