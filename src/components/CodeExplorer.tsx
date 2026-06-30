import { useState } from "react";
import { Folder, FileCode, Copy, Check, Download, BookOpen, Terminal, Shield, GitCommit } from "lucide-react";
import { pythonFiles, PythonFile } from "../pythonPackageCode";

export default function CodeExplorer() {
  const [selectedFile, setSelectedFile] = useState<PythonFile>(pythonFiles[0]);
  const [copied, setCopied] = useState(false);
  const [currentTab, setCurrentTab] = useState<"code" | "manual">("code");

  const handleCopy = () => {
    navigator.clipboard.writeText(selectedFile.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const element = document.createElement("a");
    const file = new Blob([selectedFile.content], { type: "text/plain" });
    element.href = URL.createObjectURL(file);
    element.download = selectedFile.name;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  return (
    <div id="code_explorer_container" className="bg-slate-900/60 backdrop-blur-xl border border-slate-800 rounded-2xl overflow-hidden shadow-2xl">
      <div className="border-b border-slate-800 bg-slate-950/40 p-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Folder className="w-5 h-5 text-indigo-400" />
            <h2 className="text-lg font-bold text-slate-100 font-sans">FitCheck v2.0 Python Core Library</h2>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Browse and package the core files of the fully compliant offline SDK under Apache License 2.0.
          </p>
        </div>
        
        <div className="flex items-center gap-2 self-start sm:self-auto bg-slate-900 p-1 rounded-lg border border-slate-800">
          <button
            onClick={() => setCurrentTab("code")}
            className={`flex items-center gap-1 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              currentTab === "code"
                ? "bg-indigo-600 text-white shadow"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <FileCode className="w-3.5 h-3.5" />
            Source Files
          </button>
          <button
            onClick={() => setCurrentTab("manual")}
            className={`flex items-center gap-1 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              currentTab === "manual"
                ? "bg-indigo-600 text-white shadow"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
            Hiring & Git Manual
          </button>
        </div>
      </div>

      {currentTab === "code" ? (
        <div className="grid grid-cols-1 lg:grid-cols-4 min-h-[550px] divide-y lg:divide-y-0 lg:divide-x divide-slate-800">
          {/* File Sidebar Tree */}
          <div className="lg:col-span-1 bg-slate-950/60 p-4 overflow-y-auto max-h-[250px] lg:max-h-[550px]">
            <div className="text-xs font-bold text-indigo-400 uppercase tracking-wider mb-3 px-2 flex items-center justify-between">
              <span>fitcheck/ workspace</span>
              <span className="text-[10px] bg-slate-800 text-slate-300 px-1.5 py-0.5 rounded border border-slate-700">v2.0.0</span>
            </div>
            <nav className="space-y-1">
              {pythonFiles.map((file) => (
                <button
                  key={file.path}
                  onClick={() => setSelectedFile(file)}
                  className={`w-full flex items-center gap-2.5 px-3 py-2 text-xs text-left rounded-lg transition-all ${
                    selectedFile.path === file.path
                      ? "bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 font-medium"
                      : "border border-transparent text-slate-400 hover:bg-slate-800/50 hover:text-slate-200"
                  }`}
                >
                  <FileCode className={`w-4 h-4 ${selectedFile.path === file.path ? "text-indigo-400" : "text-slate-500"}`} />
                  <div className="truncate">
                    <div className="font-mono">{file.name}</div>
                    <div className="text-[10px] text-slate-500 truncate mt-0.5">{file.path}</div>
                  </div>
                </button>
              ))}
            </nav>
          </div>

          {/* Code Viewer Panel */}
          <div className="lg:col-span-3 flex flex-col bg-slate-950/20">
            <div className="bg-slate-900/40 p-3 px-4 border-b border-slate-800/80 flex items-center justify-between">
              <div className="flex flex-col">
                <span className="text-xs font-mono text-slate-200 font-bold">{selectedFile.path}</span>
                <span className="text-[10px] text-slate-400 mt-0.5">{selectedFile.description}</span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-slate-300 hover:text-white bg-slate-800 border border-slate-700 hover:bg-slate-700 rounded-lg transition-all"
                >
                  {copied ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                      Copied!
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5 text-indigo-400" />
                      Copy Code
                    </>
                  )}
                </button>
                <button
                  onClick={handleDownload}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-slate-300 hover:text-white bg-indigo-600/20 border border-indigo-500/30 hover:bg-indigo-600/30 rounded-lg transition-all"
                >
                  <Download className="w-3.5 h-3.5 text-indigo-400" />
                  Download
                </button>
              </div>
            </div>
            <div className="flex-1 p-4 overflow-x-auto overflow-y-auto max-h-[480px]">
              <pre className="text-xs text-slate-300 leading-relaxed font-mono whitespace-pre select-all bg-slate-950/80 p-4 rounded-xl border border-slate-900">
                <code>{selectedFile.content}</code>
              </pre>
            </div>
          </div>
        </div>
      ) : (
        /* Hiring & Git manual */
        <div className="p-6 md:p-8 space-y-8 max-h-[550px] overflow-y-auto">
          <div>
            <h3 className="text-lg font-bold text-indigo-300 flex items-center gap-2">
              <Shield className="w-5 h-5 text-indigo-400" />
              FitCheck Publishing & Git Workflow Manual
            </h3>
            <p className="text-sm text-slate-400 mt-1.5">
              Follow these standard terminal recipes to build, verify, commit, and distribute this library under Apache-2.0 to GitHub and PyPI.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800">
                <div className="flex items-center gap-2 text-slate-200 font-bold text-sm mb-3">
                  <Terminal className="w-4 h-4 text-emerald-400" />
                  1. Local Verification and Testing
                </div>
                <p className="text-xs text-slate-400 mb-3">
                  Initialize a clean virtual environment and run the complete test coverage suite:
                </p>
                <pre className="bg-slate-950 p-3 rounded-lg border border-slate-900 text-xs text-slate-300 font-mono space-y-1 overflow-x-auto">
                  <div>python3 -m venv .venv</div>
                  <div>source .venv/bin/activate</div>
                  <div>pip install -e &quot;.[dev]&quot;</div>
                  <div>pytest --cov=fitcheck --cov-report=term-missing</div>
                </pre>
              </div>

              <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800">
                <div className="flex items-center gap-2 text-slate-200 font-bold text-sm mb-3">
                  <GitCommit className="w-4 h-4 text-amber-400" />
                  2. Git Pre-Commit Hook Config
                </div>
                <p className="text-xs text-slate-400 mb-3">
                  Validate incoming data files inside your pull requests with pre-commit gates:
                </p>
                <pre className="bg-slate-950 p-3 rounded-lg border border-slate-900 text-xs text-slate-300 font-mono space-y-1 overflow-x-auto">
                  <div>pip install pre-commit</div>
                  <div>pre-commit install</div>
                  <div># Run hooks manually to verify checks:</div>
                  <div>pre-commit run --all-files</div>
                </pre>
              </div>
            </div>

            <div className="space-y-4">
              <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800">
                <div className="flex items-center gap-2 text-slate-200 font-bold text-sm mb-3">
                  <Terminal className="w-4 h-4 text-indigo-400" />
                  3. Packing and Hatch Building
                </div>
                <p className="text-xs text-slate-400 mb-3">
                  Compile Python source files to wheel archives according to `pyproject.toml` specs:
                </p>
                <pre className="bg-slate-950 p-3 rounded-lg border border-slate-900 text-xs text-slate-300 font-mono space-y-1 overflow-x-auto">
                  <div>pip install build</div>
                  <div>python -m build</div>
                  <div># Verifies dist/ contains tar.gz and .whl files</div>
                  <div>ls -l dist/</div>
                </pre>
              </div>

              <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800">
                <div className="flex items-center gap-2 text-slate-200 font-bold text-sm mb-3">
                  <Terminal className="w-4 h-4 text-purple-400" />
                  4. PyPI Distribution (Social Release)
                </div>
                <p className="text-xs text-slate-400 mb-3">
                  Upload release wheels securely to PyPI registries using twine credentials:
                </p>
                <pre className="bg-slate-950 p-3 rounded-lg border border-slate-900 text-xs text-slate-300 font-mono space-y-1 overflow-x-auto">
                  <div>pip install twine</div>
                  <div># Push to TestPyPI first:</div>
                  <div>twine upload --repository testpypi dist/*</div>
                  <div># Production upload:</div>
                  <div>twine upload dist/*</div>
                </pre>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
