import React, { useState, useRef, useCallback } from "react";
import "./styles.css";
import WordCloud from "./WordCloud";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:5000";

const SENTIMENT_META = {
  positive: { icon: "▲", label: "Positive", color: "#22c55e", bg: "#dcfce7" },
  negative: { icon: "▼", label: "Negative", color: "#ef4444", bg: "#fee2e2" },
  neutral:  { icon: "●", label: "Neutral",  color: "#94a3b8", bg: "#f1f5f9" },
};

export default function App() {
  const [mode, setMode]           = useState("single");
  const [text, setText]           = useState("");
  const [file, setFile]           = useState(null);
  const [isDragOver, setDragOver] = useState(false);
  const [lastInput, setLastInput] = useState("");
  const [result, setResult]       = useState(null);
  const [error, setError]         = useState(null);
  const [loading, setLoading]     = useState(false);
  const [darkMode, setDarkMode]   = useState(false);
  const fileInputRef              = useRef(null);

  /* ─── drag & drop ─────────────────────────────────────────────── */
  const handleDragOver  = useCallback((e) => { e.preventDefault(); setDragOver(true);  }, []);
  const handleDragLeave = useCallback(()  => { setDragOver(false); }, []);
  const handleDrop      = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f && (f.name.endsWith(".txt") || f.name.endsWith(".docx"))) setFile(f);
  }, []);

  /* ─── predict ──────────────────────────────────────────────────── */
  async function handlePredict() {
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      if (mode === "single") {
        setLastInput(text.trim());
        const resp    = await fetch(`${API_BASE}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        const payload = await resp.json();
        payload?.success ? setResult(payload.data) : setError(payload?.error || `HTTP ${resp.status}`);
      } else {
        setLastInput(file ? file.name : "");
        const form = new FormData();
        form.append("file", file);
        const resp    = await fetch(`${API_BASE}/batch_predict`, { method: "POST", body: form });
        const payload = await resp.json();
        payload?.success ? setResult(payload.data) : setError(payload?.error || `HTTP ${resp.status}`);
      }
    } catch (err) {
      setError(String(err));
    }
    setLoading(false);
  }

  /* ─── csv export ───────────────────────────────────────────────── */
  function makeCsv() {
    if (!result) return null;
    const esc  = (v) => {
      const s = String(v ?? "");
      return s.includes(",") || s.includes('"') || s.includes("\n") ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const rows = [["Sentence", "Aspect", "Sentiment", "Category"]];
    if (isBatchResult) {
      result.forEach((e) =>
        e.aspects.length === 0
          ? rows.push([e.text, "", "", ""])
          : e.aspects.forEach((r) => rows.push([e.text, r.aspect, r.sentiment, r.category]))
      );
    } else {
      const s = lastInput || "";
      result.length > 0
        ? result.forEach((r) => rows.push([s, r.aspect, r.sentiment, r.category]))
        : rows.push([s, "", "", ""]);
    }
    return rows.map((r) => r.map(esc).join(",")).join("\n");
  }

  function downloadCsv() {
    const csv = makeCsv();
    if (!csv) return;
    const a    = Object.assign(document.createElement("a"), {
      href:     URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8;" })),
      download: `apc_results_${Date.now()}.csv`,
    });
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  /* ─── derived state ────────────────────────────────────────────── */
  const isBatchResult  = Array.isArray(result) && result.length > 0 && result[0]?.text !== undefined;
  const isSingleResult = Array.isArray(result) && (result.length === 0 || result[0]?.aspect !== undefined);

  const flatAspects = isBatchResult
    ? result.flatMap((e) => e.aspects)
    : Array.isArray(result) ? result : [];

  const sentimentCount = flatAspects.reduce((acc, r) => {
    const k = r.sentiment?.toLowerCase();
    acc[k]  = (acc[k] || 0) + 1;
    return acc;
  }, {});

  const categoryCount = flatAspects.reduce((acc, r) => {
    if (r.category) acc[r.category] = (acc[r.category] || 0) + 1;
    return acc;
  }, {});

  const aspectTermCount = flatAspects.reduce((acc, r) => {
    if (r.aspect) acc[r.aspect.toLowerCase()] = (acc[r.aspect.toLowerCase()] || 0) + 1;
    return acc;
  }, {});

  const topCategories  = Object.entries(categoryCount).sort((a, b) => b[1] - a[1]);
  const topAspectTerms = Object.entries(aspectTermCount).sort((a, b) => b[1] - a[1]).slice(0, 20);

  const totalSentences = isBatchResult ? result.length : (result && !isBatchResult ? 1 : 0);

  const canPredict = mode === "single" ? !!text.trim() : !!file;

  /* ─── render ───────────────────────────────────────────────────── */
  return (
    <div className={`app${darkMode ? " dark-mode" : ""}`}>

      {/* Dark mode toggle */}
      <button className="theme-toggle" onClick={() => setDarkMode(!darkMode)} aria-label="Toggle theme">
        {darkMode ? "☀" : "☽"}
      </button>

      {/* Hero */}
      <header className="hero">
        <div className="hero-badge">ATE → APC Pipeline</div>
        <h1 className="hero-title">Attention-Based Token Merging for Aspect-Based Sentiment Analysis</h1>
        <p className="hero-description">
          Extract aspect terms, sentiment polarity and categories from text using an end-to-end deep-learning pipeline.
        </p>
      </header>

      {/* Input card */}
      <div className="card input-card">

        {/* Tab switcher */}
        <div className="tab-switcher">
          <button
            className={`tab-btn${mode === "single" ? " active" : ""}`}
            onClick={() => setMode("single")}
          >
            <span className="tab-icon">✎</span> Single sentence
          </button>
          <button
            className={`tab-btn${mode === "file" ? " active" : ""}`}
            onClick={() => setMode("file")}
          >
            <span className="tab-icon">⬆</span> Upload file
          </button>
        </div>

        {/* Input area */}
        <div className="input-area">
          {mode === "single" ? (
            <textarea
              className="textarea"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="e.g. The food was great but the service was slow."
              rows={4}
            />
          ) : (
            <div
              className={`drop-zone${isDragOver ? " drag-over" : ""}${file ? " has-file" : ""}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => !file && fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".txt,.docx"
                style={{ display: "none" }}
                onChange={(e) => setFile(e.target.files[0] || null)}
              />
              {file ? (
                <div className="drop-zone-file">
                  <span className="file-icon">📄</span>
                  <div className="file-info">
                    <span className="file-name">{file.name}</span>
                    <span className="file-size">{(file.size / 1024).toFixed(1)} KB</span>
                  </div>
                  <button
                    className="file-clear"
                    onClick={(e) => { e.stopPropagation(); setFile(null); }}
                    aria-label="Remove file"
                  >✕</button>
                </div>
              ) : (
                <div className="drop-zone-empty">
                  <span className="drop-icon">☁</span>
                  <p className="drop-title">Drag & drop your file here</p>
                  <p className="drop-sub">or <span className="drop-link">browse</span> — .txt or .docx</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="card-actions">
          <button
            className="btn btn-primary"
            onClick={handlePredict}
            disabled={loading || !canPredict}
          >
            {loading ? <span className="spinner" /> : <span>▶</span>}
            {loading ? "Analyzing…" : "Analyze"}
          </button>
        </div>
      </div>

      {/* Stats */}
      {result && flatAspects.length > 0 && (
        <div className="stats-grid">
          {isBatchResult && (
            <div className="stat-card">
              <div className="stat-value">{totalSentences}</div>
              <div className="stat-label">Sentences</div>
            </div>
          )}
          <div className="stat-card">
            <div className="stat-value">{flatAspects.length}</div>
            <div className="stat-label">Aspects found</div>
          </div>
          {["positive", "negative", "neutral"].map((s) =>
            sentimentCount[s] ? (
              <div key={s} className={`stat-card stat-${s}`}>
                <div className="stat-value">{sentimentCount[s]}</div>
                <div className="stat-label">{SENTIMENT_META[s].label}</div>
              </div>
            ) : null
          )}
        </div>
      )}

      {/* Charts */}
      {result && flatAspects.length > 0 && (
        <ChartsSection
          flatAspects={flatAspects}
          sentimentCount={sentimentCount}
          topCategories={topCategories}
          topAspectTerms={topAspectTerms}
        />
      )}

      {/* Word Cloud — shown after extraction */}
      {result && topAspectTerms.length > 0 && (
        <WordCloud terms={topAspectTerms} />
      )}

      {/* Result card */}
      <div className="card result-card">
        {error && (
          <div className="error-box">
            <span className="error-icon">⚠</span>
            <span>{error}</span>
          </div>
        )}

        {!result && !error && (
          <div className="empty-state">
            <div className="empty-icon">◎</div>
            <div className="empty-title">Ready to Analyze</div>
            <div className="empty-text">Enter a sentence or upload a file, then click Analyze.</div>
          </div>
        )}

        {result && flatAspects.length === 0 && !error && (
          <div className="empty-state">
            <div className="empty-icon">⊘</div>
            <div className="empty-title">No Aspects Found</div>
            <div className="empty-text">The model found no aspect terms. Try a different sentence.</div>
          </div>
        )}

        {result && flatAspects.length > 0 && (
          <>
            <div className="result-header">
              <div>
                <h2 className="result-title">Analysis Results</h2>
                <p className="result-meta">
                  {mode === "single" ? "Sentence" : "File"}:&nbsp;
                  <strong>{lastInput}</strong>
                </p>
              </div>
              <button className="btn btn-export" onClick={downloadCsv}>
                ↓ Export CSV
              </button>
            </div>

            <div className="result-container">
              {isBatchResult && result.map((entry, ei) => (
                <div key={ei} className="result-item">
                  <div className="sentence-block">
                    <span className="sentence-num">#{ei + 1}</span>
                    {entry.text}
                  </div>
                  {entry.aspects.length === 0 ? (
                    <p className="no-aspects">No aspects found for this sentence.</p>
                  ) : (
                    <div className="aspects-list">
                      {entry.aspects.map((r, j) => (
                        <AspectRow key={j} r={r} />
                      ))}
                    </div>
                  )}
                </div>
              ))}

              {!isBatchResult && isSingleResult && result.length > 0 && (
                <div className="result-item">
                  <div className="sentence-block">{lastInput}</div>
                  <div className="aspects-list">
                    {result.map((r, i) => <AspectRow key={i} r={r} />)}
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </div>

      <footer className="footer">
        Powered by T5 ATE → BERT APC Pipeline &nbsp;·&nbsp; Attention-Based Token Merging for Aspect-Based Sentiment Analysis
      </footer>
    </div>
  );
}

function AspectRow({ r }) {
  const s    = r.sentiment?.toLowerCase();
  const meta = SENTIMENT_META[s] || { icon: "●", label: r.sentiment };
  return (
    <div className="aspect-row">
      <span className="aspect-term">{r.aspect}</span>
      <span className={`sentiment-badge sentiment-${s}`}>
        <span className="sentiment-icon">{meta.icon}</span>
        {meta.label}
      </span>
      <span className="category-badge">{r.category}</span>
    </div>
  );
}

/* ─── Charts section ─────────────────────────────────────────────────────── */
function ChartsSection({ flatAspects, sentimentCount, topCategories, topAspectTerms }) {
  const total       = flatAspects.length;
  const sentiments  = ["positive", "negative", "neutral"];
  const maxCatCount = topCategories[0]?.[1] || 1;
  const maxTermFreq = topAspectTerms[0]?.[1] || 1;

  return (
    <div className="charts-section">
      {/* Sentiment distribution */}
      <div className="chart-card">
        <h3 className="chart-title">Sentiment Distribution</h3>

        {/* Stacked bar */}
        <div className="stacked-bar">
          {sentiments.map((s) => {
            const cnt = sentimentCount[s] || 0;
            const pct = total > 0 ? (cnt / total) * 100 : 0;
            return pct > 0 ? (
              <div
                key={s}
                className="stacked-segment"
                style={{ width: `${pct}%`, background: SENTIMENT_META[s].color }}
                title={`${SENTIMENT_META[s].label}: ${cnt} (${pct.toFixed(1)}%)`}
              />
            ) : null;
          })}
        </div>

        {/* Legend rows */}
        <div className="sentiment-rows">
          {sentiments.map((s) => {
            const cnt = sentimentCount[s] || 0;
            const pct = total > 0 ? ((cnt / total) * 100).toFixed(1) : "0.0";
            return (
              <div key={s} className="sentiment-row">
                <div className="sentiment-row-left">
                  <span className="legend-dot" style={{ background: SENTIMENT_META[s].color }} />
                  <span className="sentiment-row-label">{SENTIMENT_META[s].label}</span>
                </div>
                <div className="sentiment-row-right">
                  <div className="mini-bar-track">
                    <div
                      className="mini-bar-fill"
                      style={{ width: `${pct}%`, background: SENTIMENT_META[s].color }}
                    />
                  </div>
                  <span className="sentiment-row-count">{cnt}</span>
                  <span className="sentiment-row-pct">{pct}%</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Category distribution */}
      <div className="chart-card">
        <h3 className="chart-title">Category Distribution</h3>
        <div className="category-rows">
          {topCategories.length === 0 ? (
            <p className="chart-empty">No categories found.</p>
          ) : (
            topCategories.map(([cat, cnt]) => {
              const pct = ((cnt / maxCatCount) * 100).toFixed(1);
              return (
                <div key={cat} className="category-row">
                  <span className="category-row-name">{cat}</span>
                  <div className="category-bar-wrap">
                    <div className="category-bar-track">
                      <div
                        className="category-bar-fill"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <span className="category-row-count">{cnt}</span>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Top aspect terms */}
      <div className="chart-card chart-card-full">
        <h3 className="chart-title">Top Aspect Terms</h3>
        {topAspectTerms.length === 0 ? (
          <p className="chart-empty">No aspect terms found.</p>
        ) : (
          <div className="term-cloud">
            {topAspectTerms.map(([term, cnt]) => {
              const weight = cnt / maxTermFreq;
              const size   = 12 + Math.round(weight * 12);
              const alpha  = 0.15 + weight * 0.65;
              return (
                <span
                  key={term}
                  className="term-pill"
                  style={{ fontSize: `${size}px`, opacity: 0.6 + weight * 0.4 }}
                  title={`"${term}" — ${cnt} occurrence${cnt > 1 ? "s" : ""}`}
                >
                  {term}
                  <span className="term-freq" style={{ background: `rgba(79,70,229,${alpha})` }}>
                    {cnt}
                  </span>
                </span>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
