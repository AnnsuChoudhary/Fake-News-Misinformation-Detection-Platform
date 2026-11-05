import { useEffect, useMemo, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const API_URL = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');

const modes = ['content', 'propagation', 'hybrid'];

function getBadgeTone(label) {
  if (label === 'Fake') return 'danger';
  if (label === 'Real') return 'success';
  return 'warning';
}

function App() {
  const [mode, setMode] = useState('propagation');
  const [text, setText] = useState('');
  const [graphJson, setGraphJson] = useState('{\n  "nodes": [{"id": 0}, {"id": 1}],\n  "edges": [[0, 1]]\n}');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [apiStatus, setApiStatus] = useState('checking');
  const [models, setModels] = useState(null);
  const [modelErrors, setModelErrors] = useState({});

  useEffect(() => {
    let active = true;

    async function checkApi() {
      try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Backend health check failed');
        if (active) {
          setModels(data.models || {});
          setModelErrors(data.errors || {});
          setApiStatus('online');
        }
      } catch {
        if (active) setApiStatus('offline');
      }
    }

    checkApi();
    const retryTimer = window.setInterval(checkApi, 5000);

    return () => {
      active = false;
      window.clearInterval(retryTimer);
    };
  }, []);

  const chartData = useMemo(() => {
    if (!result) return [];

    return Object.entries(result).map(([name, entry]) => {
      const confidence = entry?.confidence ?? 0;
      return { name: name.toUpperCase(), confidence };
    });
  }, [result]);

  async function runAnalysis(analysisMode = mode) {
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const endpoints = {
        content: '/predict/content',
        propagation: '/predict/propagation',
        hybrid: '/predict/hybrid',
        all: '/predict/all',
      };
      let requestBody = { text };

      if (analysisMode !== 'content') {
        let graph;
        try {
          graph = JSON.parse(graphJson);
          if (!graph || typeof graph !== 'object' || Array.isArray(graph)) throw new Error();
        } catch {
          throw new Error('Propagation data must be a JSON object with nodes and edges.');
        }
        requestBody = analysisMode === 'propagation' ? graph : { text, ...graph };
      }

      const response = await fetch(`${API_URL}${endpoints[analysisMode]}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || 'Request failed');
      }

      if (analysisMode === 'all') {
        setResult(data);
      } else {
        setResult({ [analysisMode]: data });
      }

      const summary = analysisMode === 'all' ? data.hybrid : data;
      const historyEntry = {
        id: Date.now(),
        mode: analysisMode,
        label: summary?.label || 'Unknown',
        confidence: summary?.confidence ?? 0,
        text: text.slice(0, 60),
      };
      setHistory((prev) => [historyEntry, ...prev].slice(0, 4));
    } catch (err) {
      setError(err.message || 'Unable to reach the backend service.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="hero-card">
        <div>
          <p className="eyebrow">AI-powered verification</p>
          <h1>Misinformation Detection System</h1>
          <p className="subtitle">Assess credibility from content and social propagation signals in one place.</p>
        </div>
        <div className={`api-status ${apiStatus}`} role="status">
          <span />
          {apiStatus === 'checking' ? 'Checking API…' : apiStatus === 'online' ? 'API connected' : 'API unavailable'}
        </div>
      </header>

      <section className="grid-layout">
        <div className="card">
          {apiStatus === 'online' ? (
            <div className="model-status-list" aria-label="Model availability">
              {Object.entries(models || {}).map(([name, available]) => (
                <span key={name} className={available ? 'available' : 'unavailable'} title={modelErrors[name] || undefined}>
                  {name}: {available ? 'ready' : 'unavailable'}
                </span>
              ))}
            </div>
          ) : null}
          <div className="mode-switcher">
            {modes.map((item) => {
              const available = models?.[item] !== false;
              return <button key={item} className={mode === item ? 'active' : ''} onClick={() => setMode(item)} disabled={!available} title={modelErrors[item] || undefined}>
                {item === 'content' ? 'Content Analysis' : item === 'propagation' ? 'Propagation Analysis' : 'Hybrid (Combined)'}
              </button>;
            })}
          </div>
          {apiStatus === 'online' && modelErrors[mode] ? <p className="model-warning">{modelErrors[mode]}</p> : null}

          <label className="field-label">Article or claim</label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste an article, post, or claim here..."
            rows={8}
          />

          <label className="field-label">Propagation data (JSON)</label>
          <textarea
            value={graphJson}
            onChange={(e) => setGraphJson(e.target.value)}
            rows={6}
            className="json-box"
          />

          <div className="actions">
            <button onClick={() => runAnalysis(mode)} disabled={loading || !text.trim()}>
              {loading ? 'Analyzing...' : 'Run Analysis'}
            </button>
            <button className="secondary" onClick={() => runAnalysis('all')} disabled={loading || !text.trim() || (models && !Object.values(models).every(Boolean))} title={Object.values(modelErrors).join('; ') || undefined}>
              Analyze with All Models
            </button>
          </div>

          {error ? <div className="error-box">{error}</div> : null}
        </div>

        <div className="card result-card">
          <h2>Results</h2>
          {loading ? (
            <div className="empty-state">Loading prediction…</div>
          ) : result ? (
            <>
              <div className="result-summary">
                {Object.entries(result).map(([key, value]) => {
                  const label = value?.label || 'Unknown';
                  const confidence = value?.confidence || 0;
                  return (
                    <div key={key} className="result-pill">
                      <span className="pill-title">{key.toUpperCase()}</span>
                      <span className={`status-badge ${getBadgeTone(label)}`}>{label}</span>
                      <div className="progress-bar">
                        <div style={{ width: `${confidence}%` }} />
                      </div>
                      <small>{confidence.toFixed(1)}%</small>
                    </div>
                  );
                })}
              </div>

              <div className="chart-box">
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Bar dataKey="confidence" fill="#4c6ef5" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          ) : (
            <div className="empty-state">No analysis has been run yet. Enter text and press the button.</div>
          )}
        </div>
      </section>

      <section className="card history-card">
        <h2>Recent Analysis</h2>
        <div className="history-list">
          {history.length === 0 ? <p className="empty-state">Session history will appear here.</p> : history.map((item) => (
            <div key={item.id} className="history-item">
              <div>
                <strong>{item.mode.toUpperCase()}</strong>
                <p>{item.text}</p>
              </div>
              <div className="history-meta">
                <span className={`status-badge ${getBadgeTone(item.label)}`}>{item.label}</span>
                <small>{item.confidence.toFixed(1)}%</small>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

export default App;
