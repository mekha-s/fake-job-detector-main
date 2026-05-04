import { useState, useEffect } from 'react';
import './index.css';

function SplashPage({ onEnter }) {
  const [glitch, setGlitch] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setGlitch(true);
      setTimeout(() => setGlitch(false), 150);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="splash" onClick={onEnter}>
      <div className="splash-bg">
        <div className="scan-line"></div>
        <div className="grid-overlay"></div>
      </div>

      <div className="splash-content">
        <div className="warning-badge">⚠ THREAT INTELLIGENCE SYSTEM</div>
        <h1 className={`splash-title ${glitch ? 'glitch' : ''}`} data-text="COUNTERFEIT PREDICTION">
          COUNTERFEIT PREDICTION
        </h1>
        <p className="splash-subtitle">BASED ON JOBS</p>
        <div className="splash-divider"></div>
        <p className="splash-hint">CLICK ANYWHERE TO INITIATE ANALYSIS</p>
        <div className="pulse-ring"></div>
      </div>

      <div className="splash-footer">
        <span>AI-POWERED FRAUD DETECTION</span>
        <span className="dot-sep">◆</span>
        <span>DISTILBERT NEURAL ENGINE</span>
        <span className="dot-sep">◆</span>
        <span>REAL-TIME THREAT SCORING</span>
      </div>
    </div>
  );
}

function AnalyzerPage() {
  const [text, setText] = useState('');
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      const reader = new FileReader();
      reader.onload = (ev) => setPreview(ev.target.result);
      reader.readAsDataURL(selected);
    }
  };

  const analyze = async () => {
    if (!text.trim() && !file) {
      setError('Provide job text or upload an image.');
      return;
    }
    setError(null);
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    if (text.trim()) formData.append('text', text);
    if (file) formData.append('image', file);

    try {
      const response = await fetch('http://127.0.0.1:8001/analyze', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
      }
    } catch (err) {
      setError('Cannot connect to backend. Ensure uvicorn is running.');
    } finally {
      setLoading(false);
    }
  };

  const getPredictionClass = (pred) => {
    if (pred === 'REAL JOB') return 'pred-real';
    if (pred === 'SUSPICIOUS') return 'pred-suspicious';
    return 'pred-fake';
  };

  const createHighlightedText = (extractedText, words) => {
    if (!words || words.length === 0) return { __html: extractedText };
    let newText = extractedText;
    words.forEach(word => {
      const safeWord = word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const reg = new RegExp(safeWord, 'gi');
      newText = newText.replace(reg, `<span class="highlight">$&</span>`);
    });
    return { __html: newText };
  };

  const getScoreColor = (score) => {
    if (score >= 66) return '#ef4444';
    if (score >= 35) return '#f59e0b';
    return '#22c55e';
  };

  return (
    <div className="analyzer-page">
      <header className="analyzer-header">
        <div className="header-left">
          <span className="header-badge">⚠</span>
          <div>
            <h1 className="header-title">COUNTERFEIT PREDICTION</h1>
            <p className="header-sub">BASED ON JOBS — THREAT ANALYSIS CONSOLE</p>
          </div>
        </div>
        <div className="status-indicator">
          <span className="status-dot"></span>
          SYSTEM ACTIVE
        </div>
      </header>

      <div className="analyzer-body">
        {/* Input Panel */}
        <div className="panel input-panel">
          <div className="panel-header">
            <span className="panel-icon">📋</span>
            <span>INPUT TARGET DATA</span>
          </div>

          <label className="field-label">PASTE JOB DESCRIPTION</label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter the job description text for threat analysis..."
          />

          <div className="or-divider"><span>OR</span></div>

          <label className="field-label">UPLOAD POSTER IMAGE</label>
          <label className="file-upload">
            <input type="file" onChange={handleFileChange} accept="image/*" />
            <span>📎 Upload Job Poster Image</span>
          </label>
          {preview && <img src={preview} alt="Preview" className="preview-image" />}

          {error && <div className="error-box">⚠ {error}</div>}

          <button className="analyze-btn" onClick={analyze} disabled={loading}>
            {loading ? (
              <><span className="loader"></span> SCANNING...</>
            ) : (
              '⚡ RUN THREAT ANALYSIS'
            )}
          </button>
        </div>

        {/* Result Panel */}
        {result && (
          <div className="panel result-panel fadeIn">
            <div className="panel-header">
              <span className="panel-icon">🎯</span>
              <span>THREAT ASSESSMENT REPORT</span>
            </div>

            {/* Prediction Badge */}
            <div className="prediction-block">
              <div className={`prediction-badge ${getPredictionClass(result.prediction)}`}>
                {result.prediction === 'FAKE JOB' && '☠ '}
                {result.prediction === 'REAL JOB' && '✓ '}
                {result.prediction === 'SUSPICIOUS' && '⚠ '}
                {result.prediction}
              </div>
              <div className="score-display">
                <div className="score-label">FRAUD PROBABILITY</div>
                <div className="score-bar-track">
                  <div
                    className="score-bar-fill"
                    style={{
                      width: `${result.score}%`,
                      background: `linear-gradient(90deg, ${getScoreColor(result.score)}, ${getScoreColor(result.score)}88)`,
                    }}
                  ></div>
                </div>
                <div className="score-number" style={{ color: getScoreColor(result.score) }}>
                  {result.score}%
                </div>
              </div>
            </div>

            {/* Info Grid - Conditional Visibility */}
            <div className="info-grid">
              {result.prediction === 'REAL JOB' && (
                <div className="info-card">
                  <div className="info-card-label">HIRING INTENTION</div>
                  <div className="info-card-value">{result.intention}</div>
                </div>
              )}
              {result.prediction !== 'REAL JOB' && (
                <div className="info-card">
                  <div className="info-card-label">SCAM PATTERN</div>
                  <div className="info-card-value">{result.pattern}</div>
                </div>
              )}
            </div>

            {/* Forensic Analysis */}
            <div className="analysis-section">
              <div className="section-title">
                {result.prediction === 'REAL JOB' ? '🛡 SAFETY ANALYSIS' : '🔬 FRAUD INDICATORS'}
              </div>
              <ul className="analysis-list">
                {result.analysis && result.analysis.map((a, idx) => (
                  <li key={idx}>
                    {a.startsWith('CRITICAL') || a.startsWith('EVIDENCE') || a.startsWith('RISK') ? '🔴 ' : '🔵 '}
                    {a}
                  </li>
                ))}
              </ul>
            </div>

            {/* Suspicious Words */}
            <div className="analysis-section">
              <div className="section-title">🚨 SUSPICIOUS KEYWORDS</div>
              <div className="tags-container">
                {result.words && result.words.length > 0 ? (
                  result.words.map((w, idx) => (
                    <span key={idx} className="threat-tag">{w}</span>
                  ))
                ) : (
                  <span className="no-threat">No suspicious keywords detected</span>
                )}
              </div>
            </div>

            {/* OCR Text */}
            {result.ocr_text && (
              <div className="analysis-section">
                <div className="section-title">📷 EXTRACTED OCR TEXT</div>
                <p
                  className="highlighted-text"
                  dangerouslySetInnerHTML={createHighlightedText(result.ocr_text, result.words)}
                />
              </div>
            )}

            {/* Input Text Highlighted */}
            {text.trim() && (
              <div className="analysis-section">
                <div className="section-title">📄 TEXT WITH THREAT HIGHLIGHTS</div>
                <p
                  className="highlighted-text"
                  dangerouslySetInnerHTML={createHighlightedText(text, result.words)}
                />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function App() {
  const [page, setPage] = useState('splash');

  return (
    <>
      {page === 'splash' ? (
        <SplashPage onEnter={() => setPage('analyzer')} />
      ) : (
        <AnalyzerPage />
      )}
    </>
  );
}

export default App;
