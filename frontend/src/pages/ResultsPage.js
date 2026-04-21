import React, { useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

export default function ResultsPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { result, previewUrl } = location.state || {};

  useEffect(() => {
    if (!result) {
      navigate('/upload');
    }
  }, [result, navigate]);

  if (!result) return null;

  const isPositive = result.prediction > 0.5;
  const confidence = result.confidence ?? Math.round(Math.max(result.prediction, 1 - result.prediction) * 100);

  const barColor = isPositive
    ? 'var(--color-danger)'
    : 'var(--color-success)';

  return (
    <main className="results-page container" role="main">
      <header className="page-header">
        <h1>Analysis Results</h1>
        <p className="text-muted">AI-powered TB screening with Grad-CAM explainability</p>
      </header>

      {/* Prediction Summary Card */}
      <div className="card mb-4" id="result-summary-card">
        <div
          className={`result-badge ${isPositive ? 'positive' : 'negative'}`}
          id="result-badge"
          role="status"
          aria-live="polite"
        >
          {isPositive ? 'TB Positive' : 'TB Negative'}
        </div>

        <p className="text-muted text-sm mb-4">
          Prediction Score: <strong style={{ color: 'var(--color-text)' }}>{(result.prediction * 100).toFixed(2)}%</strong>
          {' '}· Confidence: <strong style={{ color: 'var(--color-text)' }}>{confidence}%</strong>
          {result.processingTimeMs && (
            <> · Processing time: <strong style={{ color: 'var(--color-text)' }}>{result.processingTimeMs} ms</strong></>
          )}
        </p>

        {/* Confidence Bar */}
        <div style={{ marginBottom: 8 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
            <span className="text-sm text-muted">Confidence Level</span>
            <span className="text-sm" style={{ color: barColor, fontWeight: 600 }}>{confidence}%</span>
          </div>
          <div className="confidence-bar-track" role="progressbar" aria-valuenow={confidence} aria-valuemin={0} aria-valuemax={100}>
            <div
              id="confidence-bar"
              className="confidence-bar-fill"
              style={{ width: `${confidence}%`, background: barColor }}
            />
          </div>
        </div>

        {isPositive && (
          <div className="alert alert-error text-sm mt-4" role="alert">
            High probability of Tuberculosis detected. Please consult a qualified radiologist or physician for clinical confirmation.
          </div>
        )}
      </div>

      {/* Images Grid */}
      <div className="images-grid" id="images-grid">
        {/* Original */}
        {previewUrl && (
          <div className="card image-card">
            <h3>Original X-Ray</h3>
            <img
              id="result-original"
              src={previewUrl}
              alt="Original chest X-ray"
            />
          </div>
        )}

        {/* Heatmap */}
        {result.heatmap && (
          <div className="card image-card">
            <h3>Grad-CAM Heatmap</h3>
            <img
              id="result-heatmap"
              src={`data:image/jpeg;base64,${result.heatmap}`}
              alt="Grad-CAM heatmap showing regions influencing the prediction"
            />
            <p className="text-muted text-sm mt-4">
              Red areas indicate regions most influential in the prediction.
            </p>
          </div>
        )}

        {/* Overlay */}
        {result.overlay && (
          <div className="card image-card">
            <h3>Overlay (Heatmap + X-Ray)</h3>
            <img
              id="result-overlay"
              src={`data:image/jpeg;base64,${result.overlay}`}
              alt="Grad-CAM overlay blended with the original X-ray"
            />
            <p className="text-muted text-sm mt-4">
              Heatmap blended with original for contextual interpretation.
            </p>
          </div>
        )}
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', gap: 12, marginTop: 32, flexWrap: 'wrap' }}>
        <button id="analyze-new-btn" className="btn btn-primary btn-lg" onClick={() => navigate('/upload')}>
          Analyze New Image
        </button>
        <button
          id="download-report-btn"
          className="btn btn-ghost"
          onClick={() => {
            const report = {
              timestamp: new Date().toISOString(),
              prediction: result.prediction,
              label: result.label || (isPositive ? 'TB Positive' : 'TB Negative'),
              confidence,
              processingTimeMs: result.processingTimeMs,
            };
            const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `tb-report-${Date.now()}.json`;
            a.click();
          }}
        >
          Download Report
        </button>
      </div>

      <div className="divider" />
      <p className="text-sm text-muted text-center">
        This AI system is for research and educational purposes only. Results must be interpreted by a qualified medical professional.
      </p>
    </main>
  );
}
