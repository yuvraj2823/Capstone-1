import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

export default function ResultsPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { result, previewUrl } = location.state || {};

  const [controls, setControls] = useState({
    brightness: 100,
    contrast: 100,
    zoom: 1,
    panX: 0,
    panY: 0,
    invert: false,
    heatmapOpacity: 0.5,
  });

  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  useEffect(() => {
    if (!result) {
      navigate('/upload');
    }
  }, [result, navigate]);

  if (!result) return null;

  const handleZoom = (delta) => setControls(c => ({ ...c, zoom: Math.max(0.5, Math.min(3, c.zoom + delta)) }));
  
  const handleMouseDown = (e) => {
    e.preventDefault();
    setIsDragging(true);
    setDragStart({ x: e.clientX - controls.panX, y: e.clientY - controls.panY });
  };
  
  const handleMouseMove = (e) => {
    if (!isDragging) return;
    setControls(c => ({ ...c, panX: e.clientX - dragStart.x, panY: e.clientY - dragStart.y }));
  };
  
  const handleMouseUp = () => setIsDragging(false);

  const imgStyle = {
    filter: `brightness(${controls.brightness}%) contrast(${controls.contrast}%) ${controls.invert ? 'invert(100%)' : ''}`,
    transform: `scale(${controls.zoom}) translate(${controls.panX / controls.zoom}px, ${controls.panY / controls.zoom}px)`,
    transition: isDragging ? 'none' : 'transform 0.1s ease',
    transformOrigin: 'center center',
    width: '100%',
    display: 'block',
    cursor: isDragging ? 'grabbing' : 'grab'
  };

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

      {/* Interactive Image Viewer */}
      <div className="card" style={{ padding: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        
        {/* Controls Toolbar */}
        <div style={{ background: 'var(--color-bg)', padding: '16px', borderBottom: '1px solid var(--color-border)', display: 'flex', flexWrap: 'wrap', gap: '20px', alignItems: 'center' }}>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button className="btn btn-ghost" style={{ padding: '6px 12px' }} onClick={() => handleZoom(0.2)} title="Zoom In">+</button>
            <button className="btn btn-ghost" style={{ padding: '6px 12px' }} onClick={() => handleZoom(-0.2)} title="Zoom Out">-</button>
            <button 
              className="btn btn-ghost" 
              style={{ padding: '6px 12px' }} 
              onClick={() => setControls({
                brightness: 100,
                contrast: 100,
                zoom: 1,
                panX: 0,
                panY: 0,
                invert: false,
                heatmapOpacity: 0.5,
              })} 
              title="Reset All Controls"
            >
              Reset
            </button>
            <button className="btn btn-ghost" style={{ padding: '6px 12px', background: controls.invert ? 'var(--color-primary)' : 'transparent', color: controls.invert ? '#000' : 'var(--color-muted)' }} onClick={() => setControls(c => ({...c, invert: !c.invert}))} title="Invert Colors">Invert</button>
          </div>
          
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flex: 1, minWidth: '200px' }}>
            <label className="text-sm text-muted" style={{ minWidth: '75px' }}>Brightness:</label>
            <input type="range" min="50" max="200" value={controls.brightness} onChange={e => setControls(c => ({...c, brightness: Number(e.target.value)}))} style={{ flex: 1 }} />
          </div>
          
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flex: 1, minWidth: '200px' }}>
            <label className="text-sm text-muted" style={{ minWidth: '75px' }}>Contrast:</label>
            <input type="range" min="50" max="200" value={controls.contrast} onChange={e => setControls(c => ({...c, contrast: Number(e.target.value)}))} style={{ flex: 1 }} />
          </div>

          {result.heatmap && (
            <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flex: 1, minWidth: '200px' }}>
              <label className="text-sm text-muted" style={{ minWidth: '100px' }}>Heatmap Opacity:</label>
              <input type="range" min="0" max="1" step="0.05" value={controls.heatmapOpacity} onChange={e => setControls(c => ({...c, heatmapOpacity: Number(e.target.value)}))} style={{ flex: 1 }} />
            </div>
          )}
        </div>

        {/* Image Display Area */}
        <div 
          style={{ height: '600px', backgroundColor: '#000', overflow: 'hidden', position: 'relative' }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          <div style={{ position: 'relative', width: '100%', height: '100%' }}>
            {previewUrl && (
              <img
                src={previewUrl}
                alt="Original chest X-ray"
                style={{ ...imgStyle, position: 'absolute', inset: 0, height: '100%', objectFit: 'contain', pointerEvents: 'none' }}
                draggable={false}
              />
            )}
            
            {result.heatmap && controls.heatmapOpacity > 0 && (
              <img
                src={`data:image/jpeg;base64,${result.heatmap}`}
                alt="Grad-CAM Heatmap Overlay"
                style={{ ...imgStyle, position: 'absolute', inset: 0, height: '100%', objectFit: 'contain', opacity: controls.heatmapOpacity, mixBlendMode: 'screen', pointerEvents: 'none' }}
                draggable={false}
              />
            )}
          </div>
        </div>
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', gap: 12, marginTop: 32, flexWrap: 'wrap' }}>
        <button id="analyze-new-btn" className="btn btn-primary btn-lg" onClick={() => navigate('/upload')}>
          Analyze New Image
        </button>
        <button
          id="download-report-btn"
          className="btn btn-ghost"
          onClick={async () => {
            if (result.id) {
              try {
                // Import api dynamically or rely on a top-level import
                const api = require('../services/api').default;
                const response = await api.get(`/history/${result.id}/report`, {
                  responseType: 'blob',
                });
                
                const url = window.URL.createObjectURL(new Blob([response.data]));
                const link = document.createElement('a');
                link.href = url;
                link.setAttribute('download', `TB-Report-${result.id}.pdf`);
                document.body.appendChild(link);
                link.click();
                link.remove();
              } catch (err) {
                alert('Failed to download PDF report. Try again later.');
              }
            } else {
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
            }
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
