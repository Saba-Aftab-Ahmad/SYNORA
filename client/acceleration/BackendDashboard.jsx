/**
 * BackendDashboard.jsx
 * US-06 – Backend status + training metrics dashboard (React)
 *
 * Props
 * ─────
 * backendInfo   BackendDetectionResult | null   from backendDetector.js
 * benchmark     { gpuMsPerEpoch, cpuMsPerEpoch, ratio } | null
 * metrics       TrainingMetrics | null           latest epoch metrics
 * statusMsg     string                           init / training status
 * onStartDemo   () => void                       trigger a demo training run
 */

import { useState, useEffect, useRef } from 'react';

// ── Palette constants ──────────────────────────────────────────────────────
const COLOR = {
  webgpu: { bg: '#EAF3DE', border: '#639922', text: '#3B6D11', dot: '#639922' },
  webnn: { bg: '#E1F5EE', border: '#1D9E75', text: '#0F6E56', dot: '#1D9E75' },
  cpu: { bg: '#FAEEDA', border: '#BA7517', text: '#854F0B', dot: '#BA7517' },
};

// ── Sub-components ─────────────────────────────────────────────────────────

function StatusDot({ active }) {
  return (
    <span style={{
      display: 'inline-block',
      width: 10, height: 10,
      borderRadius: '50%',
      background: active ? '#639922' : '#B4B2A9',
      boxShadow: active ? '0 0 0 3px #C0DD9740' : 'none',
      marginRight: 8,
      flexShrink: 0,
    }} />
  );
}

function BackendBadge({ backendInfo }) {
  if (!backendInfo) return null;
  const pal = COLOR[backendInfo.backend] ?? COLOR.cpu;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center',
      padding: '4px 12px',
      borderRadius: 20,
      fontSize: 12, fontWeight: 600, letterSpacing: '0.03em',
      background: pal.bg, border: `1.5px solid ${pal.border}`, color: pal.text,
      gap: 6,
    }}>
      <span style={{ width: 8, height: 8, borderRadius: '50%', background: pal.dot }} />
      {backendInfo.label}
    </span>
  );
}

function MetricCard({ label, value, sub, accent = false }) {
  return (
    <div style={{
      background: accent ? '#F2F6FC' : 'var(--color-background-secondary, #F7F6F2)',
      borderRadius: 10,
      padding: '14px 18px',
      minWidth: 120,
      flex: 1,
      border: accent ? '1.5px solid #B5D4F4' : '0.5px solid rgba(0,0,0,0.08)',
    }}>
      <div style={{ fontSize: 11, color: '#888780', marginBottom: 4, fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
        {label}
      </div>
      <div style={{ fontSize: 24, fontWeight: 600, color: accent ? '#185FA5' : '#2C2C2A', lineHeight: 1.1 }}>
        {value ?? '—'}
      </div>
      {sub && <div style={{ fontSize: 11, color: '#888780', marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function SpeedupBar({ ratio }) {
  if (!ratio) return null;
  const pct = Math.min((ratio / 4) * 100, 100); // 4× = full bar
  const meetsAC = ratio >= 2;
  const color = meetsAC ? '#639922' : '#BA7517';
  const label = meetsAC ? `✓ ${ratio.toFixed(1)}× faster than CPU` : `${ratio.toFixed(1)}× (target: ≥ 2×)`;

  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#5F5E5A', marginBottom: 6 }}>
        <span>GPU speed-up vs CPU</span>
        <span style={{ color, fontWeight: 600 }}>{label}</span>
      </div>
      <div style={{ height: 8, borderRadius: 4, background: '#D3D1C7', overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 4, transition: 'width 0.8s ease' }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#B4B2A9', marginTop: 4 }}>
        <span>1×</span><span>2× (AC)</span><span>3×</span><span>4×+</span>
      </div>
    </div>
  );
}

function LossChart({ history }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || history.length < 2) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const pad = { t: 12, r: 12, b: 28, l: 40 };
    const iW = W - pad.l - pad.r;
    const iH = H - pad.t - pad.b;

    ctx.clearRect(0, 0, W, H);

    const maxLoss = Math.max(...history.map(h => h.loss));
    const minLoss = Math.min(...history.map(h => h.loss));
    const range = maxLoss - minLoss || 1;

    const toX = (i) => pad.l + (i / (history.length - 1)) * iW;
    const toY = (v) => pad.t + iH - ((v - minLoss) / range) * iH;

    // Grid lines
    ctx.strokeStyle = '#D3D1C740';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + (i / 4) * iH;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    }

    // Loss line
    ctx.beginPath();
    ctx.strokeStyle = '#185FA5';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    history.forEach((h, i) => {
      i === 0 ? ctx.moveTo(toX(i), toY(h.loss)) : ctx.lineTo(toX(i), toY(h.loss));
    });
    ctx.stroke();

    // Dots
    ctx.fillStyle = '#185FA5';
    history.forEach((h, i) => {
      ctx.beginPath();
      ctx.arc(toX(i), toY(h.loss), 3, 0, Math.PI * 2);
      ctx.fill();
    });

    // Y-axis labels
    ctx.fillStyle = '#888780';
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const v = minLoss + (1 - i / 4) * range;
      ctx.fillText(v.toFixed(3), pad.l - 6, pad.t + (i / 4) * iH + 4);
    }

    // X-axis labels
    ctx.textAlign = 'center';
    history.forEach((h, i) => {
      if (i % Math.ceil(history.length / 5) === 0 || i === history.length - 1) {
        ctx.fillText(`Ep ${h.epoch}`, toX(i), H - 8);
      }
    });
  }, [history]);

  if (history.length < 2) return null;

  return (
    <div style={{ marginTop: 20 }}>
      <div style={{ fontSize: 12, color: '#5F5E5A', marginBottom: 8, fontWeight: 500 }}>Training loss</div>
      <canvas ref={canvasRef} width={560} height={140} style={{ width: '100%', display: 'block' }} />
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────

/**
 * @param {{
 *   backendInfo:  import('./backendDetector').BackendDetectionResult | null,
 *   benchmark:    { gpuMsPerEpoch: number, cpuMsPerEpoch: number, ratio: number } | null,
 *   metrics:      import('../../src/acceleration/federatedTrainer').TrainingMetrics | null,
 *   statusMsg:    string,
 *   onStartDemo?: () => void,
 * }} props
 */
export default function BackendDashboard({ backendInfo, benchmark, metrics, statusMsg, onStartDemo }) {
  const [lossHistory, setLossHistory] = useState([]);

  useEffect(() => {
    if (!metrics) return;
    setLossHistory(prev =>
      metrics.epoch === 1
        ? [metrics]
        : [...prev.filter(h => h.epoch < metrics.epoch), metrics]
    );
  }, [metrics]);

  const isTraining = metrics && metrics.epoch < metrics.totalEpochs;
  const progress = metrics ? (metrics.epoch / metrics.totalEpochs) * 100 : 0;

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 620, margin: '0 auto', padding: '24px 0' }}>

      {/* ── Header ── */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 24 }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 18, fontWeight: 600, color: '#2C2C2A' }}>
            Acceleration Dashboard
          </h2>
          <p style={{ margin: '4px 0 0', fontSize: 13, color: '#888780' }}>
            US-06 · Hardware backend detection &amp; federated training
          </p>
        </div>
        <BackendBadge backendInfo={backendInfo} />
      </div>

      {/* ── Status bar ── */}
      <div style={{
        display: 'flex', alignItems: 'center',
        background: 'var(--color-background-secondary, #F7F6F2)',
        border: '0.5px solid rgba(0,0,0,0.08)',
        borderRadius: 8, padding: '10px 14px', marginBottom: 20,
        fontSize: 13, color: '#5F5E5A', gap: 8,
      }}>
        <StatusDot active={!!backendInfo} />
        <span>{statusMsg || 'Initialising…'}</span>
        {onStartDemo && !isTraining && backendInfo && (
          <button
            onClick={onStartDemo}
            style={{
              marginLeft: 'auto', padding: '5px 14px', fontSize: 12,
              borderRadius: 6, border: '0.5px solid rgba(0,0,0,0.18)',
              background: 'transparent', cursor: 'pointer', color: '#2C2C2A',
            }}
          >
            Run demo ↗
          </button>
        )}
      </div>

      {/* ── Backend capability cards ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginBottom: 20 }}>
        <MetricCard
          label="Active backend"
          value={backendInfo?.label ?? '—'}
          sub={backendInfo?.isGPU ? 'Hardware accelerated' : 'Software fallback'}
          accent={backendInfo?.isGPU}
        />
        <MetricCard
          label="WebGPU"
          value={backendInfo == null ? '—' : backendInfo.webgpuAvailable ? 'Available' : 'Not found'}
          sub="navigator.gpu"
          accent={backendInfo?.webgpuAvailable}
        />
        <MetricCard
          label="WebNN"
          value={backendInfo == null ? '—' : backendInfo.webnnAvailable ? 'Available' : 'Not found'}
          sub="navigator.ml"
          accent={backendInfo?.webnnAvailable}
        />
      </div>

      {/* ── Speed benchmark ── */}
      {benchmark && (
        <div style={{
          background: 'var(--color-background-secondary, #F7F6F2)',
          border: '0.5px solid rgba(0,0,0,0.08)',
          borderRadius: 10, padding: '16px 18px', marginBottom: 20,
        }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#444441', marginBottom: 12 }}>
            Speed benchmark (512 samples, {3} epochs)
          </div>
          <div style={{ display: 'flex', gap: 10, marginBottom: 4 }}>
            <MetricCard
              label="GPU ms / epoch"
              value={`${Math.round(benchmark.gpuMsPerEpoch)} ms`}
              accent
            />
            <MetricCard
              label="CPU ms / epoch"
              value={`${Math.round(benchmark.cpuMsPerEpoch)} ms`}
            />
          </div>
          <SpeedupBar ratio={benchmark.ratio} />
        </div>
      )}

      {/* ── Training progress ── */}
      {metrics && (
        <div style={{
          background: 'var(--color-background-secondary, #F7F6F2)',
          border: '0.5px solid rgba(0,0,0,0.08)',
          borderRadius: 10, padding: '16px 18px',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: '#444441' }}>
              Training — epoch {metrics.epoch} / {metrics.totalEpochs}
            </span>
            {isTraining && (
              <span style={{ fontSize: 11, color: '#888780' }}>running on {metrics.backendLabel}</span>
            )}
          </div>

          {/* Progress bar */}
          <div style={{ height: 6, borderRadius: 3, background: '#D3D1C7', marginBottom: 14, overflow: 'hidden' }}>
            <div style={{
              height: '100%', width: `${progress}%`,
              background: backendInfo?.isGPU ? '#185FA5' : '#BA7517',
              borderRadius: 3, transition: 'width 0.4s ease',
            }} />
          </div>

          <div style={{ display: 'flex', gap: 10, marginBottom: 8 }}>
            <MetricCard label="Loss" value={metrics.loss.toFixed(4)} accent />
            <MetricCard label="Accuracy" value={metrics.acc != null ? `${(metrics.acc * 100).toFixed(1)}%` : '—'} />
            <MetricCard label="Backend" value={metrics.backendLabel} />
          </div>

          <LossChart history={lossHistory} />
        </div>
      )}

      {/* ── Reason / fallback note ── */}
      {backendInfo && (
        <p style={{ fontSize: 11, color: '#B4B2A9', marginTop: 14, lineHeight: 1.5 }}>
          {backendInfo.reason}
        </p>
      )}
    </div>
  );
}