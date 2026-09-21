import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';

export default function WaveformPlot({ visualizationData, seizureSegment }) {
  const plotRef = useRef(null);

  useEffect(() => {
    if (!plotRef.current || !visualizationData) return;

    const { channels, times, data } = visualizationData;
    if (!channels || !times || !data) return;

    // Stacked EEG layout calculations
    // Each channel is offset by a fixed margin so they appear as separate horizontal lines
    const offsetSpacing = 120; // spacing in microvolts/units
    const traces = [];
    const yTickVals = [];
    const yTickText = [];

    channels.forEach((ch, idx) => {
      // Calculate offset trace
      const offsetData = data[idx].map(val => val + (idx * offsetSpacing));
      
      traces.push({
        x: times,
        y: offsetData,
        name: ch,
        mode: 'lines',
        line: {
          width: 1.2,
          color: idx % 2 === 0 ? '#3B82F6' : '#60A5FA' // alternate blue colors
        },
        hoverinfo: 'x+y'
      });

      yTickVals.push(idx * offsetSpacing);
      yTickText.push(ch);
    });

    // Shapes for layout (highlighting seizure)
    const shapes = [];
    if (seizureSegment && seizureSegment.start !== null && seizureSegment.end !== null) {
      shapes.push({
        type: 'rect',
        xref: 'x',
        yref: 'paper', // spans full height of plot
        x0: seizureSegment.start,
        x1: seizureSegment.end,
        y0: 0,
        y1: 1,
        fillcolor: 'rgba(239, 68, 68, 0.2)', // Semi-transparent red highlight
        line: {
          width: 0
        }
      });
    }

    const maxTime = times[times.length - 1] || 0;
    const isVeryLong = maxTime > 1800; // 30 minutes

    const layout = {
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'rgba(15, 23, 42, 0.3)',
      xaxis: {
        title: {
          text: 'Time (seconds)',
          font: { color: '#94A3B8', size: 11 }
        },
        gridcolor: 'rgba(255, 255, 255, 0.05)',
        tickcolor: '#64748B',
        font: { color: '#94A3B8' },
        range: isVeryLong ? (
          (seizureSegment && seizureSegment.start !== null && seizureSegment.end !== null) ?
            [Math.max(0, seizureSegment.start - 30), Math.min(maxTime, seizureSegment.end + 30)] :
            [0, Math.min(120, maxTime)]
        ) : undefined
      },
      yaxis: {
        gridcolor: 'rgba(255, 255, 255, 0.03)',
        tickvals: yTickVals,
        ticktext: yTickText,
        tickcolor: '#64748B',
        font: { color: '#94A3B8', size: 9 },
        fixedrange: false
      },
      margin: {
        t: 40,
        b: 40,
        l: 80,
        r: 30
      },
      showlegend: false,
      shapes: shapes,
      dragmode: 'pan' // default to pan for easy waveform inspection
    };

    const config = {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['select2d', 'lasso2d'],
      displaylogo: false
    };

    Plotly.newPlot(plotRef.current, traces, layout, config);

    // Cleanup plot on unmount
    return () => {
      if (plotRef.current) {
        Plotly.purge(plotRef.current);
      }
    };
  }, [visualizationData, seizureSegment]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <div ref={plotRef} style={{ width: '100%', height: '100%' }} />
      {seizureSegment && seizureSegment.start !== null && (
        <div 
          style={{
            position: 'absolute',
            top: '12px',
            right: '12px',
            background: 'rgba(239, 68, 68, 0.9)',
            color: 'white',
            fontSize: '0.75rem',
            fontWeight: 'bold',
            padding: '4px 8px',
            borderRadius: '4px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.5)',
            zIndex: 5
          }}
        >
          Seizure Window: {seizureSegment.start.toFixed(1)}s - {seizureSegment.end.toFixed(1)}s
        </div>
      )}
    </div>
  );
}
