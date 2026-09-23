import React, { useState, useEffect, useRef } from 'react';
import { 
  Droplets, Thermometer, CloudRain, Wind, 
  Sprout, Cpu, Database, Server, Activity, 
  UploadCloud, CheckCircle, AlertTriangle 
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import './index.css';

const API_BASE = 'http://localhost:8000/api';

function App() {
  const [dashboardData, setDashboardData] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [healthPrediction, setHealthPrediction] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const fileInputRef = useRef(null);

  const fetchDashboardData = async () => {
    try {
      const response = await fetch(`${API_BASE}/dashboard/summary`);
      if (!response.ok) throw new Error('API unavailable');
      const data = await response.json();
      setDashboardData(data);
      setError(null);
      
      // Update history for chart
      const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second:'2-digit' });
      setHistory(prev => {
        const newHistory = [...prev, {
          time: timeStr,
          moisture: data.current_sensors.soil_moisture,
          temperature: data.current_sensors.temperature
        }];
        return newHistory.slice(-10); // keep last 10 readings
      });
    } catch (err) {
      console.error(err);
      setError('Cannot connect to FastAPI backend. Please ensure it is running on http://localhost:8000.');
    }
  };

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 6000);
    return () => clearInterval(interval);
  }, []);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setHealthPrediction(null);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_BASE}/crop-health/predict`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('Prediction failed');
      const result = await response.json();
      setHealthPrediction(result);
    } catch (err) {
      console.error(err);
      alert('Failed to analyze image. Ensure FastAPI is running.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  if (!dashboardData && !error) {
    return <div className="loading-state">Initializing Smart Agriculture System...</div>;
  }

  const sensors = dashboardData?.current_sensors || {};
  const rec = dashboardData?.irrigation_recommendation || {};

  return (
    <div className="dashboard-container">
      {/* HEADER */}
      <header className="header">
        <div className="header-titles">
          <h1>AI-Powered Smart Irrigation & Crop Health</h1>
          <p>AI + IoT Smart Agriculture Platform</p>
        </div>
        <div className="status-indicator">
          {error ? (
            <><AlertTriangle size={18} color="var(--danger)" /> System Offline</>
          ) : (
            <><div className="status-dot"></div> System Online</>
          )}
        </div>
      </header>

      {error && <div className="error-message">{error}</div>}

      {/* SENSOR OVERVIEW */}
      <h2 className="section-title"><Activity size={20} /> Live Virtual Sensor Data</h2>
      <div className="grid-layout">
        <div className="card sensor-card">
          <div className="sensor-header">
            <span>Soil Moisture</span>
            <Droplets size={20} color="var(--primary)" />
          </div>
          <div className="sensor-value">
            {sensors.soil_moisture?.toFixed(1) || '--'} <span className="sensor-unit">%</span>
          </div>
        </div>
        <div className="card sensor-card">
          <div className="sensor-header">
            <span>Temperature</span>
            <Thermometer size={20} color="var(--warning)" />
          </div>
          <div className="sensor-value">
            {sensors.temperature?.toFixed(1) || '--'} <span className="sensor-unit">°C</span>
          </div>
        </div>
        <div className="card sensor-card">
          <div className="sensor-header">
            <span>Humidity</span>
            <Wind size={20} color="var(--secondary)" />
          </div>
          <div className="sensor-value">
            {sensors.humidity?.toFixed(1) || '--'} <span className="sensor-unit">%</span>
          </div>
        </div>
        <div className="card sensor-card">
          <div className="sensor-header">
            <span>Rainfall</span>
            <CloudRain size={20} color="#7f8c8d" />
          </div>
          <div className="sensor-value">
            {sensors.rainfall?.toFixed(1) || '--'} <span className="sensor-unit">mm/hr</span>
          </div>
        </div>
      </div>

      <div className="grid-layout" style={{ gridTemplateColumns: '1fr 1fr' }}>
        {/* IRRIGATION RECOMMENDATION */}
        <div className={`card recommendation-card ${rec.irrigation_required ? 'irrigate' : 'do-not-irrigate'}`}>
          <h2 className="section-title"><Sprout size={20} /> Irrigation Engine (AI)</h2>
          <div className={`rec-status ${rec.irrigation_required ? 'yes' : 'no'}`}>
            {rec.irrigation_required ? 'Irrigation Required' : 'Irrigation Not Required'}
          </div>
          <p style={{ color: 'var(--text-muted)' }}>{rec.reason}</p>
          
          <div className="rec-details">
            <p><strong>ML Prediction:</strong> {rec.ml_prediction === 1 ? 'Irrigate' : 'Do Not Irrigate'}</p>
            <p><strong>Confidence:</strong> {rec.confidence ? (rec.confidence * 100).toFixed(1) + '%' : 'N/A'}</p>
          </div>
        </div>

        {/* SENSOR TREND CHART */}
        <div className="card">
          <h2 className="section-title">Sensor Trend</h2>
          <div style={{ width: '100%', height: 200 }}>
            <ResponsiveContainer>
              <LineChart data={history} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="time" tick={{fontSize: 12}} />
                <YAxis yAxisId="left" tick={{fontSize: 12}} domain={['auto', 'auto']} />
                <YAxis yAxisId="right" orientation="right" tick={{fontSize: 12}} domain={['auto', 'auto']} />
                <Tooltip />
                <Legend iconType="circle" wrapperStyle={{fontSize: 12}} />
                <Line yAxisId="left" type="monotone" dataKey="moisture" stroke="var(--primary)" name="Moisture (%)" strokeWidth={2} dot={false} />
                <Line yAxisId="right" type="monotone" dataKey="temperature" stroke="var(--warning)" name="Temp (°C)" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* CROP HEALTH ANALYSIS */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <h2 className="section-title"><Sprout size={20} /> Tomato Crop Health Analysis</h2>
        <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
          <div style={{ flex: '1', minWidth: '300px' }}>
            <div className="upload-area" onClick={handleUploadClick}>
              <UploadCloud size={32} color="var(--primary)" style={{ marginBottom: '0.5rem' }} />
              <p>Click to select a tomato leaf image</p>
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileSelect} 
                accept="image/*" 
                style={{ display: 'none' }} 
              />
            </div>
            <button 
              className="btn" 
              onClick={handleAnalyze} 
              disabled={!selectedFile || isAnalyzing}
              style={{ width: '100%', justifyContent: 'center' }}
            >
              {isAnalyzing ? 'Analyzing...' : 'Analyze Crop Health'}
            </button>
          </div>
          
          <div style={{ flex: '1', minWidth: '300px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
            {previewUrl ? (
              <img src={previewUrl} alt="Preview" className="preview-image" />
            ) : (
              <p style={{ color: 'var(--text-muted)' }}>Image preview will appear here</p>
            )}
          </div>

          <div style={{ flex: '1', minWidth: '300px' }}>
            {healthPrediction ? (
              <div className="rec-details" style={{ 
                height: '100%', 
                marginTop: 0, 
                border: healthPrediction.predicted_class.toLowerCase().includes('healthy') ? '2px solid var(--success)' : '2px solid #f59e0b',
                backgroundColor: healthPrediction.predicted_class.toLowerCase().includes('healthy') ? '#f0fdf4' : '#fffbeb'
              }}>
                <h3 style={{ marginBottom: '1rem', color: 'var(--primary-dark)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  {healthPrediction.predicted_class.toLowerCase().includes('healthy') ? <CheckCircle color="var(--success)" /> : <AlertTriangle color="#d97706" />}
                  AI Model Prediction
                </h3>
                
                <div style={{ marginBottom: '1rem' }}>
                  <div style={{ fontSize: '1.4rem', fontWeight: 700, color: healthPrediction.predicted_class.toLowerCase().includes('healthy') ? 'var(--success)' : '#d97706' }}>
                    {healthPrediction.predicted_class.toLowerCase().includes('healthy') ? 'HEALTHY' : 'DISEASE DETECTED'}
                  </div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 500, marginTop: '0.25rem', color: 'var(--text-main)' }}>
                    {healthPrediction.predicted_class.toLowerCase().includes('healthy') 
                      ? 'No disease detected by the model.' 
                      : `Predicted condition: ${healthPrediction.predicted_class.replace(/___/g, ' - ').replace(/_/g, ' ')}`}
                  </div>
                </div>

                <div style={{ marginBottom: '1rem', padding: '0.75rem', backgroundColor: 'rgba(0,0,0,0.03)', borderRadius: '6px' }}>
                  <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>Confidence</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 600 }}>{(healthPrediction.confidence * 100).toFixed(1)}%</div>
                  {healthPrediction.confidence < 0.5 && (
                    <div style={{ fontSize: '0.85rem', color: '#856404', marginTop: '0.25rem', display: 'flex', gap: '0.25rem', alignItems: 'flex-start' }}>
                      <AlertTriangle size={14} style={{ marginTop: '2px', flexShrink: 0 }}/>
                      <span>Low model confidence — prediction should be interpreted cautiously.</span>
                    </div>
                  )}
                </div>
                
                <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
                  <strong>Crop:</strong> Tomato <br/>
                  <strong>Model:</strong> ResNet18
                </div>
                
                <h4 style={{ marginTop: '1rem', fontSize: '0.9rem', color: 'var(--text-muted)', borderTop: '1px solid var(--border)', paddingTop: '1rem' }}>Top Probabilities:</h4>
                <div style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
                  {Object.entries(healthPrediction.probabilities)
                    .sort(([,a], [,b]) => b - a)
                    .slice(0, 3)
                    .map(([cls, prob]) => (
                      <div key={cls} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.2rem' }}>
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '70%' }}>
                          {cls.replace(/___/g, ' - ').replace(/_/g, ' ')}
                        </span>
                        <span>{(prob * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                </div>
              </div>
            ) : (
              <div className="rec-details" style={{ height: '100%', marginTop: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
                Upload an image to see the prediction.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* SYSTEM PIPELINE */}
      <div className="pipeline-section">
        <h2 className="section-title"><Database size={20} /> System Architecture Pipeline</h2>
        
        <div className="pipeline-flow">
          <div className="pipeline-node">
            <Cpu size={24} color="var(--secondary)" />
            <span>Virtual Sensors</span>
          </div>
          <div className="pipeline-arrow">→</div>
          <div className="pipeline-node">
            <Server size={24} color="var(--primary)" />
            <span>FastAPI Backend</span>
          </div>
          <div className="pipeline-arrow">→</div>
          <div className="pipeline-node">
            <Database size={24} color="var(--warning)" />
            <span>AI Recommendation Engine</span>
          </div>
          <div className="pipeline-arrow">→</div>
          <div className="pipeline-node">
            <Activity size={24} color="var(--success)" />
            <span>Dashboard View</span>
          </div>
        </div>

        <div className="pipeline-flow" style={{ marginTop: '1rem' }}>
          <div className="pipeline-node">
            <UploadCloud size={24} color="var(--text-muted)" />
            <span>Leaf Image</span>
          </div>
          <div className="pipeline-arrow">→</div>
          <div className="pipeline-node">
            <Server size={24} color="var(--primary)" />
            <span>FastAPI Backend</span>
          </div>
          <div className="pipeline-arrow">→</div>
          <div className="pipeline-node">
            <Sprout size={24} color="var(--success)" />
            <span>ResNet18 AI Model</span>
          </div>
          <div className="pipeline-arrow">→</div>
          <div className="pipeline-node">
            <CheckCircle size={24} color="var(--primary-light)" />
            <span>Health Prediction</span>
          </div>
        </div>
      </div>

    </div>
  );
}

export default App;
