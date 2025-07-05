"""
Web-based monitoring dashboard for Ijon RAG system.

Provides real-time metrics visualization and system health monitoring.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from src.evaluation.evaluation_orchestrator import get_evaluation_orchestrator
from src.evaluation.metrics_collector import get_metrics_collector
from src.utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Ijon RAG Monitoring Dashboard")


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    orchestrator = get_evaluation_orchestrator()
    await orchestrator.initialize()
    logger.info("Monitoring dashboard started")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ijon RAG Monitoring</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background: #f5f5f5;
            }
            .header {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
            }
            .metric-card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .metric-value {
                font-size: 2em;
                font-weight: bold;
                color: #333;
            }
            .metric-label {
                color: #666;
                margin-top: 5px;
            }
            .status-healthy { color: #4CAF50; }
            .status-degraded { color: #FF9800; }
            .status-unhealthy { color: #F44336; }
            .chart-container {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-top: 20px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                padding: 10px;
                text-align: left;
                border-bottom: 1px solid #eee;
            }
            th {
                background: #f9f9f9;
                font-weight: 600;
            }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <div class="header">
            <h1>Ijon RAG System Monitoring</h1>
            <p id="last-update">Loading...</p>
        </div>
        
        <div class="metrics-grid" id="metrics-grid">
            <!-- Metrics will be populated here -->
        </div>
        
        <div class="chart-container">
            <h2>Performance Trends</h2>
            <canvas id="performance-chart"></canvas>
        </div>
        
        <div class="chart-container">
            <h2>Component Metrics</h2>
            <table id="metrics-table">
                <thead>
                    <tr>
                        <th>Component</th>
                        <th>Operation</th>
                        <th>Avg Latency (ms)</th>
                        <th>Success Rate</th>
                        <th>Count</th>
                    </tr>
                </thead>
                <tbody id="metrics-tbody">
                    <!-- Rows will be populated here -->
                </tbody>
            </table>
        </div>
        
        <script>
            // Initialize chart
            const ctx = document.getElementById('performance-chart').getContext('2d');
            const performanceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Avg Latency (ms)',
                            data: [],
                            borderColor: 'rgb(75, 192, 192)',
                            tension: 0.1
                        },
                        {
                            label: 'Success Rate (%)',
                            data: [],
                            borderColor: 'rgb(54, 162, 235)',
                            tension: 0.1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
            
            // Update functions
            async function updateDashboard() {
                try {
                    // Fetch health data
                    const healthResponse = await fetch('/api/health');
                    const health = await healthResponse.json();
                    
                    // Fetch metrics data
                    const metricsResponse = await fetch('/api/metrics');
                    const metrics = await metricsResponse.json();
                    
                    // Update header
                    document.getElementById('last-update').textContent = 
                        `Last updated: ${new Date().toLocaleString()} | Status: ${health.status}`;
                    
                    // Update metrics grid
                    updateMetricsGrid(health);
                    
                    // Update metrics table
                    updateMetricsTable(metrics);
                    
                    // Update chart
                    updateChart(metrics);
                    
                } catch (error) {
                    console.error('Failed to update dashboard:', error);
                }
            }
            
            function updateMetricsGrid(health) {
                const grid = document.getElementById('metrics-grid');
                grid.innerHTML = '';
                
                // Overall status card
                const statusCard = createMetricCard(
                    'System Status',
                    health.status.toUpperCase(),
                    '',
                    `status-${health.status}`
                );
                grid.appendChild(statusCard);
                
                // Component status cards
                for (const [component, data] of Object.entries(health.components)) {
                    const card = createMetricCard(
                        component.charAt(0).toUpperCase() + component.slice(1),
                        `${(data.success_rate * 100).toFixed(1)}%`,
                        'Success Rate',
                        `status-${data.status}`
                    );
                    grid.appendChild(card);
                }
            }
            
            function createMetricCard(title, value, label, statusClass = '') {
                const card = document.createElement('div');
                card.className = 'metric-card';
                card.innerHTML = `
                    <h3>${title}</h3>
                    <div class="metric-value ${statusClass}">${value}</div>
                    <div class="metric-label">${label}</div>
                `;
                return card;
            }
            
            function updateMetricsTable(metrics) {
                const tbody = document.getElementById('metrics-tbody');
                tbody.innerHTML = '';
                
                for (const metric of metrics.summary) {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${metric.component}</td>
                        <td>${metric.operation}</td>
                        <td>${metric.avg.toFixed(2)}</td>
                        <td>${((metric.success_rate || 0) * 100).toFixed(1)}%</td>
                        <td>${metric.count}</td>
                    `;
                    tbody.appendChild(row);
                }
            }
            
            function updateChart(metrics) {
                // Add new data point
                const now = new Date().toLocaleTimeString();
                performanceChart.data.labels.push(now);
                
                // Calculate average latency and success rate
                let totalLatency = 0;
                let totalSuccess = 0;
                let count = 0;
                
                for (const metric of metrics.summary) {
                    if (metric.metric_name === 'latency_ms') {
                        totalLatency += metric.avg;
                        count++;
                    }
                    if (metric.metric_name === 'success_rate') {
                        totalSuccess += metric.avg;
                    }
                }
                
                const avgLatency = count > 0 ? totalLatency / count : 0;
                const avgSuccess = metrics.summary.length > 0 ? 
                    (totalSuccess / metrics.summary.length) * 100 : 0;
                
                performanceChart.data.datasets[0].data.push(avgLatency);
                performanceChart.data.datasets[1].data.push(avgSuccess);
                
                // Keep only last 20 data points
                if (performanceChart.data.labels.length > 20) {
                    performanceChart.data.labels.shift();
                    performanceChart.data.datasets[0].data.shift();
                    performanceChart.data.datasets[1].data.shift();
                }
                
                performanceChart.update();
            }
            
            // Initial update and set interval
            updateDashboard();
            setInterval(updateDashboard, 5000);  // Update every 5 seconds
        </script>
    </body>
    </html>
    """


@app.get("/api/health")
async def get_health():
    """Get system health status."""
    orchestrator = get_evaluation_orchestrator()
    health = await orchestrator.get_system_health()
    return health


@app.get("/api/metrics")
async def get_metrics(
    component: Optional[str] = None,
    operation: Optional[str] = None,
    hours: int = 1
):
    """Get metrics summary."""
    collector = get_metrics_collector()
    summary = await collector.get_metrics_summary(component, operation, hours)
    return summary


@app.get("/api/metrics/live")
async def get_live_metrics():
    """Get live metrics from memory."""
    collector = get_metrics_collector()
    metrics = collector.get_aggregated_metrics()
    return metrics


def run_dashboard(host: str = "0.0.0.0", port: int = 8080):
    """Run the monitoring dashboard."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_dashboard()