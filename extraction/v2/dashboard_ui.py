"""
Unified dashboard UI for extraction monitoring and management.

Extends the validation UI to provide comprehensive extraction monitoring.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from jinja2 import Template

from extraction.v2.state import StateStore, ExtractionState
from extraction.models import ExtractedKnowledge
from src.utils.logging import get_logger

logger = get_logger(__name__)

# FastAPI app for dashboard
app = FastAPI(title="Ijon Extraction Dashboard", version="2.0")

# State store for extraction management
state_store = StateStore()

# Metrics storage (simple in-memory for now)
metrics_store = {
    "total_extractions": 0,
    "successful_extractions": 0,
    "failed_extractions": 0,
    "average_quality_score": 0.0,
    "extraction_times": [],
    "model_usage": {},
    "enhancer_usage": {},
    "hourly_stats": {}
}


class ExtractionListItem(BaseModel):
    """Summary of an extraction for list views."""
    id: str
    pdf_path: str
    status: str
    quality_score: Optional[float]
    created_at: str
    updated_at: str
    topics_count: int = 0
    facts_count: int = 0


class SystemMetrics(BaseModel):
    """System-wide metrics."""
    total_extractions: int
    successful_extractions: int
    failed_extractions: int
    pending_extractions: int
    average_quality_score: float
    average_processing_time: float
    busiest_hour: Optional[str]
    most_used_model: Optional[str]


# Main dashboard template
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ijon Extraction Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f0f2f5;
            color: #333;
            line-height: 1.6;
        }
        
        .header {
            background: #2c3e50;
            color: white;
            padding: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .header-content {
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .nav {
            display: flex;
            gap: 30px;
        }
        
        .nav a {
            color: white;
            text-decoration: none;
            font-weight: 500;
            transition: opacity 0.3s;
        }
        
        .nav a:hover {
            opacity: 0.8;
        }
        
        .nav a.active {
            border-bottom: 2px solid #3498db;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            text-align: center;
            transition: transform 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .metric-value {
            font-size: 36px;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px 0;
        }
        
        .metric-label {
            font-size: 14px;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .metric-change {
            font-size: 12px;
            margin-top: 5px;
        }
        
        .metric-change.positive {
            color: #27ae60;
        }
        
        .metric-change.negative {
            color: #e74c3c;
        }
        
        .section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 30px;
        }
        
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .section h2 {
            color: #2c3e50;
            font-size: 24px;
        }
        
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s;
            text-decoration: none;
            display: inline-block;
        }
        
        .btn-primary {
            background: #3498db;
            color: white;
        }
        
        .btn-primary:hover {
            background: #2980b9;
        }
        
        .btn-secondary {
            background: #ecf0f1;
            color: #2c3e50;
        }
        
        .btn-secondary:hover {
            background: #bdc3c7;
        }
        
        .extraction-list {
            overflow-x: auto;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }
        
        th {
            font-weight: 600;
            color: #7f8c8d;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 1px;
        }
        
        tr:hover {
            background: #f8f9fa;
        }
        
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }
        
        .status-running {
            background: #3498db;
            color: white;
        }
        
        .status-completed {
            background: #27ae60;
            color: white;
        }
        
        .status-failed {
            background: #e74c3c;
            color: white;
        }
        
        .status-pending_validation {
            background: #f39c12;
            color: white;
        }
        
        .quality-score {
            font-weight: bold;
        }
        
        .quality-high {
            color: #27ae60;
        }
        
        .quality-medium {
            color: #f39c12;
        }
        
        .quality-low {
            color: #e74c3c;
        }
        
        .chart-container {
            height: 300px;
            margin-top: 20px;
        }
        
        .activity-feed {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .activity-item {
            padding: 15px;
            border-bottom: 1px solid #ecf0f1;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .activity-icon {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }
        
        .activity-icon.success {
            background: #d5f4e6;
            color: #27ae60;
        }
        
        .activity-icon.error {
            background: #fadbd8;
            color: #e74c3c;
        }
        
        .activity-icon.info {
            background: #d6eaf8;
            color: #3498db;
        }
        
        .activity-details {
            flex: 1;
        }
        
        .activity-title {
            font-weight: 500;
            margin-bottom: 2px;
        }
        
        .activity-time {
            font-size: 12px;
            color: #7f8c8d;
        }
        
        .refresh-indicator {
            display: inline-block;
            margin-left: 10px;
            font-size: 12px;
            color: #7f8c8d;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
</head>
<body>
    <header class="header">
        <div class="header-content">
            <h1>Ijon Extraction Dashboard</h1>
            <nav class="nav">
                <a href="/" class="active">Overview</a>
                <a href="/extractions">Extractions</a>
                <a href="/validator/dashboard">Validations</a>
                <a href="/metrics">Metrics</a>
            </nav>
        </div>
    </header>
    
    <div class="container">
        <!-- Key Metrics -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Extractions</div>
                <div class="metric-value">{{ metrics.total_extractions }}</div>
                <div class="metric-change positive">+12% from yesterday</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Success Rate</div>
                <div class="metric-value">{{ "%.1f"|format(metrics.success_rate) }}%</div>
                <div class="metric-change {% if metrics.success_rate > 90 %}positive{% else %}negative{% endif %}">
                    Target: 90%
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Avg Quality Score</div>
                <div class="metric-value">{{ "%.2f"|format(metrics.average_quality_score) }}</div>
                <div class="metric-change">out of 1.0</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Active Jobs</div>
                <div class="metric-value">{{ metrics.pending_extractions }}</div>
                <div class="metric-change">
                    <span class="refresh-indicator">Live <span class="spinner"></span></span>
                </div>
            </div>
        </div>
        
        <!-- Recent Extractions -->
        <div class="section">
            <div class="section-header">
                <h2>Recent Extractions</h2>
                <a href="/extractions" class="btn btn-secondary">View All</a>
            </div>
            
            <div class="extraction-list">
                <table>
                    <thead>
                        <tr>
                            <th>Document</th>
                            <th>Status</th>
                            <th>Quality</th>
                            <th>Topics</th>
                            <th>Facts</th>
                            <th>Time</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody hx-get="/api/recent-extractions" hx-trigger="every 5s" hx-swap="innerHTML">
                        {% for extraction in recent_extractions %}
                        <tr>
                            <td>{{ extraction.pdf_name }}</td>
                            <td>
                                <span class="status-badge status-{{ extraction.status }}">
                                    {{ extraction.status }}
                                </span>
                            </td>
                            <td>
                                {% if extraction.quality_score %}
                                <span class="quality-score quality-{{ extraction.quality_class }}">
                                    {{ "%.2f"|format(extraction.quality_score) }}
                                </span>
                                {% else %}
                                -
                                {% endif %}
                            </td>
                            <td>{{ extraction.topics_count }}</td>
                            <td>{{ extraction.facts_count }}</td>
                            <td>{{ extraction.time_ago }}</td>
                            <td>
                                <a href="/extraction/{{ extraction.id }}" class="btn btn-secondary" style="padding: 5px 10px; font-size: 12px;">
                                    View
                                </a>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Activity & Charts -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
            <!-- Activity Feed -->
            <div class="section">
                <h2>Recent Activity</h2>
                <div class="activity-feed" hx-get="/api/activity-feed" hx-trigger="every 10s" hx-swap="innerHTML">
                    {% for activity in activities %}
                    <div class="activity-item">
                        <div class="activity-icon {{ activity.type }}">
                            {{ activity.icon }}
                        </div>
                        <div class="activity-details">
                            <div class="activity-title">{{ activity.title }}</div>
                            <div class="activity-time">{{ activity.time_ago }}</div>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <!-- Performance Chart -->
            <div class="section">
                <h2>Extraction Performance</h2>
                <div class="chart-container">
                    <canvas id="performanceChart"></canvas>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Performance Chart
        const ctx = document.getElementById('performanceChart').getContext('2d');
        const performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: {{ hourly_labels | tojson }},
                datasets: [{
                    label: 'Extractions',
                    data: {{ hourly_counts | tojson }},
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.4
                }, {
                    label: 'Avg Quality',
                    data: {{ hourly_quality | tojson }},
                    borderColor: '#27ae60',
                    backgroundColor: 'rgba(39, 174, 96, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Extractions'
                        }
                    },
                    y1: {
                        beginAtZero: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Quality Score'
                        },
                        max: 1
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
        
        // Auto-refresh metrics every 30 seconds
        setInterval(() => {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    // Update metric cards
                    document.querySelectorAll('.metric-value').forEach((el, idx) => {
                        const values = [
                            data.total_extractions,
                            data.success_rate.toFixed(1) + '%',
                            data.average_quality_score.toFixed(2),
                            data.pending_extractions
                        ];
                        el.textContent = values[idx];
                    });
                });
        }, 30000);
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard_home():
    """Main dashboard overview."""
    # Get current metrics
    metrics = await get_system_metrics()
    
    # Get recent extractions
    recent_extractions = await get_recent_extractions(limit=10)
    
    # Get activity feed
    activities = await get_activity_feed(limit=10)
    
    # Prepare chart data (last 24 hours)
    hourly_stats = await get_hourly_statistics()
    
    context = {
        "metrics": {
            "total_extractions": metrics.total_extractions,
            "success_rate": (metrics.successful_extractions / max(metrics.total_extractions, 1)) * 100,
            "average_quality_score": metrics.average_quality_score,
            "pending_extractions": metrics.pending_extractions
        },
        "recent_extractions": recent_extractions,
        "activities": activities,
        "hourly_labels": hourly_stats["labels"],
        "hourly_counts": hourly_stats["counts"],
        "hourly_quality": hourly_stats["quality"]
    }
    
    template = Template(DASHBOARD_TEMPLATE)
    return template.render(**context)


@app.get("/api/metrics")
async def api_get_metrics():
    """API endpoint for metrics."""
    metrics = await get_system_metrics()
    return {
        "total_extractions": metrics.total_extractions,
        "success_rate": (metrics.successful_extractions / max(metrics.total_extractions, 1)) * 100,
        "average_quality_score": metrics.average_quality_score,
        "pending_extractions": metrics.pending_extractions
    }


@app.get("/api/recent-extractions")
async def api_recent_extractions():
    """API endpoint for recent extractions (for HTMX updates)."""
    extractions = await get_recent_extractions(limit=10)
    
    # Generate HTML for table rows
    html = ""
    for ext in extractions:
        quality_class = "high" if ext.get("quality_score", 0) > 0.8 else "medium" if ext.get("quality_score", 0) > 0.6 else "low"
        
        html += f"""
        <tr>
            <td>{ext['pdf_name']}</td>
            <td><span class="status-badge status-{ext['status']}">{ext['status']}</span></td>
            <td>
                {"<span class='quality-score quality-" + quality_class + "'>" + f"{ext['quality_score']:.2f}" + "</span>" if ext.get('quality_score') else "-"}
            </td>
            <td>{ext['topics_count']}</td>
            <td>{ext['facts_count']}</td>
            <td>{ext['time_ago']}</td>
            <td>
                <a href="/extraction/{ext['id']}" class="btn btn-secondary" style="padding: 5px 10px; font-size: 12px;">View</a>
            </td>
        </tr>
        """
    
    return HTMLResponse(content=html)


@app.get("/api/activity-feed")
async def api_activity_feed():
    """API endpoint for activity feed (for HTMX updates)."""
    activities = await get_activity_feed(limit=10)
    
    # Generate HTML for activity items
    html = ""
    for activity in activities:
        html += f"""
        <div class="activity-item">
            <div class="activity-icon {activity['type']}">
                {activity['icon']}
            </div>
            <div class="activity-details">
                <div class="activity-title">{activity['title']}</div>
                <div class="activity-time">{activity['time_ago']}</div>
            </div>
        </div>
        """
    
    return HTMLResponse(content=html)


# Helper functions

async def get_system_metrics() -> SystemMetrics:
    """Calculate current system metrics."""
    all_states = await state_store.list_active()
    
    total = len(all_states)
    successful = len([s for s in all_states if s.status == "completed"])
    failed = len([s for s in all_states if s.status == "failed"])
    pending = len([s for s in all_states if s.status in ["running", "paused", "pending_validation"]])
    
    # Calculate average quality score
    quality_scores = []
    for state in all_states:
        if state.quality_report and state.status == "completed":
            quality_scores.append(state.quality_report.get("overall_score", 0))
    
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    # Calculate average processing time
    processing_times = []
    for state in all_states:
        if state.metadata.get("processing_time"):
            processing_times.append(state.metadata["processing_time"])
    
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    return SystemMetrics(
        total_extractions=total,
        successful_extractions=successful,
        failed_extractions=failed,
        pending_extractions=pending,
        average_quality_score=avg_quality,
        average_processing_time=avg_time,
        busiest_hour=None,  # TODO: Implement
        most_used_model=None  # TODO: Implement
    )


async def get_recent_extractions(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent extractions for display."""
    all_states = await state_store.list_active()
    
    # Sort by updated_at
    sorted_states = sorted(
        all_states,
        key=lambda s: s.updated_at,
        reverse=True
    )[:limit]
    
    extractions = []
    for state in sorted_states:
        # Extract filename from path
        pdf_name = Path(state.pdf_path).name if state.pdf_path else "Unknown"
        
        # Calculate time ago
        updated = datetime.fromisoformat(state.updated_at)
        time_ago = format_time_ago(updated)
        
        # Get counts
        topics_count = 0
        facts_count = 0
        if state.extraction:
            topics_count = len(state.extraction.get("topics", []))
            facts_count = len(state.extraction.get("facts", []))
        
        extractions.append({
            "id": state.id,
            "pdf_name": pdf_name,
            "status": state.status,
            "quality_score": state.quality_report.get("overall_score") if state.quality_report else None,
            "topics_count": topics_count,
            "facts_count": facts_count,
            "time_ago": time_ago,
            "quality_class": "high" if state.quality_report and state.quality_report.get("overall_score", 0) > 0.8 else "medium"
        })
    
    return extractions


async def get_activity_feed(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent activity for the feed."""
    activities = []
    
    # Get recent state changes
    all_states = await state_store.list_active()
    sorted_states = sorted(
        all_states,
        key=lambda s: s.updated_at,
        reverse=True
    )[:limit]
    
    for state in sorted_states:
        pdf_name = Path(state.pdf_path).name if state.pdf_path else "Unknown"
        updated = datetime.fromisoformat(state.updated_at)
        
        if state.status == "completed":
            icon = "✓"
            activity_type = "success"
            title = f"Extraction completed: {pdf_name}"
        elif state.status == "failed":
            icon = "✗"
            activity_type = "error"
            title = f"Extraction failed: {pdf_name}"
        elif state.status == "running":
            icon = "⟳"
            activity_type = "info"
            title = f"Extraction started: {pdf_name}"
        else:
            icon = "•"
            activity_type = "info"
            title = f"Status changed to {state.status}: {pdf_name}"
        
        activities.append({
            "type": activity_type,
            "icon": icon,
            "title": title,
            "time_ago": format_time_ago(updated)
        })
    
    return activities


async def get_hourly_statistics() -> Dict[str, List]:
    """Get hourly statistics for charts."""
    # For demo, return mock data
    # In production, this would query actual metrics
    now = datetime.now()
    labels = []
    counts = []
    quality = []
    
    for i in range(24):
        hour = now - timedelta(hours=23-i)
        labels.append(hour.strftime("%H:00"))
        counts.append(3 + (i % 5))  # Mock data
        quality.append(0.7 + (i % 3) * 0.1)  # Mock data
    
    return {
        "labels": labels,
        "counts": counts,
        "quality": quality
    }


def format_time_ago(dt: datetime) -> str:
    """Format datetime as 'X ago'."""
    now = datetime.utcnow()
    diff = now - dt
    
    if diff.days > 0:
        return f"{diff.days}d ago"
    elif diff.seconds > 3600:
        return f"{diff.seconds // 3600}h ago"
    elif diff.seconds > 60:
        return f"{diff.seconds // 60}m ago"
    else:
        return "just now"


# Mount validation UI routes
from extraction.v2.validation_ui import app as validation_app
app.mount("/validator", validation_app)


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Ijon Dashboard on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)