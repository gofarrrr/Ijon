"""
Structured feedback UI for human validators.

Provides a web interface for reviewing and validating extractions.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from jinja2 import Template

from extraction.models import ExtractedKnowledge
from extraction.v2.state import StateStore
from src.utils.logging import get_logger

logger = get_logger(__name__)

# FastAPI app for validation UI
app = FastAPI(title="Extraction Validator", version="1.0")

# State store for managing validations
validation_store = StateStore()


class ValidationRequest(BaseModel):
    """Validation request data."""
    extraction_id: str
    pdf_path: str
    extraction: Dict[str, Any]
    quality_report: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ValidationFeedback(BaseModel):
    """Structured feedback from validator."""
    extraction_id: str
    overall_assessment: str = Field(..., pattern="^(approve|reject|needs_revision)$")
    confidence_adjustment: float = Field(default=1.0, ge=0.1, le=2.0)
    
    # Detailed feedback
    topic_feedback: List[Dict[str, Any]] = []
    fact_feedback: List[Dict[str, Any]] = []
    question_feedback: List[Dict[str, Any]] = []
    relationship_feedback: List[Dict[str, Any]] = []
    
    # General comments
    comments: str = ""
    suggested_improvements: List[str] = []
    
    # Validator info
    validator_id: str
    validated_at: datetime = Field(default_factory=datetime.utcnow)


class ValidationStats(BaseModel):
    """Statistics for validation dashboard."""
    total_validations: int = 0
    pending_validations: int = 0
    approved: int = 0
    rejected: int = 0
    needs_revision: int = 0
    average_confidence_adjustment: float = 1.0


# HTML template for the validation UI
VALIDATION_UI_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Extraction Validator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .header h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .quality-score {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 20px;
        }
        
        .score-high { background: #d4edda; color: #155724; }
        .score-medium { background: #fff3cd; color: #856404; }
        .score-low { background: #f8d7da; color: #721c24; }
        
        .content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .extraction-panel, .feedback-panel {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .section {
            margin-bottom: 25px;
        }
        
        .section h3 {
            color: #34495e;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #e0e0e0;
        }
        
        .item {
            background: #f8f9fa;
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 6px;
            border-left: 3px solid #3498db;
        }
        
        .fact-item {
            border-left-color: #27ae60;
        }
        
        .question-item {
            border-left-color: #f39c12;
        }
        
        .relationship-item {
            border-left-color: #9b59b6;
        }
        
        .confidence {
            float: right;
            font-size: 0.9em;
            color: #666;
        }
        
        .evidence {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
            font-style: italic;
        }
        
        .feedback-form {
            margin-top: 20px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        
        .form-control {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        
        .form-control:focus {
            outline: none;
            border-color: #3498db;
            box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
        }
        
        .radio-group {
            display: flex;
            gap: 20px;
            margin-top: 10px;
        }
        
        .radio-group label {
            display: flex;
            align-items: center;
            font-weight: normal;
            cursor: pointer;
        }
        
        .radio-group input[type="radio"] {
            margin-right: 5px;
        }
        
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-primary {
            background: #3498db;
            color: white;
        }
        
        .btn-primary:hover {
            background: #2980b9;
        }
        
        .btn-success {
            background: #27ae60;
            color: white;
            margin-right: 10px;
        }
        
        .btn-danger {
            background: #e74c3c;
            color: white;
            margin-right: 10px;
        }
        
        .btn-warning {
            background: #f39c12;
            color: white;
        }
        
        .item-feedback {
            margin-top: 10px;
            display: none;
        }
        
        .item.selected {
            background: #e3f2fd;
            border-left-color: #2196f3;
        }
        
        .weakness-item {
            background: #ffebee;
            padding: 10px;
            margin-bottom: 8px;
            border-radius: 4px;
            border-left: 3px solid #f44336;
        }
        
        .stats-bar {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            display: flex;
            justify-content: space-around;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .stat-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Extraction Validator</h1>
            <p>Review and provide feedback on extraction quality</p>
            <span class="quality-score {{ score_class }}">Quality Score: {{ quality_score }}%</span>
        </div>
        
        <div class="stats-bar">
            <div class="stat-item">
                <div class="stat-value">{{ stats.pending_validations }}</div>
                <div class="stat-label">Pending</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{{ stats.approved }}</div>
                <div class="stat-label">Approved</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{{ stats.rejected }}</div>
                <div class="stat-label">Rejected</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{{ stats.needs_revision }}</div>
                <div class="stat-label">Needs Revision</div>
            </div>
        </div>
        
        <div class="content">
            <div class="extraction-panel">
                <h2>Extraction Results</h2>
                
                <div class="section">
                    <h3>Document Info</h3>
                    <p><strong>Path:</strong> {{ pdf_path }}</p>
                    <p><strong>Extraction ID:</strong> {{ extraction_id }}</p>
                </div>
                
                <div class="section">
                    <h3>Quality Issues</h3>
                    {% for weakness in weaknesses %}
                    <div class="weakness-item">
                        <strong>{{ weakness.dimension }}:</strong> {{ weakness.severity }}
                        {% if weakness.details %}
                        <div class="evidence">{{ weakness.details }}</div>
                        {% endif %}
                    </div>
                    {% endfor %}
                </div>
                
                <div class="section">
                    <h3>Topics ({{ topics|length }})</h3>
                    {% for topic in topics %}
                    <div class="item topic-item" onclick="toggleFeedback(this, 'topic', {{ loop.index0 }})">
                        <strong>{{ topic.name }}</strong>
                        <span class="confidence">{{ "%.0f"|format(topic.confidence * 100) }}%</span>
                        <div class="evidence">{{ topic.description }}</div>
                    </div>
                    {% endfor %}
                </div>
                
                <div class="section">
                    <h3>Facts ({{ facts|length }})</h3>
                    {% for fact in facts %}
                    <div class="item fact-item" onclick="toggleFeedback(this, 'fact', {{ loop.index0 }})">
                        {{ fact.claim }}
                        <span class="confidence">{{ "%.0f"|format(fact.confidence * 100) }}%</span>
                        {% if fact.evidence %}
                        <div class="evidence">{{ fact.evidence }}</div>
                        {% endif %}
                    </div>
                    {% endfor %}
                </div>
                
                <div class="section">
                    <h3>Questions ({{ questions|length }})</h3>
                    {% for question in questions %}
                    <div class="item question-item" onclick="toggleFeedback(this, 'question', {{ loop.index0 }})">
                        {{ question.question_text }}
                        <span class="confidence">Level: {{ question.cognitive_level }}</span>
                    </div>
                    {% endfor %}
                </div>
                
                <div class="section">
                    <h3>Relationships ({{ relationships|length }})</h3>
                    {% for rel in relationships %}
                    <div class="item relationship-item" onclick="toggleFeedback(this, 'relationship', {{ loop.index0 }})">
                        <strong>{{ rel.source_entity }}</strong> → <strong>{{ rel.target_entity }}</strong>
                        <span class="confidence">{{ "%.0f"|format(rel.confidence * 100) }}%</span>
                        <div class="evidence">{{ rel.relationship_type }}: {{ rel.description }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="feedback-panel">
                <h2>Validation Feedback</h2>
                
                <form id="validationForm" class="feedback-form">
                    <div class="form-group">
                        <label>Overall Assessment</label>
                        <div class="radio-group">
                            <label>
                                <input type="radio" name="assessment" value="approve" required>
                                Approve
                            </label>
                            <label>
                                <input type="radio" name="assessment" value="needs_revision" required>
                                Needs Revision
                            </label>
                            <label>
                                <input type="radio" name="assessment" value="reject" required>
                                Reject
                            </label>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="confidence">Confidence Adjustment</label>
                        <input type="range" id="confidence" class="form-control" 
                               min="0.1" max="2.0" step="0.1" value="1.0"
                               oninput="document.getElementById('confValue').textContent = this.value">
                        <span id="confValue">1.0</span>x
                    </div>
                    
                    <div class="form-group">
                        <label for="comments">General Comments</label>
                        <textarea id="comments" class="form-control" rows="4" 
                                  placeholder="Provide general feedback about the extraction quality..."></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label for="improvements">Suggested Improvements</label>
                        <textarea id="improvements" class="form-control" rows="3" 
                                  placeholder="List specific improvements (one per line)..."></textarea>
                    </div>
                    
                    <div class="form-group">
                        <button type="submit" class="btn btn-success">Approve</button>
                        <button type="button" class="btn btn-warning" onclick="submitWithRevision()">Request Revision</button>
                        <button type="button" class="btn btn-danger" onclick="submitRejection()">Reject</button>
                    </div>
                </form>
            </div>
        </div>
    </div>
    
    <script>
        let selectedItems = {
            topic: {},
            fact: {},
            question: {},
            relationship: {}
        };
        
        function toggleFeedback(element, type, index) {
            element.classList.toggle('selected');
            selectedItems[type][index] = element.classList.contains('selected');
        }
        
        function collectFeedback() {
            const assessment = document.querySelector('input[name="assessment"]:checked')?.value;
            const confidence = parseFloat(document.getElementById('confidence').value);
            const comments = document.getElementById('comments').value;
            const improvements = document.getElementById('improvements').value
                .split('\\n')
                .filter(line => line.trim())
                .map(line => line.trim());
            
            return {
                extraction_id: '{{ extraction_id }}',
                overall_assessment: assessment,
                confidence_adjustment: confidence,
                comments: comments,
                suggested_improvements: improvements,
                validator_id: 'validator_001', // In production, get from auth
                topic_feedback: Object.entries(selectedItems.topic)
                    .filter(([_, selected]) => selected)
                    .map(([index, _]) => ({index: parseInt(index), needs_review: true})),
                fact_feedback: Object.entries(selectedItems.fact)
                    .filter(([_, selected]) => selected)
                    .map(([index, _]) => ({index: parseInt(index), needs_review: true})),
                question_feedback: Object.entries(selectedItems.question)
                    .filter(([_, selected]) => selected)
                    .map(([index, _]) => ({index: parseInt(index), needs_review: true})),
                relationship_feedback: Object.entries(selectedItems.relationship)
                    .filter(([_, selected]) => selected)
                    .map(([index, _]) => ({index: parseInt(index), needs_review: true}))
            };
        }
        
        document.getElementById('validationForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            await submitValidation('approve');
        });
        
        async function submitWithRevision() {
            await submitValidation('needs_revision');
        }
        
        async function submitRejection() {
            if (confirm('Are you sure you want to reject this extraction?')) {
                await submitValidation('reject');
            }
        }
        
        async function submitValidation(assessment) {
            const feedback = collectFeedback();
            feedback.overall_assessment = assessment;
            
            try {
                const response = await fetch('/api/validate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(feedback)
                });
                
                if (response.ok) {
                    alert('Validation submitted successfully!');
                    window.location.href = '/validator/dashboard';
                } else {
                    alert('Error submitting validation');
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Failed to submit validation');
            }
        }
    </script>
</body>
</html>
"""


@app.get("/validator/{extraction_id}")
async def validation_ui(extraction_id: str, request: Request):
    """Render validation UI for a specific extraction."""
    # Load validation request from store
    state = await validation_store.load(extraction_id)
    if not state:
        raise HTTPException(status_code=404, detail="Extraction not found")
    
    # Parse extraction data
    extraction_data = state.extraction
    quality_report = state.quality_report
    
    # Calculate quality score class
    quality_score = quality_report["overall_score"] * 100
    if quality_score >= 80:
        score_class = "score-high"
    elif quality_score >= 60:
        score_class = "score-medium"
    else:
        score_class = "score-low"
    
    # Get validation stats
    stats = await get_validation_stats()
    
    # Prepare template context
    context = {
        "request": request,
        "extraction_id": extraction_id,
        "pdf_path": state.pdf_path,
        "quality_score": f"{quality_score:.0f}",
        "score_class": score_class,
        "stats": stats,
        "weaknesses": quality_report.get("weaknesses", []),
        "topics": extraction_data.get("topics", []),
        "facts": extraction_data.get("facts", []),
        "questions": extraction_data.get("questions", []),
        "relationships": extraction_data.get("relationships", [])
    }
    
    # Use Jinja2 to render template
    from jinja2 import Template
    template = Template(VALIDATION_UI_TEMPLATE)
    rendered_html = template.render(**context)
    return HTMLResponse(content=rendered_html)


@app.get("/validator/dashboard")
async def validation_dashboard():
    """Show dashboard with pending validations."""
    # Get all pending validations
    active_states = await validation_store.list_active()
    
    # Filter for those needing validation
    pending = [
        state for state in active_states 
        if state.metadata.get("needs_validation") and state.status == "pending_validation"
    ]
    
    # Create dashboard HTML
    dashboard_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Validation Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .validation-list { margin-top: 20px; }
            .validation-item { 
                background: #f5f5f5; 
                padding: 15px; 
                margin-bottom: 10px;
                border-radius: 5px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .btn { 
                padding: 8px 16px; 
                background: #3498db; 
                color: white; 
                text-decoration: none;
                border-radius: 4px;
            }
            .btn:hover { background: #2980b9; }
        </style>
    </head>
    <body>
        <h1>Validation Dashboard</h1>
        <p>Pending validations: {count}</p>
        <div class="validation-list">
            {items}
        </div>
    </body>
    </html>
    """
    
    items_html = ""
    for state in pending:
        items_html += f"""
        <div class="validation-item">
            <div>
                <strong>{state.pdf_path}</strong><br>
                Quality Score: {state.quality_report.get('overall_score', 0)*100:.0f}%<br>
                Created: {state.created_at}
            </div>
            <a href="/validator/{state.id}" class="btn">Review</a>
        </div>
        """
    
    rendered_dashboard = dashboard_html.format(
        count=len(pending),
        items=items_html or "<p>No pending validations</p>"
    )
    return HTMLResponse(content=rendered_dashboard)


@app.post("/api/validate")
async def submit_validation(feedback: ValidationFeedback):
    """Submit validation feedback."""
    # Load the extraction state
    state = await validation_store.load(feedback.extraction_id)
    if not state:
        raise HTTPException(status_code=404, detail="Extraction not found")
    
    # Update state with validation result
    state.metadata["validation_result"] = feedback.dict()
    state.metadata["validated_at"] = datetime.utcnow().isoformat()
    state.status = "validated"
    
    # Apply confidence adjustment to extraction
    if state.extraction and feedback.confidence_adjustment != 1.0:
        state.extraction["overall_confidence"] *= feedback.confidence_adjustment
    
    # Save updated state
    await validation_store.save(state)
    
    logger.info(f"Validation submitted for {feedback.extraction_id}: {feedback.overall_assessment}")
    
    return {"status": "success", "extraction_id": feedback.extraction_id}


async def get_validation_stats() -> ValidationStats:
    """Get validation statistics."""
    all_states = await validation_store.list_active()
    
    stats = ValidationStats()
    confidence_adjustments = []
    
    for state in all_states:
        if state.metadata.get("needs_validation"):
            if state.status == "pending_validation":
                stats.pending_validations += 1
            elif state.status == "validated":
                result = state.metadata.get("validation_result", {})
                assessment = result.get("overall_assessment")
                
                if assessment == "approve":
                    stats.approved += 1
                elif assessment == "reject":
                    stats.rejected += 1
                elif assessment == "needs_revision":
                    stats.needs_revision += 1
                
                if result.get("confidence_adjustment"):
                    confidence_adjustments.append(result["confidence_adjustment"])
        
        stats.total_validations = stats.approved + stats.rejected + stats.needs_revision
    
    if confidence_adjustments:
        stats.average_confidence_adjustment = sum(confidence_adjustments) / len(confidence_adjustments)
    
    return stats


# Add this to the triggers.py to integrate with validation UI
def create_validation_request(state_id: str, extraction: ExtractedKnowledge, 
                            quality_report: Dict[str, Any], pdf_path: str):
    """Create a validation request for the UI."""
    from extraction.v2.state import ExtractionState
    
    # Create state for validation
    val_state = ExtractionState.create_new(pdf_path, state_id)
    val_state.extraction = extraction.dict() if hasattr(extraction, 'dict') else extraction
    val_state.quality_report = quality_report
    val_state.status = "pending_validation"
    val_state.metadata["needs_validation"] = True
    
    return val_state


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)