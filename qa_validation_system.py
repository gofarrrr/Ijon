#!/usr/bin/env python3
"""
QA Validation System - Automated quality checks for QA pairs.
Provides continuous monitoring and validation of QA generation quality.
"""

import os
import sys
from pathlib import Path
import psycopg2
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from src.utils.logging import get_logger

load_dotenv()
logger = get_logger(__name__)

class QAValidationSystem:
    """Automated validation system for QA pair quality monitoring."""
    
    def __init__(self):
        """Initialize QA validation system."""
        self.connection_string = os.getenv('NEON_CONNECTION_STRING')
        if not self.connection_string:
            raise ValueError("NEON_CONNECTION_STRING not found in environment")
        
        self.validation_thresholds = {
            "min_source_alignment": 0.9,      # 90% minimum alignment
            "min_confidence_accuracy": 0.8,    # Confidence should match quality
            "max_overconfidence_rate": 0.1,    # Max 10% overconfident answers
            "min_validation_pass_rate": 0.85,  # 85% minimum validation rate
            "max_hallucination_rate": 0.05     # Max 5% hallucination rate
        }
    
    def get_qa_quality_metrics(self, hours_back: int = 24) -> Dict:
        """Get quality metrics for QA pairs generated in the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # Get QA pairs with metadata
                cur.execute("""
                    SELECT qa.id, qa.question, qa.answer, qa.answer_confidence,
                           qa.answer_type, qa.created_at,
                           cc.extraction_metadata
                    FROM qa_pairs qa
                    JOIN content_chunks cc ON cc.id = ANY(qa.source_chunk_ids)
                    WHERE qa.created_at >= %s
                    ORDER BY qa.created_at DESC
                """, (cutoff_time,))
                
                qa_data = []
                for row in cur.fetchall():
                    qa_id, question, answer, confidence, answer_type, created_at, metadata = row
                    qa_data.append({
                        "id": str(qa_id),
                        "question": question,
                        "answer": answer,
                        "confidence": confidence,
                        "answer_type": answer_type,
                        "created_at": created_at,
                        "metadata": metadata or {}
                    })
                
                return self.analyze_qa_quality_metrics(qa_data)
    
    def analyze_qa_quality_metrics(self, qa_data: List[Dict]) -> Dict:
        """Analyze quality metrics from QA data."""
        if not qa_data:
            return {"error": "No QA data to analyze"}
        
        # Basic statistics
        total_qa = len(qa_data)
        confidence_scores = [qa["confidence"] for qa in qa_data]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Validation metrics (for improved system)
        validated_qa = [qa for qa in qa_data if qa["metadata"].get("validation_enabled")]
        validation_rates = [qa["metadata"].get("validation_pass_rate", 0) for qa in validated_qa]
        avg_validation_rate = sum(validation_rates) / len(validation_rates) if validation_rates else 0
        
        # Source grounding metrics
        source_grounded = [qa for qa in qa_data if qa["metadata"].get("source_grounded_generation")]
        source_grounding_rate = len(source_grounded) / total_qa
        
        # Confidence distribution
        confidence_distribution = {
            "very_high (0.9-1.0)": len([c for c in confidence_scores if c >= 0.9]),
            "high (0.8-0.9)": len([c for c in confidence_scores if 0.8 <= c < 0.9]),
            "medium (0.6-0.8)": len([c for c in confidence_scores if 0.6 <= c < 0.8]),
            "low (0.4-0.6)": len([c for c in confidence_scores if 0.4 <= c < 0.6]),
            "very_low (<0.4)": len([c for c in confidence_scores if c < 0.4])
        }
        
        # Quality flags
        quality_flags = []
        if avg_confidence > 0.95:
            quality_flags.append("HIGH_CONFIDENCE_WARNING: Average confidence suspiciously high")
        if source_grounding_rate < self.validation_thresholds["min_source_alignment"]:
            quality_flags.append(f"SOURCE_GROUNDING_LOW: Only {source_grounding_rate:.1%} source grounded")
        if avg_validation_rate < self.validation_thresholds["min_validation_pass_rate"]:
            quality_flags.append(f"VALIDATION_RATE_LOW: Only {avg_validation_rate:.1%} validation rate")
        
        # Processing time analysis
        processing_times = []
        for qa in qa_data:
            total_time = qa["metadata"].get("total_time_ms", 0)
            if total_time > 0:
                processing_times.append(total_time)
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        metrics = {
            "analysis_timestamp": datetime.now().isoformat(),
            "time_window_hours": 24,
            "total_qa_pairs": total_qa,
            "average_confidence": round(avg_confidence, 3),
            "confidence_distribution": confidence_distribution,
            "validation_metrics": {
                "validated_qa_pairs": len(validated_qa),
                "average_validation_rate": round(avg_validation_rate, 3),
                "source_grounding_rate": round(source_grounding_rate, 3)
            },
            "performance_metrics": {
                "average_processing_time_ms": round(avg_processing_time, 1),
                "processing_efficiency": "good" if avg_processing_time < 30000 else "slow"
            },
            "quality_flags": quality_flags,
            "quality_score": self.calculate_overall_quality_score(
                avg_validation_rate, source_grounding_rate, avg_confidence
            ),
            "recommendations": self.generate_quality_recommendations(
                avg_validation_rate, source_grounding_rate, quality_flags
            )
        }
        
        return metrics
    
    def calculate_overall_quality_score(self, validation_rate: float, source_rate: float, avg_confidence: float) -> float:
        """Calculate overall quality score (0-1)."""
        # Penalize overconfidence
        confidence_penalty = 1.0
        if avg_confidence > 0.95:
            confidence_penalty = 0.8  # 20% penalty for overconfidence
        elif avg_confidence > 0.9:
            confidence_penalty = 0.9  # 10% penalty for high confidence
        
        # Weighted average of key metrics
        quality_score = (
            validation_rate * 0.4 +      # 40% weight on validation
            source_rate * 0.4 +          # 40% weight on source grounding
            confidence_penalty * 0.2     # 20% weight on confidence calibration
        )
        
        return round(quality_score, 3)
    
    def generate_quality_recommendations(self, validation_rate: float, source_rate: float, flags: List[str]) -> List[str]:
        """Generate specific recommendations based on quality metrics."""
        recommendations = []
        
        if validation_rate < 0.9:
            recommendations.append("IMPROVE: Enhance validation prompts to increase pass rate")
        
        if source_rate < 0.9:
            recommendations.append("CRITICAL: Strengthen source grounding requirements")
        
        if any("HIGH_CONFIDENCE" in flag for flag in flags):
            recommendations.append("ADJUST: Recalibrate confidence scoring to be more realistic")
        
        if validation_rate >= 0.9 and source_rate >= 0.9:
            recommendations.append("EXCELLENT: Quality metrics meet targets - continue current approach")
        
        return recommendations
    
    def run_quality_validation_check(self, hours_back: int = 24) -> Dict:
        """Run comprehensive quality validation check."""
        logger.info(f"🔍 Running quality validation check (last {hours_back} hours)")
        
        metrics = self.get_qa_quality_metrics(hours_back)
        
        if "error" in metrics:
            return metrics
        
        # Determine overall system health
        quality_score = metrics["quality_score"]
        if quality_score >= 0.9:
            system_health = "EXCELLENT"
        elif quality_score >= 0.8:
            system_health = "GOOD"
        elif quality_score >= 0.7:
            system_health = "FAIR"
        else:
            system_health = "POOR"
        
        validation_report = {
            "validation_timestamp": datetime.now().isoformat(),
            "system_health": system_health,
            "quality_score": quality_score,
            "metrics": metrics,
            "pass_fail_status": {
                "validation_rate": "PASS" if metrics["validation_metrics"]["average_validation_rate"] >= self.validation_thresholds["min_validation_pass_rate"] else "FAIL",
                "source_grounding": "PASS" if metrics["validation_metrics"]["source_grounding_rate"] >= self.validation_thresholds["min_source_alignment"] else "FAIL",
                "confidence_calibration": "PASS" if metrics["average_confidence"] <= 0.95 else "FAIL"
            },
            "action_required": quality_score < 0.8
        }
        
        return validation_report
    
    def monitor_processing_pipeline(self, document_id: str) -> Dict:
        """Monitor the processing pipeline for a specific document."""
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # Get document processing status
                cur.execute("""
                    SELECT d.title, d.import_status, d.processed_at,
                           COUNT(cc.id) as chunk_count,
                           COUNT(qa.id) as qa_count,
                           COUNT(dk.id) as knowledge_count
                    FROM documents d
                    LEFT JOIN content_chunks cc ON d.id = cc.document_id
                    LEFT JOIN qa_pairs qa ON d.id = qa.document_id
                    LEFT JOIN distilled_knowledge dk ON d.id = dk.document_id
                    WHERE d.id = %s
                    GROUP BY d.id, d.title, d.import_status, d.processed_at
                """, (document_id,))
                
                result = cur.fetchone()
                if not result:
                    return {"error": f"Document {document_id} not found"}
                
                title, status, processed_at, chunk_count, qa_count, knowledge_count = result
                
                # Check for improved processing
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM content_chunks cc
                    WHERE cc.document_id = %s 
                    AND cc.extraction_metadata->>'improved_generation' = 'true'
                """, (document_id,))
                
                improved_chunks = cur.fetchone()[0]
                
                pipeline_status = {
                    "document_id": document_id,
                    "title": title,
                    "status": status,
                    "processed_at": processed_at,
                    "pipeline_completion": {
                        "chunks_extracted": chunk_count,
                        "qa_pairs_generated": qa_count,
                        "knowledge_extracted": knowledge_count,
                        "improved_processing": improved_chunks
                    },
                    "pipeline_health": "COMPLETE" if qa_count > 0 and knowledge_count > 0 else "INCOMPLETE",
                    "quality_indicators": {
                        "has_improved_processing": improved_chunks > 0,
                        "qa_to_chunk_ratio": qa_count / chunk_count if chunk_count > 0 else 0,
                        "knowledge_to_chunk_ratio": knowledge_count / chunk_count if chunk_count > 0 else 0
                    }
                }
                
                return pipeline_status
    
    def generate_daily_quality_report(self) -> str:
        """Generate a daily quality report."""
        report_data = self.run_quality_validation_check(24)
        
        report_file = Path(f"daily_qa_quality_report_{datetime.now().strftime('%Y%m%d')}.json")
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Create summary
        summary = f"""
📊 DAILY QA QUALITY REPORT - {datetime.now().strftime('%Y-%m-%d')}
{'='*60}

🎯 SYSTEM HEALTH: {report_data['system_health']}
📈 QUALITY SCORE: {report_data['quality_score']}/1.0

📋 KEY METRICS:
- Total QA Pairs: {report_data['metrics']['total_qa_pairs']}
- Validation Rate: {report_data['metrics']['validation_metrics']['average_validation_rate']:.1%}
- Source Grounding: {report_data['metrics']['validation_metrics']['source_grounding_rate']:.1%}
- Average Confidence: {report_data['metrics']['average_confidence']:.3f}

🚦 PASS/FAIL STATUS:
- Validation Rate: {report_data['pass_fail_status']['validation_rate']}
- Source Grounding: {report_data['pass_fail_status']['source_grounding']}
- Confidence Calibration: {report_data['pass_fail_status']['confidence_calibration']}

🎯 RECOMMENDATIONS:
{chr(10).join('- ' + rec for rec in report_data['metrics']['recommendations'])}

⚠️ ACTION REQUIRED: {'YES' if report_data['action_required'] else 'NO'}

📄 Full report: {report_file}
"""
        
        print(summary)
        return str(report_file)
    
    def validate_system_deployment(self) -> Dict:
        """Validate that the improved QA system is properly deployed."""
        logger.info("🔍 Validating improved QA system deployment")
        
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # Check for improved generation markers
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM content_chunks 
                    WHERE extraction_metadata->>'improved_generation' = 'true'
                """)
                improved_chunks = cur.fetchone()[0]
                
                # Check for validation metadata
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM content_chunks 
                    WHERE extraction_metadata->>'validation_enabled' = 'true'
                """)
                validated_chunks = cur.fetchone()[0]
                
                # Check recent QA quality
                recent_metrics = self.get_qa_quality_metrics(1)  # Last hour
                
                deployment_status = {
                    "deployment_timestamp": datetime.now().isoformat(),
                    "improved_system_active": improved_chunks > 0,
                    "validation_system_active": validated_chunks > 0,
                    "improved_chunks_count": improved_chunks,
                    "validated_chunks_count": validated_chunks,
                    "recent_quality_metrics": recent_metrics,
                    "deployment_health": "HEALTHY" if improved_chunks > 0 and validated_chunks > 0 else "NEEDS_ATTENTION"
                }
                
                return deployment_status

def main():
    """Main function for QA validation system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QA Validation System")
    parser.add_argument("--daily-report", action="store_true", help="Generate daily quality report")
    parser.add_argument("--check-deployment", action="store_true", help="Validate system deployment")
    parser.add_argument("--monitor-document", help="Monitor specific document ID")
    parser.add_argument("--hours-back", type=int, default=24, help="Hours to look back for metrics")
    
    args = parser.parse_args()
    
    try:
        validator = QAValidationSystem()
        
        if args.daily_report:
            report_file = validator.generate_daily_quality_report()
            print(f"✅ Daily report generated: {report_file}")
        
        elif args.check_deployment:
            status = validator.validate_system_deployment()
            print(f"🔍 Deployment Status: {status['deployment_health']}")
            print(f"📊 Improved chunks: {status['improved_chunks_count']}")
            print(f"✅ Validated chunks: {status['validated_chunks_count']}")
        
        elif args.monitor_document:
            status = validator.monitor_processing_pipeline(args.monitor_document)
            if "error" in status:
                print(f"❌ {status['error']}")
            else:
                print(f"📋 Document: {status['title']}")
                print(f"🎯 Pipeline Health: {status['pipeline_health']}")
                print(f"📊 QA Pairs: {status['pipeline_completion']['qa_pairs_generated']}")
                print(f"🧠 Knowledge: {status['pipeline_completion']['knowledge_extracted']}")
        
        else:
            # Default: run quality check
            report = validator.run_quality_validation_check(args.hours_back)
            if "error" in report:
                print(f"❌ {report['error']}")
            else:
                print(f"🎯 System Health: {report['system_health']}")
                print(f"📈 Quality Score: {report['quality_score']}/1.0")
                print(f"⚠️ Action Required: {'YES' if report['action_required'] else 'NO'}")
        
    except Exception as e:
        print(f"❌ Error in QA validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()