#!/usr/bin/env python3
"""
QA Quality Analysis Tool - Automated analysis of QA pair quality issues.
Identifies duplicates, source mismatches, and other quality problems.
"""

import os
import sys
from pathlib import Path
import psycopg2
import json
from datetime import datetime
from typing import List, Dict, Set
from collections import defaultdict
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from src.utils.logging import get_logger

load_dotenv()
logger = get_logger(__name__)

class QAQualityAnalyzer:
    """Automated analysis of QA pair quality issues."""
    
    def __init__(self):
        """Initialize the QA quality analyzer."""
        self.connection_string = os.getenv('NEON_CONNECTION_STRING')
        if not self.connection_string:
            raise ValueError("NEON_CONNECTION_STRING not found in environment")
    
    def get_all_qa_pairs(self) -> List[Dict]:
        """Get all QA pairs with their source chunks."""
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT qa.id, qa.question, qa.answer, qa.answer_type, qa.answer_confidence,
                           qa.source_chunk_ids, d.title as document_title, d.id as document_id,
                           cc.content as source_chunk_content, cc.chunk_index
                    FROM qa_pairs qa
                    JOIN documents d ON qa.document_id = d.id
                    LEFT JOIN content_chunks cc ON cc.id = ANY(qa.source_chunk_ids)
                    ORDER BY qa.created_at DESC
                """)
                
                qa_pairs = []
                for row in cur.fetchall():
                    qa_id, question, answer, answer_type, confidence, source_chunk_ids, doc_title, doc_id, source_content, chunk_index = row
                    
                    qa_pairs.append({
                        "id": str(qa_id),
                        "question": question,
                        "answer": answer,
                        "answer_type": answer_type,
                        "confidence": confidence,
                        "document_title": doc_title,
                        "document_id": str(doc_id),
                        "source_chunk_content": source_content or "",
                        "chunk_index": chunk_index,
                        "source_chunk_ids": source_chunk_ids or []
                    })
                
                return qa_pairs
    
    def analyze_duplicates(self, qa_pairs: List[Dict]) -> Dict:
        """Identify duplicate or very similar QA pairs."""
        print("🔍 Analyzing duplicates...")
        
        # Group by exact question match
        question_groups = defaultdict(list)
        for qa in qa_pairs:
            question_groups[qa["question"]].append(qa)
        
        # Find exact duplicates
        exact_duplicates = {q: qas for q, qas in question_groups.items() if len(qas) > 1}
        
        # Group by exact answer match
        answer_groups = defaultdict(list)
        for qa in qa_pairs:
            answer_groups[qa["answer"]].append(qa)
        
        exact_answer_duplicates = {a: qas for a, qas in answer_groups.items() if len(qas) > 1}
        
        duplicate_analysis = {
            "exact_question_duplicates": len(exact_duplicates),
            "exact_answer_duplicates": len(exact_answer_duplicates),
            "duplicate_question_groups": exact_duplicates,
            "duplicate_answer_groups": exact_answer_duplicates,
            "total_duplicate_questions": sum(len(qas) - 1 for qas in exact_duplicates.values()),  # Subtract 1 for original
            "total_duplicate_answers": sum(len(qas) - 1 for qas in exact_answer_duplicates.values())
        }
        
        return duplicate_analysis
    
    def analyze_source_alignment(self, qa_pairs: List[Dict]) -> Dict:
        """Analyze how well questions align with their source chunks."""
        print("📖 Analyzing source alignment...")
        
        alignment_issues = []
        missing_sources = []
        
        for qa in qa_pairs:
            # Check if source chunk exists
            if not qa["source_chunk_content"]:
                missing_sources.append({
                    "qa_id": qa["id"],
                    "question": qa["question"][:100] + "...",
                    "issue": "No source chunk found"
                })
                continue
            
            # Basic keyword overlap analysis
            question_words = set(qa["question"].lower().split())
            source_words = set(qa["source_chunk_content"].lower().split())
            
            # Remove common words
            common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "how", "what", "when", "where", "why", "can", "should", "would", "could", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "this", "that", "these", "those"}
            question_keywords = question_words - common_words
            source_keywords = source_words - common_words
            
            # Calculate overlap
            if question_keywords:
                overlap_ratio = len(question_keywords & source_keywords) / len(question_keywords)
                
                if overlap_ratio < 0.1:  # Less than 10% keyword overlap
                    alignment_issues.append({
                        "qa_id": qa["id"],
                        "question": qa["question"][:100] + "...",
                        "overlap_ratio": overlap_ratio,
                        "issue": "Very low keyword overlap with source",
                        "source_preview": qa["source_chunk_content"][:100] + "..."
                    })
        
        alignment_analysis = {
            "total_qa_pairs": len(qa_pairs),
            "missing_sources": len(missing_sources),
            "alignment_issues": len(alignment_issues),
            "missing_source_details": missing_sources,
            "alignment_issue_details": alignment_issues,
            "alignment_rate": (len(qa_pairs) - len(alignment_issues) - len(missing_sources)) / len(qa_pairs) if qa_pairs else 0
        }
        
        return alignment_analysis
    
    def analyze_confidence_distribution(self, qa_pairs: List[Dict]) -> Dict:
        """Analyze confidence score distribution and patterns."""
        print("📊 Analyzing confidence distribution...")
        
        confidences = [qa["confidence"] for qa in qa_pairs]
        
        if not confidences:
            return {"message": "No confidence scores found"}
        
        confidence_buckets = {
            "very_high (0.9-1.0)": len([c for c in confidences if c >= 0.9]),
            "high (0.8-0.9)": len([c for c in confidences if 0.8 <= c < 0.9]),
            "medium (0.6-0.8)": len([c for c in confidences if 0.6 <= c < 0.8]),
            "low (0.4-0.6)": len([c for c in confidences if 0.4 <= c < 0.6]),
            "very_low (<0.4)": len([c for c in confidences if c < 0.4])
        }
        
        # Analyze by question type
        type_confidence = defaultdict(list)
        for qa in qa_pairs:
            type_confidence[qa["answer_type"]].append(qa["confidence"])
        
        type_avg_confidence = {}
        for answer_type, confs in type_confidence.items():
            type_avg_confidence[answer_type] = sum(confs) / len(confs) if confs else 0
        
        confidence_analysis = {
            "total_qa_pairs": len(qa_pairs),
            "average_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "confidence_distribution": confidence_buckets,
            "confidence_by_type": type_avg_confidence,
            "suspiciously_high_confidence": len([c for c in confidences if c >= 0.95])  # Very high confidence might indicate overconfidence
        }
        
        return confidence_analysis
    
    def analyze_content_quality(self, qa_pairs: List[Dict]) -> Dict:
        """Analyze content quality indicators."""
        print("📝 Analyzing content quality...")
        
        quality_issues = []
        
        for qa in qa_pairs:
            issues = []
            
            # Check question length
            if len(qa["question"]) < 20:
                issues.append("Very short question")
            elif len(qa["question"]) > 300:
                issues.append("Very long question")
            
            # Check answer length
            if len(qa["answer"]) < 50:
                issues.append("Very short answer")
            elif len(qa["answer"]) > 1000:
                issues.append("Very long answer")
            
            # Check for specific content issues
            if "nicholas winton" in qa["question"].lower() or "nicholas winton" in qa["answer"].lower():
                if "nicholas winton" not in qa["source_chunk_content"].lower():
                    issues.append("References Nicholas Winton not in source")
            
            # Check for vague questions
            vague_indicators = ["this", "that", "these", "those", "it", "they"]
            question_start = qa["question"].lower().split()[:3]
            if any(indicator in question_start for indicator in vague_indicators):
                issues.append("Question starts with vague reference")
            
            # Check for mental model terminology
            mental_model_terms = ["mental model", "cognitive bias", "framework", "first principles", "constraint analysis", "second-order thinking"]
            has_mental_model_terms = any(term in qa["question"].lower() or term in qa["answer"].lower() for term in mental_model_terms)
            
            if issues or not has_mental_model_terms:
                quality_issues.append({
                    "qa_id": qa["id"],
                    "question_preview": qa["question"][:100] + "...",
                    "issues": issues,
                    "has_mental_model_terms": has_mental_model_terms
                })
        
        content_analysis = {
            "total_qa_pairs": len(qa_pairs),
            "qa_pairs_with_issues": len(quality_issues),
            "quality_issue_details": quality_issues,
            "quality_rate": (len(qa_pairs) - len(quality_issues)) / len(qa_pairs) if qa_pairs else 0
        }
        
        return content_analysis
    
    def generate_quality_report(self) -> Dict:
        """Generate comprehensive quality analysis report."""
        print("🚀 Starting comprehensive QA quality analysis...")
        
        qa_pairs = self.get_all_qa_pairs()
        
        if not qa_pairs:
            return {"error": "No QA pairs found for analysis"}
        
        print(f"📊 Analyzing {len(qa_pairs)} QA pairs...")
        
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_qa_pairs": len(qa_pairs),
            "duplicate_analysis": self.analyze_duplicates(qa_pairs),
            "source_alignment_analysis": self.analyze_source_alignment(qa_pairs),
            "confidence_analysis": self.analyze_confidence_distribution(qa_pairs),
            "content_quality_analysis": self.analyze_content_quality(qa_pairs)
        }
        
        return report
    
    def print_quality_summary(self, report: Dict):
        """Print a summary of quality analysis findings."""
        print(f"\n📋 QA QUALITY ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        print(f"📊 Total QA Pairs Analyzed: {report['total_qa_pairs']}")
        
        # Duplicates
        dup_analysis = report["duplicate_analysis"]
        print(f"\n🔄 DUPLICATE ANALYSIS:")
        print(f"  Exact question duplicates: {dup_analysis['exact_question_duplicates']} groups")
        print(f"  Exact answer duplicates: {dup_analysis['exact_answer_duplicates']} groups")
        print(f"  Total duplicate questions: {dup_analysis['total_duplicate_questions']}")
        
        # Source alignment
        align_analysis = report["source_alignment_analysis"]
        print(f"\n📖 SOURCE ALIGNMENT:")
        print(f"  Missing sources: {align_analysis['missing_sources']}")
        print(f"  Alignment issues: {align_analysis['alignment_issues']}")
        print(f"  Alignment rate: {align_analysis['alignment_rate']:.1%}")
        
        # Confidence
        conf_analysis = report["confidence_analysis"]
        print(f"\n📊 CONFIDENCE DISTRIBUTION:")
        print(f"  Average confidence: {conf_analysis['average_confidence']:.3f}")
        print(f"  Suspiciously high confidence (≥0.95): {conf_analysis['suspiciously_high_confidence']}")
        for bucket, count in conf_analysis["confidence_distribution"].items():
            print(f"  {bucket}: {count}")
        
        # Content quality
        content_analysis = report["content_quality_analysis"]
        print(f"\n📝 CONTENT QUALITY:")
        print(f"  QA pairs with issues: {content_analysis['qa_pairs_with_issues']}")
        print(f"  Quality rate: {content_analysis['quality_rate']:.1%}")
        
        # Top recommendations
        print(f"\n🎯 TOP RECOMMENDATIONS:")
        if dup_analysis['total_duplicate_questions'] > 0:
            print(f"  🔴 CRITICAL: Remove {dup_analysis['total_duplicate_questions']} duplicate questions")
        if align_analysis['alignment_rate'] < 0.8:
            print(f"  🟡 MEDIUM: Improve source-question alignment ({align_analysis['alignment_rate']:.1%} current rate)")
        if conf_analysis['suspiciously_high_confidence'] > report['total_qa_pairs'] * 0.8:
            print(f"  🟡 MEDIUM: Review confidence scoring (too many high-confidence questions)")
        if content_analysis['quality_rate'] < 0.7:
            print(f"  🔴 CRITICAL: Address content quality issues ({content_analysis['quality_rate']:.1%} quality rate)")
    
    def save_quality_report(self, report: Dict) -> str:
        """Save quality report to file."""
        report_file = Path(f"qa_quality_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"💾 Quality analysis report saved to: {report_file}")
        return str(report_file)

def main():
    """Main function for QA quality analysis."""
    try:
        analyzer = QAQualityAnalyzer()
        
        # Generate quality report
        report = analyzer.generate_quality_report()
        
        if "error" in report:
            print(f"❌ {report['error']}")
            return
        
        # Print summary
        analyzer.print_quality_summary(report)
        
        # Save detailed report
        report_file = analyzer.save_quality_report(report)
        
        print(f"\n✅ Quality analysis complete!")
        print(f"📄 Detailed report: {report_file}")
        
    except Exception as e:
        print(f"❌ Error in quality analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()