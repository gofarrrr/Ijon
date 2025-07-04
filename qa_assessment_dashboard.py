#!/usr/bin/env python3
"""
QA Assessment Dashboard - Interactive tool for reviewing and rating QA pairs quality.
Allows side-by-side comparison of questions with their source chunks.
"""

import os
import sys
from pathlib import Path
import psycopg2
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from src.utils.logging import get_logger

load_dotenv()
logger = get_logger(__name__)

class QAAssessmentDashboard:
    """Interactive dashboard for assessing QA pair quality."""
    
    def __init__(self):
        """Initialize the QA assessment dashboard."""
        self.connection_string = os.getenv('NEON_CONNECTION_STRING')
        if not self.connection_string:
            raise ValueError("NEON_CONNECTION_STRING not found in environment")
        
        self.assessment_file = Path("qa_assessments.json")
        self.assessments = self.load_assessments()
    
    def load_assessments(self) -> Dict:
        """Load existing assessments from file."""
        if self.assessment_file.exists():
            try:
                with open(self.assessment_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load assessments: {e}")
        
        return {
            "assessments": {},  # qa_pair_id -> assessment_data
            "summary": {
                "total_assessed": 0,
                "average_quality": 0.0,
                "issues_found": [],
                "last_updated": None
            }
        }
    
    def save_assessments(self):
        """Save assessments to file."""
        try:
            self.assessments["summary"]["last_updated"] = datetime.now().isoformat()
            with open(self.assessment_file, 'w') as f:
                json.dump(self.assessments, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save assessments: {e}")
    
    def get_qa_pairs_for_assessment(self, limit: int = 10, document_id: str = None) -> List[Dict]:
        """Get QA pairs with their source chunks for assessment."""
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # Build query
                base_query = """
                    SELECT qa.id, qa.question, qa.answer, qa.answer_type, qa.answer_confidence,
                           qa.source_chunk_ids, d.title as document_title,
                           cc.content as source_chunk_content, cc.chunk_index
                    FROM qa_pairs qa
                    JOIN documents d ON qa.document_id = d.id
                    LEFT JOIN content_chunks cc ON cc.id = ANY(qa.source_chunk_ids)
                """
                
                if document_id:
                    query = base_query + " WHERE qa.document_id = %s ORDER BY qa.created_at DESC LIMIT %s"
                    cur.execute(query, (document_id, limit))
                else:
                    query = base_query + " ORDER BY qa.created_at DESC LIMIT %s"
                    cur.execute(query, (limit,))
                
                qa_pairs = []
                for row in cur.fetchall():
                    qa_id, question, answer, answer_type, confidence, source_chunk_ids, doc_title, source_content, chunk_index = row
                    
                    qa_pairs.append({
                        "id": str(qa_id),
                        "question": question,
                        "answer": answer,
                        "answer_type": answer_type,
                        "confidence": confidence,
                        "document_title": doc_title,
                        "source_chunk_content": source_content or "No source chunk found",
                        "chunk_index": chunk_index,
                        "source_chunk_ids": source_chunk_ids or [],
                        "assessed": str(qa_id) in self.assessments["assessments"]
                    })
                
                return qa_pairs
    
    def display_qa_pair(self, qa_pair: Dict, index: int, total: int):
        """Display a QA pair for assessment."""
        print(f"\n{'='*100}")
        print(f"🔍 QA PAIR ASSESSMENT ({index}/{total})")
        print(f"{'='*100}")
        
        print(f"📚 Document: {qa_pair['document_title']}")
        print(f"🎯 Type: {qa_pair['answer_type']} | Confidence: {qa_pair['confidence']}")
        print(f"📝 Chunk Index: {qa_pair['chunk_index']}")
        
        if qa_pair["assessed"]:
            assessment = self.assessments["assessments"][qa_pair["id"]]
            print(f"⭐ Previous Rating: {assessment['quality_rating']}/5 | Issues: {', '.join(assessment['issues'])}")
        
        print(f"\n❓ QUESTION:")
        print(f"{qa_pair['question']}")
        
        print(f"\n💬 ANSWER:")
        print(f"{qa_pair['answer']}")
        
        print(f"\n📖 SOURCE CHUNK:")
        print(f"{qa_pair['source_chunk_content'][:500]}{'...' if len(qa_pair['source_chunk_content']) > 500 else ''}")
        
        print(f"\n{'='*100}")
    
    def assess_qa_pair(self, qa_pair: Dict) -> Dict:
        """Interactive assessment of a single QA pair."""
        print(f"\n🔍 ASSESS THIS QA PAIR:")
        print(f"ID: {qa_pair['id']}")
        
        # Quality rating
        while True:
            try:
                quality_rating = input("\n⭐ Quality Rating (1-5, where 5=excellent): ").strip()
                quality_rating = int(quality_rating)
                if 1 <= quality_rating <= 5:
                    break
                else:
                    print("Please enter a number between 1 and 5")
            except ValueError:
                print("Please enter a valid number")
        
        # Issue identification
        print(f"\n🚨 Identify any issues (enter numbers, space-separated):")
        print(f"1. Question doesn't match source content")
        print(f"2. Answer contains hallucinations/made-up content")
        print(f"3. Duplicate or very similar to other questions")
        print(f"4. Question is too vague or unclear")
        print(f"5. Answer is not practically useful")
        print(f"6. Mental model not accurately represented")
        print(f"7. Poor grammar or formatting")
        print(f"8. No issues - good quality")
        
        issues_input = input("Issues (e.g., '1 3' for issues 1 and 3, or '8' for no issues): ").strip()
        
        issue_map = {
            "1": "source_mismatch",
            "2": "hallucination",
            "3": "duplicate",
            "4": "unclear_question",
            "5": "not_useful",
            "6": "inaccurate_mental_model",
            "7": "poor_formatting",
            "8": "no_issues"
        }
        
        issues = []
        if issues_input:
            for issue_num in issues_input.split():
                if issue_num in issue_map:
                    issues.append(issue_map[issue_num])
        
        # Additional comments
        comments = input("\n💭 Additional comments (optional): ").strip()
        
        # Source relevance
        while True:
            source_relevant = input("\n📖 Does the question accurately reflect the source chunk? (y/n): ").strip().lower()
            if source_relevant in ['y', 'yes', 'n', 'no']:
                source_relevant = source_relevant in ['y', 'yes']
                break
            else:
                print("Please enter 'y' or 'n'")
        
        # Practical utility
        while True:
            practical_utility = input("\n🎯 Is this question practically useful for learning mental models? (y/n): ").strip().lower()
            if practical_utility in ['y', 'yes', 'n', 'no']:
                practical_utility = practical_utility in ['y', 'yes']
                break
            else:
                print("Please enter 'y' or 'n'")
        
        assessment = {
            "quality_rating": quality_rating,
            "issues": issues,
            "comments": comments,
            "source_relevant": source_relevant,
            "practical_utility": practical_utility,
            "assessed_at": datetime.now().isoformat(),
            "qa_pair_id": qa_pair["id"]
        }
        
        return assessment
    
    def run_assessment_session(self, limit: int = 10, document_id: str = None):
        """Run an interactive assessment session."""
        print(f"🚀 Starting QA Assessment Session")
        print(f"📊 Assessing up to {limit} QA pairs")
        
        qa_pairs = self.get_qa_pairs_for_assessment(limit, document_id)
        
        if not qa_pairs:
            print("❌ No QA pairs found for assessment")
            return
        
        print(f"✅ Found {len(qa_pairs)} QA pairs for assessment")
        
        assessed_count = 0
        for i, qa_pair in enumerate(qa_pairs, 1):
            self.display_qa_pair(qa_pair, i, len(qa_pairs))
            
            # Check if already assessed
            if qa_pair["assessed"]:
                reassess = input(f"\n🔄 This QA pair was already assessed. Reassess? (y/n): ").strip().lower()
                if reassess not in ['y', 'yes']:
                    continue
            
            # Assess the QA pair
            try:
                assessment = self.assess_qa_pair(qa_pair)
                self.assessments["assessments"][qa_pair["id"]] = assessment
                assessed_count += 1
                
                print(f"✅ Assessment saved for QA pair {qa_pair['id']}")
                
                # Save after each assessment
                self.save_assessments()
                
            except KeyboardInterrupt:
                print(f"\n🛑 Assessment session interrupted")
                break
            except Exception as e:
                print(f"❌ Error assessing QA pair: {e}")
                continue
            
            # Continue or stop
            if i < len(qa_pairs):
                continue_assessment = input(f"\n➡️  Continue to next QA pair? (y/n): ").strip().lower()
                if continue_assessment not in ['y', 'yes']:
                    break
        
        # Update summary
        self.update_assessment_summary()
        self.save_assessments()
        
        print(f"\n🎉 Assessment session complete!")
        print(f"📊 Assessed {assessed_count} QA pairs in this session")
        self.display_assessment_summary()
    
    def update_assessment_summary(self):
        """Update the assessment summary statistics."""
        assessments = self.assessments["assessments"]
        
        if not assessments:
            return
        
        total_assessed = len(assessments)
        quality_ratings = [a["quality_rating"] for a in assessments.values()]
        average_quality = sum(quality_ratings) / len(quality_ratings)
        
        # Count issues
        all_issues = []
        for assessment in assessments.values():
            all_issues.extend(assessment["issues"])
        
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        self.assessments["summary"] = {
            "total_assessed": total_assessed,
            "average_quality": round(average_quality, 2),
            "quality_distribution": {
                "excellent (5)": len([r for r in quality_ratings if r == 5]),
                "good (4)": len([r for r in quality_ratings if r == 4]),
                "average (3)": len([r for r in quality_ratings if r == 3]),
                "poor (2)": len([r for r in quality_ratings if r == 2]),
                "very_poor (1)": len([r for r in quality_ratings if r == 1])
            },
            "common_issues": issue_counts,
            "source_relevance_rate": len([a for a in assessments.values() if a["source_relevant"]]) / total_assessed,
            "practical_utility_rate": len([a for a in assessments.values() if a["practical_utility"]]) / total_assessed,
            "last_updated": datetime.now().isoformat()
        }
    
    def display_assessment_summary(self):
        """Display summary of all assessments."""
        summary = self.assessments["summary"]
        
        print(f"\n📊 QA ASSESSMENT SUMMARY")
        print(f"{'='*50}")
        print(f"Total QA pairs assessed: {summary['total_assessed']}")
        print(f"Average quality rating: {summary['average_quality']}/5")
        
        if "quality_distribution" in summary:
            print(f"\n⭐ Quality Distribution:")
            for rating, count in summary["quality_distribution"].items():
                print(f"  {rating}: {count}")
        
        if "common_issues" in summary and summary["common_issues"]:
            print(f"\n🚨 Most Common Issues:")
            sorted_issues = sorted(summary["common_issues"].items(), key=lambda x: x[1], reverse=True)
            for issue, count in sorted_issues[:5]:
                print(f"  {issue}: {count} occurrences")
        
        if "source_relevance_rate" in summary:
            print(f"\n📖 Source Relevance Rate: {summary['source_relevance_rate']:.1%}")
            print(f"🎯 Practical Utility Rate: {summary['practical_utility_rate']:.1%}")
        
        print(f"\n💾 Assessments saved to: {self.assessment_file}")
    
    def export_assessment_report(self) -> str:
        """Export a detailed assessment report."""
        report_file = Path(f"qa_assessment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Get additional data for report
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # Get document info
                cur.execute("""
                    SELECT d.title, COUNT(qa.id) as qa_count
                    FROM documents d
                    LEFT JOIN qa_pairs qa ON d.id = qa.document_id
                    WHERE d.source_type LIKE '%pdf'
                    GROUP BY d.id, d.title
                """)
                
                document_stats = {}
                for title, qa_count in cur.fetchall():
                    document_stats[title] = qa_count
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "assessment_summary": self.assessments["summary"],
            "document_statistics": document_stats,
            "detailed_assessments": self.assessments["assessments"],
            "recommendations": self.generate_recommendations()
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Detailed assessment report exported to: {report_file}")
        return str(report_file)
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on assessment results."""
        recommendations = []
        summary = self.assessments["summary"]
        
        if summary["total_assessed"] == 0:
            return ["No assessments completed yet"]
        
        # Quality recommendations
        if summary["average_quality"] < 3.0:
            recommendations.append("🔴 CRITICAL: Average quality is below 3/5. QA generation prompts need significant improvement.")
        elif summary["average_quality"] < 4.0:
            recommendations.append("🟡 Average quality is moderate. Consider refining QA generation prompts.")
        else:
            recommendations.append("🟢 Good average quality! Continue current approach with minor optimizations.")
        
        # Issue-specific recommendations
        if "common_issues" in summary:
            issues = summary["common_issues"]
            
            if issues.get("source_mismatch", 0) > summary["total_assessed"] * 0.2:
                recommendations.append("🔧 High source mismatch rate. Improve source chunk grounding in prompts.")
            
            if issues.get("hallucination", 0) > 0:
                recommendations.append("⚠️ Hallucinations detected. Add stricter content validation requirements.")
            
            if issues.get("duplicate", 0) > summary["total_assessed"] * 0.1:
                recommendations.append("🔄 Multiple duplicates found. Implement deduplication logic.")
            
            if issues.get("not_useful", 0) > summary["total_assessed"] * 0.2:
                recommendations.append("🎯 Many questions lack practical utility. Focus on actionable mental models.")
        
        # Relevance recommendations
        if "source_relevance_rate" in summary:
            if summary["source_relevance_rate"] < 0.8:
                recommendations.append("📖 Low source relevance. Strengthen chunk-to-question alignment.")
            
            if summary["practical_utility_rate"] < 0.7:
                recommendations.append("💡 Low practical utility. Focus on real-world application questions.")
        
        return recommendations

def main():
    """Main function for running QA assessment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QA Assessment Dashboard")
    parser.add_argument("--limit", type=int, default=10, help="Number of QA pairs to assess")
    parser.add_argument("--document-id", help="Assess QA pairs from specific document")
    parser.add_argument("--export-report", action="store_true", help="Export detailed assessment report")
    parser.add_argument("--summary-only", action="store_true", help="Show summary without starting assessment")
    
    args = parser.parse_args()
    
    try:
        dashboard = QAAssessmentDashboard()
        
        if args.export_report:
            dashboard.export_assessment_report()
            return
        
        if args.summary_only:
            dashboard.display_assessment_summary()
            return
        
        # Run assessment session
        dashboard.run_assessment_session(limit=args.limit, document_id=args.document_id)
        
    except KeyboardInterrupt:
        print("\n🛑 QA Assessment interrupted by user")
    except Exception as e:
        print(f"❌ Error in QA Assessment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()