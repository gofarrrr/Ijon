#!/usr/bin/env python3
"""
Create a real PDF file for testing.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from pathlib import Path

def create_test_pdf():
    """Create a test PDF document."""
    output_dir = Path("test_pdfs")
    output_dir.mkdir(exist_ok=True)
    
    pdf_path = output_dir / "ai_research_paper.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
    )
    
    # Content
    story = []
    
    # Title page
    story.append(Paragraph("Advances in Large Language Models", title_style))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("A Comprehensive Review of Transformer Architectures", styles['Heading2']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Abstract", styles['Heading3']))
    story.append(Paragraph(
        """This paper presents a comprehensive review of recent advances in large language models (LLMs), 
        focusing on transformer architectures and their applications. We examine the evolution from 
        early attention mechanisms to modern architectures like GPT, BERT, and T5, analyzing their 
        strengths, limitations, and real-world applications.""",
        styles['BodyText']
    ))
    story.append(PageBreak())
    
    # Introduction
    story.append(Paragraph("1. Introduction", styles['Heading2']))
    story.append(Paragraph(
        """Large language models have revolutionized natural language processing by demonstrating 
        remarkable capabilities in understanding and generating human-like text. The transformer 
        architecture, introduced by Vaswani et al. in 2017, has become the foundation for most 
        modern LLMs. These models use self-attention mechanisms to process input sequences in 
        parallel, enabling them to capture long-range dependencies more effectively than previous 
        recurrent architectures.""",
        styles['BodyText']
    ))
    story.append(Spacer(1, 0.2*inch))
    
    # Key Concepts
    story.append(Paragraph("2. Key Concepts", styles['Heading2']))
    story.append(Paragraph("2.1 Self-Attention Mechanism", styles['Heading3']))
    story.append(Paragraph(
        """The self-attention mechanism allows models to weigh the importance of different words 
        in a sequence when processing each word. This is computed using queries (Q), keys (K), 
        and values (V) matrices. The attention scores are calculated as:
        Attention(Q,K,V) = softmax(QK^T/√d_k)V""",
        styles['BodyText']
    ))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("2.2 Multi-Head Attention", styles['Heading3']))
    story.append(Paragraph(
        """Multi-head attention allows the model to attend to different positions simultaneously, 
        learning various types of relationships. Each head performs attention independently, and 
        their outputs are concatenated and linearly transformed.""",
        styles['BodyText']
    ))
    story.append(PageBreak())
    
    # Applications
    story.append(Paragraph("3. Applications", styles['Heading2']))
    story.append(Paragraph(
        """LLMs have found applications in numerous domains including:
        • Text Generation: Creating human-like text for various purposes
        • Translation: High-quality machine translation between languages
        • Question Answering: Understanding and responding to queries
        • Code Generation: Writing functional code from natural language descriptions
        • Summarization: Condensing long documents into key points""",
        styles['BodyText']
    ))
    
    # Table of model comparisons
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Table 1: Comparison of Popular LLMs", styles['Heading3']))
    
    data = [
        ['Model', 'Parameters', 'Architecture', 'Key Innovation'],
        ['GPT-3', '175B', 'Decoder-only', 'Scaling laws, few-shot learning'],
        ['BERT', '340M', 'Encoder-only', 'Bidirectional pre-training'],
        ['T5', '11B', 'Encoder-Decoder', 'Text-to-text framework'],
        ['LLaMA', '65B', 'Decoder-only', 'Efficient training'],
    ]
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(table)
    
    # Conclusion
    story.append(PageBreak())
    story.append(Paragraph("4. Conclusion", styles['Heading2']))
    story.append(Paragraph(
        """Large language models represent a significant breakthrough in AI, demonstrating 
        capabilities that were thought impossible just a few years ago. As we continue to 
        scale these models and improve their architectures, we can expect even more impressive 
        applications. However, challenges remain in terms of computational requirements, 
        bias mitigation, and ensuring safe deployment.""",
        styles['BodyText']
    ))
    
    # Build PDF
    doc.build(story)
    print(f"✅ Created PDF: {pdf_path}")
    print(f"   Size: {pdf_path.stat().st_size:,} bytes")
    
    return pdf_path

if __name__ == "__main__":
    # Check if reportlab is installed
    try:
        import reportlab
        pdf_path = create_test_pdf()
    except ImportError:
        print("❌ reportlab not installed. Installing...")
        import subprocess
        subprocess.run(["pip", "install", "reportlab"])
        print("✅ Installed reportlab. Please run again.")