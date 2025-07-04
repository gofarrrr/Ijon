#!/usr/bin/env python3
"""
Generate sample PDFs for testing the RAG system.

This utility creates realistic test PDFs with varied content types
including ML/AI, medical, and legal documents.
"""

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class PDFGenerator:
    """Generate sample PDFs with realistic content."""
    
    def __init__(self, output_dir: str = "sample_pdfs"):
        """Initialize PDF generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Set up custom paragraph styles."""
        # Chapter title style
        self.styles.add(ParagraphStyle(
            name='ChapterTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1a1a1a'),
        ))
        
        # Section title style
        self.styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=self.styles['Heading2'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.HexColor('#333333'),
        ))
        
        # Subsection style
        self.styles.add(ParagraphStyle(
            name='SubsectionTitle',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=15,
            textColor=colors.HexColor('#555555'),
        ))
    
    def generate_ml_textbook(self, filename: str = "ml_textbook.pdf") -> Path:
        """Generate a machine learning textbook PDF."""
        output_path = self.output_dir / filename
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        
        story = []
        
        # Title page
        story.append(Paragraph(
            "Machine Learning: Theory and Practice",
            self.styles['Title']
        ))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(
            "A Comprehensive Guide to Modern ML Techniques",
            self.styles['Heading2']
        ))
        story.append(PageBreak())
        
        # Chapter 1: Introduction to Machine Learning
        story.append(Paragraph("Chapter 1: Introduction to Machine Learning", self.styles['ChapterTitle']))
        story.append(Paragraph(
            """Machine learning is a subset of artificial intelligence that focuses on the development 
            of algorithms and statistical models that enable computer systems to improve their performance 
            on a specific task through experience. Unlike traditional programming where we explicitly 
            program rules, machine learning systems learn patterns from data.""",
            self.styles['BodyText']
        ))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("1.1 Types of Machine Learning", self.styles['SectionTitle']))
        story.append(Paragraph(
            """There are three main types of machine learning: supervised learning, unsupervised learning, 
            and reinforcement learning. Each type addresses different kinds of problems and uses different 
            approaches to learn from data.""",
            self.styles['BodyText']
        ))
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph("1.1.1 Supervised Learning", self.styles['SubsectionTitle']))
        story.append(Paragraph(
            """Supervised learning is a type of machine learning where models are trained on labeled data. 
            This means the training data includes both input features and the correct outputs. The model 
            learns to map inputs to outputs based on these examples. Common algorithms include linear 
            regression, decision trees, and neural networks.""",
            self.styles['BodyText']
        ))
        story.append(Spacer(1, 0.1*inch))
        
        # Add a table
        data = [
            ['Algorithm', 'Type', 'Use Case'],
            ['Linear Regression', 'Regression', 'Predicting continuous values'],
            ['Logistic Regression', 'Classification', 'Binary classification'],
            ['Decision Trees', 'Both', 'Non-linear relationships'],
            ['Neural Networks', 'Both', 'Complex patterns'],
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
        story.append(PageBreak())
        
        # Chapter 2: Neural Networks
        story.append(Paragraph("Chapter 2: Neural Networks and Deep Learning", self.styles['ChapterTitle']))
        story.append(Paragraph(
            """Neural networks are computing systems inspired by biological neural networks. They consist 
            of interconnected nodes (neurons) organized in layers that process information using connectionist 
            approaches to computation. Deep learning refers to neural networks with multiple hidden layers.""",
            self.styles['BodyText']
        ))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("2.1 Architecture of Neural Networks", self.styles['SectionTitle']))
        story.append(Paragraph(
            """A typical neural network consists of an input layer, one or more hidden layers, and an output 
            layer. Each layer contains neurons that receive inputs, apply weights and biases, pass the result 
            through an activation function, and send the output to the next layer.""",
            self.styles['BodyText']
        ))
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph("2.2 Backpropagation Algorithm", self.styles['SectionTitle']))
        story.append(Paragraph(
            """Backpropagation calculates gradients of the loss function with respect to network weights by 
            applying the chain rule backwards through the network. It propagates error signals from output 
            to input layers, enabling weight updates that minimize the loss. This is the foundation of 
            training deep neural networks.""",
            self.styles['BodyText']
        ))
        story.append(PageBreak())
        
        # Chapter 3: Transformers
        story.append(Paragraph("Chapter 3: Transformer Architecture", self.styles['ChapterTitle']))
        story.append(Paragraph(
            """Transformers use self-attention mechanisms for parallel processing of sequences, capturing 
            long-range dependencies efficiently. Unlike RNNs that process sequentially, transformers can 
            process all positions simultaneously, making them faster to train and better at capturing 
            long-range dependencies.""",
            self.styles['BodyText']
        ))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("3.1 Self-Attention Mechanism", self.styles['SectionTitle']))
        story.append(Paragraph(
            """The self-attention mechanism allows the model to weight the importance of different parts 
            of the input when processing each element. This creates direct connections between any positions 
            in a sequence, eliminating the vanishing gradient problem that affects RNNs.""",
            self.styles['BodyText']
        ))
        
        # Build PDF
        doc.build(story)
        print(f"Generated: {output_path}")
        return output_path
    
    def generate_medical_handbook(self, filename: str = "medical_handbook.pdf") -> Path:
        """Generate a medical handbook PDF."""
        output_path = self.output_dir / filename
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        
        story = []
        
        # Title
        story.append(Paragraph("Clinical Medicine Handbook", self.styles['Title']))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("Essential Reference for Healthcare Professionals", self.styles['Heading2']))
        story.append(PageBreak())
        
        # Chapter: Diabetes
        story.append(Paragraph("Chapter 5: Diabetes Mellitus", self.styles['ChapterTitle']))
        story.append(Paragraph(
            """Diabetes mellitus is a group of metabolic disorders characterized by chronic hyperglycemia 
            resulting from defects in insulin secretion, insulin action, or both. The chronic hyperglycemia 
            of diabetes is associated with long-term damage to various organs.""",
            self.styles['BodyText']
        ))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("5.1 Clinical Presentation", self.styles['SectionTitle']))
        story.append(Paragraph(
            """Primary symptoms of diabetes include frequent urination (polyuria), excessive thirst 
            (polydipsia), unexplained weight loss, extreme fatigue, blurred vision, and slow wound healing. 
            These symptoms result from the body's inability to properly use glucose for energy.""",
            self.styles['BodyText']
        ))
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph("5.2 Diagnostic Criteria", self.styles['SectionTitle']))
        
        # Diagnostic criteria table
        data = [
            ['Test', 'Normal', 'Prediabetes', 'Diabetes'],
            ['Fasting Glucose', '<100 mg/dL', '100-125 mg/dL', '≥126 mg/dL'],
            ['2-hr OGTT', '<140 mg/dL', '140-199 mg/dL', '≥200 mg/dL'],
            ['HbA1c', '<5.7%', '5.7-6.4%', '≥6.5%'],
        ]
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F2F2F2')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(table)
        story.append(PageBreak())
        
        # Chapter: Hypertension
        story.append(Paragraph("Chapter 7: Hypertension", self.styles['ChapterTitle']))
        story.append(Paragraph(
            """Hypertension, or high blood pressure, is a chronic medical condition in which the blood 
            pressure in the arteries is persistently elevated. It is a major risk factor for cardiovascular 
            disease, stroke, and kidney disease.""",
            self.styles['BodyText']
        ))
        
        # Build PDF
        doc.build(story)
        print(f"Generated: {output_path}")
        return output_path
    
    def generate_legal_document(self, filename: str = "contract_law.pdf") -> Path:
        """Generate a legal document PDF."""
        output_path = self.output_dir / filename
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        
        story = []
        
        # Title
        story.append(Paragraph("Principles of Contract Law", self.styles['Title']))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("Fundamentals and Applications", self.styles['Heading2']))
        story.append(PageBreak())
        
        # Chapter: Contract Formation
        story.append(Paragraph("Chapter 3: Contract Formation", self.styles['ChapterTitle']))
        story.append(Paragraph(
            """A contract is a legally binding agreement between two or more parties that creates mutual 
            obligations enforceable by law. The formation of a valid contract requires several essential 
            elements that must all be present.""",
            self.styles['BodyText']
        ))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("3.1 Essential Elements", self.styles['SectionTitle']))
        story.append(Paragraph(
            """A valid contract requires: (1) Offer - a clear proposal to enter into an agreement, 
            (2) Acceptance - unqualified agreement to the offer's terms, (3) Consideration - something 
            of value exchanged between parties, (4) Capacity - legal ability of parties to contract, 
            and (5) Legal Purpose - the contract must be for a lawful objective. All elements must be 
            present for enforceability.""",
            self.styles['BodyText']
        ))
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph("3.2 Types of Contracts", self.styles['SectionTitle']))
        story.append(Paragraph(
            """Contracts can be classified in several ways: Express vs. Implied, Bilateral vs. Unilateral, 
            Valid vs. Void vs. Voidable, and Executed vs. Executory. Understanding these classifications 
            is crucial for determining the rights and obligations of parties.""",
            self.styles['BodyText']
        ))
        
        # Build PDF
        doc.build(story)
        print(f"Generated: {output_path}")
        return output_path
    
    def generate_all_samples(self) -> List[Path]:
        """Generate all sample PDFs."""
        print(f"Generating sample PDFs in {self.output_dir}/")
        
        pdfs = [
            self.generate_ml_textbook(),
            self.generate_medical_handbook(),
            self.generate_legal_document(),
        ]
        
        print(f"\nGenerated {len(pdfs)} sample PDFs")
        return pdfs


def main():
    """Generate sample PDFs for testing."""
    generator = PDFGenerator()
    pdfs = generator.generate_all_samples()
    
    print("\nSample PDFs generated successfully!")
    print("\nYou can now test the system with these PDFs:")
    for pdf in pdfs:
        print(f"  - {pdf}")


if __name__ == "__main__":
    main()