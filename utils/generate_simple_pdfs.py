#!/usr/bin/env python3
"""
Generate simple test PDFs without external dependencies.
Creates text files that simulate PDF content for testing.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import List

class SimplePDFGenerator:
    """Generate simple test content files."""
    
    def __init__(self, output_dir: str = "sample_pdfs"):
        """Initialize generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_ml_content(self) -> Path:
        """Generate ML textbook content."""
        content = """Machine Learning: Theory and Practice

Chapter 1: Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on the development 
of algorithms and statistical models that enable computer systems to improve their performance 
on a specific task through experience. Unlike traditional programming where we explicitly 
program rules, machine learning systems learn patterns from data.

1.1 Types of Machine Learning

There are three main types of machine learning: supervised learning, unsupervised learning, 
and reinforcement learning. Each type addresses different kinds of problems and uses different 
approaches to learn from data.

1.1.1 Supervised Learning

Supervised learning is a type of machine learning where models are trained on labeled data. 
This means the training data includes both input features and the correct outputs. The model 
learns to map inputs to outputs based on these examples. Common algorithms include linear 
regression, decision trees, and neural networks.

Chapter 2: Neural Networks and Deep Learning

Neural networks are computing systems inspired by biological neural networks. They consist 
of interconnected nodes (neurons) organized in layers that process information using connectionist 
approaches to computation. Deep learning refers to neural networks with multiple hidden layers.

2.1 Architecture of Neural Networks

A typical neural network consists of an input layer, one or more hidden layers, and an output 
layer. Each layer contains neurons that receive inputs, apply weights and biases, pass the result 
through an activation function, and send the output to the next layer.

2.2 Backpropagation Algorithm

Backpropagation calculates gradients of the loss function with respect to network weights by 
applying the chain rule backwards through the network. It propagates error signals from output 
to input layers, enabling weight updates that minimize the loss. This is the foundation of 
training deep neural networks.

Chapter 3: Transformer Architecture

Transformers use self-attention mechanisms for parallel processing of sequences, capturing 
long-range dependencies efficiently. Unlike RNNs that process sequentially, transformers can 
process all positions simultaneously, making them faster to train and better at capturing 
long-range dependencies.

3.1 Self-Attention Mechanism

The self-attention mechanism allows the model to weight the importance of different parts 
of the input when processing each element. This creates direct connections between any positions 
in a sequence, eliminating the vanishing gradient problem that affects RNNs.

Chapter 4: Convolutional Neural Networks

CNNs excel at image processing due to their unique architecture. They use local connectivity 
to capture spatial relationships, parameter sharing through filters to reduce model complexity, 
translation invariance from pooling layers, and hierarchical feature learning from edges to 
complex patterns.
"""
        
        output_path = self.output_dir / "ml_textbook.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Generated: {output_path}")
        return output_path
    
    def generate_medical_content(self) -> Path:
        """Generate medical handbook content."""
        content = """Clinical Medicine Handbook

Chapter 5: Diabetes Mellitus

Diabetes mellitus is a group of metabolic disorders characterized by chronic hyperglycemia 
resulting from defects in insulin secretion, insulin action, or both. The chronic hyperglycemia 
of diabetes is associated with long-term damage to various organs.

5.1 Clinical Presentation

Primary symptoms of diabetes include frequent urination (polyuria), excessive thirst 
(polydipsia), unexplained weight loss, extreme fatigue, blurred vision, and slow wound healing. 
These symptoms result from the body's inability to properly use glucose for energy.

5.2 Diagnostic Criteria

Diagnostic tests for diabetes include:
- Fasting Glucose: Normal <100 mg/dL, Prediabetes 100-125 mg/dL, Diabetes ≥126 mg/dL
- 2-hr OGTT: Normal <140 mg/dL, Prediabetes 140-199 mg/dL, Diabetes ≥200 mg/dL
- HbA1c: Normal <5.7%, Prediabetes 5.7-6.4%, Diabetes ≥6.5%

5.3 Management

Management includes lifestyle modifications (diet, exercise), oral medications (metformin, 
sulfonylureas), and insulin therapy for advanced cases. Regular monitoring of blood glucose 
and HbA1c is essential.

Chapter 7: Hypertension

Hypertension, or high blood pressure, is a chronic medical condition in which the blood 
pressure in the arteries is persistently elevated. It is a major risk factor for cardiovascular 
disease, stroke, and kidney disease.

7.1 Classification

Blood pressure categories:
- Normal: <120/80 mmHg
- Elevated: 120-129/<80 mmHg
- Stage 1 Hypertension: 130-139/80-89 mmHg
- Stage 2 Hypertension: ≥140/90 mmHg

7.2 Treatment

Treatment includes lifestyle changes (salt restriction, weight loss, exercise) and medications 
(ACE inhibitors, ARBs, beta-blockers, diuretics, calcium channel blockers).
"""
        
        output_path = self.output_dir / "medical_handbook.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Generated: {output_path}")
        return output_path
    
    def generate_legal_content(self) -> Path:
        """Generate legal document content."""
        content = """Principles of Contract Law

Chapter 3: Contract Formation

A contract is a legally binding agreement between two or more parties that creates mutual 
obligations enforceable by law. The formation of a valid contract requires several essential 
elements that must all be present.

3.1 Essential Elements

A valid contract requires: 
(1) Offer - a clear proposal to enter into an agreement
(2) Acceptance - unqualified agreement to the offer's terms
(3) Consideration - something of value exchanged between parties
(4) Capacity - legal ability of parties to contract
(5) Legal Purpose - the contract must be for a lawful objective

All elements must be present for enforceability.

3.2 Types of Contracts

Contracts can be classified in several ways: 
- Express vs. Implied
- Bilateral vs. Unilateral
- Valid vs. Void vs. Voidable
- Executed vs. Executory

Understanding these classifications is crucial for determining the rights and obligations of parties.

3.3 Breach of Contract

A breach occurs when one party fails to perform their contractual obligations. Remedies include:
- Damages (compensatory, consequential, punitive)
- Specific performance
- Rescission
- Reformation

Chapter 4: Contract Interpretation

Courts interpret contracts to determine the parties' intent. Principles include:
- Plain meaning rule
- Parol evidence rule
- Contra proferentem
- Course of dealing and usage of trade
"""
        
        output_path = self.output_dir / "contract_law.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Generated: {output_path}")
        return output_path
    
    def generate_info_file(self) -> None:
        """Generate info file about the test documents."""
        info_content = f"""Test Documents Information
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

These are simplified text files that simulate PDF content for testing the RAG system.
In a production environment, these would be actual PDFs processed by PyMuPDF or pdfplumber.

Files generated:
1. ml_textbook.txt - Machine learning concepts and techniques
2. medical_handbook.txt - Medical conditions and treatments  
3. contract_law.txt - Legal principles and contract formation

The RAG system can process these text files directly for testing purposes.
When actual PDF processing is needed, the system will use the PDF extraction pipeline.
"""
        
        info_path = self.output_dir / "README.txt"
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(info_content)
        
        print(f"Generated: {info_path}")
    
    def generate_all(self) -> List[Path]:
        """Generate all test documents."""
        print(f"\nGenerating test documents in {self.output_dir}/")
        print("=" * 50)
        
        files = [
            self.generate_ml_content(),
            self.generate_medical_content(),
            self.generate_legal_content(),
        ]
        
        self.generate_info_file()
        
        print("=" * 50)
        print(f"Generated {len(files)} test documents")
        print("\nThese text files simulate PDF content for testing.")
        print("The RAG system can process them directly.")
        
        return files


def main():
    """Generate test documents."""
    generator = SimplePDFGenerator()
    generator.generate_all()


if __name__ == "__main__":
    main()