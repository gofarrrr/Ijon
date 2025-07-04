"""
Create test data for evaluation and calibration.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List

from tests.test_evaluation import TestCase, TestDataset


def create_ml_test_dataset() -> TestDataset:
    """Create a comprehensive ML-focused test dataset."""
    test_cases = [
        # Easy factual questions
        TestCase(
            id="ml_easy_1",
            question="What is supervised learning?",
            expected_answer="Supervised learning is a type of machine learning where models are trained on labeled data, meaning the training data includes both input features and correct outputs. The model learns to map inputs to outputs based on these examples.",
            relevant_chunks=["chunk_ml_001", "chunk_ml_002"],
            required_entities=["supervised learning", "labeled data", "training"],
            required_facts=["labeled data", "input-output mapping", "training examples"],
            difficulty="easy",
            category="factual",
        ),
        TestCase(
            id="ml_easy_2",
            question="What are neural networks?",
            expected_answer="Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process information using connectionist approaches to computation.",
            relevant_chunks=["chunk_nn_001", "chunk_nn_002"],
            required_entities=["neural networks", "neurons", "layers"],
            required_facts=["interconnected nodes", "layers", "biological inspiration"],
            difficulty="easy",
            category="factual",
        ),
        
        # Medium analytical questions
        TestCase(
            id="ml_medium_1",
            question="How does backpropagation work in neural networks?",
            expected_answer="Backpropagation calculates gradients of the loss function with respect to network weights by applying the chain rule backwards through the network. It propagates error signals from output to input layers, enabling weight updates that minimize the loss.",
            relevant_chunks=["chunk_bp_001", "chunk_bp_002", "chunk_bp_003"],
            required_entities=["backpropagation", "gradients", "chain rule", "weights"],
            required_facts=["chain rule", "error propagation", "gradient calculation", "weight updates"],
            difficulty="medium",
            category="analytical",
        ),
        TestCase(
            id="ml_medium_2",
            question="What are the advantages of convolutional neural networks for image processing?",
            expected_answer="CNNs excel at image processing due to: 1) Local connectivity that captures spatial relationships, 2) Parameter sharing through filters reducing model complexity, 3) Translation invariance from pooling layers, and 4) Hierarchical feature learning from edges to complex patterns.",
            relevant_chunks=["chunk_cnn_001", "chunk_cnn_002", "chunk_cnn_003"],
            required_entities=["CNN", "convolution", "filters", "pooling"],
            required_facts=["spatial relationships", "parameter sharing", "translation invariance", "hierarchical features"],
            difficulty="medium",
            category="analytical",
        ),
        
        # Hard comparative questions
        TestCase(
            id="ml_hard_1",
            question="Compare and contrast transformer architectures with recurrent neural networks for sequence modeling.",
            expected_answer="Transformers use self-attention for parallel processing of sequences, capturing long-range dependencies efficiently, while RNNs process sequentially, maintaining hidden states. Transformers are faster to train and better at long sequences but require more memory. RNNs are more memory-efficient but suffer from vanishing gradients and slower training.",
            relevant_chunks=["chunk_trans_001", "chunk_trans_002", "chunk_rnn_001", "chunk_rnn_002"],
            required_entities=["transformers", "RNN", "attention", "sequence modeling"],
            required_facts=["parallel processing", "self-attention", "sequential processing", "vanishing gradients", "memory efficiency"],
            difficulty="hard",
            category="comparative",
        ),
        TestCase(
            id="ml_hard_2",
            question="Analyze the trade-offs between model complexity and generalization in deep learning.",
            expected_answer="Complex models with many parameters can fit training data well but risk overfitting. Simpler models may underfit but generalize better. The bias-variance tradeoff suggests finding optimal complexity through techniques like regularization, dropout, early stopping, and cross-validation to balance fitting capacity with generalization.",
            relevant_chunks=["chunk_complex_001", "chunk_complex_002", "chunk_regular_001"],
            required_entities=["overfitting", "underfitting", "regularization", "bias-variance"],
            required_facts=["overfitting risk", "underfitting", "bias-variance tradeoff", "regularization techniques", "cross-validation"],
            difficulty="hard",
            category="analytical",
        ),
        
        # Multi-hop reasoning questions
        TestCase(
            id="ml_multihop_1",
            question="How do attention mechanisms in transformers relate to the vanishing gradient problem in RNNs?",
            expected_answer="Attention mechanisms allow direct connections between any positions in a sequence, creating shorter gradient paths compared to RNNs where gradients must flow through all intermediate states. This architectural difference eliminates the vanishing gradient problem that plagues RNNs in long sequences.",
            relevant_chunks=["chunk_att_001", "chunk_grad_001", "chunk_rnn_003"],
            required_entities=["attention", "vanishing gradient", "gradient flow", "RNN"],
            required_facts=["direct connections", "gradient paths", "intermediate states", "architectural difference"],
            difficulty="hard",
            category="reasoning",
        ),
    ]
    
    return TestDataset(
        name="ml_comprehensive",
        description="Comprehensive ML test dataset for evaluation",
        test_cases=test_cases,
        pdf_ids=["ml_textbook.pdf", "deep_learning_book.pdf", "attention_paper.pdf"],
        created_at=datetime.utcnow(),
    )


def create_domain_specific_dataset(domain: str) -> TestDataset:
    """Create domain-specific test datasets."""
    if domain == "medical":
        test_cases = [
            TestCase(
                id="med_1",
                question="What are the primary symptoms of diabetes?",
                expected_answer="Primary symptoms include frequent urination, excessive thirst, unexplained weight loss, fatigue, blurred vision, and slow wound healing.",
                relevant_chunks=["chunk_diab_001", "chunk_diab_002"],
                required_entities=["diabetes", "symptoms"],
                required_facts=["frequent urination", "excessive thirst", "weight loss"],
                difficulty="easy",
                category="factual",
            ),
        ]
        return TestDataset(
            name="medical_basic",
            description="Medical domain test dataset",
            test_cases=test_cases,
            pdf_ids=["medical_handbook.pdf"],
            created_at=datetime.utcnow(),
        )
    
    elif domain == "legal":
        test_cases = [
            TestCase(
                id="legal_1",
                question="What constitutes a valid contract?",
                expected_answer="A valid contract requires offer, acceptance, consideration, capacity of parties, and legal purpose. All elements must be present for enforceability.",
                relevant_chunks=["chunk_contract_001"],
                required_entities=["contract", "offer", "acceptance"],
                required_facts=["offer", "acceptance", "consideration", "capacity", "legal purpose"],
                difficulty="medium",
                category="factual",
            ),
        ]
        return TestDataset(
            name="legal_basic",
            description="Legal domain test dataset",
            test_cases=test_cases,
            pdf_ids=["contract_law.pdf"],
            created_at=datetime.utcnow(),
        )
    
    else:
        raise ValueError(f"Unknown domain: {domain}")


def save_test_datasets():
    """Save test datasets to disk."""
    # Create directory
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Create and save ML dataset
    ml_dataset = create_ml_test_dataset()
    ml_path = test_data_dir / "ml_comprehensive.json"
    
    with open(ml_path, "w") as f:
        json.dump(
            {
                "name": ml_dataset.name,
                "description": ml_dataset.description,
                "test_cases": [
                    {
                        "id": tc.id,
                        "question": tc.question,
                        "expected_answer": tc.expected_answer,
                        "relevant_chunks": tc.relevant_chunks,
                        "required_entities": tc.required_entities,
                        "required_facts": tc.required_facts,
                        "difficulty": tc.difficulty,
                        "category": tc.category,
                    }
                    for tc in ml_dataset.test_cases
                ],
                "pdf_ids": ml_dataset.pdf_ids,
                "created_at": ml_dataset.created_at.isoformat(),
            },
            f,
            indent=2,
        )
    
    print(f"Saved ML dataset to {ml_path}")
    
    # Create and save domain-specific datasets
    for domain in ["medical", "legal"]:
        dataset = create_domain_specific_dataset(domain)
        path = test_data_dir / f"{domain}_basic.json"
        
        with open(path, "w") as f:
            json.dump(
                {
                    "name": dataset.name,
                    "description": dataset.description,
                    "test_cases": [
                        {
                            "id": tc.id,
                            "question": tc.question,
                            "expected_answer": tc.expected_answer,
                            "relevant_chunks": tc.relevant_chunks,
                            "required_entities": tc.required_entities,
                            "required_facts": tc.required_facts,
                            "difficulty": tc.difficulty,
                            "category": tc.category,
                        }
                        for tc in dataset.test_cases
                    ],
                    "pdf_ids": dataset.pdf_ids,
                    "created_at": dataset.created_at.isoformat(),
                },
                f,
                indent=2,
            )
        
        print(f"Saved {domain} dataset to {path}")


if __name__ == "__main__":
    save_test_datasets()
