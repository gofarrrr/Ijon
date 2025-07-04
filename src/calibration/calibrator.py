"""
Calibration system for optimizing RAG system parameters.

This module provides tools for tuning and optimizing the RAG system
based on evaluation results.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize

from src.config import Settings, get_settings
from src.rag.pipeline import RAGPipeline
from src.utils.logging import get_logger
from tests.test_evaluation import RAGEvaluator, TestDataset

logger = get_logger(__name__)


@dataclass
class CalibrationParameter:
    """Parameter to be calibrated."""
    
    name: str
    current_value: Any
    min_value: Any
    max_value: Any
    step_size: Any
    description: str
    parameter_type: str  # float, int, bool, categorical
    

@dataclass 
class CalibrationResult:
    """Result of calibration process."""
    
    parameter_name: str
    original_value: Any
    optimal_value: Any
    improvement: float  # Percentage improvement
    evaluation_scores: List[float]
    tested_values: List[Any]
    

@dataclass
class CalibrationProfile:
    """Complete calibration profile."""
    
    name: str
    description: str
    parameters: Dict[str, Any]
    evaluation_score: float
    created_at: datetime = field(default_factory=datetime.utcnow)
    test_dataset: Optional[str] = None
    

class RAGCalibrator:
    """Calibration system for RAG parameters."""
    
    def __init__(self, pipeline: RAGPipeline, evaluator: RAGEvaluator):
        """Initialize calibrator."""
        self.pipeline = pipeline
        self.evaluator = evaluator
        self.settings = get_settings()
        self.calibration_history: List[CalibrationResult] = []
        
    def get_tunable_parameters(self) -> List[CalibrationParameter]:
        """Get list of tunable parameters."""
        return [
            # Chunking parameters
            CalibrationParameter(
                name="chunk_size",
                current_value=self.settings.chunk_size,
                min_value=200,
                max_value=2000,
                step_size=100,
                description="Size of text chunks",
                parameter_type="int",
            ),
            CalibrationParameter(
                name="chunk_overlap",
                current_value=self.settings.chunk_overlap,
                min_value=0,
                max_value=500,
                step_size=50,
                description="Overlap between chunks",
                parameter_type="int",
            ),
            
            # Retrieval parameters
            CalibrationParameter(
                name="retrieval_top_k",
                current_value=5,
                min_value=3,
                max_value=15,
                step_size=1,
                description="Number of chunks to retrieve",
                parameter_type="int",
            ),
            CalibrationParameter(
                name="retrieval_min_score",
                current_value=0.5,
                min_value=0.0,
                max_value=0.9,
                step_size=0.1,
                description="Minimum similarity score for retrieval",
                parameter_type="float",
            ),
            
            # Knowledge graph parameters
            CalibrationParameter(
                name="entity_confidence_threshold",
                current_value=0.7,
                min_value=0.5,
                max_value=0.95,
                step_size=0.05,
                description="Minimum confidence for entity extraction",
                parameter_type="float",
            ),
            CalibrationParameter(
                name="graph_traversal_depth",
                current_value=2,
                min_value=1,
                max_value=4,
                step_size=1,
                description="Maximum graph traversal depth",
                parameter_type="int",
            ),
            
            # Agent parameters
            CalibrationParameter(
                name="agent_temperature",
                current_value=0.7,
                min_value=0.0,
                max_value=1.0,
                step_size=0.1,
                description="Agent response temperature",
                parameter_type="float",
            ),
            CalibrationParameter(
                name="agent_max_iterations",
                current_value=3,
                min_value=1,
                max_value=5,
                step_size=1,
                description="Maximum agent reasoning iterations",
                parameter_type="int",
            ),
        ]
    
    async def calibrate_parameter(
        self,
        parameter: CalibrationParameter,
        test_dataset: TestDataset,
        optimization_metric: str = "overall_score",
    ) -> CalibrationResult:
        """
        Calibrate a single parameter.
        
        Args:
            parameter: Parameter to calibrate
            test_dataset: Dataset for evaluation
            optimization_metric: Metric to optimize
            
        Returns:
            Calibration result
        """
        logger.info(f"Calibrating parameter: {parameter.name}")
        
        # Generate test values
        test_values = self._generate_test_values(parameter)
        evaluation_scores = []
        
        # Test each value
        for value in test_values:
            # Update parameter
            self._update_parameter(parameter.name, value)
            
            # Evaluate
            results = await self.evaluator.evaluate_dataset(
                test_dataset,
                use_agent=True,
                save_results=False,
            )
            
            # Extract score
            if optimization_metric == "overall_score":
                score = results["overall_score"]["mean"]
            else:
                score = results["metrics"].get(f"avg_{optimization_metric}", 0)
            
            evaluation_scores.append(score)
            
            logger.debug(
                f"Tested {parameter.name}={value}, score={score:.3f}"
            )
        
        # Find optimal value
        optimal_idx = np.argmax(evaluation_scores)
        optimal_value = test_values[optimal_idx]
        optimal_score = evaluation_scores[optimal_idx]
        
        # Calculate improvement
        original_idx = test_values.index(parameter.current_value) if parameter.current_value in test_values else 0
        original_score = evaluation_scores[original_idx]
        improvement = ((optimal_score - original_score) / original_score) * 100 if original_score > 0 else 0
        
        # Create result
        result = CalibrationResult(
            parameter_name=parameter.name,
            original_value=parameter.current_value,
            optimal_value=optimal_value,
            improvement=improvement,
            evaluation_scores=evaluation_scores,
            tested_values=test_values,
        )
        
        # Update to optimal value
        self._update_parameter(parameter.name, optimal_value)
        
        self.calibration_history.append(result)
        
        return result
    
    async def auto_calibrate(
        self,
        test_dataset: TestDataset,
        parameters: Optional[List[str]] = None,
        optimization_metric: str = "overall_score",
    ) -> CalibrationProfile:
        """
        Automatically calibrate multiple parameters.
        
        Args:
            test_dataset: Dataset for evaluation
            parameters: List of parameter names to calibrate (None for all)
            optimization_metric: Metric to optimize
            
        Returns:
            Optimal calibration profile
        """
        logger.info("Starting automatic calibration")
        
        # Get parameters to calibrate
        all_params = self.get_tunable_parameters()
        if parameters:
            params_to_calibrate = [
                p for p in all_params if p.name in parameters
            ]
        else:
            params_to_calibrate = all_params
        
        # Baseline evaluation
        baseline_results = await self.evaluator.evaluate_dataset(
            test_dataset,
            use_agent=True,
            save_results=False,
        )
        baseline_score = baseline_results["overall_score"]["mean"]
        
        logger.info(f"Baseline score: {baseline_score:.3f}")
        
        # Calibrate each parameter
        optimal_values = {}
        for param in params_to_calibrate:
            result = await self.calibrate_parameter(
                param,
                test_dataset,
                optimization_metric,
            )
            optimal_values[param.name] = result.optimal_value
            
            logger.info(
                f"Optimized {param.name}: {result.original_value} -> {result.optimal_value} "
                f"(+{result.improvement:.1f}%)"
            )
        
        # Final evaluation with all optimal values
        final_results = await self.evaluator.evaluate_dataset(
            test_dataset,
            use_agent=True,
            save_results=False,
        )
        final_score = final_results["overall_score"]["mean"]
        
        total_improvement = ((final_score - baseline_score) / baseline_score) * 100
        
        logger.info(
            f"Calibration complete. Final score: {final_score:.3f} "
            f"(+{total_improvement:.1f}% improvement)"
        )
        
        # Create calibration profile
        profile = CalibrationProfile(
            name=f"auto_calibration_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            description=f"Automatic calibration optimizing {optimization_metric}",
            parameters=optimal_values,
            evaluation_score=final_score,
            test_dataset=test_dataset.name,
        )
        
        # Save profile
        self._save_profile(profile)
        
        return profile
    
    async def grid_search(
        self,
        parameters: List[CalibrationParameter],
        test_dataset: TestDataset,
        optimization_metric: str = "overall_score",
    ) -> CalibrationProfile:
        """
        Perform grid search over multiple parameters.
        
        Args:
            parameters: Parameters to search over
            test_dataset: Dataset for evaluation  
            optimization_metric: Metric to optimize
            
        Returns:
            Optimal calibration profile
        """
        logger.info(f"Starting grid search over {len(parameters)} parameters")
        
        # Generate all combinations
        param_values = {}
        for param in parameters:
            param_values[param.name] = self._generate_test_values(param)
        
        # Create grid
        from itertools import product
        
        param_names = list(param_values.keys())
        value_combinations = list(product(*param_values.values()))
        
        logger.info(f"Testing {len(value_combinations)} combinations")
        
        # Test each combination
        best_score = -float('inf')
        best_combination = None
        
        for i, values in enumerate(value_combinations):
            # Update all parameters
            for name, value in zip(param_names, values):
                self._update_parameter(name, value)
            
            # Evaluate
            results = await self.evaluator.evaluate_dataset(
                test_dataset,
                use_agent=True,
                save_results=False,
            )
            
            if optimization_metric == "overall_score":
                score = results["overall_score"]["mean"]
            else:
                score = results["metrics"].get(f"avg_{optimization_metric}", 0)
            
            logger.debug(
                f"Combination {i+1}/{len(value_combinations)}: "
                f"{dict(zip(param_names, values))}, score={score:.3f}"
            )
            
            if score > best_score:
                best_score = score
                best_combination = dict(zip(param_names, values))
        
        # Create profile with best combination
        profile = CalibrationProfile(
            name=f"grid_search_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            description=f"Grid search optimizing {optimization_metric}",
            parameters=best_combination,
            evaluation_score=best_score,
            test_dataset=test_dataset.name,
        )
        
        # Apply best combination
        for name, value in best_combination.items():
            self._update_parameter(name, value)
        
        # Save profile
        self._save_profile(profile)
        
        return profile
    
    def load_profile(self, profile_name: str) -> CalibrationProfile:
        """Load and apply a calibration profile."""
        profile_path = Path("calibration_profiles") / f"{profile_name}.json"
        
        with open(profile_path, "r") as f:
            data = json.load(f)
        
        profile = CalibrationProfile(**data)
        
        # Apply parameters
        for name, value in profile.parameters.items():
            self._update_parameter(name, value)
        
        logger.info(f"Loaded calibration profile: {profile_name}")
        
        return profile
    
    def _generate_test_values(self, parameter: CalibrationParameter) -> List[Any]:
        """Generate test values for a parameter."""
        if parameter.parameter_type in ["int", "float"]:
            # Generate range of values
            values = []
            current = parameter.min_value
            
            while current <= parameter.max_value:
                if parameter.parameter_type == "int":
                    values.append(int(current))
                else:
                    values.append(float(current))
                current += parameter.step_size
            
            # Ensure current value is included
            if parameter.current_value not in values:
                values.append(parameter.current_value)
                values.sort()
            
            return values
            
        elif parameter.parameter_type == "bool":
            return [True, False]
            
        else:
            # Categorical - would need custom handling
            return [parameter.current_value]
    
    def _update_parameter(self, name: str, value: Any) -> None:
        """Update a parameter value."""
        # Update settings
        if hasattr(self.settings, name):
            setattr(self.settings, name, value)
        
        # Update pipeline components as needed
        if name == "chunk_size" or name == "chunk_overlap":
            # Would update chunking strategy
            pass
        elif name == "retrieval_top_k":
            # Would update retrieval settings
            pass
        elif name == "entity_confidence_threshold":
            # Would update knowledge extractor
            if hasattr(self.pipeline, 'knowledge_extractor'):
                self.pipeline.knowledge_extractor.confidence_threshold = value
        # Add more parameter updates as needed
    
    def _save_profile(self, profile: CalibrationProfile) -> None:
        """Save calibration profile."""
        # Create directory
        profiles_dir = Path("calibration_profiles")
        profiles_dir.mkdir(exist_ok=True)
        
        # Save profile
        profile_path = profiles_dir / f"{profile.name}.json"
        
        with open(profile_path, "w") as f:
            json.dump(
                {
                    "name": profile.name,
                    "description": profile.description,
                    "parameters": profile.parameters,
                    "evaluation_score": profile.evaluation_score,
                    "created_at": profile.created_at.isoformat(),
                    "test_dataset": profile.test_dataset,
                },
                f,
                indent=2,
            )
        
        logger.info(f"Saved calibration profile: {profile_path}")


class ConfidenceCalibrator:
    """Calibrate confidence scores for better reliability."""
    
    def __init__(self):
        """Initialize confidence calibrator."""
        self.calibration_data: List[Tuple[float, bool]] = []
        
    def add_sample(self, predicted_confidence: float, was_correct: bool) -> None:
        """Add a calibration sample."""
        self.calibration_data.append((predicted_confidence, was_correct))
    
    def calibrate(self) -> Dict[str, Any]:
        """Calculate calibration metrics and adjustments."""
        if not self.calibration_data:
            return {"error": "No calibration data"}
        
        # Group by confidence bins
        bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(len(bins) - 1):
            bin_samples = [
                (conf, correct)
                for conf, correct in self.calibration_data
                if bins[i] <= conf < bins[i + 1]
            ]
            
            if bin_samples:
                accuracies = [correct for _, correct in bin_samples]
                confidences = [conf for conf, _ in bin_samples]
                
                bin_accuracies.append(np.mean(accuracies))
                bin_confidences.append(np.mean(confidences))
                bin_counts.append(len(bin_samples))
            else:
                bin_accuracies.append(None)
                bin_confidences.append(None)
                bin_counts.append(0)
        
        # Calculate calibration error
        ece = 0  # Expected Calibration Error
        total_samples = len(self.calibration_data)
        
        for i in range(len(bin_accuracies)):
            if bin_accuracies[i] is not None:
                ece += (bin_counts[i] / total_samples) * abs(
                    bin_accuracies[i] - bin_confidences[i]
                )
        
        # Fit calibration curve
        valid_points = [
            (conf, acc)
            for conf, acc, count in zip(bin_confidences, bin_accuracies, bin_counts)
            if conf is not None and count > 5
        ]
        
        if len(valid_points) >= 3:
            x = [p[0] for p in valid_points]
            y = [p[1] for p in valid_points]
            
            # Fit isotonic regression
            from sklearn.isotonic import IsotonicRegression
            
            iso_reg = IsotonicRegression(out_of_bounds="clip")
            iso_reg.fit(x, y)
            
            calibration_function = iso_reg
        else:
            calibration_function = None
        
        return {
            "expected_calibration_error": ece,
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
            "calibration_function": calibration_function,
            "num_samples": len(self.calibration_data),
        }
