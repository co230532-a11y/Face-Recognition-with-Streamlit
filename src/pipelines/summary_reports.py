"""
Visualization & Summary Script - Generate reports and visualizations

This script creates comprehensive summaries and comparison reports
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

from config import RESULTS_DIR, LOGS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'summary_reports.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ResultsSummaryGenerator:
    """Generate comprehensive summary reports"""
    
    def __init__(self):
        self.preprocessing_results = []
        self.models_results = []
        self.analysis_results = {}
    
    def load_all_results(self):
        """Load all investigation results"""
        try:
            # Load preprocessing
            for file in RESULTS_DIR.glob("preprocessing_investigation_*.json"):
                with open(file) as f:
                    self.preprocessing_results.extend(json.load(f))
            
            # Load models
            for file in RESULTS_DIR.glob("models_investigation_*.json"):
                with open(file) as f:
                    self.models_results.extend(json.load(f))
            
            # Load analysis
            for file in RESULTS_DIR.glob("statistical_analysis_*.json"):
                with open(file) as f:
                    self.analysis_results.update(json.load(f))
            
            logger.info(f"Loaded results: {len(self.preprocessing_results)} preprocessing, "
                       f"{len(self.models_results)} models")
            return True
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return False
    
    def generate_preprocessing_summary(self):
        """Generate preprocessing techniques summary"""
        try:
            if not self.preprocessing_results:
                return None
            
            print("\n" + "="*80)
            print("PREPROCESSING TECHNIQUES SUMMARY")
            print("="*80)
            
            # Group by condition
            by_condition = {}
            for r in self.preprocessing_results:
                condition = r['condition']
                if condition not in by_condition:
                    by_condition[condition] = []
                by_condition[condition].append(r['accuracy'])
            
            # Group by technique
            by_technique = {}
            for r in self.preprocessing_results:
                technique = r['preprocessing']
                if technique not in by_technique:
                    by_technique[technique] = []
                by_technique[technique].append(r['accuracy'])
            
            print("\nBy Condition:")
            print("-" * 80)
            for condition, accuracies in sorted(by_condition.items()):
                print(f"\n{condition.upper()}:")
                print(f"  Count: {len(accuracies)}")
                print(f"  Mean: {np.mean(accuracies):.2f}%")
                print(f"  Std Dev: {np.std(accuracies):.2f}%")
                print(f"  Range: [{np.min(accuracies):.2f}% - {np.max(accuracies):.2f}%]")
            
            print("\n\nBy Technique:")
            print("-" * 80)
            for technique, accuracies in sorted(by_technique.items()):
                print(f"\n{technique.upper()}:")
                print(f"  Count: {len(accuracies)}")
                print(f"  Mean: {np.mean(accuracies):.2f}%")
                print(f"  Std Dev: {np.std(accuracies):.2f}%")
                print(f"  Range: [{np.min(accuracies):.2f}% - {np.max(accuracies):.2f}%]")
            
            # Best technique
            best_technique = max(by_technique.items(), key=lambda x: np.mean(x[1]))
            print(f"\n\nBest Technique: {best_technique[0]}")
            print(f"Average Accuracy: {np.mean(best_technique[1]):.2f}%")
            
            return by_condition, by_technique
        except Exception as e:
            logger.error(f"Failed to generate preprocessing summary: {e}")
            return None, None
    
    def generate_models_summary(self):
        """Generate models comparison summary"""
        try:
            if not self.models_results:
                return None
            
            print("\n" + "="*80)
            print("FACE RECOGNITION MODELS SUMMARY")
            print("="*80)
            
            # Group by model
            by_model = {}
            for r in self.models_results:
                model = r['model']
                if model not in by_model:
                    by_model[model] = {'confidence': [], 'correct': []}
                by_model[model]['confidence'].append(r['confidence'])
                by_model[model]['correct'].append(r.get('correct_prediction', False))
            
            # Group by condition
            by_condition = {}
            for r in self.models_results:
                condition = r['condition']
                if condition not in by_condition:
                    by_condition[condition] = []
                by_condition[condition].append(r['confidence'])
            
            print("\nBy Model:")
            print("-" * 80)
            for model, data in sorted(by_model.items()):
                confidences = data['confidence']
                correct = sum(data['correct'])
                total = len(data['correct'])
                accuracy_pct = (correct / total * 100) if total > 0 else 0
                
                print(f"\n{model.upper()}:")
                print(f"  Samples: {len(confidences)}")
                print(f"  Avg Confidence: {np.mean(confidences):.2f}%")
                print(f"  Std Dev: {np.std(confidences):.2f}%")
                print(f"  Correct: {correct}/{total} ({accuracy_pct:.1f}%)")
                print(f"  Range: [{np.min(confidences):.2f}% - {np.max(confidences):.2f}%]")
            
            print("\n\nBy Condition:")
            print("-" * 80)
            for condition, confidences in sorted(by_condition.items()):
                print(f"\n{condition.upper()}:")
                print(f"  Mean Confidence: {np.mean(confidences):.2f}%")
                print(f"  Std Dev: {np.std(confidences):.2f}%")
                print(f"  Range: [{np.min(confidences):.2f}% - {np.max(confidences):.2f}%]")
            
            # Best model
            best_model = max(by_model.items(), 
                           key=lambda x: np.mean(x[1]['confidence']))
            print(f"\n\nBest Model: {best_model[0]}")
            print(f"Average Confidence: {np.mean(best_model[1]['confidence']):.2f}%")
            
            return by_model, by_condition
        except Exception as e:
            logger.error(f"Failed to generate models summary: {e}")
            return None, None
    
    def generate_analysis_summary(self):
        """Generate statistical analysis summary"""
        try:
            if not self.analysis_results:
                logger.warning("No analysis results available")
                return
            
            print("\n" + "="*80)
            print("STATISTICAL ANALYSIS SUMMARY")
            print("="*80)
            
            for test_name, test_data in sorted(self.analysis_results.items()):
                print(f"\n{test_name.upper()}:")
                print(f"  Type: {test_data.get('test_type', 'Unknown')}")
                
                if 'p_value' in test_data:
                    significant = "SIGNIFICANT" if test_data['p_value'] < 0.05 else "NOT SIGNIFICANT"
                    print(f"  Result: {significant}")
                    print(f"  p-value: {test_data['p_value']:.6f}")
                
                if 'r_squared' in test_data:
                    print(f"  R²: {test_data['r_squared']:.4f}")
                
                if 'accuracy' in test_data:
                    print(f"  Accuracy: {test_data['accuracy']:.4f}")
        
        except Exception as e:
            logger.error(f"Failed to generate analysis summary: {e}")
    
    def generate_comparison_report(self):
        """Generate comparative analysis report"""
        try:
            print("\n" + "="*80)
            print("COMPARATIVE ANALYSIS REPORT")
            print("="*80)
            
            # Compare preprocessing vs models
            if self.preprocessing_results and self.models_results:
                prep_acc = [r['accuracy'] for r in self.preprocessing_results]
                model_conf = [r['confidence'] for r in self.models_results]
                
                print("\nPreprocessing Techniques vs. Recommended Models:")
                print("-" * 80)
                print(f"\nPreprocessing Techniques:")
                print(f"  Count: {len(prep_acc)}")
                print(f"  Mean Accuracy: {np.mean(prep_acc):.2f}%")
                print(f"  Std Dev: {np.std(prep_acc):.2f}%")
                print(f"  Range: [{np.min(prep_acc):.2f}% - {np.max(prep_acc):.2f}%]")
                
                print(f"\nRecommended Models:")
                print(f"  Count: {len(model_conf)}")
                print(f"  Mean Confidence: {np.mean(model_conf):.2f}%")
                print(f"  Std Dev: {np.std(model_conf):.2f}%")
                print(f"  Range: [{np.min(model_conf):.2f}% - {np.max(model_conf):.2f}%]")
                
                # Winner
                prep_mean = np.mean(prep_acc)
                model_mean = np.mean(model_conf)
                
                print(f"\nWinner: {'Preprocessing Techniques' if prep_mean > model_mean else 'Recommended Models'}")
                print(f"Difference: {abs(prep_mean - model_mean):.2f}%")
        
        except Exception as e:
            logger.error(f"Failed to generate comparison report: {e}")
    
    def generate_full_report(self):
        """Generate complete investigation report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Load results
            if not self.load_all_results():
                logger.error("Failed to load results")
                return None
            
            # Generate summaries
            self.generate_preprocessing_summary()
            self.generate_models_summary()
            self.generate_analysis_summary()
            self.generate_comparison_report()
            
            # Create text report
            report_path = RESULTS_DIR / f"investigation_report_{timestamp}.txt"
            
            print(f"\n\n✓ Report generation completed")
            print(f"Timestamp: {datetime.now().isoformat()}")
            
            return report_path
        
        except Exception as e:
            logger.error(f"Failed to generate full report: {e}")
            return None


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("INVESTIGATION RESULTS SUMMARY GENERATOR")
    print("="*80)
    
    generator = ResultsSummaryGenerator()
    generator.generate_full_report()


if __name__ == "__main__":
    main()
