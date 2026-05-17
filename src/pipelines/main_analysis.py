"""
Main Analysis Script - Statistical Testing & Results Aggregation

Performs all statistical tests:
- Test I: Independent t-test
- Test II: Paired t-test  
- Test III: ANOVA
- Test IV: Simple Regression
- Test V: Multiple Regression
- Test VI: Logistic Regression
"""

import json
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import sys

# Add parent to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from statistical.statistical_tests import StatisticalTests
from core.evaluation import AccuracyCalculator
from core.config import RESULTS_DIR, LOGS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ComprehensiveAnalysis:
    """Perform comprehensive statistical analysis on investigation results"""
    
    def __init__(self):
        self.stats_tests = StatisticalTests()
        self.evaluator = AccuracyCalculator()
        self.all_results = []
        self.preprocessing_results = []
        self.models_results = []
        self.analysis_report = {}
    
    def load_results(self):
        """Load all investigation results"""
        try:
            # Load preprocessing results
            preprocessing_files = sorted(RESULTS_DIR.glob("preprocessing_investigation_*.json"))
            for file in preprocessing_files:
                with open(file, 'r') as f:
                    data = json.load(f)
                    self.preprocessing_results.extend(data)
                    logger.info(f"Loaded preprocessing results: {file.name}")
            
            # Load models results
            models_files = sorted(RESULTS_DIR.glob("models_investigation_*.json"))
            for file in models_files:
                with open(file, 'r') as f:
                    data = json.load(f)
                    self.models_results.extend(data)
                    logger.info(f"Loaded models results: {file.name}")
            
            logger.info(f"Total results loaded: {len(self.preprocessing_results)} preprocessing, "
                       f"{len(self.models_results)} models")
            return True
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return False
    
    def test_i_independent_ttest(self):
        """
        Test I: Independent t-test
        Compare Preprocessing Techniques vs. Recommended Models
        """
        try:
            logger.info("\n" + "=" * 60)
            logger.info("TEST I: INDEPENDENT T-TEST")
            logger.info("Preprocessing Techniques vs. Recommended Models")
            logger.info("=" * 60)
            
            if not self.preprocessing_results or not self.models_results:
                logger.warning("Insufficient data for independent t-test")
                return None
            
            # Extract accuracies
            preprocessing_accuracies = [r['before']['ssim'] for r in self.preprocessing_results]
            models_accuracies = [r.get('confidence', 0) for r in self.models_results]
            
            result = self.stats_tests.independent_ttest(
                preprocessing_accuracies,
                models_accuracies,
                "Preprocessing Techniques",
                "Recommended Models"
            )
            
            if result:
                self.analysis_report['test_i'] = result
                self._print_test_result(result)
            
            return result
        except Exception as e:
            logger.error(f"Test I failed: {e}")
            return None
    
    def test_ii_paired_ttest(self):
        """
        Test II: Paired t-test
        Compare same technique/model before and after preprocessing
        """
        try:
            logger.info("\n" + "=" * 60)
            logger.info("TEST II: PAIRED T-TEST")
            logger.info("Same Techniques Before vs. After Preprocessing")
            logger.info("=" * 60)
            
            if not self.preprocessing_results:
                logger.warning("Insufficient data for paired t-test")
                return None
            
            # Group by condition
            conditions_data = {}
            for result in self.preprocessing_results:
                condition = result['condition']
                if condition not in conditions_data:
                    conditions_data[condition] = []
                conditions_data[condition].append(result['accuracy'])
            
            # Compare normal vs. other conditions
            if 'normal' in conditions_data and len(conditions_data) > 1:
                normal_accuracies = conditions_data['normal']
                
                for condition, accuracies in conditions_data.items():
                    if condition != 'normal' and len(accuracies) == len(normal_accuracies):
                        result = self.stats_tests.paired_ttest(
                            normal_accuracies,
                            accuracies,
                            f"Normal vs. {condition}"
                        )
                        
                        if result:
                            key = f"test_ii_{condition}"
                            self.analysis_report[key] = result
                            self._print_test_result(result)
                            return result
            
            logger.warning("Unable to perform paired t-test with available data")
            return None
        except Exception as e:
            logger.error(f"Test II failed: {e}")
            return None
    
    def test_iii_anova(self):
        """
        Test III: ANOVA
        Compare three or more groups (Preprocessing vs. Models vs. Hybrid)
        """
        try:
            logger.info("\n" + "=" * 60)
            logger.info("TEST III: ONE-WAY ANOVA")
            logger.info("Multiple Approaches Comparison")
            logger.info("=" * 60)
            
            if not self.preprocessing_results or not self.models_results:
                logger.warning("Insufficient data for ANOVA")
                return None
            
            # Group models results by condition
            conditions = {}
            for result in self.models_results:
                condition = result['condition']
                if condition not in conditions:
                    conditions[condition] = {'eigenfaces': [], 'lbph': [], 'fisherface': []}
                
                model = result['model']
                if model in conditions[condition]:
                    conditions[condition][model].append(result['confidence'])
            
            # Perform ANOVA on a specific condition
            if conditions:
                condition = list(conditions.keys())[0]
                groups = []
                group_names = []
                
                for model, confidences in conditions[condition].items():
                    if confidences:
                        groups.append(confidences)
                        group_names.append(model)
                
                if len(groups) >= 3:
                    result = self.stats_tests.anova_test(*groups, group_names=group_names)
                    
                    if result:
                        self.analysis_report['test_iii'] = result
                        self._print_test_result(result)
                    
                    return result
            
            logger.warning("Unable to perform ANOVA with available data")
            return None
        except Exception as e:
            logger.error(f"Test III failed: {e}")
            return None
    
    def test_iv_simple_regression(self):
        """
        Test IV: Simple Linear Regression
        Can one model affect accuracy of others?
        Example: Eigenfaces accuracy vs. Ensemble accuracy
        """
        try:
            logger.info("\n" + "=" * 60)
            logger.info("TEST IV: SIMPLE LINEAR REGRESSION")
            logger.info("Model Accuracy Prediction (One Variable)")
            logger.info("=" * 60)
            
            if not self.models_results:
                logger.warning("Insufficient data for simple regression")
                return None
            
            # Extract specific models
            eigenfaces_scores = []
            ensemble_scores = []
            
            for result in self.models_results:
                if result['model'] == 'eigenfaces':
                    eigenfaces_scores.append(result['confidence'])
                elif result['model'] == 'ensemble':
                    ensemble_scores.append(result['confidence'])
            
            if eigenfaces_scores and ensemble_scores:
                # Ensure same length
                min_len = min(len(eigenfaces_scores), len(ensemble_scores))
                eigenfaces_scores = eigenfaces_scores[:min_len]
                ensemble_scores = ensemble_scores[:min_len]
                
                result = self.stats_tests.simple_regression(
                    eigenfaces_scores,
                    ensemble_scores,
                    "Eigenfaces Accuracy",
                    "Ensemble Accuracy"
                )
                
                if result:
                    self.analysis_report['test_iv'] = result
                    self._print_test_result(result)
                
                return result
            
            logger.warning("Unable to perform simple regression with available data")
            return None
        except Exception as e:
            logger.error(f"Test IV failed: {e}")
            return None
    
    def test_v_multiple_regression(self):
        """
        Test V: Multiple Linear Regression
        Multiple models combined to predict accuracy
        """
        try:
            logger.info("\n" + "=" * 60)
            logger.info("TEST V: MULTIPLE LINEAR REGRESSION")
            logger.info("Combined Model Prediction")
            logger.info("=" * 60)
            
            if not self.models_results:
                logger.warning("Insufficient data for multiple regression")
                return None
            
            # Group results by condition
            conditions_data = {}
            for result in self.models_results:
                condition = result['condition']
                if condition not in conditions_data:
                    conditions_data[condition] = {
                        'eigenfaces': [], 'lbph': [], 'fisherface': [],
                        'ensemble': []
                    }
                
                model = result['model']
                if model in conditions_data[condition]:
                    conditions_data[condition][model].append(result['confidence'])
            
            # Get first condition with sufficient data
            for condition, models_data in conditions_data.items():
                # Check if we have ensemble scores
                if models_data['ensemble']:
                    ensemble_scores = models_data['ensemble']
                    
                    # Build predictor matrix
                    predictor_data = []
                    for i in range(len(ensemble_scores)):
                        row = []
                        for model in ['eigenfaces', 'lbph', 'fisherface']:
                            if models_data[model] and i < len(models_data[model]):
                                row.append(models_data[model][i])
                        if len(row) == 3:
                            predictor_data.append(row)
                    
                    if predictor_data:
                        result = self.stats_tests.multiple_regression(
                            predictor_data,
                            ensemble_scores[:len(predictor_data)],
                            ['Eigenfaces', 'LBPH', 'Fisherface']
                        )
                        
                        if result:
                            self.analysis_report['test_v'] = result
                            self._print_test_result(result)
                        
                        return result
            
            logger.warning("Unable to perform multiple regression with available data")
            return None
        except Exception as e:
            logger.error(f"Test V failed: {e}")
            return None
    
    def test_vi_logistic_regression(self):
        """
        Test VI: Logistic Regression
        Predict high vs. low accuracy
        """
        try:
            logger.info("\n" + "=" * 60)
            logger.info("TEST VI: LOGISTIC REGRESSION")
            logger.info("High vs. Low Accuracy Prediction")
            logger.info("=" * 60)
            
            if not self.preprocessing_results:
                logger.warning("Insufficient data for logistic regression")
                return None
            
            # Create binary classification: high (>=median) vs. low (<median)
            accuracies = [r['accuracy'] for r in self.preprocessing_results]
            median_accuracy = np.median(accuracies)
            
            y_binary = [1 if acc >= median_accuracy else 0 for acc in accuracies]
            
            # Use numeric representation of preprocessing techniques as features
            technique_map = {'normal': 0, 'low_light': 1, 'motion_blur': 2, 'sensor_noise': 3, 'extreme_angle': 4}
            condition_map = {'normal': 0, 'low_light': 1, 'motion_blur': 2, 'sensor_noise': 3, 'extreme_angle': 4}
            
            X_data = []
            for r in self.preprocessing_results:
                technique_num = technique_map.get(r['preprocessing'], 0)
                condition_num = condition_map.get(r['condition'], 0)
                X_data.append([technique_num, condition_num])
            
            result = self.stats_tests.logistic_regression(
                X_data,
                y_binary,
                ['Preprocessing_Technique', 'Condition']
            )
            
            if result:
                self.analysis_report['test_vi'] = result
                self._print_test_result(result)
            
            return result
        except Exception as e:
            logger.error(f"Test VI failed: {e}")
            return None
    
    def run_all_tests(self):
        """Run all statistical tests"""
        try:
            logger.info("\n" + "=" * 80)
            logger.info("COMPREHENSIVE STATISTICAL ANALYSIS")
            logger.info("=" * 80)
            
            if not self.load_results():
                logger.error("Failed to load results")
                return False
            
            # Run all tests
            self.test_i_independent_ttest()
            self.test_ii_paired_ttest()
            self.test_iii_anova()
            self.test_iv_simple_regression()
            self.test_v_multiple_regression()
            self.test_vi_logistic_regression()
            
            # Save report
            self._save_analysis_report()
            self._print_final_summary()
            
            return True
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _print_test_result(self, result):
        """Print formatted test result"""
        try:
            print(f"\nTest Type: {result.get('test_type', 'Unknown')}")
            
            if result.get('test_type') == 'Independent t-test':
                print(f"Group 1: {result['group1_name']} (mean={result['group1_mean']:.2f})")
                print(f"Group 2: {result['group2_name']} (mean={result['group2_mean']:.2f})")
                print(f"t-statistic: {result['t_statistic']:.4f}")
                print(f"p-value: {result['p_value']:.4f}")
                print(f"Significant: {'Yes' if result['significant'] else 'No'}")
                print(f"Cohen's d: {result['cohens_d']:.4f}")
            
            elif result.get('test_type') == 'Paired t-test':
                print(f"Before mean: {result['before_mean']:.2f}")
                print(f"After mean: {result['after_mean']:.2f}")
                print(f"t-statistic: {result['t_statistic']:.4f}")
                print(f"p-value: {result['p_value']:.4f}")
                print(f"Cohen's d: {result['cohens_d']:.4f}")
            
            elif result.get('test_type') == 'One-way ANOVA':
                print(f"Number of groups: {result['num_groups']}")
                print(f"F-statistic: {result['f_statistic']:.4f}")
                print(f"p-value: {result['p_value']:.4f}")
                print(f"Eta-squared: {result['eta_squared']:.4f}")
            
            elif result.get('test_type') == 'Simple Linear Regression':
                print(f"Predictor: {result['predictor']}")
                print(f"Response: {result['response']}")
                print(f"Slope: {result['slope']:.4f}")
                print(f"R²: {result['r_squared']:.4f}")
                print(f"RMSE: {result['rmse']:.4f}")
            
            elif result.get('test_type') == 'Multiple Linear Regression':
                print(f"Number of features: {result['num_features']}")
                print(f"R²: {result['r_squared']:.4f}")
                print(f"Adj R²: {result['adj_r_squared']:.4f}")
                print(f"RMSE: {result['rmse']:.4f}")
            
            elif result.get('test_type') == 'Logistic Regression':
                print(f"Accuracy: {result['accuracy']:.4f}")
                print(f"Sensitivity: {result['sensitivity']:.4f}")
                print(f"Specificity: {result['specificity']:.4f}")
            
            print()
        except Exception as e:
            logger.error(f"Failed to print result: {e}")
    
    def _save_analysis_report(self):
        """Save analysis report to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = RESULTS_DIR / f"statistical_analysis_{timestamp}.json"
            
            with open(filepath, 'w') as f:
                json.dump(self.analysis_report, f, indent=2)
            
            logger.info(f"Analysis report saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save analysis report: {e}")
            return None
    
    def _print_final_summary(self):
        """Print final analysis summary"""
        try:
            print("\n" + "=" * 80)
            print("STATISTICAL ANALYSIS SUMMARY")
            print("=" * 80)
            
            print(f"\nTests Performed: {len(self.analysis_report)}")
            
            print("\nKey Findings:")
            print("-" * 80)
            
            if 'test_i' in self.analysis_report:
                t1 = self.analysis_report['test_i']
                significant = "SIGNIFICANT" if t1['significant'] else "NOT SIGNIFICANT"
                print(f"\n• Test I (Independent t-test): {significant}")
                print(f"  - p-value: {t1['p_value']:.4f}")
                print(f"  - Effect size (Cohen's d): {t1['cohens_d']:.4f}")
            
            if 'test_iv' in self.analysis_report:
                t4 = self.analysis_report['test_iv']
                print(f"\n• Test IV (Simple Regression): R² = {t4['r_squared']:.4f}")
                print(f"  - Equation: {t4.get('equation', 'N/A')}")
            
            if 'test_v' in self.analysis_report:
                t5 = self.analysis_report['test_v']
                print(f"\n• Test V (Multiple Regression): R² = {t5['r_squared']:.4f}")
                print(f"  - Adjusted R²: {t5['adj_r_squared']:.4f}")
            
            if 'test_vi' in self.analysis_report:
                t6 = self.analysis_report['test_vi']
                print(f"\n• Test VI (Logistic Regression): Accuracy = {t6['accuracy']:.4f}")
                print(f"  - Sensitivity: {t6['sensitivity']:.4f}, Specificity: {t6['specificity']:.4f}")
            
            print("\n" + "=" * 80 + "\n")
        except Exception as e:
            logger.error(f"Failed to print summary: {e}")


def main():
    """Main entry point"""
    analysis = ComprehensiveAnalysis()
    analysis.run_all_tests()


if __name__ == "__main__":
    main()
