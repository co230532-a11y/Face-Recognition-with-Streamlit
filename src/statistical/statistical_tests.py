"""
Statistical Testing Module
Tests: Independent t-test, Paired t-test, ANOVA, Regression, Logistic Regression
"""

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import json
import logging
from pathlib import Path
import sys

# Add parent to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.config import LOGS_DIR, RESULTS_DIR, SIGNIFICANCE_LEVEL

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'statistics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StatisticalTests:
    """Perform statistical tests on recognition results"""
    
    def __init__(self, significance_level=SIGNIFICANCE_LEVEL):
        self.significance_level = significance_level
        self.test_results = {}
    
    def independent_ttest(self, group1, group2, group1_name="Group1", group2_name="Group2"):
        """
        Test I: Independent t-test
        Compare means of two independent groups
        Example: Preprocessing Techniques vs. Recommended Models
        """
        try:
            group1 = np.array(group1)
            group2 = np.array(group2)
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(group1, group2)
            
            # Levene's test for equal variances
            levene_stat, levene_p = stats.levene(group1, group2)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(group1)-1)*np.std(group1)**2 + 
                                  (len(group2)-1)*np.std(group2)**2) / 
                                 (len(group1) + len(group2) - 2))
            cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
            
            result = {
                'test_type': 'Independent t-test',
                'group1_name': group1_name,
                'group2_name': group2_name,
                'group1_mean': float(np.mean(group1)),
                'group2_mean': float(np.mean(group2)),
                'group1_std': float(np.std(group1)),
                'group2_std': float(np.std(group2)),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < self.significance_level,
                'significance_level': self.significance_level,
                'cohens_d': float(cohens_d),
                'levene_p_value': float(levene_p),
                'equal_variances': levene_p > self.significance_level
            }
            
            logger.info(f"Independent t-test: {group1_name} vs {group2_name}, p-value={p_value:.4f}")
            return result
        except Exception as e:
            logger.error(f"Independent t-test failed: {e}")
            return None
    
    def paired_ttest(self, before, after, name="Paired Measurements"):
        """
        Test II: Paired t-test
        Compare same group measured twice
        Example: Same method before and after preprocessing
        """
        try:
            before = np.array(before)
            after = np.array(after)
            
            if len(before) != len(after):
                raise ValueError("Before and after arrays must have same length")
            
            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(before, after)
            
            # Wilcoxon signed-rank test (non-parametric alternative)
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(before, after)
            
            # Cohen's d for paired samples
            differences = after - before
            cohens_d = np.mean(differences) / np.std(differences) if np.std(differences) > 0 else 0
            
            result = {
                'test_type': 'Paired t-test',
                'name': name,
                'before_mean': float(np.mean(before)),
                'after_mean': float(np.mean(after)),
                'before_std': float(np.std(before)),
                'after_std': float(np.std(after)),
                'mean_difference': float(np.mean(differences)),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < self.significance_level,
                'cohens_d': float(cohens_d),
                'wilcoxon_p_value': float(wilcoxon_p)
            }
            
            logger.info(f"Paired t-test: {name}, p-value={p_value:.4f}")
            return result
        except Exception as e:
            logger.error(f"Paired t-test failed: {e}")
            return None
    
    def anova_test(self, *groups, group_names=None):
        """
        Test III: One-way ANOVA
        Compare three or more groups
        Example: Preprocessing Techniques vs. Recommended Models vs. Hybrid Approach
        """
        try:
            # Convert to numpy arrays
            groups_array = [np.array(g) for g in groups]
            
            if group_names is None:
                group_names = [f"Group_{i+1}" for i in range(len(groups_array))]
            
            # Perform ANOVA
            f_stat, p_value = stats.f_oneway(*groups_array)
            
            # Kruskal-Wallis test (non-parametric alternative)
            kw_stat, kw_p = stats.kruskal(*groups_array)
            
            # Effect size (eta squared)
            grand_mean = np.mean(np.concatenate(groups_array))
            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups_array)
            ss_total = sum((x - grand_mean)**2 for g in groups_array for x in g)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            # Group statistics
            group_stats = {}
            for i, (group, name) in enumerate(zip(groups_array, group_names)):
                group_stats[name] = {
                    'mean': float(np.mean(group)),
                    'std': float(np.std(group)),
                    'n': len(group),
                    'min': float(np.min(group)),
                    'max': float(np.max(group))
                }
            
            result = {
                'test_type': 'One-way ANOVA',
                'num_groups': len(groups_array),
                'group_names': group_names,
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < self.significance_level,
                'eta_squared': float(eta_squared),
                'kruskal_wallis_p': float(kw_p),
                'group_statistics': group_stats
            }
            
            logger.info(f"ANOVA: {len(groups_array)} groups, p-value={p_value:.4f}")
            return result
        except Exception as e:
            logger.error(f"ANOVA test failed: {e}")
            return None
    
    def correlation_analysis(self, x_data, y_data, x_name="X", y_name="Y"):
        """
        Correlation analysis between two variables
        """
        try:
            x = np.array(x_data)
            y = np.array(y_data)
            
            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(x, y)
            
            # Spearman correlation (non-parametric)
            spearman_r, spearman_p = stats.spearmanr(x, y)
            
            # Kendall tau correlation
            kendall_tau, kendall_p = stats.kendalltau(x, y)
            
            result = {
                'test_type': 'Correlation Analysis',
                'x_variable': x_name,
                'y_variable': y_name,
                'pearson_r': float(pearson_r),
                'pearson_p_value': float(pearson_p),
                'spearman_r': float(spearman_r),
                'spearman_p_value': float(spearman_p),
                'kendall_tau': float(kendall_tau),
                'kendall_p_value': float(kendall_p),
                'x_mean': float(np.mean(x)),
                'y_mean': float(np.mean(y))
            }
            
            logger.info(f"Correlation: {x_name} vs {y_name}, Pearson r={pearson_r:.4f}, p={pearson_p:.4f}")
            return result
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return None
    
    def simple_regression(self, x_data, y_data, x_name="Predictor", y_name="Response"):
        """
        Test IV: Simple Linear Regression
        See if one variable predicts another
        Example: Can one model accuracy affect all other models?
        """
        try:
            x = np.array(x_data).reshape(-1, 1)
            y = np.array(y_data)
            
            # Fit regression model
            model = LinearRegression()
            model.fit(x, y)
            
            # Predictions and residuals
            y_pred = model.predict(x)
            residuals = y - y_pred
            
            # R-squared (coefficient of determination)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Adjusted R-squared
            n = len(y)
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - 2) if n > 2 else r_squared
            
            # Standard error
            mse = np.mean(residuals**2)
            rmse = np.sqrt(mse)
            
            # Correlation coefficient
            correlation = np.corrcoef(x.flatten(), y)[0, 1]
            
            result = {
                'test_type': 'Simple Linear Regression',
                'predictor': x_name,
                'response': y_name,
                'slope': float(model.coef_[0]),
                'intercept': float(model.intercept_),
                'r_squared': float(r_squared),
                'adj_r_squared': float(adj_r_squared),
                'rmse': float(rmse),
                'correlation': float(correlation),
                'num_samples': n,
                'equation': f"y = {model.coef_[0]:.4f}*x + {model.intercept_:.4f}"
            }
            
            logger.info(f"Simple Regression: {x_name} → {y_name}, R²={r_squared:.4f}")
            return result
        except Exception as e:
            logger.error(f"Simple regression failed: {e}")
            return None
    
    def multiple_regression(self, X_data, y_data, feature_names=None):
        """
        Test V: Multiple Linear Regression
        See if two or more things predict another
        Example: Multiple models combined to predict accuracy
        """
        try:
            X = np.array(X_data)
            y = np.array(y_data)
            
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            
            if feature_names is None:
                feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
            
            # Fit regression model
            model = LinearRegression()
            model.fit(X, y)
            
            # Predictions and residuals
            y_pred = model.predict(X)
            residuals = y - y_pred
            
            # R-squared
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Adjusted R-squared
            n = len(y)
            p = X.shape[1]
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else r_squared
            
            # RMSE
            rmse = np.sqrt(np.mean(residuals**2))
            
            # Feature coefficients
            coefficients = {}
            for name, coef in zip(feature_names, model.coef_):
                coefficients[name] = float(coef)
            
            result = {
                'test_type': 'Multiple Linear Regression',
                'num_features': p,
                'feature_names': feature_names,
                'coefficients': coefficients,
                'intercept': float(model.intercept_),
                'r_squared': float(r_squared),
                'adj_r_squared': float(adj_r_squared),
                'rmse': float(rmse),
                'num_samples': n
            }
            
            logger.info(f"Multiple Regression: {p} features → Response, R²={r_squared:.4f}")
            return result
        except Exception as e:
            logger.error(f"Multiple regression failed: {e}")
            return None
    
    def logistic_regression(self, X_data, y_data, feature_names=None):
        """
        Test VI: Logistic Regression
        Predict if accuracy is high or low
        Example: What predicts high vs. low accuracy?
        """
        try:
            X = np.array(X_data)
            y = np.array(y_data)
            
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            
            if feature_names is None:
                feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit logistic regression
            model = LogisticRegression()
            model.fit(X_scaled, y)
            
            # Predictions
            y_pred = model.predict(X_scaled)
            y_pred_proba = model.predict_proba(X_scaled)
            
            # Accuracy
            accuracy = np.mean(y_pred == y)
            
            # Classification report
            tn, fp, fn, tp = 0, 0, 0, 0
            for true, pred in zip(y, y_pred):
                if true == 1 and pred == 1:
                    tp += 1
                elif true == 0 and pred == 0:
                    tn += 1
                elif true == 1 and pred == 0:
                    fn += 1
                else:
                    fp += 1
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # Feature coefficients
            coefficients = {}
            for name, coef in zip(feature_names, model.coef_[0]):
                coefficients[name] = float(coef)
            
            result = {
                'test_type': 'Logistic Regression',
                'num_features': X.shape[1],
                'feature_names': feature_names,
                'coefficients': coefficients,
                'intercept': float(model.intercept_[0]),
                'accuracy': float(accuracy),
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'precision': float(precision),
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'num_samples': len(y)
            }
            
            logger.info(f"Logistic Regression: {X.shape[1]} features, Accuracy={accuracy:.4f}")
            return result
        except Exception as e:
            logger.error(f"Logistic regression failed: {e}")
            return None
    
    def save_test_results(self, filename="statistical_tests.json"):
        """Save all test results to file"""
        try:
            filepath = RESULTS_DIR / filename
            
            with open(filepath, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            
            logger.info(f"Test results saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")
            return None


if __name__ == "__main__":
    print("Statistical tests module loaded successfully")
