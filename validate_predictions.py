import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import argparse
import logging
import datetime
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_biocondensat_analysis.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Expanded positive controls with more S. cerevisiae proteins
positive_controls = {
    # S. cerevisiae condensate-forming proteins
    'MTR3': {
        'forms_condensate': True,
        'condensate_type': ['Nuclear body', 'Stress granule'],
        'has_prion_domain': False,
        'is_amyloidogenic': False
    },
    'PABP': {
        'forms_condensate': True,
        'condensate_type': 'Stress granule',
        'has_prion_domain': False,
        'is_amyloidogenic': False
    },
    'PUB1': {
        'forms_condensate': True,
        'condensate_type': 'Stress granule',
        'has_prion_domain': True,  # Can form [PUB1+]
        'is_amyloidogenic': True
    },
    'LSM4': {
        'forms_condensate': True,
        'condensate_type': 'P-body',
        'has_prion_domain': False,
        'is_amyloidogenic': False
    },
    'DCP2': {
        'forms_condensate': True,
        'condensate_type': 'P-body',
        'has_prion_domain': False,
        'is_amyloidogenic': False
    },
    'EDC3': {
        'forms_condensate': True,
        'condensate_type': 'P-body',
        'has_prion_domain': False,
        'is_amyloidogenic': False
    },
    'WHI3': {
        'forms_condensate': True,
        'condensate_type': 'mRNP granule',
        'has_prion_domain': False,
        'is_amyloidogenic': False
    },
    'GIS2': {
        'forms_condensate': True,
        'condensate_type': 'Stress granule',
        'has_prion_domain': False,
        'is_amyloidogenic': False
    },
    
    # Prion proteins
    'ERF3': {
        'forms_condensate': True,  # Can form condensates
        'condensate_type': 'Prion aggregate',
        'has_prion_domain': True,
        'prion_type': 'Q/N-rich',
        'is_amyloidogenic': True,
        'amyloid_type': 'Functional amyloid'
    },
    'URE2': {
        'forms_condensate': False,
        'has_prion_domain': True,
        'prion_type': 'Q/N-rich',
        'is_amyloidogenic': True,
        'amyloid_type': 'Functional amyloid'
    },
    'RNQ1': {
        'forms_condensate': False,
        'has_prion_domain': True,
        'prion_type': 'Q/N-rich',
        'is_amyloidogenic': True,
        'amyloid_type': 'Functional amyloid'
    },
    'NEW1': {
        'forms_condensate': False,
        'has_prion_domain': True,
        'prion_type': 'Q/N-rich',
        'is_amyloidogenic': True,
        'amyloid_type': 'Functional amyloid'
    },
    'SWI1': {
        'forms_condensate': False,
        'has_prion_domain': True,
        'prion_type': 'Q/N-rich',
        'is_amyloidogenic': True,
        'amyloid_type': 'Functional amyloid'
    },
    
    # Human disease proteins
    'FUS': {
        'forms_condensate': True,
        'condensate_type': 'Stress granule',
        'has_prion_domain': True,
        'prion_type': 'SG-rich',
        'is_amyloidogenic': True,
        'amyloid_type': 'Pathological amyloid'
    },
    'NSR1': {
        'forms_condensate': True,
        'condensate_type': 'Stress granule',
        'has_prion_domain': True,
        'prion_type': 'Mixed',
        'is_amyloidogenic': True,
        'amyloid_type': 'Pathological amyloid'
    },
    'TDP43': {
        'forms_condensate': True,
        'condensate_type': 'Stress granule',
        'has_prion_domain': True,
        'prion_type': 'Mixed',
        'is_amyloidogenic': True,
        'amyloid_type': 'Pathological amyloid'
    }
}

# Expanded negative controls
negative_controls = {
    # S. cerevisiae stable proteins
    'ADH1': {
        'forms_condensate': False,
        'has_prion_domain': False,
        'is_amyloidogenic': False,
        'expected_disorder_percent': '<20'
    },
    'PGK': {
        'forms_condensate': False,
        'has_prion_domain': False,
        'is_amyloidogenic': False,
        'expected_disorder_percent': '<15'
    },
    'ENO2': {
        'forms_condensate': False,
        'has_prion_domain': False,
        'is_amyloidogenic': False,
        'expected_disorder_percent': '<20'
    },
    'TPIS': {
        'forms_condensate': False,
        'has_prion_domain': False,
        'is_amyloidogenic': False,
        'expected_disorder_percent': '<15'
    },
    'CDC19': {
        'forms_condensate': False,
        'has_prion_domain': False,
        'is_amyloidogenic': False,
        'expected_disorder_percent': '<20'
    },
    
    # Human stable proteins
    'CYC7': {  # Cytochrome c
        'forms_condensate': False,
        'has_prion_domain': False,
        'is_amyloidogenic': False,
        'expected_disorder_percent': '<10'
    }
}

def extract_protein_id(full_id):
    """Extract clean protein ID from various formats"""
    if '|' in full_id:
        # Handle UniProt format: sp|P00000|PROTEIN_SPECIES
        parts = full_id.split('|')
        if len(parts) >= 3:
            protein_part = parts[2]
            # Remove species suffix if present
            if '_' in protein_part:
                return protein_part.split('_')[0]
            return protein_part
        return parts[-1]
    return full_id

def validate_predictions(results_df, control_dict, control_type='positive'):
    """Validate predictions against known controls"""
    validation_results = []
    
    # Create a clean protein ID column
    results_df['protein_id_clean'] = results_df['protein_id'].apply(extract_protein_id)
    
    for protein_id, expected in control_dict.items():
        try_search = results_df[results_df['protein_id_clean'] == protein_id]
        if try_search.empty:
            try_search = results_df[results_df['protein_id_clean'] == protein_id[::-1]]
        
        matching_rows = try_search.copy()
        
        if not matching_rows.empty:
            pred = matching_rows.iloc[0]
            
            result = {
                'protein_id': protein_id,
                'protein_id_full': pred['protein_id'],
                'control_type': control_type,
                
                # Condensate validation
                'condensate_expected': expected.get('forms_condensate', None),
                'condensate_predicted': pred['forms_condensate'],
                'condensate_score': pred['condensate_score'],
                'condensate_type_pred': pred['condensate_type'],
                'condensate_type_expected': expected.get('condensate_type', None),
                
                # Prion validation
                'prion_expected': expected.get('has_prion_domain', None),
                'prion_predicted': pred['has_prion_domain'],
                'prion_score': pred['prion_score'],
                'prion_type_pred': pred['prion_type'],
                'prion_type_expected': expected.get('prion_type', None),
                
                # Amyloid validation
                'amyloid_expected': expected.get('is_amyloidogenic', None),
                'amyloid_predicted': pred['is_amyloidogenic'],
                'amyloid_propensity': pred['amyloid_propensity'],
                'amyloid_type_pred': pred['amyloid_type'],
                'amyloid_type_expected': expected.get('amyloid_type', None),
                
                # Additional features
                'disorder_percent': pred['disorder_percent'],
                'lcr_percent': pred['lcr_percent'],
                'aggregation_type': pred['aggregation_type'],
                'structure_type': pred['structure_type'],
                'type_confidence': pred.get('type_confidence', 'N/A')
            }
            
            validation_results.append(result)
        else:
            logger.warning(f"{protein_id} not found in predictions")
    
    return pd.DataFrame(validation_results)

def calculate_metrics_per_feature(validation_df):
    """Calculate comprehensive metrics for each feature"""
    metrics = {}
    
    for feature in ['condensate', 'prion', 'amyloid']:
        expected_col = f'{feature}_expected'
        predicted_col = f'{feature}_predicted'
        score_col = f'{feature}_score' if feature != 'amyloid' else 'amyloid_propensity'
        
        # Filter out None values
        mask = validation_df[expected_col].notna()
        if mask.sum() == 0:
            continue
            
        y_true = validation_df[mask][expected_col].astype(bool).astype(int)
        y_pred = validation_df[mask][predicted_col].astype(bool).astype(int)
        y_scores = validation_df[mask][score_col]
        
        if len(np.unique(y_true)) < 2:
            logger.warning(f"Only one class present for {feature}, skipping metrics")
            continue
            
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle single class case
            if y_true.sum() == 0:  # All negatives
                tn = (y_pred == 0).sum()
                fp = (y_pred == 1).sum()
                fn = tp = 0
            else:  # All positives
                tp = (y_pred == 1).sum()
                fn = (y_pred == 0).sum()
                tn = fp = 0
        
        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        metrics[feature] = {
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
            'mcc': calculate_mcc(tp, tn, fp, fn),
            'n_samples': len(y_true),
            'n_positive': y_true.sum(),
            'n_negative': len(y_true) - y_true.sum()
        }
        
        # Add ROC and PR metrics if we have scores
        if len(np.unique(y_true)) == 2:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                metrics[feature]['auc_roc'] = auc(fpr, tpr)
                
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
                metrics[feature]['auc_pr'] = auc(recall_curve, precision_curve)
            except:
                metrics[feature]['auc_roc'] = 'N/A'
                metrics[feature]['auc_pr'] = 'N/A'
    
    return metrics

def calculate_mcc(tp, tn, fp, fn):
    """Calculate Matthews Correlation Coefficient"""
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator if denominator != 0 else 0

def identify_misclassifications(validation_df):
    """Identify and categorize misclassified proteins"""
    misclassifications = {
        'condensate': {'FP': [], 'FN': [], 'details': []},
        'prion': {'FP': [], 'FN': [], 'details': []},
        'amyloid': {'FP': [], 'FN': [], 'details': []}
    }
    
    for feature in ['condensate', 'prion', 'amyloid']:
        expected_col = f'{feature}_expected'
        predicted_col = f'{feature}_predicted'
        score_col = f'{feature}_score' if feature != 'amyloid' else 'amyloid_propensity'
        
        # Find false positives and false negatives
        mask = validation_df[expected_col].notna()
        feature_df = validation_df[mask]
        
        # False Positives
        fp_mask = (feature_df[expected_col] == False) & (feature_df[predicted_col] == True)
        fp_proteins = feature_df[fp_mask]
        
        # False Negatives  
        fn_mask = (feature_df[expected_col] == True) & (feature_df[predicted_col] == False)
        fn_proteins = feature_df[fn_mask]
        
        # Store protein IDs
        misclassifications[feature]['FP'] = fp_proteins['protein_id'].tolist()
        misclassifications[feature]['FN'] = fn_proteins['protein_id'].tolist()
        
        # Detailed analysis
        for _, protein in fp_proteins.iterrows():
            misclassifications[feature]['details'].append({
                'protein_id': protein['protein_id'],
                'type': 'FP',
                'score': protein[score_col],
                'disorder_percent': protein['disorder_percent'],
                'lcr_percent': protein['lcr_percent'],
                'structure_type': protein['structure_type'],
                'aggregation_type': protein['aggregation_type'],
                'confidence': protein.get('type_confidence', 'N/A')
            })
            
        for _, protein in fn_proteins.iterrows():
            misclassifications[feature]['details'].append({
                'protein_id': protein['protein_id'],
                'type': 'FN',
                'score': protein[score_col],
                'disorder_percent': protein['disorder_percent'],
                'lcr_percent': protein['lcr_percent'],
                'structure_type': protein['structure_type'],
                'aggregation_type': protein['aggregation_type'],
                'confidence': protein.get('type_confidence', 'N/A')
            })
    
    return misclassifications

def analyze_misclassification_patterns(misclassifications, validation_df):
    """Find common patterns in misclassified proteins"""
    patterns = {}
    
    for feature in ['condensate', 'prion', 'amyloid']:
        feature_patterns = {
            'FP_patterns': {},
            'FN_patterns': {},
            'recommendations': []
        }
        
        # Analyze False Positives
        fp_details = [d for d in misclassifications[feature]['details'] if d['type'] == 'FP']
        if fp_details:
            fp_df = pd.DataFrame(fp_details)
            
            feature_patterns['FP_patterns'] = {
                'count': len(fp_df),
                'avg_score': fp_df['score'].mean(),
                'std_score': fp_df['score'].std(),
                'avg_disorder': fp_df['disorder_percent'].mean(),
                'avg_lcr': fp_df['lcr_percent'].mean(),
                'common_structure_type': fp_df['structure_type'].mode().iloc[0] if not fp_df['structure_type'].mode().empty else 'N/A',
                'score_range': (fp_df['score'].min(), fp_df['score'].max())
            }
            
            if fp_df['score'].mean() < 0.6:
                feature_patterns['recommendations'].append(
                    f"Consider raising threshold for {feature} (current FP avg: {fp_df['score'].mean():.3f})"
                )
        
        # Analyze False Negatives
        fn_details = [d for d in misclassifications[feature]['details'] if d['type'] == 'FN']
        if fn_details:
            fn_df = pd.DataFrame(fn_details)
            
            feature_patterns['FN_patterns'] = {
                'count': len(fn_df),
                'avg_score': fn_df['score'].mean(),
                'std_score': fn_df['score'].std(),
                'avg_disorder': fn_df['disorder_percent'].mean(),
                'avg_lcr': fn_df['lcr_percent'].mean(),
                'common_structure_type': fn_df['structure_type'].mode().iloc[0] if not fn_df['structure_type'].mode().empty else 'N/A',
                'score_range': (fn_df['score'].min(), fn_df['score'].max())
            }
            
            if fn_df['score'].mean() > 0.4:
                feature_patterns['recommendations'].append(
                    f"Consider lowering threshold for {feature} (current FN avg: {fn_df['score'].mean():.3f})"
                )
        
        patterns[feature] = feature_patterns
    
    return patterns

def create_visualizations(validation_df, misclassifications, output_dir):
    """Create comprehensive visualizations"""
    # Create output directory
    viz_dir = Path(output_dir) / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Score distributions
    fig1 = analyze_score_distributions(validation_df)
    fig1.savefig(viz_dir / 'score_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC curves
    fig2 = plot_roc_curves(validation_df)
    fig2.savefig(viz_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Precision-Recall curves
    fig3 = plot_pr_curves(validation_df)
    fig3.savefig(viz_dir / 'pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Misclassification analysis
    fig4 = visualize_misclassifications(misclassifications, validation_df)
    fig4.savefig(viz_dir / 'misclassification_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Feature correlation heatmap
    fig5 = plot_feature_correlations(validation_df)
    fig5.savefig(viz_dir / 'feature_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_score_distributions(validation_df):
    """Analyze score distributions for true/false predictions"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    features = ['condensate', 'prion', 'amyloid']
    score_cols = ['condensate_score', 'prion_score', 'amyloid_propensity']
    
    for idx, (feature, score_col) in enumerate(zip(features, score_cols)):
        # Top row: histograms
        ax_top = axes[0, idx]
        
        # Get masks for different classes
        expected_col = f'{feature}_expected'
        mask = validation_df[expected_col].notna()
        
        if mask.sum() > 0:
            tp_mask = (validation_df[expected_col] == True) & mask
            tn_mask = (validation_df[expected_col] == False) & mask
            
            if tp_mask.sum() > 0:
                ax_top.hist(validation_df[tp_mask][score_col], alpha=0.7, 
                           label=f'Positive ({tp_mask.sum()})', bins=20, color='green')
            if tn_mask.sum() > 0:
                ax_top.hist(validation_df[tn_mask][score_col], alpha=0.7, 
                           label=f'Negative ({tn_mask.sum()})', bins=20, color='red')
            
            ax_top.set_xlabel(f'{feature.capitalize()} Score')
            ax_top.set_ylabel('Count')
            ax_top.set_title(f'{feature.capitalize()} Score Distribution')
            ax_top.legend()
        
        # Bottom row: box plots
        ax_bottom = axes[1, idx]
        
        if mask.sum() > 0:
            data_for_box = []
            labels_for_box = []
            
            if tp_mask.sum() > 0:
                data_for_box.append(validation_df[tp_mask][score_col])
                labels_for_box.append('Positive')
            if tn_mask.sum() > 0:
                data_for_box.append(validation_df[tn_mask][score_col])
                labels_for_box.append('Negative')
            
            if data_for_box:
                ax_bottom.boxplot(data_for_box, labels=labels_for_box)
                ax_bottom.set_ylabel(f'{feature.capitalize()} Score')
                ax_bottom.set_title(f'{feature.capitalize()} Score by Class')
    
    plt.tight_layout()
    return fig

def plot_roc_curves(validation_df):
    """Plot ROC curves for all features"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    features = ['condensate', 'prion', 'amyloid']
    score_cols = ['condensate_score', 'prion_score', 'amyloid_propensity']
    colors = ['blue', 'green', 'red']
    
    for feature, score_col, color in zip(features, score_cols, colors):
        expected_col = f'{feature}_expected'
        
        mask = validation_df[expected_col].notna()
        if mask.sum() == 0:
            continue
            
        y_true = validation_df[mask][expected_col].astype(bool).astype(int)
        y_scores = validation_df[mask][score_col]
        
        if len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=color, lw=2, 
                   label=f'{feature.capitalize()} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_pr_curves(validation_df):
    """Plot Precision-Recall curves for all features"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    features = ['condensate', 'prion', 'amyloid']
    score_cols = ['condensate_score', 'prion_score', 'amyloid_propensity']
    colors = ['blue', 'green', 'red']
    
    for feature, score_col, color in zip(features, score_cols, colors):
        expected_col = f'{feature}_expected'
        
        mask = validation_df[expected_col].notna()
        if mask.sum() == 0:
            continue
            
        y_true = validation_df[mask][expected_col].astype(bool).astype(int)
        y_scores = validation_df[mask][score_col]
        
        if len(np.unique(y_true)) == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recall, precision)
            
            ax.plot(recall, precision, color=color, lw=2, 
                   label=f'{feature.capitalize()} (AUC = {pr_auc:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    return fig

def visualize_misclassifications(misclassifications, validation_df):
    """Create comprehensive misclassification visualizations"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    features = ['condensate', 'prion', 'amyloid']
    
    for idx, feature in enumerate(features):
        # Top row: Score distributions for misclassifications
        ax_top = axes[0, idx]
        
        all_details = misclassifications[feature]['details']
        if all_details:
            detail_df = pd.DataFrame(all_details)
            
            if not detail_df.empty:
                # Box plot of scores by misclassification type
                fp_scores = detail_df[detail_df['type'] == 'FP']['score']
                fn_scores = detail_df[detail_df['type'] == 'FN']['score']
                
                data = []
                labels = []
                if len(fp_scores) > 0:
                    data.append(fp_scores)
                    labels.append(f'FP (n={len(fp_scores)})')
                if len(fn_scores) > 0:
                    data.append(fn_scores)
                    labels.append(f'FN (n={len(fn_scores)})')
                
                if data:
                    ax_top.boxplot(data, labels=labels)
                    ax_top.set_ylabel('Prediction Score')
                    ax_top.set_title(f'{feature.capitalize()} Misclassification Scores')
        
        # Bottom row: Feature analysis
        ax_bottom = axes[1, idx]
        
        # Scatter plot of disorder vs LCR for all predictions
        expected_col = f'{feature}_expected'
        predicted_col = f'{feature}_predicted'
        
        mask = validation_df[expected_col].notna()
        if mask.sum() > 0:
            feature_df = validation_df[mask]
            
            # Define groups
            tp_mask = (feature_df[expected_col] == True) & (feature_df[predicted_col] == True)
            tn_mask = (feature_df[expected_col] == False) & (feature_df[predicted_col] == False)
            fp_mask = (feature_df[expected_col] == False) & (feature_df[predicted_col] == True)
            fn_mask = (feature_df[expected_col] == True) & (feature_df[predicted_col] == False)
            
            # Plot each group
            if tp_mask.sum() > 0:
                ax_bottom.scatter(feature_df[tp_mask]['disorder_percent'], 
                                feature_df[tp_mask]['lcr_percent'], 
                                c='green', label=f'TP ({tp_mask.sum()})', alpha=0.6, s=50)
            if tn_mask.sum() > 0:
                ax_bottom.scatter(feature_df[tn_mask]['disorder_percent'], 
                                feature_df[tn_mask]['lcr_percent'], 
                                c='blue', label=f'TN ({tn_mask.sum()})', alpha=0.6, s=50)
            if fp_mask.sum() > 0:
                ax_bottom.scatter(feature_df[fp_mask]['disorder_percent'], 
                                feature_df[fp_mask]['lcr_percent'], 
                                c='red', label=f'FP ({fp_mask.sum()})', alpha=0.8, s=100, marker='^')
            if fn_mask.sum() > 0:
                ax_bottom.scatter(feature_df[fn_mask]['disorder_percent'], 
                                feature_df[fn_mask]['lcr_percent'], 
                                c='orange', label=f'FN ({fn_mask.sum()})', alpha=0.8, s=100, marker='v')
            
            ax_bottom.set_xlabel('Disorder %')
            ax_bottom.set_ylabel('LCR %')
            ax_bottom.set_title(f'{feature.capitalize()} by Sequence Features')
            ax_bottom.legend()
    
    plt.tight_layout()
    return fig

def plot_feature_correlations(validation_df):
    """Plot correlation heatmap between features"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Select numeric columns
    numeric_cols = ['disorder_percent', 'lcr_percent', 'condensate_score', 
                    'prion_score', 'amyloid_propensity', 'type_confidence']
    
    # Create binary columns for categorical features
    binary_cols = []
    for col in ['forms_condensate', 'has_prion_domain', 'is_amyloidogenic']:
        if col in validation_df.columns:
            validation_df[f'{col}_binary'] = validation_df[col].astype(int)
            binary_cols.append(f'{col}_binary')
    
    # Combine all columns for correlation
    corr_cols = numeric_cols + binary_cols
    corr_cols = [col for col in corr_cols if col in validation_df.columns]
    
    # Calculate correlation matrix
    corr_matrix = validation_df[corr_cols].corr()
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=ax, 
                cbar_kws={"shrink": 0.8})
    
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    
    return fig

def generate_detailed_report(metrics, misclassifications, patterns, validation_df, output_dir):
    """Generate comprehensive validation report"""
    report_path = Path(output_dir) / 'validation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BIOCONDENSATE PIPELINE VALIDATION REPORT\n")
        f.write(f"Generated: {datetime.datetime.now()}\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall performance summary
        f.write("OVERALL PERFORMANCE SUMMARY\n")
        f.write("-" * 40 + "\n\n")
        
        for feature, feature_metrics in metrics.items():
            f.write(f"{feature.upper()} PREDICTIONS:\n")
            f.write(f"  Total samples: {feature_metrics['n_samples']}\n")
            f.write(f"  Positive samples: {feature_metrics['n_positive']}\n")
            f.write(f"  Negative samples: {feature_metrics['n_negative']}\n")
            f.write(f"  Sensitivity (Recall): {feature_metrics['sensitivity']:.3f}\n")
            f.write(f"  Specificity: {feature_metrics['specificity']:.3f}\n")
            f.write(f"  Precision: {feature_metrics['precision']:.3f}\n")
            f.write(f"  F1 Score: {feature_metrics['f1_score']:.3f}\n")
            f.write(f"  Accuracy: {feature_metrics['accuracy']:.3f}\n")
            f.write(f"  MCC: {feature_metrics['mcc']:.3f}\n")
            if 'auc_roc' in feature_metrics and feature_metrics['auc_roc'] != 'N/A':
                f.write(f"  AUC-ROC: {feature_metrics['auc_roc']:.3f}\n")
            if 'auc_pr' in feature_metrics and feature_metrics['auc_pr'] != 'N/A':
                f.write(f"  AUC-PR: {feature_metrics['auc_pr']:.3f}\n")
            f.write("\n")
        
        # Misclassification analysis
        f.write("\n" + "=" * 80 + "\n")
        f.write("MISCLASSIFICATION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        for feature in ['condensate', 'prion', 'amyloid']:
            f.write(f"\n{feature.upper()} MISCLASSIFICATIONS:\n")
            f.write("-" * 40 + "\n")
            
            # False Positives
            fp_list = misclassifications[feature]['FP']
            if fp_list:
                f.write(f"\nFalse Positives ({len(fp_list)}): {', '.join(fp_list)}\n")
                if 'FP_patterns' in patterns[feature] and patterns[feature]['FP_patterns']:
                    fp_pat = patterns[feature]['FP_patterns']
                    f.write(f"  Average score: {fp_pat['avg_score']:.3f} ± {fp_pat['std_score']:.3f}\n")
                    f.write(f"  Average disorder: {fp_pat['avg_disorder']:.1f}%\n")
                    f.write(f"  Average LCR: {fp_pat['avg_lcr']:.1f}%\n")
                    f.write(f"  Score range: [{fp_pat['score_range'][0]:.3f}, {fp_pat['score_range'][1]:.3f}]\n")
            
            # False Negatives
            fn_list = misclassifications[feature]['FN']
            if fn_list:
                f.write(f"\nFalse Negatives ({len(fn_list)}): {', '.join(fn_list)}\n")
                if 'FN_patterns' in patterns[feature] and patterns[feature]['FN_patterns']:
                    fn_pat = patterns[feature]['FN_patterns']
                    f.write(f"  Average score: {fn_pat['avg_score']:.3f} ± {fn_pat['std_score']:.3f}\n")
                    f.write(f"  Average disorder: {fn_pat['avg_disorder']:.1f}%\n")
                    f.write(f"  Average LCR: {fn_pat['avg_lcr']:.1f}%\n")
                    f.write(f"  Score range: [{fn_pat['score_range'][0]:.3f}, {fn_pat['score_range'][1]:.3f}]\n")
            
            # Recommendations
            if patterns[feature]['recommendations']:
                f.write("\nRecommendations:\n")
                for rec in patterns[feature]['recommendations']:
                    f.write(f"  • {rec}\n")
        
        # Biological consistency checks
        f.write("\n" + "=" * 80 + "\n")
        f.write("BIOLOGICAL CONSISTENCY CHECKS\n")
        f.write("=" * 80 + "\n\n")
        
        consistency_checks = check_biological_consistency(validation_df)
        for _, check in consistency_checks.iterrows():
            status = "✓" if check['pass'] else "✗"
            f.write(f"{status} {check['check']}: {check['result']:.3f} (expected {check['expected']})\n")

def check_biological_consistency(validation_df):
    """Verify biological expectations"""
    checks = []
    
    # Check if prion proteins are mostly amyloidogenic
    prion_mask = validation_df['prion_expected'] == True
    if prion_mask.sum() > 0:
        print(validation_df[prion_mask].columns)
        prion_amyloid_rate = validation_df[prion_mask]['amyloid_predicted'].mean()
        checks.append({
            'check': 'Prion proteins are amyloidogenic',
            'result': prion_amyloid_rate,
            'expected': '>0.7',
            'pass': prion_amyloid_rate > 0.7
        })
    
    # Check disorder correlation with condensates
    condensate_mask = validation_df['condensate_expected'] == True
    if condensate_mask.sum() > 0:
        avg_disorder = validation_df[condensate_mask]['disorder_percent'].mean()
        checks.append({
            'check': 'Condensate proteins have high disorder',
            'result': avg_disorder,
            'expected': '>40%',
            'pass': avg_disorder > 40
        })
    
    # Check negative controls have low disorder
    neg_mask = validation_df['control_type'] == 'negative'
    if neg_mask.sum() > 0:
        avg_neg_disorder = validation_df[neg_mask]['disorder_percent'].mean()
        checks.append({
            'check': 'Negative controls have low disorder',
            'result': avg_neg_disorder,
            'expected': '<30%',
            'pass': avg_neg_disorder < 30
        })
    
    return pd.DataFrame(checks)

def find_threshold_recommendations(validation_df, output_dir):
    """Find optimal thresholds based on control performance"""
    threshold_recommendations = {}
    
    features = ['condensate', 'prion', 'amyloid']
    score_cols = ['condensate_score', 'prion_score', 'amyloid_propensity']
    
    for feature, score_col in zip(features, score_cols):
        expected_col = f'{feature}_expected'
        
        mask = validation_df[expected_col].notna()
        if mask.sum() == 0:
            continue
            
        y_true = validation_df[mask][expected_col].astype(int)
        y_scores = validation_df[mask][score_col]
        
        if len(np.unique(y_true)) > 1:  # Need both classes
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            
            # Find threshold for different criteria
            # 1. Youden's J statistic (balanced)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            # 2. High sensitivity (90%)
            idx_90_sens = np.where(tpr >= 0.9)[0]
            if len(idx_90_sens) > 0:
                thresh_90_sens = thresholds[idx_90_sens[0]]
                spec_at_90_sens = 1 - fpr[idx_90_sens[0]]
            else:
                thresh_90_sens = None
                spec_at_90_sens = None
            
            # 3. High specificity (90%)
            idx_90_spec = np.where(fpr <= 0.1)[0]
            if len(idx_90_spec) > 0:
                thresh_90_spec = thresholds[idx_90_spec[-1]]
                sens_at_90_spec = tpr[idx_90_spec[-1]]
            else:
                thresh_90_spec = None
                sens_at_90_spec = None
            
            threshold_recommendations[feature] = {
                'current_threshold': 0.5,  # Assuming default
                'optimal_balanced': optimal_threshold,
                'optimal_sens': tpr[optimal_idx],
                'optimal_spec': 1 - fpr[optimal_idx],
                'high_sensitivity': {
                    'threshold': thresh_90_sens,
                    'specificity': spec_at_90_sens
                },
                'high_specificity': {
                    'threshold': thresh_90_spec,
                    'sensitivity': sens_at_90_spec
                }
            }
    
    # Save threshold recommendations
    thresh_path = Path(output_dir) / 'threshold_recommendations.txt'
    with open(thresh_path, 'w') as f:
        f.write("THRESHOLD RECOMMENDATIONS\n")
        f.write("=" * 50 + "\n\n")
        
        for feature, thresh in threshold_recommendations.items():
            f.write(f"{feature.upper()}:\n")
            f.write(f"  Current threshold: {thresh['current_threshold']:.3f}\n")
            f.write(f"  Optimal balanced threshold: {thresh['optimal_balanced']:.3f}\n")
            f.write(f"    - Sensitivity: {thresh['optimal_sens']:.3f}\n")
            f.write(f"    - Specificity: {thresh['optimal_spec']:.3f}\n")
            
            if thresh['high_sensitivity']['threshold'] is not None:
                f.write(f"  For 90% sensitivity use: {thresh['high_sensitivity']['threshold']:.3f}\n")
                f.write(f"    - Specificity: {thresh['high_sensitivity']['specificity']:.3f}\n")
            
            if thresh['high_specificity']['threshold'] is not None:
                f.write(f"  For 90% specificity use: {thresh['high_specificity']['threshold']:.3f}\n")
                f.write(f"    - Sensitivity: {thresh['high_specificity']['sensitivity']:.3f}\n")
            f.write("\n")
    
    return threshold_recommendations

def run_full_validation(results_file, output_dir='validation_results'):
    """Complete validation pipeline"""
    start_time = datetime.datetime.now()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading predictions from {results_file}")
    
    # Load predictions
    results_df = pd.read_csv(results_file, sep='\t')
    logger.info(f"Loaded {len(results_df)} predictions")
    
    # Validate positive controls
    logger.info("Validating positive controls...")
    pos_validation = validate_predictions(results_df, positive_controls, 'positive')
    logger.info(f"Found {len(pos_validation)} positive controls")
    
    # Validate negative controls
    logger.info("Validating negative controls...")
    neg_validation = validate_predictions(results_df, negative_controls, 'negative')
    logger.info(f"Found {len(neg_validation)} negative controls")
    
    # Combine validations
    all_validation = pd.concat([pos_validation, neg_validation], ignore_index=True)
    
    # Save validation data
    all_validation.to_csv(output_path / 'validation_data.csv', index=False)
    
    # Calculate metrics
    logger.info("Calculating performance metrics...")
    metrics = calculate_metrics_per_feature(all_validation)
    
    # Identify misclassifications
    logger.info("Analyzing misclassifications...")
    misclassifications = identify_misclassifications(all_validation)
    patterns = analyze_misclassification_patterns(misclassifications, all_validation)
    
    # Generate visualizations
    logger.info("Creating visualizations...")
    create_visualizations(all_validation, misclassifications, output_dir)
    
    # Find threshold recommendations
    logger.info("Computing threshold recommendations...")
    thresholds = find_threshold_recommendations(all_validation, output_dir)
    
    # Generate detailed report
    logger.info("Generating validation report...")
    generate_detailed_report(metrics, misclassifications, patterns, all_validation, output_dir)
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    for feature, feature_metrics in metrics.items():
        print(f"\n{feature.upper()}:")
        print(f"  Sensitivity: {feature_metrics['sensitivity']:.3f}")
        print(f"  Specificity: {feature_metrics['specificity']:.3f}")
        print(f"  F1 Score: {feature_metrics['f1_score']:.3f}")
        print(f"  MCC: {feature_metrics['mcc']:.3f}")
    
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    
    logger.info(f"Validation completed in {duration}")
    logger.info(f"Results saved to {output_dir}")
    
    return metrics, all_validation

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Validate Biocondensate Pipeline predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i predictions_summary.csv -o validation_results
  %(prog)s -i results.tsv -o my_validation
        """
    )
    
    parser.add_argument("-i", "--input", required=True,
                       help="Input TSV predictions summary file")
    parser.add_argument("-o", "--output", required=True,
                       help="Output directory for validation results")
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_args()
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    try:
        logger.info(f"Starting validation of predictions from {args.input}")
        metrics, validation_df = run_full_validation(args.input, args.output)
        
        logger.info("Validation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
