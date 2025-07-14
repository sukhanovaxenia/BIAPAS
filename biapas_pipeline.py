#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Biocondensate Analysis Pipeline v3 with Original Output Format

This version maintains the exact same output format as the original pipeline
while incorporating significant improvements in detection algorithms:

Key improvements:
- Better condensate sensitivity (was 30%, targeting >70%)
- Fixed amyloid detection (was 0%, targeting >70%)
- Improved prion specificity (was 50%, targeting >80%)
- Maintains all original output fields and structure

Author: Enhanced version 3 based on original format
Date: 2025-01-15
"""

import os
import sys
import argparse
import logging
import tempfile
import subprocess
import pandas as pd
import numpy as np
from Bio import SeqIO
from multiprocessing import Pool, cpu_count
import json
import re
import requests
from urllib.parse import quote
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_biocondensat_analysis_v3.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EnhancedBiocondensateAnalyzerV3:
    """
    Enhanced analyzer v3 with improved accuracy but original output format
    """
    
    def __init__(self, input_file, output_dir, rna_file=None, threads=1, config=None, visualize=False):
        """
        Initialize enhanced analyzer with optimized thresholds
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.rna_file = rna_file
        self.threads = min(threads, cpu_count())
        self.config = config if config else {}
        self.visualize = visualize
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if self.visualize:
            self.vis_dir = os.path.join(output_dir, "visualizations")
            if not os.path.exists(self.vis_dir):
                os.makedirs(self.vis_dir)
        
        # OPTIMIZED THRESHOLDS based on validation feedback
        self.base_thresholds = self.config.get('thresholds', {
            # Disorder thresholds
            'iupred_disorder': 0.5,
            'disorder_percent': 25,  # Lowered for better sensitivity
            
            # Low complexity thresholds - MORE SENSITIVE
            'lcr_complexity': 0.5,  # Raised (less strict)
            'lcr_percent': 10,  # Lowered significantly
            
            # Prion thresholds - MORE SPECIFIC
            'prion_score': 0.586,  # Raised for better specificity
            'qn_content': 0.20,  # Raised
            'prion_qn_threshold': 0.30,  # Raised
            'prion_hydrophobic_threshold': 0.4,
            'prion_aromatic_threshold': 0.35,
            'prion_syg_threshold': 0.25,
            'prion_min_length': 40,  # Increased for specificity
            
            # Biocondensate thresholds - BALANCED
            'biocondensat_score': 0.57,  # Slightly raised from 0.50
            'biocondensat_min_lcr': 0,  # No minimum LCR requirement
            'biocondensat_min_disorder': 15,  # Lower requirement
            'rg_charge_ratio': 0.2,
            'aromatic_content': 0.05,
            
            # Amyloid thresholds - PROPERLY BALANCED
            'amyloid_propensity': 0.615,  # Raised significantly from 0.35
            'cross_beta_score': 0.6,  # Raised
            'amyloid_min_score': 0.582,  # Much higher minimum
            'amyloid_window': 8,  # Slightly larger window
            'amyloid_hydrophobic_min': 0.6,  # Require high hydrophobicity
            
            # RNA binding thresholds
            'rna_binding_score': 0.5,
            
            # Domain detection thresholds
            'domain_confidence': 0.6,  # Slightly lower
            'domain_enrichment': 2.0
        })
        
        # Enhanced feature weights
        self.feature_weights = self.config.get('feature_weights', {
            'multivalency': 0.25,
            'phase_separation': 0.25,
            'specificity': 0.20,
            'aggregation_balance': 0.15,
            'disorder_lcr': 0.15
        })
        
        # Analysis parameters
        self.analysis_params = self.config.get('analysis_parameters', {
            'window_sizes': {
                'lcr_window': 20,
                'prion_window': 41,
                'disorder_window': 15,  # Smaller window
                'amyloid_window': 8,  # Increased from 6
                'domain_window': 30
            },
            'step_sizes': {
                'lcr_step': 5,
                'prion_step': 1,
                'disorder_step': 1,
                'amyloid_step': 1
            },
            'min_region_lengths': {
                'lcr_min_length': 10,  # Lowered
                'prion_min_length': 25,  # Slightly raised
                'disorder_min_length': 20,  # Lowered
                'amyloid_min_length': 5,  # Lowered
                'domain_min_length': 20
            }
        })
        
        # Background amino acid frequencies (from UniProt)
        self.background_frequencies = {
            'A': 0.0825, 'R': 0.0553, 'N': 0.0406, 'D': 0.0545, 'C': 0.0137,
            'Q': 0.0393, 'E': 0.0675, 'G': 0.0707, 'H': 0.0227, 'I': 0.0596,
            'L': 0.0966, 'K': 0.0584, 'M': 0.0242, 'F': 0.0386, 'P': 0.0470,
            'S': 0.0656, 'T': 0.0534, 'W': 0.0108, 'Y': 0.0292, 'V': 0.0687
        }
        
        # Load sequences
        self.sequences = self._load_sequences(self.input_file)
        self.rna_sequences = self._load_sequences(self.rna_file) if self.rna_file else None
        
        # Initialize statistical models
        self._initialize_statistical_models()
    
    def _initialize_statistical_models(self):
        """Initialize statistical models for enhanced analysis"""
        # Amino acid property scales
        self.aa_properties = {
            'hydrophobicity': {
                'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
                'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
                'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
                'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
            },
            'disorder_propensity': {
                'A': 0.06, 'C': -0.02, 'D': 0.192, 'E': 0.736, 'F': -0.697,
                'G': 0.166, 'H': 0.303, 'I': -0.486, 'K': 0.586, 'L': -0.326,
                'M': -0.262, 'N': 0.434, 'P': 0.987, 'Q': 0.318, 'R': 0.180,
                'S': 0.341, 'T': 0.059, 'V': -0.544, 'W': -0.884, 'Y': -0.510
            },
            'beta_propensity': {
                'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19,
                'Q': 1.10, 'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.60,
                'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55,
                'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70
            },
            # NEW: Amyloid propensity scale
            'amyloid_propensity': {
                'A': -0.17, 'R': -0.81, 'N': -0.42, 'D': -0.72, 'C': -0.13,
                'Q': -0.33, 'E': -0.65, 'G': -0.01, 'H': -0.08, 'I': 0.81,
                'L': 0.56, 'K': -0.99, 'M': 0.43, 'F': 0.61, 'P': -1.00,
                'S': -0.26, 'T': -0.18, 'W': 0.49, 'Y': 0.45, 'V': 0.64
            }
        }
        
        # Define biocondensate type signatures (same as original)
        self.biocondensate_signatures = {
            'Nucleolus': {
                'required_features': ['nuclear', 'rna_binding'],
                'enriched_features': ['ribosomal_rna_processing', 'gar_domain'],
                'typical_disorder': (25, 55),
                'typical_lcr': (20, 45),
                'characteristic_motifs': ['RGG', 'KK']
            },
            'Stress granule': {
                'required_features': ['rna_binding', 'stress_responsive'],
                'enriched_features': ['tia1_like', 'g3bp_like', 'translation_factor'],
                'typical_disorder': (40, 70),
                'typical_lcr': (25, 55),
                'characteristic_motifs': ['RGG', 'YG', 'QN']
            },
            'P-body': {
                'required_features': ['rna_decay_machinery', 'cytoplasmic'],
                'enriched_features': ['deadenylase', 'decapping', 'lsm'],
                'typical_disorder': (30, 60),
                'typical_lcr': (20, 50),
                'characteristic_motifs': ['FDF', 'HLM']
            },
            'Nuclear speckle': {
                'required_features': ['sr_rich', 'nuclear'],
                'enriched_features': ['splicing_factor', 'rrm_domain'],
                'typical_disorder': (35, 65),
                'typical_lcr': (30, 60),
                'characteristic_motifs': ['SR', 'RS']
            },
            'Nuclear pore': {
                'required_features': ['fg_repeats', 'nuclear_transport'],
                'enriched_features': ['nucleoporin'],
                'typical_disorder': (50, 80),
                'typical_lcr': (40, 70),
                'characteristic_motifs': ['FG', 'GLFG']
            },
            'RNA granule': {
                'required_features': ['rna_binding'],
                'enriched_features': ['rrm_domain', 'kh_domain'],
                'typical_disorder': (30, 60),
                'typical_lcr': (25, 50),
                'characteristic_motifs': ['RGG', 'YG']
            },
            'Protein aggregate': {
                'required_features': ['aggregation_prone'],
                'enriched_features': ['prion_like', 'amyloid_like'],
                'typical_disorder': (20, 80),
                'typical_lcr': (20, 80),
                'characteristic_motifs': ['QN', 'GY']
            }
        }
    
    def _load_sequences(self, fasta_file):
        """Load sequences from FASTA file"""
        if not fasta_file:
            return {}
            
        sequences = {}
        try:
            for record in SeqIO.parse(fasta_file, "fasta"):
                sequences[record.id] = str(record.seq).upper()
            logger.info(f"Loaded {len(sequences)} sequences from {fasta_file}")
            return sequences
        except Exception as e:
            logger.error(f"Error loading sequences: {e}")
            return {}
    
    def run_analysis(self):
        """Run enhanced analysis on all sequences"""
        logger.info("Starting enhanced biocondensate analysis v3...")
        
        results = {}
        
        # Analyze each protein
        if self.threads > 1 and len(self.sequences) > 1:
            with Pool(self.threads) as pool:
                protein_results = pool.map(self._analyze_protein_enhanced, 
                                         list(self.sequences.items()))
            
            for protein_id, result in protein_results:
                results[protein_id] = result
        else:
            for protein_id, sequence in self.sequences.items():
                protein_id, result = self._analyze_protein_enhanced((protein_id, sequence))
                results[protein_id] = result
        
        # Save results
        self._save_results(results)
        
        # Create visualizations
        if self.visualize:
            self._create_visualizations(results)
        
        logger.info("Enhanced analysis completed.")
        
        return results
    
    def _analyze_protein_enhanced(self, protein_data):
        """
        Enhanced analysis of a single protein with improved algorithms
        """
        protein_id, sequence = protein_data
        logger.info(f"Analyzing protein {protein_id} (length: {len(sequence)})")
        
        # Results dictionary (maintaining original structure)
        protein_results = {
            'protein_id': protein_id,
            'length': len(sequence),
            'sequence': sequence
        }
        
        # 1. Basic sequence properties
        protein_results['sequence_properties'] = self._analyze_sequence_properties(sequence)
        
        # 2. Disorder analysis with improved algorithm
        protein_results['disorder_analysis'] = self._analyze_disorder_enhanced(sequence)
        
        # 3. Low complexity region analysis with better sensitivity
        protein_results['lcr_analysis'] = self._analyze_lcr_enhanced(sequence)
        
        # 4. Enhanced prion domain analysis
        protein_results['prion_analysis'] = self._analyze_prion_domains_enhanced(sequence)
        
        # 5. Compositional bias analysis
        comp_bias = self._analyze_compositional_bias_for_prions(sequence)
        protein_results['compositional_bias'] = comp_bias
        
        # 6. Fixed amyloid analysis
        protein_results['amyloid_analysis'] = self._analyze_amyloidogenicity_enhanced(
            sequence,
            protein_id,
            protein_results['prion_analysis'],
            protein_results['sequence_properties'],
            protein_results['disorder_analysis']
        )
        
        # 7. Enhanced biocondensate prediction
        protein_results['biocondensat_analysis'] = self._predict_biocondensat_formation_enhanced(
            sequence, protein_id,
            protein_results['sequence_properties'],
            protein_results['disorder_analysis'],
            protein_results['lcr_analysis'],
            protein_results['prion_analysis'],
            protein_results['amyloid_analysis'],
            comp_bias
        )

        # 8. Enhanced RNA interaction analysis
        if protein_results['biocondensat_analysis'].get('contains_rna', False):
            protein_results['rna_interaction'] = self._detect_rna_binding_enhanced(sequence)
        
        # 9. Heteromeric structure prediction
        protein_results['heteromeric_structure'] = self._predict_heteromeric_structure(
            sequence,
            protein_results['biocondensat_analysis']
        )
        
        # 10. Aggregation classification
        protein_results['aggregation_classification'] = self._classify_aggregation_type(
            protein_results)
        
        return (protein_id, protein_results)
    
    def _analyze_sequence_properties(self, sequence):
        """Analyze basic sequence properties"""
        length = len(sequence)
        properties = {}
        
        # Amino acid groups
        aa_groups = {
            'charged_positive': ['R', 'K', 'H'],
            'charged_negative': ['D', 'E'],
            'polar_uncharged': ['S', 'T', 'N', 'Q'],
            'hydrophobic': ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'],
            'aromatic': ['F', 'Y', 'W'],
            'special': ['C', 'G', 'P'],
            'qn': ['Q', 'N'],
            'fg': ['F', 'G'],
            'rg': ['R', 'G'],
            'sr': ['S', 'R']
        }
        
        # Count amino acids
        aa_count = Counter(sequence)
        
        # Calculate group properties
        for group_name, group_aas in aa_groups.items():
            group_count = sum(aa_count.get(aa, 0) for aa in group_aas)
            properties[f'{group_name}_count'] = group_count
            properties[f'{group_name}_percent'] = (group_count / length) * 100 if length > 0 else 0
        
        # Calculate charge properties
        pos_charged = properties['charged_positive_count']
        neg_charged = properties['charged_negative_count']
        properties['net_charge'] = pos_charged - neg_charged
        properties['charge_density'] = abs(properties['net_charge']) / length if length > 0 else 0
        
        # Calculate ratios
        g_count = aa_count.get('G', 0)
        if g_count > 0:
            properties['rg_ratio'] = aa_count.get('R', 0) / g_count
            properties['fg_ratio'] = aa_count.get('F', 0) / g_count
        else:
            properties['rg_ratio'] = 0
            properties['fg_ratio'] = 0
        
        # Hydrophobicity and other properties
        properties['hydrophobicity'] = properties['hydrophobic_count'] / length if length > 0 else 0
        properties['aromatic_content'] = properties['aromatic_count'] / length if length > 0 else 0
        properties['qn_content'] = properties['qn_count'] / length if length > 0 else 0
        
        # Analyze motifs
        motifs = {
            'RGG': len(re.findall(r'RGG', sequence)),
            'FG': len(re.findall(r'FG', sequence)),
            'SR': len(re.findall(r'SR', sequence)),
            'YG': len(re.findall(r'YG', sequence)),
            'PY': len(re.findall(r'PY', sequence))
        }
        properties['motifs'] = motifs
        
        return properties
    
    def _analyze_disorder_enhanced(self, sequence):
        """Enhanced disorder analysis with better sensitivity"""
        length = len(sequence)
        window_size = self.analysis_params['window_sizes']['disorder_window']
        
        # Method 1: Classic disorder propensity
        disorder_scores = []
        
        for i in range(length):
            start = max(0, i - window_size // 2)
            end = min(length, i + window_size // 2 + 1)
            window = sequence[start:end]
            
            # Calculate disorder score based on AA propensities
            window_score = sum(
                self.aa_properties['disorder_propensity'].get(aa, 0) 
                for aa in window
            ) / len(window)
            
            # Normalize to [0, 1]
            normalized_score = (window_score + 1) / 2
            disorder_scores.append(normalized_score)
        
        # Method 2: Charge-hydropathy relationship
        ch_scores = []
        for i in range(length):
            start = max(0, i - window_size // 2)
            end = min(length, i + window_size // 2 + 1)
            window = sequence[start:end]
            
            # Calculate net charge
            charge = abs(sum(1 for aa in window if aa in 'KRH') - 
                       sum(1 for aa in window if aa in 'DE')) / len(window)
            
            # Calculate hydropathy
            hydropathy = sum(self.aa_properties['hydrophobicity'].get(aa, 0) 
                           for aa in window) / len(window)
            
            # Uversky boundary
            boundary = 2.785 * hydropathy - 1.151
            ch_score = 1.0 if charge > boundary else 0.0
            ch_scores.append(ch_score)
        
        # Combine methods
        combined_scores = []
        for i in range(length):
            combined = disorder_scores[i] * 0.7 + ch_scores[i] * 0.3
            combined_scores.append(combined)
        
        # Find disordered regions with lower threshold
        disordered_regions = []
        threshold = self.base_thresholds['iupred_disorder']
        in_region = False
        start_pos = 0
        
        for i, score in enumerate(combined_scores):
            if score > threshold and not in_region:
                in_region = True
                start_pos = i
            elif (score <= threshold or i == length - 1) and in_region:
                in_region = False
                end_pos = i - 1 if score <= threshold else i
                
                min_length = self.analysis_params['min_region_lengths']['disorder_min_length']
                if end_pos - start_pos + 1 >= min_length:
                    region_scores = combined_scores[start_pos:end_pos+1]
                    disordered_regions.append({
                        'start': start_pos,
                        'end': end_pos,
                        'length': end_pos - start_pos + 1,
                        'avg_score': np.mean(region_scores)
                    })
        
        # Calculate overall disorder metrics
        disordered_residues = sum(region['length'] for region in disordered_regions)
        disorder_percent = (disordered_residues / length) * 100 if length > 0 else 0
        
        return {
            'disorder_scores': combined_scores,
            'disordered_regions': disordered_regions,
            'disorder_percent': disorder_percent,
            'is_disordered': disorder_percent > self.base_thresholds['disorder_percent'],
            'max_score': max(combined_scores) if combined_scores else 0,
            'avg_score': np.mean(combined_scores) if combined_scores else 0
        }
    
    def _analyze_lcr_enhanced(self, sequence):
        """Enhanced low complexity region analysis with better sensitivity"""
        length = len(sequence)
        window_size = self.analysis_params['window_sizes']['lcr_window']
        step_size = self.analysis_params['step_sizes']['lcr_step']
        
        complexity_scores = [1.0] * length
        lcr_types = [None] * length
        
        # Analyze sequence complexity with multiple methods
        for i in range(0, length - window_size + 1, step_size):
            window = sequence[i:i+window_size]
            
            # Method 1: Shannon entropy
            aa_freq = Counter(window)
            entropy = -sum(
                (count/window_size) * np.log2(count/window_size)
                for count in aa_freq.values() if count > 0
            )
            
            # Normalize entropy
            max_entropy = np.log2(min(20, window_size))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            # Method 2: Check for specific low complexity patterns
            is_lcr = False
            lcr_type = None
            
            # Single amino acid enrichment
            max_aa_count = max(aa_freq.values())
            if max_aa_count / window_size > 0.35:  # Lowered threshold
                is_lcr = True
                lcr_type = f'poly{max(aa_freq, key=aa_freq.get)}'
            
            # Specific composition checks
            qn_fraction = (window.count('Q') + window.count('N')) / window_size
            sr_fraction = (window.count('S') + window.count('R')) / window_size
            rg_fraction = (window.count('R') + window.count('G')) / window_size
            fg_fraction = (window.count('F') + window.count('G')) / window_size
            
            if qn_fraction > 0.35:  # Lowered threshold
                is_lcr = True
                lcr_type = 'qn_rich'
            elif sr_fraction > 0.35:
                is_lcr = True
                lcr_type = 'sr_rich'
            elif rg_fraction > 0.35:
                is_lcr = True
                lcr_type = 'rg_rich'
            elif fg_fraction > 0.25:
                is_lcr = True
                lcr_type = 'fg_rich'
            
            # Update scores
            if is_lcr or normalized_entropy < self.base_thresholds['lcr_complexity']:
                for j in range(i, min(i + window_size, length)):
                    complexity_scores[j] = min(complexity_scores[j], normalized_entropy)
                    if lcr_types[j] is None and lcr_type:
                        lcr_types[j] = lcr_type
        
        # Find LCR regions
        lcr_regions = []
        threshold = self.base_thresholds['lcr_complexity']
        in_region = False
        start_pos = 0
        
        for i, score in enumerate(complexity_scores):
            if score < threshold and not in_region:
                in_region = True
                start_pos = i
            elif (score >= threshold or i == length - 1) and in_region:
                in_region = False
                end_pos = i - 1 if score >= threshold else i
                
                min_length = self.analysis_params['min_region_lengths']['lcr_min_length']
                if end_pos - start_pos + 1 >= min_length:
                    # Determine dominant LCR type
                    region_types = Counter(lcr_types[start_pos:end_pos+1])
                    if None in region_types:
                        del region_types[None]
                    dominant_type = region_types.most_common(1)[0][0] if region_types else 'generic'
                    
                    lcr_regions.append({
                        'start': start_pos,
                        'end': end_pos,
                        'length': end_pos - start_pos + 1,
                        'type': dominant_type,
                        'avg_complexity': np.mean(complexity_scores[start_pos:end_pos+1])
                    })
        
        # Calculate LCR statistics
        lcr_residues = sum(region['length'] for region in lcr_regions)
        lcr_percent = (lcr_residues / length) * 100 if length > 0 else 0
        
        # Determine primary LCR type
        type_lengths = defaultdict(int)
        for region in lcr_regions:
            type_lengths[region['type']] += region['length']
        
        primary_lcr_type = max(type_lengths.items(), key=lambda x: x[1])[0] if type_lengths else None
        
        return {
            'lcr_regions': lcr_regions,
            'lcr_percent': lcr_percent,
            'has_lcr': lcr_percent > self.base_thresholds['lcr_percent'],
            'primary_lcr_type': primary_lcr_type,
            'complexity_scores': complexity_scores,
            'lcr_types': lcr_types,
            'type_distribution': dict(type_lengths)
        }
    
    def _analyze_prion_domains_enhanced(self, sequence):
        """Enhanced prion domain analysis with better specificity"""
        length = len(sequence)
        window_size = self.analysis_params['window_sizes']['prion_window']
        
        # Prion propensities (more conservative)
        prion_propensities = {
            'Q': 1.5, 'N': 1.2, 'Y': 0.8, 'F': 0.7, 'S': 0.5,
            'G': 0.4, 'M': 0.3, 'L': 0.2, 'I': 0.2, 'V': 0.1,
            'A': 0.0, 'T': 0.0, 'W': 0.3, 'H': -0.2, 'C': -0.3,
            'P': -1.5, 'K': -0.8, 'R': -0.8, 'D': -1.0, 'E': -1.0
        }
        
        # Calculate prion scores with stricter criteria
        prion_scores = []
        component_scores = {
            'qn_scores': [],
            'hydrophobic_scores': [],
            'aromatic_scores': [],
            'charged_scores': []
        }
        
        for i in range(length - window_size + 1):
            window = sequence[i:i+window_size]
            
            # Basic prion score
            aa_score = sum(prion_propensities.get(aa, 0) for aa in window) / window_size
            
            # Component scores
            qn_score = (window.count('Q') + window.count('N')) / window_size
            hydrophobic_score = sum(1 for aa in window if aa in 'VILMFYW') / window_size
            aromatic_score = sum(1 for aa in window if aa in 'FYW') / window_size
            charged_score = 1.0 - sum(1 for aa in window if aa in 'DEKRH') / window_size
            
            # Require strong Q/N enrichment or very low charge
            if qn_score < self.base_thresholds['qn_content'] and charged_score < 0.85:
                prion_scores.append(0.0)
                component_scores['qn_scores'].append(qn_score)
                component_scores['hydrophobic_scores'].append(hydrophobic_score)
                component_scores['aromatic_scores'].append(aromatic_score)
                component_scores['charged_scores'].append(charged_score)
                continue
            
            # Check for specific patterns
            pattern_bonus = 0
            
            # Strong Q/N runs (at least 4)
            if re.search(r'[QN]{4,}', window):
                pattern_bonus += 0.2
            
            # Q/N spacing pattern
            if re.search(r'[QN].{0,2}[QN].{0,2}[QN]', window):
                pattern_bonus += 0.1
            
            # Calculate combined score with strict pathways
            if qn_score >= self.base_thresholds['prion_qn_threshold']:  # Strong Q/N pathway
                combined_score = qn_score * 0.6 + charged_score * 0.3 + pattern_bonus * 0.1
            elif charged_score > 0.9 and (qn_score > 0.15 or aromatic_score > 0.2):  # Alternative pathway
                combined_score = charged_score * 0.4 + (qn_score + aromatic_score) * 0.3 + pattern_bonus * 0.1
            else:
                combined_score = 0.0
            
            # Normalize with base score
            if aa_score > 0:
                combined_score = combined_score * 0.8 + (aa_score / 2) * 0.2
            
            prion_scores.append(combined_score)
            component_scores['qn_scores'].append(qn_score)
            component_scores['hydrophobic_scores'].append(hydrophobic_score)
            component_scores['aromatic_scores'].append(aromatic_score)
            component_scores['charged_scores'].append(charged_score)
        
        # Pad scores
        padding = [0.0] * (window_size - 1)
        full_prion_scores = prion_scores + padding
        for key in component_scores:
            component_scores[key] = component_scores[key] + padding
        
        # Find prion regions with strict criteria
        prion_regions = []
        threshold = self.base_thresholds['prion_score']
        in_region = False
        start_pos = 0
        
        for i, score in enumerate(prion_scores):
            if score > threshold and not in_region:
                in_region = True
                start_pos = i
            elif (score <= threshold or i == len(prion_scores) - 1) and in_region:
                in_region = False
                end_pos = i - 1 if score <= threshold else i
                
                # Apply strict length filter
                min_length = self.base_thresholds['prion_min_length']
                if end_pos - start_pos + window_size >= min_length:
                    # Additional validation
                    region_seq = sequence[start_pos:end_pos + window_size]
                    region_qn = (region_seq.count('Q') + region_seq.count('N')) / len(region_seq)
                    region_charge = sum(1 for aa in region_seq if aa in 'DEKRH') / len(region_seq)
                    
                    # Must have significant Q/N content OR very low charge
                    if region_qn >= self.base_thresholds['qn_content'] or region_charge < 0.1:
                        region_scores = prion_scores[start_pos:end_pos+1]
                        
                        # Require consistently high scores
                        if np.mean(region_scores) > threshold * 0.9:
                            prion_regions.append({
                                'start': start_pos,
                                'end': end_pos + window_size - 1,
                                'length': end_pos - start_pos + window_size,
                                'type': 'QN-rich' if region_qn > 0.25 else 'Low-complexity',
                                'max_score': max(region_scores),
                                'avg_score': np.mean(region_scores),
                                'qn_score': np.mean(component_scores['qn_scores'][start_pos:end_pos+1]),
                                'hydrophobic_score': np.mean(component_scores['hydrophobic_scores'][start_pos:end_pos+1]),
                                'aromatic_score': np.mean(component_scores['aromatic_scores'][start_pos:end_pos+1])
                            })
        
        # Calculate overall metrics
        has_prion_domain = len(prion_regions) > 0
        max_score = max(prion_scores) if prion_scores else 0
        qn_percent = (sequence.count('Q') + sequence.count('N')) / length * 100
        
        return {
            'prion_regions': prion_regions,
            'has_prion_domain': has_prion_domain,
            'max_score': max_score,
            'prion_scores': full_prion_scores[:length],
            'component_scores': {k: v[:length] for k, v in component_scores.items()},
            'qn_percent': qn_percent,
            'domain_type': prion_regions[0]['type'] if prion_regions else None
        }
    
    def _analyze_compositional_bias_for_prions(self, sequence, window_size=41):
        """Analyze compositional bias patterns characteristic of prions"""
        length = len(sequence)
        if length < window_size:
            return {}
        
        # Define amino acid groups
        aa_groups = {
            'polar_uncharged': set('QNSTG'),
            'hydrophobic': set('VILMFYW'),
            'aromatic': set('FYW'),
            'charged': set('DEKRH'),
            'qn': set('QN')
        }
        
        # Calculate bias profiles
        bias_profiles = {group: [] for group in aa_groups}
        
        for i in range(length - window_size + 1):
            window = sequence[i:i+window_size]
            
            for group, aas in aa_groups.items():
                count = sum(1 for aa in window if aa in aas)
                bias = count / window_size
                bias_profiles[group].append(bias)
        
        features = {}
        
        # Q/N bias analysis
        qn_bias = bias_profiles['qn']
        if qn_bias:
            features['max_qn_bias'] = max(qn_bias)
            features['mean_qn_bias'] = np.mean(qn_bias)
            features['has_qn_peak'] = any(b > 0.3 for b in qn_bias)
        
        # Charge depletion
        charge_bias = bias_profiles['charged']
        features['charge_depleted_fraction'] = sum(1 for b in charge_bias if b < 0.15) / len(charge_bias) if charge_bias else 0
        
        return features
    
    def _analyze_amyloidogenicity_enhanced(self, sequence, protein_id, prion_analysis, seq_props, disorder_analysis):
        """Balanced amyloid detection that includes Q/N-rich amyloids"""
        length = len(sequence)
        window_size = self.analysis_params['window_sizes']['amyloid_window']
        
        # Calculate amyloid scores with multiple pathways
        amyloid_scores = []
        
        for i in range(length - window_size + 1):
            window = sequence[i:i+window_size]
            
            # Method 1: Classic hydrophobic amyloid pathway
            prop_score = sum(
                self.aa_properties['amyloid_propensity'].get(aa, 0)
                for aa in window
            ) / window_size
            
            # Method 2: Beta-sheet propensity
            beta_score = sum(
                self.aa_properties['beta_propensity'].get(aa, 0)
                for aa in window
            ) / window_size
            normalized_beta = beta_score / 1.7  # Max beta score is ~1.7
            
            # Method 3: Hydrophobicity
            hydrophobic_count = sum(1 for aa in window if aa in 'VILMFYW')
            hydrophobic_score = hydrophobic_count / window_size
            
            # Method 4: Q/N content (for prion-like amyloids)
            qn_count = window.count('Q') + window.count('N')
            qn_score = qn_count / window_size
            
            # Method 5: Pattern detection
            pattern_score = 0
            
            # Q/N runs (important for yeast prions) - LOWERED THRESHOLD
            if re.search(r'[QN]{2,}', window):  # Lowered from 3 to 2
                pattern_score += 0.3
            
            # G/S runs (important for disordered amyloids)
            if re.search(r'[GS]{3,}', window):  # Glycine/Serine rich
                pattern_score += 0.2
            
            # Alternating pattern
            if self._has_alternating_pattern(window):
                pattern_score += 0.2
            
            # Hydrophobic stretches
            if re.search(r'[VILMFYW]{3,}', window):
                pattern_score += 0.2
            
            # Known amyloid motifs
            if re.search(r'V[QI]IV[YF]', window):  # VQIVY-like
                pattern_score += 0.3
            elif re.search(r'[VIL]{3,}', window):  # Poly-V/I/L
                pattern_score += 0.1
            
            # Calculate score based on different pathways
            combined_score = 0.0
            
            # Pathway 1: Classical hydrophobic amyloid
            if prop_score > -0.2 and hydrophobic_score > self.base_thresholds.get('amyloid_hydrophobic_min', 0.2):
                hydrophobic_pathway_score = (
                    (prop_score + 1) / 2 * 0.4 +  # Normalize prop_score
                    normalized_beta * 0.3 +
                    hydrophobic_score * 0.2 +
                    pattern_score * 0.1
                )
                combined_score = max(combined_score, hydrophobic_pathway_score)
            
            # Pathway 2: Q/N-rich amyloid (yeast prions)
            if qn_score > self.base_thresholds.get('amyloid_qn_threshold', 0.15):
                qn_pathway_score = (
                    qn_score * 0.5 +
                    normalized_beta * 0.2 +
                    pattern_score * 0.3
                )
                combined_score = max(combined_score, qn_pathway_score)
            
            # Pathway 3: Mixed amyloid (some hydrophobic + some polar)
            if hydrophobic_score > 0.15 and (qn_score > 0.1 or beta_score > 1.0):
                mixed_pathway_score = (
                    hydrophobic_score * 0.3 +
                    normalized_beta * 0.3 +
                    qn_score * 0.2 +
                    pattern_score * 0.2
                )
                combined_score = max(combined_score, mixed_pathway_score)

            if disorder_analysis.get('disorder_percent', 0) > 60 and prion_analysis.get('has_prion_domain', False):
                disorder_pathway_score = (
                    0.5 +  # Base score for disordered prions
                    qn_score * 0.3 +
                    pattern_score * 0.2
                )
                combined_score = max(combined_score, disorder_pathway_score)
            
            amyloid_scores.append((i, combined_score))
        
        # Find amyloidogenic regions with appropriate threshold
        amyloid_regions = []
        threshold = self.base_thresholds.get('amyloid_propensity', 0.35)
        min_score_threshold = self.base_thresholds.get('amyloid_min_score', 0.2)
        
        # Find high-scoring regions
        for start, end, score in self._find_high_scoring_regions(amyloid_scores, threshold, min_score_threshold):
            if end - start >= self.analysis_params['min_region_lengths']['amyloid_min_length']:
                region_seq = sequence[start:end]
                
                # Validate region based on type
                region_qn = (region_seq.count('Q') + region_seq.count('N')) / len(region_seq)
                region_hydrophobic = sum(1 for aa in region_seq if aa in 'VILMFYW') / len(region_seq)
                
                # Accept if it's either hydrophobic OR Q/N-rich
                if region_hydrophobic >= 0.2 or region_qn >= 0.15:
                    amyloid_regions.append({
                        'start': start,
                        'end': end,
                        'length': end - start,
                        'avg_score': score
                    })
        
        # Calculate overall amyloid propensity
        amyloid_propensity = 0.0
        
        if amyloid_regions:
            # Base score from regions
            best_region_score = max(r['avg_score'] for r in amyloid_regions)
            total_amyloid_length = sum(r['length'] for r in amyloid_regions)
            coverage = total_amyloid_length / length
            
            # Calculate propensity
            amyloid_propensity = best_region_score * 0.7 + min(coverage * 5, 0.3)  # Scale coverage
            
            # Boost for sequence features
            if seq_props['hydrophobic_percent'] > 35:
                amyloid_propensity *= 1.1
            if seq_props['qn_percent'] > 15:
                amyloid_propensity *= 1.1
            
            amyloid_propensity = min(amyloid_propensity, 1.0)
        
        # Special consideration for prion proteins (often amyloidogenic)
        # ENHANCED special consideration for prion proteins
        if prion_analysis.get('has_prion_domain', False):
            # Prions with ANY Q/N are often amyloidogenic - LOWERED THRESHOLD
            if seq_props.get('qn_percent', 0) > 15:
                amyloid_propensity = max(amyloid_propensity, 0.6)  # Higher boost
            # Even without high Q/N, prions can be amyloidogenic
            elif prion_analysis.get('max_score', 0) > 0.7:
                amyloid_propensity = max(amyloid_propensity, 0.5)  # Higher boost
            else:
                # Any prion has some amyloid potential
                amyloid_propensity = max(amyloid_propensity, 0.4)

        # NEW: Check for highly disordered proteins (like NSR1)
        if disorder_analysis.get('disorder_percent', 0) > 80:
            # Highly disordered proteins can form amyloids
            if seq_props.get('qn_percent', 0) > 5 or seq_props.get('polar_uncharged_percent', 0) > 30:
                amyloid_propensity = max(amyloid_propensity, 0.45)
        
        # Determine amyloid type
        amyloid_type = None
        if amyloid_propensity > self.base_thresholds['amyloid_propensity']:
            if seq_props['qn_percent'] > 15:
                if prion_analysis.get('has_prion_domain', False):
                    amyloid_type = "Prion-like amyloid"
                else:
                    amyloid_type = "Q/N-rich amyloid"
            elif seq_props['hydrophobic_percent'] > 35:
                amyloid_type = "Hydrophobic amyloid"
            else:
                amyloid_type = "Mixed amyloid"
        
        # Cross-beta score
        cross_beta_score = amyloid_propensity * 0.7 if amyloid_regions else 0
        
        return {
            'amyloid_regions': amyloid_regions,
            'amyloid_propensity': amyloid_propensity,
            'is_amyloidogenic': amyloid_propensity > self.base_thresholds['amyloid_propensity'],
            'amyloid_type': amyloid_type,
            'cross_beta_score': cross_beta_score,
            'forms_cross_beta': cross_beta_score > self.base_thresholds['cross_beta_score']
        }
    
    def _has_alternating_pattern(self, sequence):
        """Check for alternating hydrophobic/polar pattern"""
        if len(sequence) < 4:
            return False
        
        hydrophobic = set('VILMFYW')
        alternations = 0
        
        for i in range(len(sequence) - 1):
            if (sequence[i] in hydrophobic) != (sequence[i+1] in hydrophobic):
                alternations += 1
        
        # Less strict requirement
        return alternations >= len(sequence) * 0.3
    
    def _find_high_scoring_regions(self, scores, threshold, min_score):
        """Find regions with consistently high scores"""
        regions = []
        in_region = False
        start = 0
        region_scores = []
        
        for i, (pos, score) in enumerate(scores):
            if score >= threshold and not in_region:
                in_region = True
                start = pos
                region_scores = [score]
            elif score >= threshold and in_region:
                region_scores.append(score)
            elif (score < threshold or i == len(scores) - 1) and in_region:
                in_region = False
                end = pos if score < threshold else pos + 1
                
                # Require minimum average score across region
                if region_scores and np.mean(region_scores) >= min_score:
                    regions.append((start, end, np.mean(region_scores)))
                region_scores = []
        
        return regions
    
    def _detect_rna_binding_enhanced(self, sequence):
        """Enhanced RNA-binding detection"""
        score = 0.0
        features = []
        
        # RRM domain
        if re.search(r'[RK]G[FY].{0,2}[FY]', sequence):
            score += 0.3
            features.append('RRM')
        
        # KH domain
        if re.search(r'[ILV]IG.{2}G.{2}[ILV]', sequence):
            score += 0.3
            features.append('KH')
        
        # RGG box
        rgg_count = len(re.findall(r'RGG', sequence))
        if rgg_count > 0:
            score += min(rgg_count * 0.1, 0.3)
            features.append('RGG')
        
        # Basic patches
        for i in range(0, len(sequence) - 10):
            window = sequence[i:i+10]
            basic_count = sum(1 for aa in window if aa in 'KRH')
            if basic_count >= 4:
                score += 0.2
                features.append('Basic_patch')
                break
        
        # RS domain
        if re.search(r'[RS]{4,}', sequence):
            score += 0.2
            features.append('RS')
        
        return {
            'is_rna_binding': score > self.base_thresholds['rna_binding_score'],
            'confidence': min(score, 1.0),
            'binding_modes': list(set(features)),
            'specificity': 'high' if score > 0.7 else 'medium' if score > 0.5 else 'low'
        }
    
    def _predict_biocondensat_formation_enhanced(self, sequence, protein_id, seq_props,
                                                disorder_analysis, lcr_analysis, 
                                                prion_analysis, amyloid_analysis, comp_bias):
        """Enhanced biocondensate prediction with better balance"""
        length = len(sequence)
        
        # Extract features for scoring
        features = {
            'sequence': sequence,
            'length': length,
            'disorder_percent': disorder_analysis.get('disorder_percent', 0),
            'lcr_percent': lcr_analysis.get('lcr_percent', 0),
            'has_prion_domain': prion_analysis.get('has_prion_domain', False),
            'prion_score': prion_analysis.get('max_score', 0),
            'amyloid_propensity': amyloid_analysis.get('amyloid_propensity', 0),
            'net_charge': seq_props.get('net_charge', 0),
            'aromatic_content': seq_props.get('aromatic_content', 0),
            'qn_content': seq_props.get('qn_content', 0)
        }
        
        # Initialize scoring components
        score_components = {
            'disorder_lcr': 0.0,
            'multivalency': 0.0,
            'phase_separation': 0.0,
            'specific_features': 0.0,
            'interaction_potential': 0.0
        }
        
        # 1. Disorder/LCR component (balanced)
        if features['disorder_percent'] > 70:
            score_components['disorder_lcr'] = 1.0
        elif features['disorder_percent'] > 50:
            score_components['disorder_lcr'] = 0.8
        elif features['disorder_percent'] > 30:
            score_components['disorder_lcr'] = 0.6
        elif features['disorder_percent'] > 20:
            score_components['disorder_lcr'] = 0.4
        elif features['disorder_percent'] > 10:
            score_components['disorder_lcr'] = 0.2
        
        # Boost for LCR (but not too much)
        if features['lcr_percent'] > 30:
            score_components['disorder_lcr'] = min(score_components['disorder_lcr'] + 0.3, 1.0)
        elif features['lcr_percent'] > 15:
            score_components['disorder_lcr'] = min(score_components['disorder_lcr'] + 0.2, 1.0)
        elif features['lcr_percent'] > 5:
            score_components['disorder_lcr'] = min(score_components['disorder_lcr'] + 0.1, 1.0)
        
        # 2. Multivalency (multiple interaction modes)
        interaction_modes = 0
        
        # Check RNA binding
        rna_binding = self._detect_rna_binding_enhanced(sequence)
        if rna_binding['is_rna_binding']:
            interaction_modes += 1
            features['rna_binding'] = True
            features['rna_binding_confidence'] = rna_binding['confidence']
        
        # Multiple disordered regions
        if len(disorder_analysis.get('disordered_regions', [])) > 1:
            interaction_modes += 1
        
        # Multiple LCR regions  
        if len(lcr_analysis.get('lcr_regions', [])) > 1:
            interaction_modes += 1
        
        # Prion domain (but weight less if low confidence)
        if features['has_prion_domain'] and features['prion_score'] > 0.7:
            interaction_modes += 1
        elif features['has_prion_domain']:
            interaction_modes += 0.5
        
        # Check for specific motifs
        has_rgg = bool(re.search(r'RGG', sequence))
        has_fg = bool(re.search(r'FG', sequence))
        has_sr = bool(re.search(r'[SR]{4,}', sequence))
        
        if any([has_rgg, has_fg, has_sr]):
            interaction_modes += 0.5
        
        score_components['multivalency'] = min(interaction_modes * 0.3, 1.0)
        
        # 3. Phase separation features
        # Charge properties
        net_charge_per_residue = abs(features['net_charge']) / length
        if 0.05 < net_charge_per_residue < 0.15:  # Optimal range
            score_components['phase_separation'] += 0.3
        elif net_charge_per_residue < 0.2:  # Still acceptable
            score_components['phase_separation'] += 0.15
        
        # Aromatic content
        if features['aromatic_content'] > 0.08:
            score_components['phase_separation'] += 0.3
        elif features['aromatic_content'] > 0.05:
            score_components['phase_separation'] += 0.2
        elif features['aromatic_content'] > 0.03:
            score_components['phase_separation'] += 0.1
        
        # Charge patterning
        charge_pattern_score = self._calculate_charge_pattern_score(features)
        score_components['phase_separation'] += charge_pattern_score * 0.3
        
        score_components['phase_separation'] = min(score_components['phase_separation'], 1.0)
        
        # 4. Specific features (more selective)
        if has_rgg and features.get('rna_binding', False):
            score_components['specific_features'] += 0.4
        elif has_rgg:
            score_components['specific_features'] += 0.2
            
        if has_fg and features['disorder_percent'] > 40:
            score_components['specific_features'] += 0.3
        elif has_fg:
            score_components['specific_features'] += 0.1
            
        if has_sr and features.get('rna_binding', False):
            score_components['specific_features'] += 0.3
        elif has_sr:
            score_components['specific_features'] += 0.1
        
        score_components['specific_features'] = min(score_components['specific_features'], 1.0)
        
        # 5. General interaction potential
        # Polar content (but not too high)
        polar_percent = seq_props.get('polar_uncharged_percent', 0)
        if 20 < polar_percent < 40:
            score_components['interaction_potential'] += 0.3
        elif 15 < polar_percent < 50:
            score_components['interaction_potential'] += 0.2
        
        # Q/N content (moderate levels)
        if 0.1 < features['qn_content'] < 0.3:
            score_components['interaction_potential'] += 0.3
        elif 0.05 < features['qn_content'] < 0.4:
            score_components['interaction_potential'] += 0.2
        
        # Not too hydrophobic
        if seq_props.get('hydrophobic_percent', 0) < 45:
            score_components['interaction_potential'] += 0.2
        
        score_components['interaction_potential'] = min(score_components['interaction_potential'], 1.0)
        
        # Calculate base score
        base_score = sum(
            score_components[comp] * self.feature_weights.get(comp, 0.2)
            for comp in score_components
        )
        
        # Apply synergy bonuses (more conservative)
        synergy_multiplier = 1.0
        
        # High disorder + RNA binding
        if features['disorder_percent'] > 40 and features.get('rna_binding', False):
            synergy_multiplier *= 1.15
        
        # Multiple strong features
        strong_components = sum(1 for v in score_components.values() if v > 0.6)
        if strong_components >= 3:
            synergy_multiplier *= 1.1
        
        # Apply penalties
        penalty_multiplier = 1.0
        
        # Very low disorder AND low LCR
        if features['disorder_percent'] < 10 and features['lcr_percent'] < 5:
            penalty_multiplier *= 0.5
        
        # Very high aggregation propensity
        if features['amyloid_propensity'] > 0.9:
            penalty_multiplier *= 0.7
        elif features['amyloid_propensity'] > 0.8:
            penalty_multiplier *= 0.85
        
        # Calculate final score
        final_score = base_score * synergy_multiplier * penalty_multiplier
        final_score = min(final_score, 1.0)
        
        # Determine if forms condensates (slightly higher threshold)
        forms_condensates = final_score > self.base_thresholds['biocondensat_score']
        
        # Additional validation
        if forms_condensates:
            # Need at least minimal disorder or LCR
            if features['disorder_percent'] < 10 and features['lcr_percent'] < 5:
                forms_condensates = False
            
            # Need at least 2 meaningful features
            meaningful_features = sum(1 for v in score_components.values() if v > 0.3)
            if meaningful_features < 2:
                forms_condensates = False
        
        # Determine condensate type and other properties
        if forms_condensates:
            condensate_type = self._assign_condensate_type(features, rna_binding)
            contains_rna = condensate_type in ['Stress granule', 'P-body', 'Nuclear speckle', 
                                               'Nucleolus', 'RNA granule']
            contains_dna = condensate_type in ['Heterochromatin', 'PcG body']
        else:
            condensate_type = None
            contains_rna = False
            contains_dna = False
        
        # Evidence collection
        evidence = []
        if forms_condensates:
            if features['disorder_percent'] > 30:
                evidence.append(f"Disordered regions ({features['disorder_percent']:.1f}%)")
            if features['lcr_percent'] > 10:
                evidence.append(f"Low complexity regions ({features['lcr_percent']:.1f}%)")
            if features['has_prion_domain']:
                evidence.append(f"Prion-like domain (score: {features['prion_score']:.2f})")
            if features.get('rna_binding', False):
                evidence.append("RNA-binding capability")
            if features['aromatic_content'] > 0.05:
                evidence.append(f"Aromatic residues ({features['aromatic_content']*100:.1f}%)")
        
        # Create results in original format
        results = {
            'is_biocondensat_forming': forms_condensates,
            'biocondensat_score': final_score,
            'confidence_interval': {
                'confidence': final_score,
                'lower_bound': max(0, final_score - 0.1),
                'upper_bound': min(1, final_score + 0.1)
            },
            'biocondensat_type': condensate_type,
            'type_confidence': final_score * 0.8 if forms_condensates else 0,
            'alternative_types': [],
            'contains_rna': contains_rna,
            'contains_dna': contains_dna,
            'mechanism': self._determine_mechanism(features),
            'evidence': evidence,
            'features': features,
            'domain_analysis': {},
            'adjusted_thresholds': self.base_thresholds
        }
        
        return results
    
    def _calculate_charge_pattern_score(self, features):
        """Calculate charge patterning score for phase separation"""
        sequence = features.get('sequence', '')
        if not sequence or len(sequence) < 20:
            return 0.0
        
        # Simple charge pattern detection
        positive_blocks = 0
        negative_blocks = 0
        
        for i in range(0, len(sequence) - 10, 5):
            window = sequence[i:i+10]
            pos_charge = sum(1 for aa in window if aa in 'KRH')
            neg_charge = sum(1 for aa in window if aa in 'DE')
            
            if pos_charge > 5:
                positive_blocks += 1
            elif neg_charge > 5:
                negative_blocks += 1
        
        # Good patterning has both positive and negative blocks
        if positive_blocks > 0 and negative_blocks > 0:
            return min((positive_blocks + negative_blocks) / 10, 1.0)
        
        return 0.0
    
    def _assign_condensate_type(self, features, rna_binding):
        """Assign condensate type based on features"""
        # RNA-binding condensates
        if features.get('rna_binding', False):
            if 'RGG' in rna_binding.get('binding_modes', []):
                if features.get('qn_content', 0) > 0.1:
                    return "Stress granule"
                else:
                    return "RNA granule"
            elif 'RS' in rna_binding.get('binding_modes', []):
                return "Nuclear speckle"
            else:
                return "P-body"
        
        # FG-repeat condensates
        if re.search(r'FG', features.get('sequence', '')) and features.get('disorder_percent', 0) > 50:
            return "Nuclear pore"
        
        # Prion-like aggregates
        if features.get('has_prion_domain', False):
            return "Protein aggregate"
        
        # Default
        return "Unspecified condensate"
    
    def _determine_mechanism(self, features):
        """Determine the primary mechanism of condensate formation"""
        mechanisms = []
        
        if features.get('net_charge', 0) / features.get('length', 1) > 0.1:
            mechanisms.append('electrostatic')
        
        if features.get('aromatic_content', 0) > 0.05:
            mechanisms.append('pi-pi stacking')
        
        if features.get('has_prion_domain'):
            mechanisms.append('prion-like')
        
        if features.get('disorder_percent', 0) > 40:
            mechanisms.append('multivalent')
        
        if mechanisms:
            return ', '.join(mechanisms)
        else:
            return 'unknown'
    
    def _predict_heteromeric_structure(self, sequence, biocondensat_results):
        """Predict heteromeric vs homomeric assembly"""
        if not biocondensat_results.get('is_biocondensat_forming', False):
            return {
                'structure_type': 'none',
                'heteromeric_potential': 0.0,
                'homotypic_detail': None,
                'heterotypic_partners': []
            }
        
        features = biocondensat_results.get('features', {})
        
        # Calculate scores
        heteromeric_potential = 0.0
        homotypic_score = 0.0
        heterotypic_partners = []
        
        # Check for RNA binding
        if biocondensat_results.get('contains_rna', False):
            heteromeric_potential += 0.4
            heterotypic_partners.append({
                'type': 'RNA',
                'mechanism': 'RNA-protein interaction',
                'strength': 0.8
            })
        
        # Check charge
        net_charge = features.get('net_charge', 0)
        if abs(net_charge) > 10:
            heteromeric_potential += 0.3
            partner_type = 'Negatively charged proteins' if net_charge > 0 else 'Positively charged proteins'
            heterotypic_partners.append({
                'type': partner_type,
                'mechanism': 'Electrostatic',
                'strength': min(1.0, abs(net_charge) / 20)
            })
        
        # Check for self-assembly features
        if features.get('has_prion_domain', False):
            homotypic_score += 0.5
        
        # Determine structure type
        if heteromeric_potential > 0.5:
            structure_type = "heterotypic"
        elif homotypic_score > 0.3:
            structure_type = "homotypic"
        else:
            structure_type = "mixed"
        
        return {
            'structure_type': structure_type,
            'structure_class': "RNA-protein assembly" if biocondensat_results.get('contains_rna') else "Protein assembly",
            'heteromeric_potential': heteromeric_potential,
            'homotypic_score': homotypic_score,
            'homotypic_detail': "Prion-like self-assembly" if features.get('has_prion_domain') else None,
            'heterotypic_partners': heterotypic_partners,
            'components': {
                'multivalency_score': 0.5,
                'specificity_score': 0.5,
                'complementarity_score': heteromeric_potential
            }
        }
    
    def _classify_aggregation_type(self, protein_results):
        """Classify aggregation type with improved sensitivity"""
        prion_analysis = protein_results.get('prion_analysis', {})
        amyloid_analysis = protein_results.get('amyloid_analysis', {})
        
        # Initialize classification
        classification = {
            'primary_type': 'Non-aggregating',
            'subtype': None,
            'confidence': 0.0,
            'evidence': [],
            'is_prion': False,
            'is_amyloid': False,
            'prion_score': prion_analysis.get('max_score', 0),
            'amyloid_score': amyloid_analysis.get('amyloid_propensity', 0)
        }
        
        # Check prion
        if prion_analysis.get('has_prion_domain', False):
            classification['is_prion'] = True
            classification['primary_type'] = "Prion-like"
            classification['confidence'] = prion_analysis.get('max_score', 0)
            classification['evidence'].append("Prion-like domain detected")
            
            # Subtype
            if prion_analysis.get('domain_type') == 'QN-rich':
                classification['subtype'] = "Q/N-rich prion"
            else:
                classification['subtype'] = "Non-classical prion"
        
        # Check amyloid
        if amyloid_analysis.get('is_amyloidogenic', False):
            classification['is_amyloid'] = True
            
            if classification['is_prion']:
                classification['primary_type'] = "Hybrid"
                classification['subtype'] = "Prion-amyloid hybrid"
                classification['confidence'] = (
                    prion_analysis.get('max_score', 0) + 
                    amyloid_analysis.get('amyloid_propensity', 0)
                ) / 2
            else:
                classification['primary_type'] = "Amyloid"
                classification['subtype'] = amyloid_analysis.get('amyloid_type', "Mixed amyloid")
                classification['confidence'] = amyloid_analysis.get('amyloid_propensity', 0)
            
            classification['evidence'].append("Amyloidogenic regions detected")
        
        return classification
    
    def _save_results(self, results):
        """Save analysis results in original format"""
        # Save detailed JSON results
        json_file = os.path.join(self.output_dir, "enhanced_biocondensat_analysis.json")
        
        # Convert numpy types for JSON serialization
        json_results = {}
        for protein_id, protein_results in results.items():
            json_results[protein_id] = self._prepare_for_json(protein_results)
        
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Detailed results saved to {json_file}")
        
        # Create summary table in ORIGINAL FORMAT
        summary_rows = []
        
        for protein_id, protein_results in results.items():
            row = {
                'protein_id': protein_id,
                'length': protein_results['length'],
                
                # Disorder analysis
                'is_disordered': protein_results.get('disorder_analysis', {}).get('is_disordered', False),
                'disorder_percent': protein_results.get('disorder_analysis', {}).get('disorder_percent', 0),
                
                # LCR analysis
                'has_lcr': protein_results.get('lcr_analysis', {}).get('has_lcr', False),
                'lcr_percent': protein_results.get('lcr_analysis', {}).get('lcr_percent', 0),
                'lcr_type': protein_results.get('lcr_analysis', {}).get('primary_lcr_type', None),
                
                # Prion analysis
                'has_prion_domain': protein_results.get('prion_analysis', {}).get('has_prion_domain', False),
                'prion_score': protein_results.get('prion_analysis', {}).get('max_score', 0),
                'prion_type': protein_results.get('prion_analysis', {}).get('domain_type', None),
                
                # Biocondensate analysis
                'forms_condensate': protein_results.get('biocondensat_analysis', {}).get('is_biocondensat_forming', False),
                'condensate_score': protein_results.get('biocondensat_analysis', {}).get('biocondensat_score', 0),
                'condensate_type': protein_results.get('biocondensat_analysis', {}).get('biocondensat_type', None),
                'type_confidence': protein_results.get('biocondensat_analysis', {}).get('type_confidence', 0),
                'contains_rna': protein_results.get('biocondensat_analysis', {}).get('contains_rna', False),
                'contains_dna': protein_results.get('biocondensat_analysis', {}).get('contains_dna', False),
                
                # Amyloid analysis
                'is_amyloidogenic': protein_results.get('amyloid_analysis', {}).get('is_amyloidogenic', False),
                'amyloid_propensity': protein_results.get('amyloid_analysis', {}).get('amyloid_propensity', 0),
                'amyloid_type': protein_results.get('amyloid_analysis', {}).get('amyloid_type', None),
                
                # Structure prediction
                'structure_type': protein_results.get('heteromeric_structure', {}).get('structure_type', None),
                'heteromeric_potential': protein_results.get('heteromeric_structure', {}).get('heteromeric_potential', 0),
                
                # Aggregation classification
                'aggregation_type': protein_results.get('aggregation_classification', {}).get('primary_type', None),
                'aggregation_confidence': protein_results.get('aggregation_classification', {}).get('confidence', 0)
            }
            
            summary_rows.append(row)
        
        # Save summary table
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_file = os.path.join(self.output_dir, "enhanced_biocondensat_summary.csv")
            summary_df.to_csv(summary_file, index=False, sep='\t')
            logger.info(f"Summary table saved to {summary_file}")
        
        # Generate type distribution report
        self._generate_type_report(results)
    
    def _prepare_for_json(self, obj):
        """Prepare object for JSON serialization"""
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._prepare_for_json(item) for item in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj
    
    def _generate_type_report(self, results):
        """Generate report on biocondensate type distribution"""
        type_counts = defaultdict(int)
        aggregation_counts = defaultdict(int)
        
        for protein_id, protein_results in results.items():
            # Biocondensate types
            if protein_results.get('biocondensat_analysis', {}).get('is_biocondensat_forming', False):
                bc_type = protein_results['biocondensat_analysis'].get('biocondensat_type', 'Unknown')
                type_counts[bc_type] += 1
            
            # Aggregation types
            agg_type = protein_results.get('aggregation_classification', {}).get('primary_type', 'Unknown')
            aggregation_counts[agg_type] += 1
        
        # Write report
        report_file = os.path.join(self.output_dir, "enhanced_analysis_report.txt")
        with open(report_file, 'w') as f:
            f.write("Enhanced Biocondensate Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total proteins analyzed: {len(results)}\n\n")
            
            # Biocondensate types
            f.write("Biocondensate Type Distribution:\n")
            f.write("-" * 30 + "\n")
            total_condensates = sum(type_counts.values())
            for bc_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                percent = (count / total_condensates * 100) if total_condensates > 0 else 0
                f.write(f"{bc_type}: {count} ({percent:.1f}%)\n")
            
            f.write(f"\nTotal condensate-forming proteins: {total_condensates}\n\n")
            
            # Aggregation types
            f.write("Aggregation Type Distribution:\n")
            f.write("-" * 30 + "\n")
            for agg_type, count in sorted(aggregation_counts.items(), key=lambda x: x[1], reverse=True):
                percent = (count / len(results) * 100) if len(results) > 0 else 0
                f.write(f"{agg_type}: {count} ({percent:.1f}%)\n")
        
        logger.info(f"Analysis report saved to {report_file}")
    
    def _create_visualizations(self, results):
        """Create enhanced visualizations"""
        if not self.visualize:
            return
        
        logger.info("Creating visualizations...")
        
        # Create individual protein visualizations
        for protein_id, protein_results in results.items():
            self._visualize_protein_comprehensive(protein_id, protein_results)
        
        # Create aggregate visualizations
        self._visualize_condensate_distribution(results)
        self._visualize_feature_correlations(results)
        self._visualize_confidence_distribution(results)
        
        logger.info("Visualizations completed.")
    
    def _visualize_protein_comprehensive(self, protein_id, protein_results):
        """Create comprehensive visualization for a protein"""
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        
        length = protein_results['length']
        x_axis = list(range(length))
        
        # 1. Disorder profile
        if 'disorder_analysis' in protein_results:
            disorder_scores = protein_results['disorder_analysis']['disorder_scores']
            axes[0].plot(x_axis, disorder_scores, 'b-', alpha=0.7, label='Disorder')
            axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
            axes[0].set_ylabel('Disorder Score')
            axes[0].set_title(f'Protein Analysis: {protein_id}')
            axes[0].legend(loc='upper right')
            
            # Highlight disordered regions
            for region in protein_results['disorder_analysis'].get('disordered_regions', []):
                axes[0].axvspan(region['start'], region['end'], alpha=0.2, color='blue')
        
        # 2. LCR profile
        if 'lcr_analysis' in protein_results:
            complexity_scores = protein_results['lcr_analysis']['complexity_scores']
            axes[1].plot(x_axis, complexity_scores, 'g-', alpha=0.7, label='Complexity')
            axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
            axes[1].set_ylabel('Complexity Score')
            axes[1].legend(loc='upper right')
            
            # Highlight LCR regions
            for region in protein_results['lcr_analysis'].get('lcr_regions', []):
                axes[1].axvspan(region['start'], region['end'], alpha=0.2, color='green')
        
        # 3. Prion profile
        if 'prion_analysis' in protein_results:
            prion_scores = protein_results['prion_analysis']['prion_scores']
            axes[2].plot(x_axis, prion_scores, 'r-', alpha=0.7, label='Prion propensity')
            axes[2].axhline(y=0.55, color='r', linestyle='--', alpha=0.5, label='Threshold')
            axes[2].set_ylabel('Prion Score')
            axes[2].legend(loc='upper right')
            
            # Highlight prion regions
            for region in protein_results['prion_analysis'].get('prion_regions', []):
                axes[2].axvspan(region['start'], region['end'], alpha=0.2, color='red')
        
        # 4. Domain architecture
        axes[3].set_ylim(0, 1)
        axes[3].set_ylabel('Features')
        axes[3].set_xlabel('Position')
        
        # Show key features
        y_pos = 0.8
        if protein_results.get('biocondensat_analysis', {}).get('is_biocondensat_forming', False):
            axes[3].text(length/2, y_pos, 
                        f"Condensate: {protein_results['biocondensat_analysis']['biocondensat_type']}", 
                        ha='center', fontsize=12, weight='bold')
        
        # Show amyloid regions
        if 'amyloid_analysis' in protein_results:
            for region in protein_results['amyloid_analysis'].get('amyloid_regions', []):
                axes[3].barh(0.5, region['length'], left=region['start'], 
                           height=0.2, color='orange', alpha=0.7, label='Amyloid')
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(self.vis_dir, f"{protein_id}_comprehensive_analysis.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_condensate_distribution(self, results):
        """Visualize distribution of condensate types"""
        type_counts = defaultdict(int)
        
        for protein_id, protein_results in results.items():
            if protein_results.get('biocondensat_analysis', {}).get('is_biocondensat_forming', False):
                bc_type = protein_results['biocondensat_analysis'].get('biocondensat_type', 'Unknown')
                type_counts[bc_type] += 1
        
        if not type_counts:
            return
        
        # Create pie chart
        plt.figure(figsize=(10, 8))
        
        labels = list(type_counts.keys())
        sizes = list(type_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Distribution of Biocondensate Types')
        plt.axis('equal')
        
        output_file = os.path.join(self.vis_dir, "condensate_type_distribution.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_feature_correlations(self, results):
        """Visualize correlations between features"""
        # Extract features for correlation analysis
        feature_data = []
        
        for protein_id, protein_results in results.items():
            features = {
                'disorder_percent': protein_results.get('disorder_analysis', {}).get('disorder_percent', 0),
                'lcr_percent': protein_results.get('lcr_analysis', {}).get('lcr_percent', 0),
                'prion_score': protein_results.get('prion_analysis', {}).get('max_score', 0),
                'condensate_score': protein_results.get('biocondensat_analysis', {}).get('biocondensat_score', 0),
                'amyloid_propensity': protein_results.get('amyloid_analysis', {}).get('amyloid_propensity', 0),
                'qn_percent': protein_results.get('prion_analysis', {}).get('qn_percent', 0)
            }
            feature_data.append(features)
        
        if not feature_data:
            return
        
        # Create DataFrame and correlation matrix
        df = pd.DataFrame(feature_data)
        corr_matrix = df.corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlations in Biocondensate Analysis')
        
        output_file = os.path.join(self.vis_dir, "feature_correlations.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_confidence_distribution(self, results):
        """Visualize confidence scores for predictions"""
        condensate_scores = []
        prion_scores = []
        amyloid_scores = []
        
        for protein_id, protein_results in results.items():
            # Condensate scores
            if protein_results.get('biocondensat_analysis', {}).get('is_biocondensat_forming', False):
                condensate_scores.append(
                    protein_results['biocondensat_analysis'].get('biocondensat_score', 0)
                )
            
            # Prion scores
            if protein_results.get('prion_analysis', {}).get('has_prion_domain', False):
                prion_scores.append(
                    protein_results['prion_analysis'].get('max_score', 0)
                )
            
            # Amyloid scores
            if protein_results.get('amyloid_analysis', {}).get('is_amyloidogenic', False):
                amyloid_scores.append(
                    protein_results['amyloid_analysis'].get('amyloid_propensity', 0)
                )
        
        # Create histograms
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Condensate scores
        if condensate_scores:
            ax1.hist(condensate_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax1.set_xlabel('Condensate Score')
            ax1.set_ylabel('Count')
            ax1.set_title('Condensate Formation Scores')
            ax1.axvline(np.mean(condensate_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(condensate_scores):.2f}')
            ax1.legend()
        
        # Prion scores
        if prion_scores:
            ax2.hist(prion_scores, bins=20, alpha=0.7, color='red', edgecolor='black')
            ax2.set_xlabel('Prion Score')
            ax2.set_ylabel('Count')
            ax2.set_title('Prion Domain Scores')
            ax2.axvline(np.mean(prion_scores), color='blue', linestyle='--',
                       label=f'Mean: {np.mean(prion_scores):.2f}')
            ax2.legend()
        
        # Amyloid scores
        if amyloid_scores:
            ax3.hist(amyloid_scores, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax3.set_xlabel('Amyloid Score')
            ax3.set_ylabel('Count')
            ax3.set_title('Amyloid Propensity Scores')
            ax3.axvline(np.mean(amyloid_scores), color='red', linestyle='--',
                       label=f'Mean: {np.mean(amyloid_scores):.2f}')
            ax3.legend()
        
        plt.tight_layout()
        
        output_file = os.path.join(self.vis_dir, "score_distributions.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced Biocondensate Analysis Pipeline v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i proteins.fasta -o results/
  %(prog)s -i proteins.fasta -o results/ -v -t 4
  %(prog)s -i proteins.fasta -o results/ -r rna.fasta -c config.json
        """
    )
    
    parser.add_argument("-i", "--input", required=True,
                       help="Input FASTA file with protein sequences")
    parser.add_argument("-o", "--output", required=True,
                       help="Output directory for results")
    parser.add_argument("-r", "--rna", 
                       help="Optional RNA sequences FASTA file")
    parser.add_argument("-t", "--threads", type=int, default=1,
                       help="Number of threads for parallel processing (default: 1)")
    parser.add_argument("-c", "--config",
                       help="Configuration file (JSON format)")
    parser.add_argument("-v", "--visualize", action="store_true",
                       help="Create visualization plots")
    parser.add_argument("-p", "--protein_id",
                       help="Analyze only specified protein ID")
    parser.add_argument("--version", action="version",
                       version="%(prog)s 3.0")
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Load configuration if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            sys.exit(1)
    
    # Create analyzer instance
    analyzer = EnhancedBiocondensateAnalyzerV3(
        input_file=args.input,
        output_dir=args.output,
        rna_file=args.rna,
        threads=args.threads,
        config=config,
        visualize=args.visualize
    )
    
    # Filter by protein ID if specified
    if args.protein_id:
        if args.protein_id in analyzer.sequences:
            analyzer.sequences = {args.protein_id: analyzer.sequences[args.protein_id]}
            logger.info(f"Analyzing only protein {args.protein_id}")
        else:
            logger.error(f"Protein {args.protein_id} not found in {args.input}")
            sys.exit(1)
    
    # Run analysis
    start_time = datetime.now()
    logger.info(f"Starting enhanced analysis at {start_time}")
    
    try:
        results = analyzer.run_analysis()
        
        # Print summary statistics
        end_time = datetime.now()
        duration = end_time - start_time
        
        total_proteins = len(results)
        condensate_forming = sum(1 for r in results.values() 
                               if r.get('biocondensat_analysis', {}).get('is_biocondensat_forming', False))
        prion_like = sum(1 for r in results.values()
                        if r.get('prion_analysis', {}).get('has_prion_domain', False))
        amyloidogenic = sum(1 for r in results.values()
                           if r.get('amyloid_analysis', {}).get('is_amyloidogenic', False))
        
        logger.info(f"Analysis completed in {duration}")
        logger.info(f"Total proteins analyzed: {total_proteins}")
        logger.info(f"Condensate-forming proteins: {condensate_forming} ({condensate_forming/total_proteins*100:.1f}%)")
        logger.info(f"Proteins with prion-like domains: {prion_like} ({prion_like/total_proteins*100:.1f}%)")
        logger.info(f"Amyloidogenic proteins: {amyloidogenic} ({amyloidogenic/total_proteins*100:.1f}%)")
        logger.info(f"Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
