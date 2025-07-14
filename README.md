# BIAPAS: Enhanced Biocondensate Analysis Pipeline

BIAPAS (Biocondensate Integrated Analysis Pipeline for Amyloids and Prions) is a Python-based tool for analyzing protein sequences to predict:
- Biocondensate-forming potential
- Prion-like domains
- Amyloidogenic propensity
- RNA-binding capability
- Low complexity/disordered regions
- Aggregation types and structural assembly

This version improves detection performance while maintaining the original output format.

## 🚀 Features

| Feature                        | BIAPAS v3 Goal |
|-------------------------------|----------------|
| Condensate detection (sensitivity) | >70%           |
| Amyloid prediction (accuracy)     | >70%           |
| Prion specificity                 | >80%           |
| Output structure                 | Preserved      |

## 🧬 Input

- Protein FASTA file (`.fasta`)
- Optional RNA FASTA file (`--rna`)
- Optional config file (`.json`) to override thresholds or weights

## 📦 Installation

Clone the repo and install the dependencies:
```bash
git clone https://github.com/<your-org-or-username>/biapas.git
cd biapas
pip install -r requirements.txt
```

## 🔧 Usage
```python
python3 biapas_pipeline.py -i proteins.fasta -o output_dir
```

Optional arguments:
-r, --rna – RNA sequences in FASTA format
-c, --config – Configuration file (.json)
-t, --threads – Number of threads (default: 1)
-v, --visualize – Generate plots and figures
-p, --protein_id – Analyze only a specific protein
--version – Show version

## 📁 Output

enhanced_biocondensat_summary.csv – Summary table of features
enhanced_biocondensat_analysis.json – Full analysis per protein
enhanced_analysis_report.txt – Condensate and aggregation type stats
visualizations/ – (if --visualize) Visual plots per protein and summary plots

## 🧪 Validation Goals

Improved recall for Q/N-rich and disordered amyloids
Enhanced specificity for prion domains
Balanced detection for RNA-binding, aggregation, and condensate-forming regions

Validation command
```python
python3 validate_predictions.py -i enhanced_biocondensat_summary.csv -o output_dir
```

## 👩‍🔬 Citation

Please cite as:

Sukhanova X. et al., BIAPAS: Enhanced Biocondensate and Amyloid-Prion Analysis Pipeline, v3.0, 2025. https://github.com/<your-org-or-username>/biapas

## 📬 Contact

For feedback or questions, please contact: sukhanovaxenia@gmail.com
