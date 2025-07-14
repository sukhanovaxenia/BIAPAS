#!/bin/bash
# Example script for running BIAPAS pipeline

echo "BIAPAS - Biomolecular Assembly Prediction and Analysis Suite"
echo "==========================================================="
echo ""

# Check if input file exists
if [ ! -f "example_proteins.fasta" ]; then
    echo "Error: example_proteins.fasta not found!"
    echo "Please run this script from the test/ directory"
    exit 1
fi

# Create output directory
OUTPUT_DIR="example_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "Running BIAPAS analysis..."
echo "Input: example_proteins.fasta"
echo "Output: $OUTPUT_DIR/"
echo ""

# Run main pipeline with visualization
python ../biapas_pipeline.py \
    -i example_proteins.fasta \
    -o $OUTPUT_DIR \
    -t 4 \
    -v

# Check if analysis completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "Analysis completed successfully!"
    echo ""
    
    # Run validation if control proteins are included
    if [ -f "$OUTPUT_DIR/enhanced_biocondensat_summary.csv" ]; then
        echo "Running validation analysis..."
        python ../validate_predictions.py \
            -i $OUTPUT_DIR/enhanced_biocondensat_summary.csv \
            -o $OUTPUT_DIR/validation
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "Validation completed!"
            echo ""
            echo "Results summary:"
            echo "==============="
            # Show summary statistics
            echo "Total proteins analyzed: $(tail -n +2 $OUTPUT_DIR/enhanced_biocondensat_summary.csv | wc -l)"
            echo "Condensate-forming: $(awk -F'\t' 'NR>1 && $4=="True" {count++} END {print count}' $OUTPUT_DIR/enhanced_biocondensat_summary.csv)"
            echo "Prion-like domains: $(awk -F'\t' 'NR>1 && $7=="True" {count++} END {print count}' $OUTPUT_DIR/enhanced_biocondensat_summary.csv)"
            echo "Amyloidogenic: $(awk -F'\t' 'NR>1 && $16=="True" {count++} END {print count}' $OUTPUT_DIR/enhanced_biocondensat_summary.csv)"
        fi
    fi
    
    echo ""
    echo "All results saved to: $OUTPUT_DIR/"
    echo ""
    echo "Key output files:"
    echo "- Summary table: $OUTPUT_DIR/enhanced_biocondensat_summary.csv"
    echo "- Detailed results: $OUTPUT_DIR/enhanced_biocondensat_analysis.json"
    echo "- Report: $OUTPUT_DIR/enhanced_analysis_report.txt"
    if [ -d "$OUTPUT_DIR/visualizations" ]; then
        echo "- Visualizations: $OUTPUT_DIR/visualizations/"
    fi
    if [ -d "$OUTPUT_DIR/validation" ]; then
        echo "- Validation report: $OUTPUT_DIR/validation/validation_report.txt"
    fi
else
    echo ""
    echo "Error: Analysis failed! Check the log file for details."
    exit 1
fi
