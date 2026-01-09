#!/bin/bash
# Quick script to run comprehensive evaluation
# Usage: ./run_evaluation.sh

echo "========================================="
echo "Comprehensive Model Evaluation"
echo "========================================="
echo ""

# Check if test data exists
if [ -f "KDDTest_plus.csv" ]; then
    TEST_DATA="KDDTest_plus.csv"
elif [ -f "archive/KDDTest+.txt" ]; then
    TEST_DATA="archive/KDDTest+.txt"
else
    echo "ERROR: Test data not found!"
    echo "Please ensure KDDTest_plus.csv or archive/KDDTest+.txt exists"
    exit 1
fi

echo "Using test data: $TEST_DATA"
echo ""

# Run comprehensive evaluation
python comprehensive_evaluation.py \
    --test_data "$TEST_DATA" \
    --artifact_dir . \
    --output_dir evaluation_results

echo ""
echo "========================================="
echo "Evaluation complete!"
echo "Results saved in: evaluation_results/"
echo "========================================="

