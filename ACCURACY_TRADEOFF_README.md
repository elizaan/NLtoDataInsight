# Accuracy vs Execution Time Tradeoff Analysis

This document explains the new empirical tradeoff analysis features added to the dataset profiler.

## Overview

The dataset profiler now performs comprehensive **accuracy vs execution time tradeoff analysis** by:

1. **Empirical Benchmarking**: Testing multiple quality levels on actual data
2. **Regional Subset Testing**: Testing smaller spatial regions to show spatial scaling
3. **Accuracy Metrics**: Computing resolution loss and accuracy degradation
4. **Tradeoff Analysis**: Quantifying speedup gains vs accuracy costs
5. **Visualization**: Generating publication-quality plots for research papers

## What's New

### 1. Regional Subset Tests

The profiler now tests:
- **Full spatial extent**: Original tests with all quality levels
- **Small regions**: e.g., x=[0, 200], y=[0, 200]
- **Medium regions**: e.g., x=[0, 500], y=[0, 500]

This shows how spatial subsetting affects performance independently of quality/resolution settings.

### 2. Accuracy Metrics

For each test suite and quality level, we compute:

```json
{
  "quality_level": -10,
  "execution_time_seconds": 3.49,
  "data_points": 10497600,
  "accuracy_retained_percent": 33.3,  // NEW
  "accuracy_loss_percent": 66.7,      // NEW
  "speedup_vs_baseline": 53.0,        // NEW
  "efficiency_score": 9.54            // NEW
}
```

**Metrics explained:**
- `accuracy_retained_percent`: (current_points / baseline_points) × 100
- `accuracy_loss_percent`: 100 - accuracy_retained_percent
- `speedup_vs_baseline`: baseline_time / current_time
- `efficiency_score`: accuracy_retained / time_saved

### 3. Recommendations by Time Budget

The profiler generates recommendations for different time budgets:

```json
"recommendations_by_time_budget": {
  "10_percent": {
    "recommended_quality": -12,
    "expected_time_seconds": 1.65,
    "accuracy_retained_percent": 8.3,
    "accuracy_loss_percent": 91.7
  },
  "50_percent": {
    "recommended_quality": -8,
    "expected_time_seconds": 7.63,
    "accuracy_retained_percent": 50.0,
    "accuracy_loss_percent": 50.0
  }
}
```

## Profile JSON Structure

The enhanced profile JSON now includes:

```json
{
  "benchmark_results": {
    "tests_performed": [...],
    "failed_tests": [...],
    "by_test_suite": {...},
    
    "accuracy_tradeoff_analysis": {  // NEW
      "test_suite_name": {
        "baseline_quality": 0,
        "baseline_time_seconds": 185.3,
        "baseline_data_points": 1259712000,
        "tradeoffs": [
          {
            "quality_level": -2,
            "execution_time_seconds": 3.5,
            "accuracy_retained_percent": 95.0,
            "speedup_vs_baseline": 53.0
          }
        ],
        "recommendations_by_time_budget": {...}
      }
    },
    
    "visualization_data": [...]  // NEW: Plot-ready data
  }
}
```

## Using Tradeoff Data in Prompts

The system prompt in `dataset_insight_generator.py` now uses this data:

```python
# Example: LLM sees empirical tradeoff data
prompt = f"""
Based on empirical benchmarking:
- Quality -12: 1.65s execution, 8.3% accuracy retained (91.7% loss)
- Quality -10: 3.49s execution, 33.3% accuracy retained (66.7% loss)
- Quality -8: 7.63s execution, 50.0% accuracy retained (50.0% loss)

User's time budget: {user_time_limit} minutes

**Your task**: Choose the optimal quality level that:
1. Fits within the time budget
2. Maximizes accuracy retained
3. Balances the tradeoff intelligently
"""
```

## Generating Plots for Research Papers

Use the provided plotting script:

```bash
# Generate all plots
python agent6-web-app/src/utils/plot_accuracy_tradeoffs.py \
    agent6-web-app/ai_data/dataset_profiles/dyamond_llc2160.json \
    ./research_plots

# Output:
#   research_plots/
#     ├── tradeoff_full_resolution_single_timestep.png
#     ├── tradeoff_multi_timestep_2_timesteps.png
#     ├── speedup_vs_accuracy_loss.png
#     ├── efficiency_frontier.png
#     └── tradeoff_table.tex
```

### Plot Types

1. **Per-Suite Tradeoff Curves** (`tradeoff_*.png`)
   - X-axis: Execution time (seconds)
   - Y-axis: Accuracy retained (%)
   - Shows quality levels annotated
   - Highlights high-accuracy region (>90%)

2. **Speedup vs Accuracy Loss** (`speedup_vs_accuracy_loss.png`)
   - Compares all test suites
   - Shows efficiency of different quality settings
   - Useful for understanding cost-benefit

3. **Efficiency Frontier** (`efficiency_frontier.png`)
   - Pareto-optimal points
   - Log scale on X-axis
   - Shows "best" settings for each time budget

4. **LaTeX Table** (`tradeoff_table.tex`)
   - Ready to include in papers
   - Top 5 quality levels
   - Formatted for academic publications

## Research Contributions

This implementation enables:

1. **Empirical validation** of quality/resolution tradeoffs (no simulation!)
2. **Reproducible benchmarks** for dataset-specific optimization
3. **Intelligent query planning** based on measured performance
4. **User-driven time constraints** with accuracy guarantees
5. **Publication-quality visualizations** for research papers

## Example Usage in System

When a user asks a query with time constraints:

```python
# User: "Show me velocity trends for January 2020, I have 2 minutes"

# System uses tradeoff analysis:
tradeoffs = profile['accuracy_tradeoff_analysis']['multi_timestep_5_timesteps']

# Find best quality for 2 min = 120 seconds
for t in tradeoffs['tradeoffs']:
    if t['execution_time_seconds'] <= 120:
        selected_quality = t['quality_level']
        expected_accuracy = t['accuracy_retained_percent']
        break

# LLM prompt includes this:
"""
Based on empirical tests with your 2-minute budget:
- Recommended quality: -10
- Expected time: 6.3 seconds
- Accuracy: 33% of full resolution
- Speedup: 18.7x faster than full resolution
"""
```

## Future Enhancements

Potential additions:
- **Memory usage tradeoffs**: Show memory vs accuracy curves
- **Multi-variable tests**: Test different variables (Temperature, Salinity, etc.)
- **Temporal aggregation tests**: Compare daily/hourly/monthly aggregations
- **Interactive plots**: HTML dashboards for exploring tradeoffs

## Citation

If you use this tradeoff analysis in your research, please cite:

```bibtex
@software{nlq_to_data_insight_profiler,
  title={Empirical Quality vs Performance Tradeoff Analysis for Large-Scale Scientific Data},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo}
}
```

## Questions?

For questions or issues, contact the development team or file a GitHub issue.
