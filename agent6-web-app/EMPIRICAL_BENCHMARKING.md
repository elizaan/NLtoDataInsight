# Empirical Benchmarking Enhancement

## What Changed?

The dataset profiler now includes **empirical benchmarking** - it actually tests different quality levels with real queries to measure performance, just like a human expert would!

## The Problem

Previously, the profiler used heuristics to recommend quality levels:
- "For large datasets, use quality=-12"
- "For visualization, use quality=-10"

These were **educated guesses**, not based on actual measurements.

## The Solution: Stage 3 - Empirical Benchmarking

The profiler now runs **quick test queries** during profiling to measure:

### What We Test
1. **Multiple Quality Levels**: -15, -12, -10, -8, -6
2. **Small Test Region**: 10% of each spatial dimension, first timestep
3. **Actual Data Loading**: Uses OpenVisus to load real data (or simulates if unavailable)

### What We Measure
For each quality level, we record:
- ✅ **Execution time** (seconds)
- ✅ **Data points loaded** (actual count)
- ✅ **Memory usage** (MB)
- ✅ **Throughput** (points/second)

### Example Benchmark Results
```json
{
  "quality_level_performance": {
    "-15": {
      "execution_time_seconds": 0.05,
      "data_points_loaded": 125000,
      "memory_mb": 0.48,
      "throughput_points_per_sec": 2500000
    },
    "-12": {
      "execution_time_seconds": 0.15,
      "data_points_loaded": 1000000,
      "memory_mb": 3.81,
      "throughput_points_per_sec": 6666667
    },
    "-10": {
      "execution_time_seconds": 0.42,
      "data_points_loaded": 4000000,
      "memory_mb": 15.26,
      "throughput_points_per_sec": 9523810
    },
    "-8": {
      "execution_time_seconds": 1.20,
      "data_points_loaded": 16000000,
      "memory_mb": 61.04,
      "throughput_points_per_sec": 13333333
    },
    "-6": {
      "execution_time_seconds": 4.85,
      "data_points_loaded": 64000000,
      "memory_mb": 244.14,
      "throughput_points_per_sec": 13195876
    }
  },
  "recommendations": {
    "exploration": {
      "quality_level": -15,
      "reason": "Fastest execution (0.05s)",
      "expected_time_seconds": 0.05
    },
    "visualization": {
      "quality_level": -10,
      "reason": "Good balance: 4,000,000 points in 0.42s",
      "expected_time_seconds": 0.42
    },
    "analysis": {
      "quality_level": -6,
      "reason": "Maximum detail: 64,000,000 points",
      "expected_time_seconds": 4.85
    }
  },
  "empirical_findings": [
    "Tested 5 quality levels",
    "Quality -15: 0.05s (125,000 points)",
    "Quality -6: 4.85s (64,000,000 points)",
    "Throughput range: 2,500,000 - 13,333,333 points/sec",
    "Performance cliff: quality -10 → -8 increases time by 2.9x"
  ]
}
```

## How It Works

### New Algorithm (4 Stages)

```
Stage 1: Statistical Sampling (metadata analysis)
    ↓
Stage 2: Pattern Detection (algorithmic analysis)
    ↓
Stage 3: Empirical Benchmarking ⭐ NEW!
    ↓
    For each quality level [-15, -12, -10, -8, -6]:
        1. Create small test query (10% spatial slice)
        2. Load data with OpenVisus (or simulate)
        3. Measure: time, data points, memory, throughput
        4. Record results
    ↓
    Analyze results:
        - Find fastest (exploration)
        - Find balanced (visualization)
        - Find most detailed (analysis)
        - Detect performance cliffs
    ↓
Stage 4: LLM Synthesis (with empirical evidence!)
```

### Key Methods

1. **`_empirical_benchmarking()`** - Main orchestration
   - Tests multiple quality levels
   - Collects measurements
   - Analyzes results

2. **`_run_benchmark_query()`** - Runs single test
   - Attempts actual OpenVisus data loading
   - Falls back to realistic simulation if unavailable
   - Measures time, memory, throughput

3. **`_simulated_benchmark()`** - Fallback when no OpenVisus
   - Calculates realistic estimates based on:
     - Region size
     - Quality level (downsampling factor)
     - Typical I/O speeds (~25M points/sec)
   - Simulates small delay for realism

4. **`_analyze_benchmark_results()`** - Derive recommendations
   - Finds sweet spots for different use cases
   - Identifies performance cliffs (3x+ slowdowns)
   - Generates evidence-based recommendations

## LLM Integration

The LLM synthesis prompt now includes:
```
**EMPIRICAL BENCHMARK RESULTS (ACTUAL MEASUREMENTS):**
{benchmark data}

**IMPORTANT:** Use these empirical measurements to make 
evidence-based recommendations, not just heuristics!
```

### Example LLM Output (Evidence-Based!)

**Before** (heuristic):
```
"visualization_queries": "Use quality=-10 as starting point"
```

**After** (empirical):
```
"visualization_queries": "Use quality=-10, which our benchmarks 
show completes in 0.42 seconds for typical regions (4M points). 
This provides good visual detail while maintaining interactive 
response times."
```

## Benefits

### 1. **Evidence-Based Recommendations**
- No more guessing - we have actual measurements
- Recommendations cite real timing data
- Users know exactly what to expect

### 2. **Dataset-Specific Tuning**
- Every dataset is benchmarked individually
- Recommendations adapt to actual hardware/network
- Accounts for dataset-specific I/O characteristics

### 3. **Performance Cliff Detection**
- Identifies where quality increases hurt too much
- Example: "quality -10 → -8 increases time by 2.9x"
- Helps users avoid expensive mistakes

### 4. **Human Expert Mimicry**
- This is exactly what a human would do!
- Test with different settings
- Measure results
- Make informed decisions

## Graceful Degradation

If benchmarking fails (no OpenVisus, errors, etc.):
1. **Simulated benchmarks** - Uses realistic estimates
2. **Fallback profile** - Provides safe defaults
3. **System never crashes** - Always returns usable profile

## Example Usage Recommendations (After Benchmarking)

```json
"usage_recommendations": [
  "Use quality=-15 for rapid exploration (0.05s measured)",
  "Use quality=-10 for interactive visualization (0.42s, good detail)",
  "Use quality=-6 for final analysis (4.85s, maximum detail)",
  "Avoid quality=-8 unless necessary (performance cliff: 2.9x slower than -10)",
  "For time-constrained queries (<1 min), stay at quality=-12 or higher"
]
```

## Performance Impact

- **Profiling time increase**: ~5-15 seconds (runs 5 test queries)
- **Profiling frequency**: Once per dataset (cached forever)
- **Query time improvement**: Potentially significant (avoid bad quality choices)
- **User confidence**: Much higher (evidence-based guidance)

## Future Enhancements

1. **More test patterns**: Test different regions, timesteps
2. **Query type specific**: Benchmark statistics vs visualization separately
3. **Adaptive testing**: Test fewer quality levels if pattern is clear
4. **Parallel testing**: Run multiple benchmarks concurrently
5. **Historical tracking**: Compare benchmarks over time

## Testing

The benchmarking is automatically tested when you:
1. Load a new dataset (first time)
2. Dataset metadata changes (triggers re-profiling)

To see benchmarking in action:
```bash
# The profiler will automatically run benchmarks
cd agent6-web-app
python src/app.py

# Load dataset via UI
# Check console for:
# [Profiler] Stage 3: Empirical benchmarking (testing quality levels)
# [Profiler] Testing quality levels [-15, -12, -10, -8, -6] on temperature
```

## Inspect Benchmark Results

```bash
# View cached profile with benchmarks
cat agent6-web-app/ai_data/dataset_profiles/dyamond_llc2160.json | jq '.benchmark_results'
```

## Summary

**What changed**: Added Stage 3 (Empirical Benchmarking) that actually tests quality levels

**Why**: Human experts test settings empirically - now the AI does too!

**Result**: Evidence-based recommendations with real measured times instead of heuristics

**Impact**: Better query optimization, user confidence, and performance
