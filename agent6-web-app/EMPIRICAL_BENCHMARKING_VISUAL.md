# Empirical Benchmarking - Visual Guide

## Before vs After

### BEFORE (Heuristics Only)
```
Dataset Metadata
       ↓
[Analyze Size/Dimensions]
       ↓
[Apply Fixed Rules]
   "Large dataset → use quality=-12"
   "Visualization → use quality=-10"
       ↓
❌ Guessing - No actual measurements
```

### AFTER (Empirical + Heuristics)
```
Dataset Metadata
       ↓
[Analyze Size/Dimensions]
       ↓
[Run Test Queries] ⭐ NEW!
  • quality=-15 → measure time
  • quality=-12 → measure time
  • quality=-10 → measure time
  • quality=-8  → measure time
  • quality=-6  → measure time
       ↓
[Analyze Results]
  • Find fastest
  • Find balanced
  • Find detailed
  • Detect cliffs
       ↓
✅ Evidence-based recommendations with actual times
```

## What Gets Tested

```
Test Region (10% of dataset, 1 timestep)
┌─────────────────────────────────┐
│  Full Dataset                   │
│  X=8640, Y=6480, Z=90          │
│  10,366 timesteps               │
│                                 │
│  ┌────────────┐                │
│  │ Test Slice │                │  ← Small region
│  │ X=864      │                │     Single timestep
│  │ Y=648      │                │     Fast to test
│  │ Z=10       │                │
│  │ T=0        │                │
│  └────────────┘                │
│                                 │
└─────────────────────────────────┘
```

## Benchmark Results Example

```
Quality Level: -15 (Most Aggressive)
├── Execution: 0.05 seconds ⚡
├── Data Points: 125,000
├── Memory: 0.48 MB
└── Throughput: 2.5M points/sec

Quality Level: -12
├── Execution: 0.15 seconds
├── Data Points: 1,000,000
├── Memory: 3.81 MB
└── Throughput: 6.7M points/sec

Quality Level: -10 (Balanced)
├── Execution: 0.42 seconds ✅
├── Data Points: 4,000,000
├── Memory: 15.26 MB
└── Throughput: 9.5M points/sec

Quality Level: -8 ⚠️ PERFORMANCE CLIFF
├── Execution: 1.20 seconds (2.9x slower!)
├── Data Points: 16,000,000
├── Memory: 61.04 MB
└── Throughput: 13.3M points/sec

Quality Level: -6 (Maximum Detail)
├── Execution: 4.85 seconds
├── Data Points: 64,000,000
├── Memory: 244.14 MB
└── Throughput: 13.2M points/sec
```

## Sweet Spot Analysis

```
         Fast          Balanced        Detailed
          ↓               ↓              ↓
    ┌────────┐      ┌────────┐     ┌────────┐
    │  -15   │      │  -10   │     │   -6   │
    │ 0.05s  │      │ 0.42s  │     │ 4.85s  │
    │125k pts│      │  4M pts│     │ 64M pts│
    └────────┘      └────────┘     └────────┘
         ↓               ↓              ↓
    Exploration   Visualization    Analysis
```

## How LLM Uses This Data

### Input to LLM
```json
{
  "benchmark_results": {
    "quality_level_performance": {
      "-10": {"execution_time_seconds": 0.42, "data_points_loaded": 4000000},
      "-8": {"execution_time_seconds": 1.20, "data_points_loaded": 16000000}
    },
    "empirical_findings": [
      "Performance cliff: quality -10 → -8 increases time by 2.9x"
    ]
  }
}
```

### LLM Output (Evidence-Based!)
```json
{
  "optimization_guidance": {
    "visualization_queries": "Use quality=-10, which completed in 0.42s 
    in our benchmarks (4M points). Provides good visual detail while 
    maintaining interactive response. Avoid quality=-8 unless necessary 
    as it's 2.9x slower due to I/O bottleneck.",
    
    "statistics_queries": "Use quality=-15 for rapid statistics (0.05s). 
    Sufficient sampling for min/max/mean calculations on this dataset scale."
  },
  
  "usage_recommendations": [
    "Start with quality=-10 (0.42s measured) for exploration",
    "Increase to quality=-6 (4.85s) only for final results",
    "Performance cliff at quality=-8: 2.9x slower than -10 for only 4x more data"
  ]
}
```

## Decision Tree

```
User Query
    ↓
Need quick exploration?
    YES → quality=-15 (0.05s measured) ⚡
    NO  ↓
         Need interactive visualization?
             YES → quality=-10 (0.42s measured) ✅
             NO  ↓
                  Need maximum detail?
                      YES → quality=-6 (4.85s measured) 🎯
                      NO  → Stay at -10 (safe default)
```

## Performance Cliff Detection

```
Time vs Quality Level

Time (s)
 5.0│                                    *  (-6: 4.85s)
    │
 4.0│
    │
 3.0│
    │
 2.0│
    │                          *  (-8: 1.20s)  ⚠️ CLIFF!
 1.0│                          │              (2.9x jump)
    │                          │
 0.5│              *───────────┘  (-10: 0.42s)
    │           (-12: 0.15s)
    │    *  (-15: 0.05s)
 0.0└────┴────┴────┴────┴────┴────┴────┴────
     -15  -14  -13  -12  -11  -10  -9   -8   -7   -6
                   Quality Level →
```

## Code Flow

```python
# Stage 3: Empirical Benchmarking
def _empirical_benchmarking(dataset_info):
    results = {}
    
    # Test each quality level
    for quality in [-15, -12, -10, -8, -6]:
        # Run actual query
        start = time.time()
        data = load_data_with_quality(quality)
        elapsed = time.time() - start
        
        # Record measurements
        results[quality] = {
            'time': elapsed,
            'points': data.size,
            'memory': data.nbytes / (1024**2)
        }
    
    # Analyze results
    fastest = min(results, key=lambda q: results[q]['time'])
    balanced = find_sweet_spot(results)
    detailed = max(results, key=lambda q: results[q]['points'])
    
    return {
        'exploration': fastest,      # -15 (0.05s)
        'visualization': balanced,   # -10 (0.42s)
        'analysis': detailed         # -6 (4.85s)
    }
```

## Real World Example

**User Query**: "Show temperature in the Gulf Stream"

**Before** (Heuristic):
```
System: "Large dataset detected, using quality=-12"
→ Query takes 2.5 seconds
→ May be too slow OR unnecessarily detailed
```

**After** (Empirical):
```
System: "Based on benchmarks, quality=-10 completes in ~0.42s 
        with good detail for visualization"
→ Query takes 0.45 seconds (close to prediction!)
→ Perfect balance for user's need
```

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Recommendations** | Heuristics | Evidence-based |
| **Time Estimates** | Vague | Specific (measured) |
| **Quality Choice** | Fixed rules | Sweet spot analysis |
| **Performance Cliffs** | Unknown | Detected & avoided |
| **User Confidence** | Low | High |
| **Profiling Time** | ~5s | ~10-20s (one-time) |
| **Query Success Rate** | Variable | Higher |

## Key Insight

> **This is what a human expert does!**
> 
> When configuring a new system, humans:
> 1. Try different settings
> 2. Measure results
> 3. Find sweet spots
> 4. Avoid known pitfalls
> 
> Now the AI does the same thing automatically! 🎯
