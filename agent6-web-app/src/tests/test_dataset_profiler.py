"""
Test script for the new Dataset Profiler with caching functionality.

This tests that:
1. Profile is generated on first run
2. Profile is cached to JSON
3. Subsequent runs load from cache (fast)
4. Profile invalidation works when dataset changes
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.agents.dataset_profiler import DatasetProfiler


def test_profiler():
    """Test the profiler with a mock dataset."""
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        # Try reading from file
        api_key_file = Path(__file__).parent.parent.parent / 'ai_data' / 'openai_api_key.txt'
        if api_key_file.exists():
            api_key = api_key_file.read_text().strip()
    
    if not api_key:
        print("❌ No API key found. Set OPENAI_API_KEY environment variable.")
        return False
    
    # Create test cache directory
    cache_dir = Path(__file__).parent.parent.parent / 'ai_data' / 'dataset_profiles_test'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize profiler
    print("="*60)
    print("INITIALIZING DATASET PROFILER")
    print("="*60)
    profiler = DatasetProfiler(api_key=api_key, cache_dir=cache_dir)
    
    # Create mock dataset metadata
    dataset_info = {
        'id': 'test_ecco_llc2160',
        'name': 'Test ECCO LLC2160 Dataset',
        'size': '~100TB',
        'variables': [
            {'id': 'THETA', 'name': 'Potential Temperature', 'units': '°C', 'description': 'Ocean temperature'},
            {'id': 'SALT', 'name': 'Salinity', 'units': 'PSU', 'description': 'Ocean salinity'},
            {'id': 'U', 'name': 'Zonal Velocity', 'units': 'm/s'},
            {'id': 'V', 'name': 'Meridional Velocity', 'units': 'm/s'}
        ],
        'spatial_info': {
            'dimensions': {
                'x': 2160,
                'y': 2160,
                'z': 90
            },
            'geographic_info': {
                'has_geographic_info': True,
                'geographic_info_file': 'llc2160_latlon.nc'
            }
        },
        'temporal_info': {
            'total_time_steps': '10000',
            'time_units': 'hours since 1992-01-01',
            'time_range': {
                'start': '1992-01-01',
                'end': '2023-12-31'
            }
        }
    }
    
    # Test 1: First profile generation (should take time)
    print("\n" + "="*60)
    print("TEST 1: FIRST PROFILE GENERATION (should take ~10-15 seconds)")
    print("="*60)
    import time
    start_time = time.time()
    
    profile1 = profiler.get_or_create_profile(dataset_info)
    
    elapsed1 = time.time() - start_time
    print(f"\n✅ Profile generated in {elapsed1:.2f} seconds")
    print(f"   Quality Score: {profile1['llm_insights']['data_quality_score']}/10")
    print(f"   Primary Bottleneck: {profile1['pattern_analysis']['processing_bottleneck']}")
    
    # Check that JSON file was created
    profile_file = cache_dir / f"{dataset_info['id']}.json"
    if profile_file.exists():
        print(f"✅ Profile cached to: {profile_file}")
    else:
        print(f"❌ Profile file not found: {profile_file}")
        return False
    
    # Test 2: Load from cache (should be instant)
    print("\n" + "="*60)
    print("TEST 2: LOAD FROM CACHE (should be <1ms)")
    print("="*60)
    start_time = time.time()
    
    profile2 = profiler.get_or_create_profile(dataset_info)
    
    elapsed2 = time.time() - start_time
    print(f"\n✅ Profile loaded from cache in {elapsed2*1000:.1f}ms")
    
    if elapsed2 > 0.1:  # More than 100ms is too slow
        print(f"⚠️  Warning: Cache loading took longer than expected ({elapsed2*1000:.1f}ms)")
    
    # Verify it's the same profile
    if profile1['profiled_at'] == profile2['profiled_at']:
        print("✅ Loaded profile matches original (same timestamp)")
    else:
        print("❌ Profile was regenerated instead of loaded from cache!")
        return False
    
    # Test 3: Display profile summary
    print("\n" + "="*60)
    print("PROFILE SUMMARY")
    print("="*60)
    
    llm_insights = profile1['llm_insights']
    print(f"\n📊 Data Quality Score: {llm_insights['data_quality_score']}/10")
    print(f"\n🔬 Scientific Context:")
    print(f"   {llm_insights['scientific_context'][:200]}...")
    
    print(f"\n⚡ Processing Insights:")
    print(f"   Bottleneck: {llm_insights['processing_insights']['primary_bottleneck']}")
    print(f"   Time Expectations: {llm_insights['processing_insights']['time_expectations']}")
    
    print(f"\n🎯 Recommended Quality Levels:")
    opt_guidance = llm_insights['optimization_guidance']
    print(f"   Statistics: {opt_guidance['statistics_queries'][:80]}...")
    print(f"   Visualization: {opt_guidance['visualization_queries'][:80]}...")
    print(f"   Analytics: {opt_guidance['analytics_queries'][:80]}...")
    
    print(f"\n💡 Usage Recommendations:")
    for i, rec in enumerate(llm_insights['usage_recommendations'][:3], 1):
        print(f"   {i}. {rec}")
    
    if llm_insights.get('potential_issues'):
        print(f"\n⚠️  Potential Issues:")
        for issue in llm_insights['potential_issues']:
            print(f"   - {issue}")
    
    # Test 4: Invalidation (change dataset metadata)
    print("\n" + "="*60)
    print("TEST 4: PROFILE INVALIDATION (dataset changed)")
    print("="*60)
    
    # Modify dataset metadata
    dataset_info_modified = dataset_info.copy()
    dataset_info_modified['size'] = '~200TB'  # Changed!
    dataset_info_modified['variables'].append({
        'id': 'NEW_VAR',
        'name': 'New Variable',
        'units': 'units'
    })
    
    start_time = time.time()
    profile3 = profiler.get_or_create_profile(dataset_info_modified)
    elapsed3 = time.time() - start_time
    
    if profile3['profiled_at'] != profile1['profiled_at']:
        print(f"✅ Profile was regenerated (took {elapsed3:.2f}s)")
        print(f"   Old hash: {profile1['cache_validity']['data_hash']}")
        print(f"   New hash: {profile3['cache_validity']['data_hash']}")
        
        # Check backup was created
        backup_file = cache_dir / f"{dataset_info['id']}.backup.json"
        if backup_file.exists():
            print(f"✅ Old profile backed up to: {backup_file}")
        else:
            print(f"⚠️  No backup file found")
    else:
        print("❌ Profile should have been regenerated but wasn't!")
        return False
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print(f"\nCache directory: {cache_dir}")
    print(f"Profile files: {list(cache_dir.glob('*.json'))}")
    
    return True


if __name__ == '__main__':
    success = test_profiler()
    sys.exit(0 if success else 1)
