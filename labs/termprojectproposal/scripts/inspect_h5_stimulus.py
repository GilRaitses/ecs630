#!/usr/bin/env python3
"""
Inspect H5 files to find global quantities and LED stimulus data.
Specifically looking for:
- Red pulsing LED (led1Val)
- Blue constant LED (led2Val)
- Pulse duration (should be 10 seconds)
- Experiment structure matching real tracks
"""

import h5py
import numpy as np
from pathlib import Path

def inspect_h5_stimulus(filepath):
    """Thoroughly inspect H5 file for stimulus data and structure."""
    print(f"\n{'='*80}")
    print(f"Inspecting: {Path(filepath).name}")
    print(f"{'='*80}")
    
    with h5py.File(filepath, 'r') as f:
        # Print top-level structure
        print("\n=== TOP-LEVEL KEYS ===")
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Group):
                print(f"  {key}/ (Group)")
                try:
                    subkeys = list(obj.keys())
                    print(f"    Subkeys: {subkeys[:10]}{'...' if len(subkeys) > 10 else ''}")
                except:
                    pass
            elif isinstance(obj, h5py.Dataset):
                print(f"  {key} (Dataset): shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"  {key}: {type(obj)}")
        
        # Look for global_quantities
        print("\n=== GLOBAL QUANTITIES ===")
        gq_paths = ['global_quantities', 'globalQuantities', 'global_quantity', 'gq']
        found_gq = False
        for gq_path in gq_paths:
            if gq_path in f:
                found_gq = True
                gq = f[gq_path]
                print(f"Found: {gq_path}")
                if isinstance(gq, h5py.Group):
                    print(f"  Keys: {list(gq.keys())}")
                    for key in gq.keys():
                        item = gq[key]
                        if isinstance(item, h5py.Dataset):
                            data = item[...]
                            print(f"    {key}: shape={data.shape}, dtype={data.dtype}, range=[{data.min():.2f}, {data.max():.2f}]")
                        elif isinstance(item, h5py.Group):
                            print(f"    {key}/ (Group): {list(item.keys())}")
                break
        
        if not found_gq:
            print("  Not found at top level, searching recursively...")
        
        # Look for LED data
        print("\n=== LED/STIMULUS DATA ===")
        led_candidates = []
        
        def search_for_led(name, obj):
            if isinstance(obj, h5py.Dataset):
                name_lower = name.lower()
                if any(term in name_lower for term in ['led', 'stimulus', 'intensity', 'light']):
                    try:
                        data = obj[...]
                        if len(data.shape) <= 2 and data.size > 0:
                            led_candidates.append((name, data.shape, data.dtype, np.min(data), np.max(data)))
                    except:
                        pass
        
        f.visititems(search_for_led)
        
        if led_candidates:
            print("Found LED/stimulus candidates:")
            for name, shape, dtype, min_val, max_val in led_candidates[:20]:
                print(f"  {name}: shape={shape}, dtype={dtype}, range=[{min_val:.2f}, {max_val:.2f}]")
        else:
            print("  No LED/stimulus datasets found")
        
        # Check metadata attributes
        print("\n=== METADATA ATTRIBUTES ===")
        if 'metadata' in f:
            md = f['metadata']
            print(f"Metadata group found. Keys: {list(md.keys())}")
            if hasattr(md, 'attrs'):
                print(f"  Attributes: {dict(md.attrs)}")
        
        # Check tracks structure
        print("\n=== TRACKS STRUCTURE ===")
        if 'tracks' in f:
            tracks = f['tracks']
            track_keys = list(tracks.keys())
            print(f"Found {len(track_keys)} tracks")
            if track_keys:
                first_track = tracks[track_keys[0]]
                print(f"  First track ({track_keys[0]}) structure:")
                print(f"    Keys: {list(first_track.keys())}")
                
                # Check for LED data in tracks
                for key in first_track.keys():
                    if isinstance(first_track[key], h5py.Dataset):
                        try:
                            data = first_track[key][...]
                            if 'led' in key.lower() or 'stimulus' in key.lower():
                                print(f"    {key}: shape={data.shape}, range=[{data.min():.2f}, {data.max():.2f}]")
                        except:
                            pass
        
        # Look for experiment-level LED data
        print("\n=== EXPERIMENT-LEVEL DATA ===")
        exp_paths = ['experiment', 'expt', 'experiment_data']
        for exp_path in exp_paths:
            if exp_path in f:
                exp = f[exp_path]
                print(f"Found: {exp_path}")
                if isinstance(exp, h5py.Group):
                    print(f"  Keys: {list(exp.keys())}")
                    for key in exp.keys():
                        if 'led' in key.lower() or 'stimulus' in key.lower():
                            item = exp[key]
                            if isinstance(item, h5py.Dataset):
                                data = item[...]
                                print(f"    {key}: shape={data.shape}, range=[{data.min():.2f}, {data.max():.2f}]")
        
        # Full recursive search for led1Val and led2Val
        print("\n=== SEARCHING FOR led1Val AND led2Val ===")
        led1_found = []
        led2_found = []
        
        def find_led_values(name, obj):
            if isinstance(obj, h5py.Dataset):
                name_lower = name.lower()
                if 'led1val' in name_lower or 'led1_val' in name_lower:
                    try:
                        data = obj[...]
                        led1_found.append((name, data.shape, np.min(data), np.max(data), len(data)))
                    except:
                        pass
                elif 'led2val' in name_lower or 'led2_val' in name_lower:
                    try:
                        data = obj[...]
                        led2_found.append((name, data.shape, np.min(data), np.max(data), len(data)))
                    except:
                        pass
        
        f.visititems(find_led_values)
        
        if led1_found:
            print("LED1 (Red pulsing) found:")
            for name, shape, min_val, max_val, length in led1_found:
                print(f"  {name}: shape={shape}, length={length}, range=[{min_val:.2f}, {max_val:.2f}]")
                # Check if it's pulsing
                if length > 100:
                    sample = led1_found[0]
                    f.visititems(lambda n, o: print(f"    Full path: {n}") if n == name else None)
                    with h5py.File(filepath, 'r') as f2:
                        data = f2[name][...]
                        # Check for variability (pulsing)
                        std_dev = np.std(data)
                        print(f"    Std dev: {std_dev:.2f} (high = pulsing, low = constant)")
        
        if led2_found:
            print("LED2 (Blue constant) found:")
            for name, shape, min_val, max_val, length in led2_found:
                print(f"  {name}: shape={shape}, length={length}, range=[{min_val:.2f}, {max_val:.2f}]")
                # Check if it's constant
                if length > 100:
                    with h5py.File(filepath, 'r') as f2:
                        data = f2[name][...]
                        std_dev = np.std(data)
                        print(f"    Std dev: {std_dev:.2f} (low = constant)")

if __name__ == '__main__':
    files = [
        '/Users/gilraitses/mechanosensation/h5tests/GMR61_202509051201_tier1 1.h5',
        '/Users/gilraitses/mechanosensation/h5tests/GMR61_202509051201_tier3.h5'
    ]
    
    for filepath in files:
        if Path(filepath).exists():
            try:
                inspect_h5_stimulus(filepath)
            except Exception as e:
                print(f"Error inspecting {filepath}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"File not found: {filepath}")

