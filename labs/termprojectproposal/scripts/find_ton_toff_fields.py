#!/usr/bin/env python3
"""
Find addTonToff fields (_ton and _toff) in H5 files for LED1 and LED2.
"""

import h5py
import numpy as np
from pathlib import Path

def find_ton_toff_fields(filepath):
    """Find _ton and _toff fields for led1Val and led2Val."""
    print(f"\n{'='*80}")
    print(f"Finding addTonToff fields: {Path(filepath).name}")
    print(f"{'='*80}")
    
    ton_toff_fields = {}
    
    with h5py.File(filepath, 'r') as f:
        def search_ton_toff(name, obj):
            if isinstance(obj, h5py.Dataset):
                name_lower = name.lower()
                if '_ton' in name_lower or '_toff' in name_lower:
                    try:
                        data = obj[...]
                        ton_toff_fields[name] = {
                            'shape': data.shape,
                            'dtype': data.dtype,
                            'sample': data[:min(10, len(data))] if len(data) > 0 else []
                        }
                    except:
                        pass
        
        f.visititems(search_ton_toff)
        
        print(f"\n=== TON/TOFF FIELDS FOUND ===")
        if ton_toff_fields:
            for field_name, info in sorted(ton_toff_fields.items()):
                print(f"\n{field_name}:")
                print(f"  Shape: {info['shape']}")
                print(f"  Dtype: {info['dtype']}")
                if len(info['sample']) > 0:
                    print(f"  Sample values: {info['sample']}")
                    if info['dtype'] == 'bool' or np.issubdtype(info['dtype'], np.bool_):
                        print(f"  (Boolean field - True = ON/OFF time)")
                        print(f"  Fraction True: {np.sum(info['sample']) / len(info['sample']):.3f}")
                    elif np.issubdtype(info['dtype'], np.number):
                        print(f"  (Numeric field - likely time values)")
                        print(f"  Range: [{np.min(info['sample'])}, {np.max(info['sample'])}]")
        else:
            print("  No _ton or _toff fields found")
            print("  These fields may need to be created using MAGAT addTonToff()")
        
        # Also check tracks for ton/toff fields
        print(f"\n=== CHECKING TRACKS FOR TON/TOFF FIELDS ===")
        if 'tracks' in f:
            tracks = f['tracks']
            track_keys = list(tracks.keys())
            if track_keys:
                first_track = tracks[track_keys[0]]
                print(f"  Checking first track ({track_keys[0]})...")
                
                track_ton_toff = {}
                def search_track(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        if '_ton' in name.lower() or '_toff' in name.lower():
                            try:
                                data = obj[...]
                                track_ton_toff[name] = {
                                    'shape': data.shape,
                                    'dtype': data.dtype
                                }
                            except:
                                pass
                
                first_track.visititems(search_track)
                
                if track_ton_toff:
                    print(f"  Found ton/toff fields in track:")
                    for field_name, info in sorted(track_ton_toff.items()):
                        print(f"    {field_name}: shape={info['shape']}, dtype={info['dtype']}")
                else:
                    print(f"  No ton/toff fields found in tracks")
        
        # Check for led1Val and led2Val in global quantities or elsewhere
        print(f"\n=== CHECKING FOR LED1VAL AND LED2VAL DATA ===")
        led_fields = {}
        def search_led(name, obj):
            if isinstance(obj, h5py.Dataset):
                name_lower = name.lower()
                if 'led1val' in name_lower or 'led2val' in name_lower:
                    if '_ton' not in name_lower and '_toff' not in name_lower:
                        try:
                            data = obj[...]
                            led_fields[name] = {
                                'shape': data.shape,
                                'dtype': data.dtype,
                                'min': np.min(data),
                                'max': np.max(data),
                                'mean': np.mean(data)
                            }
                        except:
                            pass
        
        f.visititems(search_led)
        
        if led_fields:
            for field_name, info in sorted(led_fields.items()):
                print(f"\n{field_name}:")
                print(f"  Shape: {info['shape']}")
                print(f"  Range: [{info['min']:.2f}, {info['max']:.2f}], Mean: {info['mean']:.2f}")
        else:
            print("  No led1Val or led2Val fields found")

if __name__ == '__main__':
    files = [
        '/Users/gilraitses/mechanosensation/h5tests/GMR61_202509051201_tier1 1.h5',
        '/Users/gilraitses/mechanosensation/h5tests/GMR61_202509051201_tier3.h5'
    ]
    
    for filepath in files:
        if Path(filepath).exists():
            try:
                find_ton_toff_fields(filepath)
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"File not found: {filepath}")

