#!/usr/bin/env python3
"""
Thoroughly inspect H5 files to understand structure.

Usage:
    python3 scripts/inspect_h5_files.py [h5_file_path]
    
If no path provided, inspects all H5 files in h5tests directory.
"""

import sys
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any

def inspect_group(name: str, obj: h5py.Group, indent: int = 0) -> Dict[str, Any]:
    """Recursively inspect an H5 group."""
    info = {
        'name': name,
        'type': 'Group',
        'children': {}
    }
    
    prefix = '  ' * indent
    
    for key in obj.keys():
        child = obj[key]
        if isinstance(child, h5py.Group):
            child_info = inspect_group(key, child, indent + 1)
            info['children'][key] = child_info
        elif isinstance(child, h5py.Dataset):
            info['children'][key] = inspect_dataset(key, child, indent + 1)
    
    # Check attributes
    if obj.attrs:
        info['attrs'] = dict(obj.attrs)
    
    return info

def inspect_dataset(name: str, obj: h5py.Dataset, indent: int = 0) -> Dict[str, Any]:
    """Inspect an H5 dataset."""
    prefix = '  ' * indent
    
    info = {
        'name': name,
        'type': 'Dataset',
        'shape': obj.shape,
        'dtype': str(obj.dtype),
        'size': obj.size
    }
    
    # Get sample data (first few elements)
    if obj.size > 0:
        try:
            if obj.size <= 10:
                info['data'] = obj[:].tolist()
            else:
                info['sample'] = obj[:min(5, len(obj))].tolist()
        except:
            info['sample'] = 'Unable to read'
    
    # Compression info
    if obj.compression:
        info['compression'] = obj.compression
    
    # Attributes
    if obj.attrs:
        info['attrs'] = dict(obj.attrs)
    
    return info

def print_inspection(info: Dict[str, Any], indent: int = 0):
    """Print inspection results."""
    prefix = '  ' * indent
    
    if info['type'] == 'Group':
        print(f"{prefix}📁 {info['name']}/")
        if 'attrs' in info:
            for k, v in info['attrs'].items():
                print(f"{prefix}   @{k}: {v}")
        for child_name, child_info in info['children'].items():
            print_inspection(child_info, indent + 1)
    
    elif info['type'] == 'Dataset':
        shape_str = '×'.join(map(str, info['shape']))
        print(f"{prefix}📊 {info['name']}: {info['dtype']} [{shape_str}]")
        if 'attrs' in info:
            for k, v in info['attrs'].items():
                print(f"{prefix}   @{k}: {v}")
        if 'sample' in info:
            print(f"{prefix}   Sample: {info['sample']}")
        elif 'data' in info:
            print(f"{prefix}   Data: {info['data']}")

def inspect_h5_file(h5_path: Path):
    """Inspect a single H5 file."""
    print(f"\n{'='*70}")
    print(f"File: {h5_path.name}")
    print(f"Path: {h5_path}")
    print(f"Size: {h5_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*70}\n")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Root-level attributes
            if f.attrs:
                print("ROOT ATTRIBUTES:")
                for key, val in f.attrs.items():
                    print(f"  {key}: {val}")
                print()
            
            # Top-level groups
            print("STRUCTURE:")
            print("-" * 70)
            
            root_info = {
                'name': '/',
                'type': 'Group',
                'children': {}
            }
            
            for key in f.keys():
                obj = f[key]
                if isinstance(obj, h5py.Group):
                    root_info['children'][key] = inspect_group(key, obj)
                elif isinstance(obj, h5py.Dataset):
                    root_info['children'][key] = inspect_dataset(key, obj)
            
            print_inspection(root_info)
            
            # Summary statistics
            print("\n" + "-" * 70)
            print("SUMMARY:")
            print("-" * 70)
            
            # Count tracks
            if 'tracks' in f:
                track_keys = list(f['tracks'].keys())
                print(f"  Tracks found: {len(track_keys)}")
                if track_keys:
                    # Inspect first track in detail
                    first_track_key = track_keys[0]
                    first_track = f[f'tracks/{first_track_key}']
                    print(f"\n  First track ({first_track_key}):")
                    if 'positions' in first_track:
                        pos = first_track['positions']
                        print(f"    Positions: {pos.shape} {pos.dtype}")
                        if pos.size > 0:
                            pos_data = pos[:]
                            valid = ~np.isnan(pos_data[:, 0]) if len(pos_data.shape) > 1 else ~np.isnan(pos_data)
                            print(f"    Valid points: {np.sum(valid)}/{len(pos_data)}")
                            if np.sum(valid) > 0:
                                sample_pos = pos_data[valid][:3]
                                print(f"    Sample positions:\n{sample_pos}")
            
            # Check for stimulus/LED data
            print(f"\n  Stimulus/LED data:")
            for key in ['led_data', 'stimulus', 'experiment']:
                if key in f:
                    obj = f[key]
                    if isinstance(obj, h5py.Dataset):
                        print(f"    {key}: Dataset {obj.shape} {obj.dtype}")
                        if obj.size > 0 and obj.size < 100:
                            print(f"      Values: {obj[:].tolist()}")
                    elif isinstance(obj, h5py.Group):
                        print(f"    {key}: Group with keys: {list(obj.keys())}")
                        # Show LED values if available
                        for subkey in ['led1Val', 'intensity', 'value']:
                            if subkey in obj:
                                led_data = obj[subkey]
                                print(f"      {subkey}: {led_data.shape} {led_data.dtype}")
                                if led_data.size > 0:
                                    sample = led_data[:min(10, len(led_data))]
                                    print(f"        Sample: {sample.tolist()}")
                                break
            
            # Metadata
            if 'metadata' in f:
                print(f"\n  Metadata:")
                for key, val in f['metadata'].attrs.items():
                    print(f"    {key}: {val}")
    
    except Exception as e:
        print(f"ERROR reading file: {e}")
        import traceback
        traceback.print_exc()

def main():
    h5_dir = Path("/Users/gilraitses/mechanosensation/h5tests")
    
    if len(sys.argv) > 1:
        # Inspect specific file
        h5_path = Path(sys.argv[1])
        if not h5_path.exists():
            print(f"File not found: {h5_path}")
            sys.exit(1)
        inspect_h5_file(h5_path)
    else:
        # Inspect all H5 files
        h5_files = sorted(h5_dir.glob("*.h5"))
        
        if not h5_files:
            print(f"No H5 files found in {h5_dir}")
            sys.exit(1)
        
        print(f"Found {len(h5_files)} H5 file(s)\n")
        
        for h5_file in h5_files:
            inspect_h5_file(h5_file)
            print("\n")

if __name__ == '__main__':
    try:
        import h5py
    except ImportError:
        print("ERROR: h5py not installed.")
        print("Install with: pip install h5py")
        print("\nAlternatively, you can inspect H5 files using h5dump:")
        print("  h5dump -H /path/to/file.h5")
        sys.exit(1)
    
    main()

