#!/usr/bin/env python3
"""
Inspect H5 file structure to understand data organization.

Usage:
    python scripts/inspect_h5_structure.py <h5_file_path>
"""

import sys
import h5py
import numpy as np
from pathlib import Path

def print_structure(name, obj):
    """Recursively print HDF5 structure."""
    if isinstance(obj, h5py.Dataset):
        print(f"  {name}: Dataset {obj.shape} {obj.dtype}")
        if obj.size < 100:
            print(f"    Sample: {obj[:min(5, len(obj))]}")
    elif isinstance(obj, h5py.Group):
        print(f"{name}/")

def inspect_h5_file(h5_path):
    """Inspect an H5 file and print its structure."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {h5_path}")
    print(f"{'='*60}\n")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            print("File size:", f"{Path(h5_path).stat().st_size / 1024 / 1024:.2f} MB")
            print("\nStructure:")
            print("/")
            f.visititems(print_structure)
            
            # Try to get some key information
            print("\n" + "-"*60)
            print("Key information:")
            
            # Check for metadata
            if 'metadata' in f:
                print("\nMetadata:")
                for key, value in f['metadata'].attrs.items():
                    print(f"  {key}: {value}")
            
            # Check for tracks
            if 'tracks' in f:
                track_keys = list(f['tracks'].keys())
                print(f"\nTracks found: {len(track_keys)}")
                if track_keys:
                    # Inspect first track
                    first_track = track_keys[0]
                    print(f"\nFirst track structure ({first_track}):")
                    f[f'tracks/{first_track}'].visititems(print_structure)
                    
                    # Try to get positions if available
                    if 'positions' in f[f'tracks/{first_track}']:
                        pos = f[f'tracks/{first_track}/positions']
                        print(f"\n  Positions shape: {pos.shape}")
                        print(f"  Sample positions:\n{pos[:5]}")
            
            # Check for LED/stimulus data
            if 'led_data' in f:
                print("\nLED Data found:")
                f['led_data'].visititems(print_structure)
            elif 'stimulus' in f:
                print("\nStimulus Data found:")
                f['stimulus'].visititems(print_structure)
            elif 'experiment' in f:
                print("\nExperiment Data found:")
                f['experiment'].visititems(print_structure)
            
    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inspect_h5_structure.py <h5_file_path>")
        print("\nAvailable files:")
        h5_dir = Path("/Users/gilraitses/mechanosensation/h5tests")
        for h5_file in sorted(h5_dir.glob("*.h5")):
            size_mb = h5_file.stat().st_size / 1024 / 1024
            print(f"  {h5_file.name} ({size_mb:.1f} MB)")
        sys.exit(1)
    
    h5_path = sys.argv[1]
    inspect_h5_file(h5_path)

