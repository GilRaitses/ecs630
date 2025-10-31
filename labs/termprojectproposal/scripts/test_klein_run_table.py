#!/usr/bin/env python3
"""
Tests for Klein Run Table Generator

Tests all functionality including edge cases and error handling.
"""

import numpy as np
import pandas as pd
import pytest
from klein_run_table import (
    wrap_angle_diff,
    calculate_path_length,
    associate_head_swings_with_turns,
    calculate_first_head_swing,
    generate_klein_run_table,
    calculate_klein_derived_metrics,
    calculate_turn_rate
)

class TestAngleWrapping:
    """Test angle wrapping function"""
    
    def test_wrap_angle_simple(self):
        """Test simple angle differences"""
        theta1 = np.array([0.0])
        theta2 = np.array([np.pi / 2])
        result = wrap_angle_diff(theta1, theta2)
        assert np.allclose(result, np.pi / 2)
    
    def test_wrap_angle_wrap_positive(self):
        """Test wrapping when difference > π"""
        theta1 = np.array([0.0])
        theta2 = np.array([np.pi + 0.5])
        result = wrap_angle_diff(theta1, theta2)
        assert np.allclose(result, -np.pi + 0.5)
    
    def test_wrap_angle_wrap_negative(self):
        """Test wrapping when difference < -π"""
        theta1 = np.array([0.0])
        theta2 = np.array([-np.pi - 0.5])
        result = wrap_angle_diff(theta1, theta2)
        assert np.allclose(result, np.pi - 0.5)
    
    def test_wrap_angle_array(self):
        """Test with array inputs"""
        theta1 = np.array([0.0, np.pi / 2, np.pi])
        theta2 = np.array([np.pi / 2, np.pi, -np.pi / 2])
        result = wrap_angle_diff(theta1, theta2)
        # For pi to -pi/2: -pi/2 - pi = -3pi/2, wraps to pi/2
        expected = np.array([np.pi / 2, np.pi / 2, np.pi / 2])
        assert np.allclose(result, expected)

class TestPathLength:
    """Test path length calculation"""
    
    def test_path_length_straight_line(self):
        """Test path length for straight line"""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 0.0, 0.0, 0.0])
        length = calculate_path_length(x, y, 0, 3)
        assert np.allclose(length, 3.0)
    
    def test_path_length_diagonal(self):
        """Test path length for diagonal movement"""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        length = calculate_path_length(x, y, 0, 2)
        expected = 2 * np.sqrt(2)
        assert np.allclose(length, expected)
    
    def test_path_length_empty(self):
        """Test path length for single point"""
        x = np.array([0.0])
        y = np.array([0.0])
        length = calculate_path_length(x, y, 0, 0)
        assert length == 0.0

class TestHeadSwingAssociation:
    """Test head swing to turn association"""
    
    def test_associate_contained(self):
        """Test head swing contained within reorientation"""
        head_swings = [(5, 7)]
        reorientations = [(4, 8)]
        result = associate_head_swings_with_turns(head_swings, reorientations)
        assert len(result[0]) == 1
        assert result[0][0] == (5, 7)
    
    def test_associate_overlapping(self):
        """Test head swing overlapping reorientation"""
        head_swings = [(5, 10)]
        reorientations = [(8, 12)]
        result = associate_head_swings_with_turns(head_swings, reorientations)
        assert len(result[0]) == 1
    
    def test_associate_multiple(self):
        """Test multiple head swings in one turn"""
        head_swings = [(5, 7), (8, 10)]
        reorientations = [(4, 12)]
        result = associate_head_swings_with_turns(head_swings, reorientations)
        assert len(result[0]) == 2
    
    def test_associate_multiple_turns(self):
        """Test head swings in different turns"""
        head_swings = [(5, 7), (15, 17)]
        reorientations = [(4, 10), (14, 20)]
        result = associate_head_swings_with_turns(head_swings, reorientations)
        assert len(result[0]) == 1
        assert len(result[1]) == 1

class TestFirstHeadSwing:
    """Test first head swing calculation"""
    
    def test_first_head_swing_single(self):
        """Test single head swing"""
        # Need heading at start and end of head swing
        # Head swing is frames 1-3, so heading[1] and heading[3]
        heading = np.array([0.0, np.pi / 4, np.pi / 2, np.pi / 2, 0.0])
        head_swings = [(1, 3)]
        reoHS1, idx = calculate_first_head_swing(head_swings, heading, 0, 4)
        assert idx == 0
        # Heading goes from pi/4 (frame 1) to pi/2 (frame 3) = positive change (left turn)
        assert reoHS1 > 0  # Left turn
        assert np.allclose(reoHS1, np.pi / 4)  # pi/2 - pi/4 = pi/4
    
    def test_first_head_swing_multiple(self):
        """Test multiple head swings - should return first"""
        heading = np.array([0.0, np.pi / 4, 0.0, -np.pi / 4, 0.0])
        head_swings = [(1, 2), (3, 4)]
        reoHS1, idx = calculate_first_head_swing(head_swings, heading, 0, 4)
        assert idx == 0
        # First swing: pi/4 to 0.0 = negative change (right turn)
        # But we expect wrapped angle: 0.0 - pi/4 = -pi/4, which wraps to...
        # Actually the function calculates: heading_end - heading_start
        # 0.0 - pi/4 = -pi/4 (right turn, negative)
        assert reoHS1 < 0  # First swing is right (negative)
    
    def test_first_head_swing_empty(self):
        """Test no head swings"""
        heading = np.array([0.0, 0.0, 0.0])
        head_swings = []
        reoHS1, idx = calculate_first_head_swing(head_swings, heading, 0, 2)
        assert idx is None
        assert reoHS1 == 0.0

class TestKleinRunTableGeneration:
    """Test Klein run table generation"""
    
    def create_test_trajectory(self, n_frames=100):
        """Create test trajectory data"""
        time = np.linspace(0, 10, n_frames)
        x = np.linspace(0, 10, n_frames)
        y = np.sin(np.linspace(0, 2*np.pi, n_frames))
        heading = np.arctan2(np.diff(y, prepend=y[0]), np.diff(x, prepend=x[0]))
        speed = np.ones(n_frames) * 0.1
        
        return pd.DataFrame({
            'time': time,
            'x': x,
            'y': y,
            'heading': heading,
            'speed': speed
        })
    
    def create_test_segmentation(self, n_frames=100):
        """Create test MAGAT segmentation"""
        # Create 3 runs with 2 reorientations between them
        runs = [(0, 29), (40, 69), (80, 99)]
        head_swings = [(30, 35), (70, 75)]
        reorientations = [(30, 39), (70, 79)]
        
        return {
            'runs': runs,
            'head_swings': head_swings,
            'reorientations': reorientations,
            'n_runs': len(runs),
            'n_head_swings': len(head_swings),
            'n_reorientations': len(reorientations),
            'is_run': np.zeros(n_frames, dtype=bool),
            'is_head_swing': np.zeros(n_frames, dtype=bool),
            'is_reorientation': np.zeros(n_frames, dtype=bool)
        }
    
    def test_basic_generation(self):
        """Test basic run table generation"""
        df = self.create_test_trajectory(100)
        seg = self.create_test_segmentation(100)
        
        run_table = generate_klein_run_table(df, seg, track_id=1, experiment_id=1)
        
        assert len(run_table) == 3  # 3 runs
        assert 'set' in run_table.columns
        assert 'expt' in run_table.columns
        assert 'track' in run_table.columns
        assert 'time0' in run_table.columns
        assert 'reoYN' in run_table.columns
        assert all(run_table['reoYN'].values == [1, 1, 0])  # First two have turns, last doesn't
    
    def test_missing_columns(self):
        """Test error when required columns missing"""
        df = pd.DataFrame({'x': [0, 1, 2], 'y': [0, 1, 2]})  # Missing 'time' and 'heading'
        seg = self.create_test_segmentation(3)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_klein_run_table(df, seg, track_id=1)
    
    def test_missing_segmentation(self):
        """Test error when segmentation data missing"""
        df = self.create_test_trajectory(100)
        seg = {}  # Missing 'runs'
        
        with pytest.raises(ValueError, match="must contain 'runs'"):
            generate_klein_run_table(df, seg, track_id=1)
    
    def test_empty_runs(self):
        """Test error when no runs"""
        df = self.create_test_trajectory(100)
        seg = {'runs': [], 'head_swings': [], 'reorientations': []}
        
        with pytest.raises(ValueError, match="No runs found"):
            generate_klein_run_table(df, seg, track_id=1)
    
    def test_invalid_indices(self):
        """Test error when run indices invalid"""
        df = self.create_test_trajectory(100)
        seg = {'runs': [(0, 150)], 'head_swings': [], 'reorientations': []}  # end > n_frames
        
        with pytest.raises(ValueError, match="Invalid run indices"):
            generate_klein_run_table(df, seg, track_id=1)
    
    def test_run_metrics(self):
        """Test run-level metrics are calculated correctly"""
        df = self.create_test_trajectory(100)
        seg = self.create_test_segmentation(100)
        
        run_table = generate_klein_run_table(df, seg, track_id=1)
        
        # Check all 18 columns present
        expected_cols = ['set', 'expt', 'track', 'time0', 'reoYN', 'runQ', 'runL', 'runT',
                        'runX', 'reo#HS', 'reoQ1', 'reoQ2', 'reoHS1', 'runQ0',
                        'runX0', 'runY0', 'runX1', 'runY1']
        assert all(col in run_table.columns for col in expected_cols)
        
        # Check runQ0, runQ1 are valid angles
        assert np.all(np.isfinite(run_table['runQ0']))
        assert np.all(np.isfinite(run_table['runQ']))
        
        # Check runL, runT are positive
        assert np.all(run_table['runL'] >= 0)
        assert np.all(run_table['runT'] > 0)
    
    def test_turn_metrics(self):
        """Test turn-level metrics"""
        df = self.create_test_trajectory(100)
        seg = self.create_test_segmentation(100)
        
        run_table = generate_klein_run_table(df, seg, track_id=1)
        
        # Rows with reoYN == 1 should have turn metrics
        turns = run_table[run_table['reoYN'] == 1]
        assert len(turns) == 2
        
        # Check reo#HS (head swing count)
        assert np.all(turns['reo#HS'] >= 0)
        
        # Check reoQ1, reoQ2 are valid (may be NaN for rows without turns)
        assert np.all(np.isfinite(turns['reoQ1']))
        assert np.all(np.isfinite(turns['reoQ2']))

class TestDerivedMetrics:
    """Test derived metrics calculation"""
    
    def test_turn_magnitude_calculation(self):
        """Test turn magnitude is calculated correctly"""
        # Create run table with known angles
        run_table = pd.DataFrame({
            'reoYN': [1, 0, 1],
            'reoQ1': [0.0, np.nan, np.pi / 2],
            'reoQ2': [np.pi / 2, np.nan, np.pi],
            'runQ0': [0.0, 0.0, np.pi / 2],
            'runL': [10.0, 5.0, 8.0],
            'runT': [2.0, 1.0, 1.5],
            'runX0': [0.0, 0.0, 0.0],
            'runY0': [0.0, 0.0, 0.0],
            'runX1': [10.0, 5.0, 8.0],
            'runY1': [0.0, 0.0, 0.0]
        })
        
        result = calculate_klein_derived_metrics(run_table)
        
        # Check turn magnitude
        assert 'turn_magnitude' in result.columns
        assert np.isfinite(result.loc[0, 'turn_magnitude'])
        assert np.isnan(result.loc[1, 'turn_magnitude'])  # No turn
        assert np.isfinite(result.loc[2, 'turn_magnitude'])
        
        # Check turn direction
        assert 'turn_direction' in result.columns
        assert result.loc[0, 'turn_direction'] in ['LEFT', 'RIGHT']
    
    def test_run_efficiency(self):
        """Test run efficiency calculation"""
        run_table = pd.DataFrame({
            'reoYN': [0],
            'reoQ1': [np.nan],
            'reoQ2': [np.nan],
            'runQ0': [0.0],
            'runL': [10.0],  # Path length
            'runT': [2.0],
            'runX0': [0.0],
            'runY0': [0.0],
            'runX1': [10.0],  # Straight line = efficiency = 1.0
            'runY1': [0.0]
        })
        
        result = calculate_klein_derived_metrics(run_table)
        
        assert 'run_efficiency' in result.columns
        assert np.allclose(result['run_efficiency'].values, [1.0])
    
    def test_run_drift(self):
        """Test run drift calculation"""
        run_table = pd.DataFrame({
            'reoYN': [1],
            'reoQ1': [np.pi / 4],
            'reoQ2': [np.pi / 2],
            'runQ0': [0.0],
            'runL': [10.0],
            'runT': [2.0],
            'runX0': [0.0],
            'runY0': [0.0],
            'runX1': [10.0],
            'runY1': [0.0]
        })
        
        result = calculate_klein_derived_metrics(run_table)
        
        assert 'run_drift' in result.columns
        assert np.isfinite(result['run_drift'].values[0])
        assert result['run_drift_direction'].values[0] in ['LEFT', 'RIGHT']

class TestTurnRate:
    """Test turn rate calculation"""
    
    def test_turn_rate_calculation(self):
        """Test Klein turn rate formula"""
        run_table = pd.DataFrame({
            'reoYN': [1, 0, 1, 0],
            'runT': [2.0, 1.0, 1.5, 0.5]
        })
        
        turn_rate = calculate_turn_rate(run_table)
        
        # Total turns = 2, total time = 5.0s
        # Rate = 2/5 * 60 = 24 turns/min
        expected = 2.0 / 5.0 * 60.0
        assert np.allclose(turn_rate, expected)
    
    def test_turn_rate_zero_time(self):
        """Test error when total time is zero"""
        run_table = pd.DataFrame({
            'reoYN': [0],
            'runT': [0.0]
        })
        
        with pytest.raises(ValueError, match="Total run time must be > 0"):
            calculate_turn_rate(run_table)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
