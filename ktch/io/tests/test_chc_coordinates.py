"""Tests for chain code coordinate conversion functions."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from ktch.io import read_chc, write_chc


def test_chain_code_to_coordinates():
    """Test conversion of chain code to coordinates."""
    chain_code = np.array([0, 2, 4, 6])
    
    with tempfile.NamedTemporaryFile(suffix=".chc", delete=False) as f:
        temp_file = f.name
    
    try:
        write_chc(
            temp_file, 
            chain_code, 
            sample_names="Square", 
            xs=0, 
            ys=0, 
            area_per_pixels=1.0, 
            area_pixels_values=4
        )
        
        coords = read_chc(temp_file)
        
        expected_coords = np.array([
            [0, 0],  # Starting point
            [1, 0],  # After going right
            [1, -1], # After going up
            [0, -1], # After going left
            [0, 0]   # After going down (back to start)
        ])
        
        assert coords.shape == (5, 2)  # 5 points (including start and end)
        assert np.allclose(coords, expected_coords)
        
        raw_chain_code = read_chc(temp_file, as_coordinates=False)
        assert np.array_equal(raw_chain_code, chain_code)
    finally:
        os.unlink(temp_file)


def test_scaled_coordinates():
    """Test scaling of coordinates using area_per_pixel."""
    chain_code = np.array([0, 2, 4, 6])  # Simple square
    
    with tempfile.NamedTemporaryFile(suffix=".chc", delete=False) as f:
        temp_file = f.name
    
    try:
        write_chc(
            temp_file, 
            chain_code, 
            sample_names="ScaledSquare", 
            xs=10, 
            ys=20, 
            area_per_pixels=4.0, 
            area_pixels_values=4
        )
        
        coords = read_chc(temp_file)
        
        expected_coords = np.array([
            [10, 20],   # Starting point
            [12, 20],   # After going right (scaled by 2)
            [12, 18],   # After going up (scaled by 2)
            [10, 18],   # After going left (scaled by 2)
            [10, 20]    # After going down (scaled by 2)
        ])
        
        assert coords.shape == (5, 2)
        assert np.allclose(coords, expected_coords)
    finally:
        os.unlink(temp_file)


def test_dataframe_output():
    """Test DataFrame output format."""
    chain_code = np.array([0, 2, 4, 6])  # Simple square
    
    with tempfile.NamedTemporaryFile(suffix=".chc", delete=False) as f:
        temp_file = f.name
    
    try:
        write_chc(
            temp_file, 
            chain_code, 
            sample_names="Square", 
            xs=0, 
            ys=0, 
            area_per_pixels=1.0, 
            area_pixels_values=4
        )
        
        df = read_chc(temp_file, as_frame=True)
        
        assert isinstance(df, pd.DataFrame)
        assert "x" in df.columns
        assert "y" in df.columns
        assert "chain_code" in df.columns
        
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ["specimen_id", "coord_id"]
        
        assert df.shape[0] == 5  # 5 points (including start and end)
        assert df.loc[("Square", 0), "chain_code"] == -1  # First point has no direction
        assert df.loc[("Square", 1), "chain_code"] == 0   # Right
        assert df.loc[("Square", 2), "chain_code"] == 2   # Up
        assert df.loc[("Square", 3), "chain_code"] == 4   # Left
        assert df.loc[("Square", 4), "chain_code"] == 6   # Down
    finally:
        os.unlink(temp_file)


def test_multiple_chain_codes_as_coordinates():
    """Test reading multiple chain codes as coordinates."""
    chain_code1 = np.array([0, 2, 4, 6])  # Square
    chain_code2 = np.array([0, 0, 2, 2, 4, 4, 6, 6])  # Rectangle
    
    with tempfile.NamedTemporaryFile(suffix=".chc", delete=False) as f:
        temp_file = f.name
    
    try:
        write_chc(
            temp_file, 
            [chain_code1, chain_code2], 
            sample_names=["Square", "Rectangle"], 
            xs=[0, 10], 
            ys=[0, 10], 
            area_per_pixels=[1.0, 1.0], 
            area_pixels_values=[4, 8]
        )
        
        coords_list = read_chc(temp_file)
        
        assert isinstance(coords_list, list)
        assert len(coords_list) == 2
        
        square_coords = coords_list[0]
        assert square_coords.shape == (5, 2)
        
        rectangle_coords = coords_list[1]
        assert rectangle_coords.shape == (9, 2)
        
        raw_codes = read_chc(temp_file, as_coordinates=False)
        assert isinstance(raw_codes, list)
        assert len(raw_codes) == 2
        assert np.array_equal(raw_codes[0], chain_code1)
        assert np.array_equal(raw_codes[1], chain_code2)
    finally:
        os.unlink(temp_file)
