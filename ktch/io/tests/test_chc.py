"""Tests for chain code I/O functions."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from ktch.io import read_chc, write_chc


def test_read_chc():
    """Test read_chc function."""
    sample_data = "Sample1_1 100 200 1.5 1000 0 1 2 3 4 5 6 7 0 -1"
    
    with tempfile.NamedTemporaryFile(suffix=".chc", delete=False) as f:
        f.write(sample_data.encode())
        temp_file = f.name
    
    try:
        chain_code = read_chc(temp_file, validate=True)
        assert isinstance(chain_code, np.ndarray)
        assert np.array_equal(chain_code, np.array([0, 1, 2, 3, 4, 5, 6, 7, 0]))
        
        df = read_chc(temp_file, as_frame=True)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 9  # 9 chain code points
        assert len(df.index.levels[0]) == 1  # 1 sample
    finally:
        os.unlink(temp_file)


def test_write_chc():
    """Test write_chc function."""
    chain_code = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0])
    
    with tempfile.NamedTemporaryFile(suffix=".chc", delete=False) as f:
        temp_file = f.name
    
    try:
        write_chc(
            temp_file, 
            chain_code, 
            sample_names="Test", 
            numbers=1, 
            xs=100, 
            ys=200, 
            area_per_pixels=1.5, 
            area_pixels_values=1000,
            validate=True
        )
        
        chain_code_read = read_chc(temp_file)
        
        assert np.array_equal(chain_code, chain_code_read)
    finally:
        os.unlink(temp_file)


def test_multiple_chain_codes():
    """Test reading and writing multiple chain codes."""
    chain_code1 = np.array([0, 1, 2, 3, 4])
    chain_code2 = np.array([5, 6, 7, 0, 1])
    
    with tempfile.NamedTemporaryFile(suffix=".chc", delete=False) as f:
        temp_file = f.name
    
    try:
        write_chc(
            temp_file, 
            [chain_code1, chain_code2], 
            sample_names=["Test1", "Test2"], 
            numbers=[1, 2], 
            xs=[100, 200], 
            ys=[150, 250], 
            area_per_pixels=[1.0, 2.0], 
            area_pixels_values=[500, 1000]
        )
        
        chain_codes_read = read_chc(temp_file)
        
        assert isinstance(chain_codes_read, list)
        assert len(chain_codes_read) == 2
        
        assert np.array_equal(chain_code1, chain_codes_read[0])
        assert np.array_equal(chain_code2, chain_codes_read[1])
    finally:
        os.unlink(temp_file)


def test_invalid_chain_code():
    """Test validation of invalid chain code values."""
    invalid_chain_code = np.array([0, 1, 2, 8, 9])
    
    with tempfile.NamedTemporaryFile(suffix=".chc", delete=False) as f:
        temp_file = f.name
    
    try:
        with pytest.raises(ValueError):
            write_chc(
                temp_file, 
                invalid_chain_code, 
                sample_names="Test", 
                numbers=1
            )
            
        write_chc(
            temp_file, 
            invalid_chain_code, 
            sample_names="Test", 
            numbers=1,
            validate=False
        )
        
        with pytest.raises(ValueError):
            read_chc(temp_file)
            
        chain_code = read_chc(temp_file, validate=False)
        assert np.array_equal(chain_code, invalid_chain_code)
    finally:
        os.unlink(temp_file)
