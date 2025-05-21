"""Chain code file I/O functions."""

#
#
#

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd



@dataclass
class ChainCodeData:
    """Chain code data class.
    
    Chain codes represent 2D contours using directional codes from 0 to 7:
    
        3   2   1
         \  |  /
        4 - * - 0
         /  |  \
        5   6   7
    
    Parameters
    ----------
    sample_name : str
        Sample name.
    number : int
        Sample number.
    x : float
        X coordinate.
    y : float
        Y coordinate.
    area_per_pixel : float
        Area (mm2) per pixel.
    area_pixels : int
        Area in pixels.
    chain_code : np.ndarray
        Chain code sequence with values from 0 to 7 representing directions.
    """
    sample_name: str
    number: int
    x: float
    y: float
    area_per_pixel: float
    area_pixels: int
    chain_code: np.ndarray
    
    def __post_init__(self):
        if not isinstance(self.chain_code, np.ndarray):
            self.chain_code = np.array(self.chain_code)
            
        if not np.all((self.chain_code >= 0) & (self.chain_code <= 7)):
            invalid_values = self.chain_code[(self.chain_code < 0) | (self.chain_code > 7)]
            raise ValueError(f"Chain code contains invalid values: {invalid_values}. "
                             f"Values must be between 0 and 7 (inclusive).")
    
    def to_numpy(self):
        """Convert chain code to numpy array.
        
        Returns
        -------
        chain_code : np.ndarray
            Chain code.
        """
        return self.chain_code
    
    def to_dataframe(self):
        """Convert chain code to pandas DataFrame.
        
        Returns
        -------
        df : pd.DataFrame
            Chain code.
        """
        df = pd.DataFrame(
            {
                "chain_code": self.chain_code,
            },
            index=pd.MultiIndex.from_tuples(
                [[f"{self.sample_name}_{self.number}", i] for i in range(len(self.chain_code))],
                name=["specimen_id", "coord_id"],
            ),
        )
        return df




def read_chc(file_path, as_frame=False, validate=True):
    """Read chain code (.chc) file.
    
    Chain codes represent 2D contours using directional codes from 0 to 7:
    
        3   2   1
         \  |  /
        4 - * - 0
         /  |  \
        5   6   7
    
    Parameters
    ----------
    file_path : str
        Path to the chain code file.
    as_frame : bool, default=False
        If True, return pandas.DataFrame. Otherwise, return numpy.ndarray.
    validate : bool, default=True
        If True, validate that chain code values are between 0 and 7.
        Set to False to skip validation for legacy files that may contain other values.
        
    Returns
    -------
    chain_codes : list of ChainCodeData or pd.DataFrame
        Chain codes.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
        
    if not path.suffix == ".chc":
        raise ValueError(f"{path} is not a chain code file.")
        
    chc_data_list = _read_chc(path, validate=validate)
    
    if len(chc_data_list) == 1:
        if as_frame:
            return chc_data_list[0].to_dataframe()
        else:
            return chc_data_list[0].to_numpy()
    else:
        if as_frame:
            dfs = [chc_data.to_dataframe() for chc_data in chc_data_list]
            return pd.concat(dfs)
        else:
            return [chc_data.to_numpy() for chc_data in chc_data_list]


def write_chc(file_path, chain_codes, sample_names=None, numbers=None, xs=None, ys=None, 
              area_per_pixels=None, area_pixels_values=None, validate=True):
    """Write chain code to .chc file.
    
    Chain codes represent 2D contours using directional codes from 0 to 7:
    
        3   2   1
         \  |  /
        4 - * - 0
         /  |  \
        5   6   7
    
    Parameters
    ----------
    file_path : str
        Path to the chain code file.
    chain_codes : list of np.ndarray or np.ndarray
        Chain codes with values from 0 to 7 representing directions.
    sample_names : list of str or str, optional
        Sample names.
    numbers : list of int or int, optional
        Sample numbers.
    xs : list of float or float, optional
        X coordinates.
    ys : list of float or float, optional
        Y coordinates.
    area_per_pixels : list of float or float, optional
        Area (mm2) per pixel.
    area_pixels_values : list of int or int, optional
        Area in pixels.
    validate : bool, default=True
        If True, validate that chain code values are between 0 and 7.
        Set to False to skip validation for legacy files that may contain other values.
    """
    path = Path(file_path)
    
    if isinstance(chain_codes, np.ndarray):
        if chain_codes.ndim == 1:
            chain_codes = [chain_codes]
        elif chain_codes.ndim == 2:
            chain_codes = [chain_codes[i, :] for i in range(chain_codes.shape[0])]
        else:
            raise ValueError("chain_codes must be a 1D or 2D array.")
    elif not isinstance(chain_codes, list):
        raise ValueError("chain_codes must be a list of numpy arrays or a numpy array.")
        
    n_samples = len(chain_codes)
    
    if sample_names is None:
        sample_names = ["Sample"] * n_samples
    elif isinstance(sample_names, str):
        sample_names = [sample_names] * n_samples
        
    if numbers is None:
        numbers = list(range(1, n_samples + 1))
    elif isinstance(numbers, int):
        numbers = [numbers] * n_samples
        
    if xs is None:
        xs = [0] * n_samples
    elif isinstance(xs, (int, float)):
        xs = [xs] * n_samples
        
    if ys is None:
        ys = [0] * n_samples
    elif isinstance(ys, (int, float)):
        ys = [ys] * n_samples
        
    if area_per_pixels is None:
        area_per_pixels = [1.0] * n_samples
    elif isinstance(area_per_pixels, (int, float)):
        area_per_pixels = [area_per_pixels] * n_samples
        
    if area_pixels_values is None:
        area_pixels_values = [len(code) for code in chain_codes]
    elif isinstance(area_pixels_values, int):
        area_pixels_values = [area_pixels_values] * n_samples
        
    chc_data_list = []
    for i in range(n_samples):
        if validate and not np.all((chain_codes[i] >= 0) & (chain_codes[i] <= 7)):
            invalid_values = chain_codes[i][(chain_codes[i] < 0) | (chain_codes[i] > 7)]
            raise ValueError(f"Chain code contains invalid values: {invalid_values}. "
                             f"Values must be between 0 and 7 (inclusive).")
        
        chc_data_list.append(
            ChainCodeData(
                sample_name=sample_names[i],
                number=numbers[i],
                x=xs[i],
                y=ys[i],
                area_per_pixel=area_per_pixels[i],
                area_pixels=area_pixels_values[i],
                chain_code=chain_codes[i]
            )
        )
    
    _write_chc(path, chc_data_list)




def _read_chc(file_path, validate=True):
    """Read chain code file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the chain code file.
    validate : bool, default=True
        If True, validate that chain code values are between 0 and 7.
        
    Returns
    -------
    chc_data_list : list of ChainCodeData
        Chain code data.
    """
    chc_data_list = []
    
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(" ")
            
            sample_id = parts[0]
            sample_name, number = sample_id.split("_")
            number = int(number)
            
            x = float(parts[1])
            y = float(parts[2])
            area_per_pixel = float(parts[3])
            area_pixels = int(parts[4])
            
            try:
                end_idx = parts.index("-1")
                chain_code = np.array(parts[5:end_idx], dtype=int)
            except ValueError:
                chain_code = np.array(parts[5:], dtype=int)
            
            try:
                chc_data = ChainCodeData(
                    sample_name=sample_name,
                    number=number,
                    x=x,
                    y=y,
                    area_per_pixel=area_per_pixel,
                    area_pixels=area_pixels,
                    chain_code=chain_code
                )
            except ValueError as e:
                if validate:
                    raise e
                original_post_init = ChainCodeData.__post_init__
                try:
                    ChainCodeData.__post_init__ = lambda self: None if not isinstance(self.chain_code, np.ndarray) else setattr(self, 'chain_code', np.array(self.chain_code))
                    chc_data = ChainCodeData(
                        sample_name=sample_name,
                        number=number,
                        x=x,
                        y=y,
                        area_per_pixel=area_per_pixel,
                        area_pixels=area_pixels,
                        chain_code=chain_code
                    )
                finally:
                    ChainCodeData.__post_init__ = original_post_init
            
            chc_data_list.append(chc_data)
    
    return chc_data_list


def _write_chc(file_path, chc_data_list):
    """Write chain code data to a file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the chain code file.
    chc_data_list : list of ChainCodeData
        Chain code data.
    """
    with open(file_path, "w") as f:
        for chc_data in chc_data_list:
            f.write(f"{chc_data.sample_name}_{chc_data.number} {chc_data.x} {chc_data.y} "
                    f"{chc_data.area_per_pixel} {chc_data.area_pixels} ")
            
            f.write(" ".join(map(str, chc_data.chain_code.tolist())))
            
            f.write(" -1\n")
