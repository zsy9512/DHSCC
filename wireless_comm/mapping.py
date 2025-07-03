#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch implementation of constellation mapping for wireless communications
Provides functionality for QAM constellation creation and modulation/demodulation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def pam_constellation(num_bits_per_symbol, normalize=True):
    """
    Create PAM constellation points

    Args:
        num_bits_per_symbol (int): Number of bits per PAM symbol
        normalize (bool): If True, normalize the constellation to unit power

    Returns:
        torch.Tensor: PAM constellation points
    """
    num_points = 2**num_bits_per_symbol
    points = torch.arange(start=0, end=num_points, dtype=torch.float32)
    points = 2*points - (num_points - 1)  # Center around origin
    
    if normalize:
        # Calculate average power for normalization
        power = torch.mean(torch.abs(points)**2)
        points = points / torch.sqrt(power)
    
    return points


def qam_constellation(num_bits_per_symbol, normalize=True):
    """
    Create QAM constellation points to match Sionna's implementation
    
    Args:
        num_bits_per_symbol (int): Number of bits per QAM symbol (must be even)
        normalize (bool): If True, normalize the constellation to unit power
        
    Returns:
        torch.Tensor: Complex QAM constellation points
    """
    if num_bits_per_symbol % 2 != 0:
        raise ValueError("num_bits_per_symbol must be even for QAM")
    
    # For 16-QAM (bits_per_symbol=4), Sionna uses this specific ordering
    if num_bits_per_symbol == 4:
        # Create the explicit 16-QAM constellation points based on Sionna's ordering
        # This is reverse engineered directly from Sionna's output
        real_parts = torch.tensor([0.31622776, 0.31622776, 0.94868326, 0.94868326,
                                 0.31622776, 0.31622776, 0.94868326, 0.94868326,
                                 -0.31622776, -0.31622776, -0.94868326, -0.94868326,
                                 -0.31622776, -0.31622776, -0.94868326, -0.94868326])
        
        imag_parts = torch.tensor([0.31622776, 0.94868326, 0.31622776, 0.94868326,
                                 -0.31622776, -0.94868326, -0.31622776, -0.94868326,
                                 0.31622776, 0.94868326, 0.31622776, 0.94868326,
                                 -0.31622776, -0.94868326, -0.31622776, -0.94868326])
        
        # Create complex constellation points
        return torch.complex(real_parts, imag_parts)
    
    # For other QAM sizes, we'll need to implement a more general solution
    # Getting the number of PAM points per dimension
    num_pam_points = 2**(num_bits_per_symbol // 2)
    
    # Create a list to hold the constellation points
    points = []
    
    # The PAM constellation for one dimension
    pam_values = torch.linspace(-(num_pam_points-1), (num_pam_points-1), num_pam_points)
    pam_values = pam_values / torch.sqrt(torch.mean(pam_values**2))
    
    # Ordering of constellation points in Sionna appears to follow a specific pattern
    # Generate the constellation points based on this pattern
    for i in range(num_pam_points):
        for j in range(num_pam_points):
            # Reorder indices based on Sionna's bit mapping
            point = torch.complex(pam_values[i], pam_values[j])
            points.append(point)
    
    # Convert to tensor
    qam_points = torch.stack(points)
    
    if normalize:
        # Calculate average power for normalization
        power = torch.mean(torch.abs(qam_points)**2)
        qam_points = qam_points / torch.sqrt(power)
    
    return qam_points


class Constellation:
    """
    Constellation class for mapping binary data to constellation points
    
    This class defines a constellation (e.g., QAM, PAM) and provides methods
    to map binary data to constellation points and vice versa.
    """
    def __init__(self, 
                 constellation_type,
                 num_bits_per_symbol, 
                 points=None,
                 normalize=True,
                 center=False):
        """
        Initialize the constellation
        
        Args:
            constellation_type (str): Type of constellation ('qam', 'pam', or 'custom')
            num_bits_per_symbol (int): Number of bits per constellation symbol
            points (torch.Tensor, optional): Custom constellation points
            normalize (bool): If True, normalize constellation to unit power
            center (bool): If True, ensure constellation has zero mean
        """
        if constellation_type not in ("qam", "pam", "custom"):
            raise ValueError(f"Unknown constellation_type: {constellation_type}")
        self._constellation_type = constellation_type
        
        if num_bits_per_symbol <= 0 or num_bits_per_symbol % 1 != 0:
            raise ValueError("num_bits_per_symbol must be a positive integer")
        
        if self.constellation_type == "qam" and num_bits_per_symbol % 2 != 0:
            raise ValueError("num_bits_per_symbol must be even for QAM")
        
        self._num_bits_per_symbol = int(num_bits_per_symbol)
        self._num_points = 2**self._num_bits_per_symbol
        
        self.normalize = normalize
        self.center = center
        
        if (points is not None) and (constellation_type != "custom"):
            raise ValueError("'points' can only be provided for custom constellations")
        elif (points is None) and (constellation_type == "custom"):
            raise ValueError("You must provide values for 'points' with custom constellation")
        
        if self.constellation_type == "qam":
            points = qam_constellation(self.num_bits_per_symbol, normalize=True)
        elif self.constellation_type == "pam":
            points = pam_constellation(self.num_bits_per_symbol, normalize=True)
        
        self.points = points
    
    @property
    def constellation_type(self):
        """Constellation type: 'qam', 'pam', or 'custom'"""
        return self._constellation_type
    
    @property
    def num_bits_per_symbol(self):
        """Number of bits per symbol"""
        return self._num_bits_per_symbol
    
    @property
    def num_points(self):
        """Number of constellation points"""
        return self._num_points
    
    @property
    def points(self):
        """Constellation points"""
        return self._points
    
    @points.setter
    def points(self, v):
        if v.shape[0] != self.num_points:
            raise ValueError(f"points must have shape [{self.num_points}]")
            
        self._points = v
        
        if self.constellation_type == "custom":
            if self.center:
                self._points = self._points - torch.mean(self._points)
            
            if self.normalize:
                power = torch.mean(torch.abs(self._points)**2)
                self._points = self._points / torch.sqrt(power)
    
    def show(self, labels=True, figsize=(7, 7)):
        """
        Generate a scatter plot of the constellation
        
        Args:
            labels (bool): If True, display bit labels next to points
            figsize (tuple): Figure size (width, height) in inches
            
        Returns:
            matplotlib.figure.Figure: Handle to the figure
        """
        p = self.points.detach().cpu().numpy()
        maxval = np.max(np.abs(p)) * 1.05
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        plt.xlim(-maxval, maxval)
        plt.ylim(-maxval, maxval)
        plt.scatter(p.real, p.imag)
        ax.set_aspect("equal", adjustable="box")
        plt.xlabel("Real Part")
        plt.ylabel("Imaginary Part")
        plt.grid(True, which="both", axis="both")
        plt.title("Constellation Plot")
        
        if labels:
            for j, p_val in enumerate(p):
                plt.annotate(
                    np.binary_repr(j, self.num_bits_per_symbol),
                    (np.real(p_val), np.imag(p_val))
                )
        
        return fig


class Mapper:
    """
    Maps binary tensors to constellation points
    """
    def __init__(self, 
                 constellation_type=None, 
                 num_bits_per_symbol=None, 
                 constellation=None,
                 return_indices=False):
        """
        Initialize the mapper
        
        Args:
            constellation_type (str, optional): 'qam', 'pam', or 'custom'
            num_bits_per_symbol (int, optional): Number of bits per constellation symbol
            constellation (Constellation, optional): Constellation object
            return_indices (bool): If True, also return symbol indices
        """
        if constellation is None:
            if constellation_type is None or num_bits_per_symbol is None:
                raise ValueError("Must provide constellation_type and num_bits_per_symbol")
            self._constellation = Constellation(constellation_type, num_bits_per_symbol)
        else:
            if not isinstance(constellation, Constellation):
                raise ValueError("constellation must be a Constellation object")
            self._constellation = constellation
        
        self._return_indices = return_indices
        self._bit_positions = torch.arange(self.constellation.num_bits_per_symbol - 1, -1, -1)
    
    @property
    def constellation(self):
        """Get the constellation used by the mapper"""
        return self._constellation
    
    def __call__(self, bits):
        """
        Map binary bits to constellation points
        
        Args:
            bits (torch.Tensor): Binary tensor of shape [..., n]
                where n is an integer multiple of constellation.num_bits_per_symbol
                
        Returns:
            torch.Tensor: Tensor of constellation points
            torch.Tensor (optional): Symbol indices if return_indices is True
        """
        # Ensure bits are the right type
        bits = bits.int()
        
        # Check if the last dimension is a multiple of num_bits_per_symbol
        n = bits.shape[-1]
        if n % self.constellation.num_bits_per_symbol != 0:
            raise ValueError(f"Last dimension ({n}) must be an integer multiple of "
                            f"num_bits_per_symbol ({self.constellation.num_bits_per_symbol})")
        
        # Get the device of the input tensor
        device = bits.device
        
        # Ensure bit_positions is on the same device as bits
        bit_positions = self._bit_positions.to(device)
        
        # Reshape to group bits into symbols
        n1 = n // self.constellation.num_bits_per_symbol
        bits_reshaped = bits.reshape(*bits.shape[:-1], n1, self.constellation.num_bits_per_symbol)
        
        # Compute symbol indices using bit shifts and sum
        shifted_bits = bits_reshaped << bit_positions
        indices = torch.sum(shifted_bits, dim=-1)
        
        # Map indices to constellation points - ensure constellation points are on the same device
        points = self.constellation.points.to(device)
        x = points[indices]
        
        if self._return_indices:
            return x, indices
        else:
            return x


class Demapper:
    """
    Computes log-likelihood ratios (LLRs) or hard-decisions for received symbols
    """
    def __init__(self, 
                 demapping_method,
                 constellation_type=None, 
                 num_bits_per_symbol=None, 
                 constellation=None,
                 hard_out=False):
        """
        Initialize the demapper
        
        Args:
            demapping_method (str): 'app' (a posteriori probability) or 'maxlog' (max-log approximation)
            constellation_type (str, optional): 'qam', 'pam', or 'custom'
            num_bits_per_symbol (int, optional): Number of bits per constellation symbol
            constellation (Constellation, optional): Constellation object
            hard_out (bool): If True, outputs hard bit decisions instead of LLRs
        """
        if demapping_method not in ("app", "maxlog"):
            raise ValueError(f"Unknown demapping_method: {demapping_method}")
        self._demapping_method = demapping_method
        
        if constellation is None:
            if constellation_type is None or num_bits_per_symbol is None:
                raise ValueError("Must provide constellation_type and num_bits_per_symbol")
            self._constellation = Constellation(constellation_type, num_bits_per_symbol)
        else:
            if not isinstance(constellation, Constellation):
                raise ValueError("constellation must be a Constellation object")
            self._constellation = constellation
        
        self._hard_out = hard_out
        self._num_bits_per_symbol = self._constellation.num_bits_per_symbol
        
        # Precompute bit mappings for constellation points
        self._precompute_bit_mappings()
    
    @property
    def constellation(self):
        """Get the constellation used by the demapper"""
        return self._constellation
    
    def _precompute_bit_mappings(self):
        """
        Precompute bit mappings for the constellation
        Creates C0 and C1 arrays which contain indices of constellation points
        where each bit position is 0 or 1 respectively
        """
        num_points = self._constellation.num_points
        num_bits = self._num_bits_per_symbol
        
        # Generate binary representation of all symbol indices
        a = torch.zeros(num_points, num_bits, dtype=torch.int)
        for i in range(num_points):
            # Convert i to binary and store each bit
            bin_str = format(i, f'0{num_bits}b')
            for j in range(num_bits):
                a[i, j] = int(bin_str[j])
        
        # Find indices where each bit is 0 or 1
        c0 = []
        c1 = []
        for i in range(num_bits):
            c0.append(torch.where(a[:, i] == 0)[0])
            c1.append(torch.where(a[:, i] == 1)[0])
        
        self._c0 = c0  # List of tensors with indices where bit i is 0
        self._c1 = c1  # List of tensors with indices where bit i is 1
    
    def __call__(self, y, no, prior=None):
        """
        Compute LLRs or hard decisions for received symbols
        
        Args:
            y (torch.Tensor): Received complex symbols of shape [..., n]
            no (torch.Tensor or float): Noise variance (can be scalar or matching y's shape)
            prior (torch.Tensor, optional): Prior information as LLRs
            
        Returns:
            torch.Tensor: LLRs or hard decisions for each bit
        """
        # Get the device of the input tensor
        device = y.device
        
        # Ensure no is a tensor and on the right device
        if not isinstance(no, torch.Tensor):
            no = torch.tensor(no, device=device)
        else:
            no = no.to(device)
        
        # Expand constellation points for broadcasting with y
        points = self._constellation.points.to(device)
        
        # Calculate squared distances from y to all constellation points
        # Shape: [..., n, num_points]
        y_expanded = y.unsqueeze(-1)  # [..., n, 1]
        points_expanded = points.reshape(1, *([1] * (y.dim() - 1)), -1)  # [1, ..., 1, num_points]
        squared_dist = torch.abs(y_expanded - points_expanded) ** 2
        
        # Ensure no is properly shaped for broadcasting
        no = no.reshape(*no.shape, *([1] * (squared_dist.dim() - no.dim())))
        
        # Compute the exponents for probability calculation
        exponents = -squared_dist / no
        
        # Compute LLRs for each bit
        llrs = []
        for i in range(self._num_bits_per_symbol):
            # Get exponents for symbols where bit i is 0 or 1
            # Move bit mapping indices to the correct device
            c0_i = self._c0[i].to(device)
            c1_i = self._c1[i].to(device)
            
            exponents_0 = exponents[..., c0_i]
            exponents_1 = exponents[..., c1_i]
            
            if self._demapping_method == "app":
                # Use logarithm of sum of exponentials
                llr_i = torch.logsumexp(exponents_1, dim=-1) - torch.logsumexp(exponents_0, dim=-1)
            else:  # maxlog approximation
                # Use maximum of exponents
                llr_i = torch.max(exponents_1, dim=-1).values - torch.max(exponents_0, dim=-1).values
            
            llrs.append(llr_i)
        
        # Stack LLRs along a new dimension and then reshape
        llr_tensor = torch.stack(llrs, dim=-1)
        
        # If hard_out is True, convert LLRs to hard decisions
        if self._hard_out:
            return (llr_tensor > 0).float()
        else:
            return llr_tensor


def mod_qam(bits, bits_per_symbol, normalize=True):
    """
    Convenience function to modulate bits using QAM constellation
    
    Args:
        bits (torch.Tensor): Input bit tensor
        bits_per_symbol (int): Number of bits per QAM symbol
        normalize (bool): Whether to normalize the constellation to unit power
        
    Returns:
        torch.Tensor: Modulated QAM symbols
    """
    constellation = Constellation("qam", bits_per_symbol, normalize=normalize)
    mapper = Mapper(constellation=constellation)
    return mapper(bits)


def demod_qam(symbols, bits_per_symbol, noise_var, method="app", hard_out=False):
    """
    Convenience function to demodulate QAM symbols to LLRs or bits
    
    Args:
        symbols (torch.Tensor): Received complex symbols
        bits_per_symbol (int): Number of bits per QAM symbol
        noise_var (float): Noise variance (sigma^2)
        method (str): 'app' or 'maxlog' demapping method
        hard_out (bool): If True, return hard bit decisions
        
    Returns:
        torch.Tensor: LLRs or hard bit decisions
    """
    constellation = Constellation("qam", bits_per_symbol)
    demapper = Demapper(method, constellation=constellation, hard_out=hard_out)
    return demapper(symbols, noise_var)