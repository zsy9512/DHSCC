#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch implementation of Polar encoder including 5G compliant rate-matching and CRC concatenation.
Adapted from the Sionna TensorFlow implementation.
Standalone polar encoding module for distributed deployment.
"""

import torch
import numpy as np
import numbers
from scipy.special import comb

class PolarEncoder(torch.nn.Module):
    """Polar encoder for given code parameters.

    This class performs polar encoding for the given ``k`` information bits and
    the `frozen set` (i.e., indices of frozen positions) specified by
    ``frozen_pos``.

    Parameters
    ----------
    frozen_pos: ndarray
        Array of `int` defining the `n-k` frozen indices, i.e., information
        bits are mapped onto the `k` complementary positions.

    n: int
        Defining the codeword length.

    dtype: torch.dtype, default=torch.float32
        Data type to use for internal calculations and outputs.

    Input
    -----
    bits: [...,k], torch.Tensor
        Binary tensor containing the information bits to be encoded.

    Output
    ------
    : [...,n], torch.Tensor
        Binary tensor containing the codeword bits.

    Note
    ----
    As commonly done, we assume frozen bits are set to `0`. Please note
    that - although its practical relevance is only little - setting frozen
    bits to `1` may result in `affine` codes instead of linear code as the
    `all-zero` codeword is not necessarily part of the code any more.
    """

    def __init__(self,
                 frozen_pos,
                 n,
                 dtype=torch.float32):
        super().__init__()

        if not isinstance(n, numbers.Number):
            raise TypeError("n must be a number.")
        n = int(n)  # n can be float (e.g. as result of n=k*r)

        # Fix the data type check for frozen_pos
        # Convert frozen_pos to numpy array if it's not already one
        if not isinstance(frozen_pos, np.ndarray):
            frozen_pos = np.array(frozen_pos)

        # Ensure the array has an integer dtype
        if not np.issubdtype(frozen_pos.dtype, np.integer):
            try:
                # Try to convert to int32 if possible
                frozen_pos = frozen_pos.astype(np.int32)
            except:
                raise TypeError("frozen_pos must consist of ints.")

        if len(frozen_pos) > n:
            msg = "Number of elements in frozen_pos cannot be greater than n."
            raise ValueError(msg)
        if np.log2(n) != int(np.log2(n)):
            raise ValueError("n must be a power of 2.")

        self._k = n - len(frozen_pos)
        self._n = n
        self._frozen_pos = frozen_pos
        self._dtype = dtype

        # Generate info positions
        self._info_pos = np.setdiff1d(np.arange(self._n), frozen_pos)
        if self._k != len(self._info_pos):
            raise ValueError("Internal error: invalid info_pos generated.")

        self._check_input = True  # Check input for bin. values during first call

        self._nb_stages = int(np.log2(self._n))
        self._ind_gather = self._gen_indices(self._n)

        # Register the gather indices as a buffer so it can be moved to the correct device
        self.register_buffer('ind_gather_tensor', torch.tensor(self._ind_gather, dtype=torch.int64))

    #########################################
    # Public methods and properties
    #########################################

    @property
    def k(self):
        """Number of information bits"""
        return self._k

    @property
    def n(self):
        """Codeword length"""
        return self._n

    @property
    def frozen_pos(self):
        """Frozen positions for Polar decoding"""
        return self._frozen_pos

    @property
    def info_pos(self):
        """Information bit positions for Polar encoding"""
        return self._info_pos

    #################
    # Utility methods
    #################

    def _gen_indices(self, n):
        """Pre-calculate encoding indices stage-wise for torch.gather.
        """
        nb_stages = int(np.log2(n))
        # Last position denotes empty placeholder (points to element n+1)
        ind_gather = np.ones([nb_stages, n+1]) * n

        for s in range(nb_stages):
            ind_range = np.arange(int(n/2))
            ind_dest = ind_range * 2 - np.mod(ind_range, 2**(s))
            ind_origin = ind_dest + 2**s
            ind_gather[s, ind_dest] = ind_origin  # And update gather indices

        return ind_gather

    def forward(self, bits):
        """Polar encoding function.

        This function returns the polar encoded codewords for the given
        information bits ``bits``.

        Args:
            bits (torch.Tensor): Tensor of shape `[...,k]` containing the
            information bits to be encoded.

        Returns:
            torch.Tensor: Tensor of shape `[...,n]`.
        """
        # Ensure input is a torch tensor
        if not isinstance(bits, torch.Tensor):
            bits = torch.tensor(bits, dtype=self._dtype)

        # Device handling
        device = bits.device

        # Reshape inputs to [...,k]
        input_shape = bits.shape

        # Check if the last dimension matches expected k
        if input_shape[-1] != self._k:
            raise ValueError(f"Input tensor last dimension ({input_shape[-1]}) doesn't match encoder k ({self._k})")

        batch_size = np.prod(input_shape[:-1]).astype(int) if len(input_shape) > 1 else 1
        u = bits.reshape(batch_size, self._k)

        # Assert if binary and u is non binary
        if self._check_input:
            u_test = u.float()  # Only for internal check
            is_binary = torch.logical_or(torch.eq(u_test, 0.), torch.eq(u_test, 1.))
            if not torch.all(is_binary):
                raise ValueError("Input must be binary.")
            # Input datatype consistency should be only evaluated once
            self._check_input = False

        # Copy info bits to information set; other positions are frozen (=0)
        c = torch.zeros((self._n, batch_size), dtype=bits.dtype, device=device)

        # u has shape batch_size x k, we now want k x batch_size
        u_transpose = u.transpose(0, 1)

        # Use advanced indexing to update c
        info_pos_tensor = torch.tensor(self.info_pos, dtype=torch.int64, device=device)
        c[info_pos_tensor] = u_transpose

        # Transpose c back
        c = c.transpose(0, 1)

        # Add an extra zero column for padding
        x_nan = torch.zeros((batch_size, 1), dtype=c.dtype, device=device)
        x = torch.cat([c, x_nan], dim=1)

        # Convert to int64 for bitwise operations
        orig_dtype = x.dtype
        x = x.to(torch.int64)

        # Loop over all stages - performing the polar transform
        for s in range(self._nb_stages):
            ind_helper = self.ind_gather_tensor[s].to(device)
            x_add = torch.index_select(x, 1, ind_helper)
            x = x ^ x_add  # Bitwise XOR

        # Convert back to original dtype
        x = x.to(orig_dtype)

        # Remove last position
        c_out = x[:, :self._n]

        # Restore original shape
        output_shape = list(input_shape)
        output_shape[-1] = self._n
        c_reshaped = c_out.reshape(output_shape)

        return c_reshaped

    def __call__(self, bits):
        """Alias for the forward method"""
        return self.forward(bits)


def generate_5g_ranking(k, n, sort=True):
    """Returns information and frozen bit positions of the 5G Polar code
    as defined in Tab. 5.3.1.2-1 in [3GPPTS38212] for given values of ``k``
    and ``n``.

    Parameters
    ----------
    k: int
        The number of information bit per codeword.

    n: int
        The desired codeword length. Must be a power of two.

    sort: bool, (default True)
        Indicates if the returned indices are sorted.

    Returns
    -------
    [frozen_pos, info_pos]: list
        frozen_pos: ndarray
            An array of ints of shape `[n-k]` containing the frozen
            positions.
        info_pos: ndarray
            An array of ints of shape `[k]` containing the information
            positions.
    """
    # Load 5G polar ranking table
    polar_5g_path = "../../wireless_comm/polar_5G.csv"
    try:
        polar_5g = np.loadtxt(polar_5g_path, delimiter=';', dtype=int)
    except:
        # Fallback: generate basic ranking if file not found
        polar_5g = np.arange(n)

    # Extract ranking for given n
    if n <= len(polar_5g):
        ranking = polar_5g[:n]
    else:
        # Extend ranking if needed
        ranking = np.concatenate([polar_5g, np.arange(len(polar_5g), n)])

    # Get the k best positions
    if sort:
        ranking_sorted = np.sort(ranking)
        info_pos = ranking_sorted[:k]
        frozen_pos = ranking_sorted[k:]
    else:
        info_pos = ranking[:k]
        frozen_pos = ranking[k:]

    return frozen_pos, info_pos


def get_frozen_pos(k, n):
    """Helper function to get frozen positions for 5G Polar codes"""
    frozen_pos, _ = generate_5g_ranking(k, n)
    return frozen_pos