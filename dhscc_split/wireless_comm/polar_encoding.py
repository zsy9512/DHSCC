#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch implementation of Polar encoder including 5G compliant rate-matching and CRC concatenation.
Adapted from the Sionna TensorFlow implementation.
"""

import torch
import numpy as np
import numbers
from importlib_resources import files, as_file
import os
import matplotlib.pyplot as plt
from scipy.special import comb

# Import CRC encoder/decoder from crc.py
from .crc import CRCEncoder, CRCDecoder

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
            position indices.

        info_pos: ndarray
            An array of ints of shape `[k]` containing the information
            position indices.
    """
    if not isinstance(k, int):
        raise TypeError("k must be integer.")
    if not isinstance(n, int):
        raise TypeError("n must be integer.")
    if not isinstance(sort, bool):
        raise TypeError("sort must be bool.")
    if k < 0:
        raise ValueError("k cannot be negative.")
    if k > 1024:
        raise ValueError("k cannot be larger than 1024.")
    if n > 1024:
        raise ValueError("n cannot be larger than 1024.")
    if n < 32:
        raise ValueError("n must be >=32.")
    if n < k and k > 0:
        raise ValueError("Invalid coderate (>1).")
    if np.log2(n) != int(np.log2(n)):
        raise ValueError("n must be a power of 2.")

    # Load the channel ranking from csv format TS 38.212 5.3.1.2-1
    csv_path = "polar_5G.csv"
    try:
        ch_order = np.genfromtxt(csv_path, delimiter=";")
        ch_order = ch_order.astype(int)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find 5G polar code ranking file at {csv_path}. Please ensure Sionna is installed correctly.")

    # Find n smallest values of channel order (2nd row)
    ind = np.argsort(ch_order[:,1])
    ch_order_sort = ch_order[ind,:]
    # Only consider the first n channels
    ch_order_sort_n = ch_order_sort[0:n,:]
    # And sort again according to reliability
    ind_n = np.argsort(ch_order_sort_n[:,0])
    ch_order_n = ch_order_sort_n[ind_n,:]

    # Special case: if k=0, return full sequence for ranking purpose
    if k == 0:
        return ch_order_n[:,1].astype(int), np.array([], dtype=int)

    # Calculate frozen/information positions for given n, k
    frozen_pos = np.zeros(n-k)
    info_pos = np.zeros(k)
    # The n-k smallest positions of ch_order denote frozen pos.
    for i in range(n-k):
        frozen_pos[i] = ch_order_n[i,1]  # 2nd column yields index to freeze
    for i in range(n-k, n):
        info_pos[i-(n-k)] = ch_order_n[i,1]  # 2nd column yields index to freeze

    # Sort to have channels in ascending order
    if sort:
        info_pos = np.sort(info_pos)
        frozen_pos = np.sort(frozen_pos)

    return frozen_pos.astype(int), info_pos.astype(int)


class Polar5GEncoder(PolarEncoder):
    """5G compliant Polar encoder including rate-matching for the uplink scenario (`UCI`) and downlink scenario (`DCI`).

    This class performs polar encoding for ``k`` information bits and
    rate-matching such that the codeword length is ``n``. This includes the CRC
    concatenation and the interleaving as defined in 3GPP TS 38.212.

    Parameters
    ----------
    k: int
        Defining the number of information bit per codeword.

    n: int
        Defining the codeword length.

    channel_type: 'uplink' (default) | 'downlink'
        Can be 'uplink' or 'downlink'.

    verbose: bool, (default False)
        If True, rate-matching parameters will be printed.

    dtype: torch.dtype, default=torch.float32
        Data type to use for internal calculations and outputs.
    """
    
    def __init__(self,
                 k,
                 n,
                 channel_type="uplink",
                 verbose=False,
                 dtype=torch.float32):
        
        if not isinstance(k, numbers.Number):
            raise TypeError("k must be a number.")
        if not isinstance(n, numbers.Number):
            raise TypeError("n must be a number.")
        k = int(k)  # k or n can be float (e.g. as result of n=k*r)
        n = int(n)  # k or n can be float (e.g. as result of n=k*r)
        if n < k:
            raise ValueError("Invalid coderate (>1).")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be bool.")

        if channel_type not in ("uplink", "downlink"):
            raise ValueError("Unsupported channel_type.")
        self._channel_type = channel_type

        self._k_target = k
        self._n_target = n
        self._verbose = verbose
        self._dtype = dtype

        # Initialize rate-matcher
        crc_degree, n_polar, frozen_pos, idx_rm, idx_input = self._init_rate_match(k, n)
        
        self._frozen_pos = frozen_pos  # Required for decoder
        self._ind_rate_matching = idx_rm  # Store temporarily
        self._ind_input_int = idx_input  # Index for input interleaver (not a tensor)

        # Initialize the parent class (PolarEncoder) with frozen positions and n_polar
        super().__init__(frozen_pos, n_polar, dtype=dtype)
        
        # Initialize CRC encoder after the parent initialization
        self._enc_crc = CRCEncoder(crc_degree, dtype=dtype)
        
        # Now register buffers after parent's __init__
        self.register_buffer('ind_rate_matching', self._ind_rate_matching)  # Index for gather-based rate-matching

    ###############################
    # Public methods and properties
    ###############################

    @property
    def enc_crc(self):
        """CRC encoder block used for CRC concatenation"""
        return self._enc_crc

    @property
    def k_target(self):
        """Number of information bits including rate-matching"""
        return self._k_target

    @property
    def n_target(self):
        """Codeword length including rate-matching"""
        return self._n_target

    @property
    def k_polar(self):
        """Number of information bits of the underlying Polar code"""
        return self._k

    @property
    def n_polar(self):
        """Codeword length of the underlying Polar code"""
        return self._n

    @property
    def k(self):
        """Number of information bits including rate-matching"""
        return self._k_target

    @property
    def n(self):
        """Codeword length including rate-matching"""
        return self._n_target

    def subblock_interleaving(self, u):
        """Input bit interleaving as defined in Sec 5.4.1.1 in 3GPP TS 38.212.

        Input
        -----
        u: ndarray
            1D array to be interleaved. Length of ``u`` must be a multiple
            of 32.

        Output
        ------
        : ndarray
            Interleaved version of ``u`` with same shape and dtype as ``u``.
        """

        k = u.shape[-1]
        if np.mod(k, 32) != 0:
            msg = "length for sub-block interleaving must be a multiple of 32."
            raise ValueError(msg)
        y = np.zeros_like(u)

        # Permutation according to Tab 5.4.1.1.1-1 in 38.212
        perm = np.array([0, 1, 2, 4, 3, 5, 6, 7, 8, 16, 9, 17, 10, 18, 11, 19,
                         12, 20, 13, 21, 14, 22, 15, 23, 24, 25, 26, 28, 27,
                         29, 30, 31])

        for n in range(k):
            i = int(np.floor(32*n/k))
            j = perm[i] * k/32 + np.mod(n, k/32)
            j = int(j)
            y[n] = u[j]

        return y

    def channel_interleaver(self, c):
        """Triangular interleaver following Sec. 5.4.1.3 in 3GPP TS 38.212.

        Input
        -----
        c: ndarray
            1D array to be interleaved.

        Output
        ------
        : ndarray
            Interleaved version of ``c`` with same shape and dtype as ``c``.
        """

        n = c.shape[-1]  # Denoted as E in 38.212
        c_int = np.zeros_like(c)

        # Find smallest T s.t. T*(T+1)/2 >= n
        t = 0
        while t*(t+1)/2 < n:
            t += 1

        v = np.zeros([t, t])
        ind_k = 0
        for ind_i in range(t):
            for ind_j in range(t-ind_i):
                if ind_k < n:
                    v[ind_i, ind_j] = c[ind_k]
                else:
                    v[ind_i, ind_j] = np.nan  # NULL
                # Store nothing otherwise
                ind_k += 1
        ind_k = 0
        for ind_j in range(t):
            for ind_i in range(t-ind_j):
                if not np.isnan(v[ind_i, ind_j]):
                    c_int[ind_k] = v[ind_i, ind_j]
                    ind_k += 1
        return c_int

    def input_interleaver(self, c):
        """Input interleaver following Sec. 5.4.1.1 in 3GPP TS 38.212.

        Input
        -----
        c: ndarray
            1D array to be interleaved.

        Output
        ------
        : ndarray
            Interleaved version of ``c`` with same shape and dtype as ``c``.
        """

        # 38.212 Table 5.3.1.1-1
        p_il_max_table = [0, 2, 4, 7, 9, 14, 19, 20, 24, 25, 26, 28, 31, 34,
            42, 45, 49, 50, 51, 53, 54, 56, 58, 59, 61, 62, 65, 66, 67, 69,
            70, 71, 72, 76, 77, 81, 82, 83, 87, 88, 89, 91, 93, 95, 98, 101,
            104, 106, 108, 110, 111, 113, 115, 118, 119, 120, 122, 123, 126,
            127, 129, 132, 134, 138, 139, 140, 1, 3, 5, 8, 10, 15, 21, 27, 29,
            32, 35, 43, 46, 52, 55, 57, 60, 63, 68, 73, 78, 84, 90, 92, 94, 96,
            99, 102, 105, 107, 109, 112, 114, 116, 121, 124, 128, 130, 133,
            135, 141, 6, 11, 16, 22, 30, 33, 36, 44, 47, 64, 74, 79, 85, 97,
            100, 103, 117, 125, 131, 136, 142, 12, 17, 23, 37, 48, 75, 80, 86,
            137, 143, 13, 18, 38, 144, 39, 145, 40, 146, 41, 147, 148, 149,
            150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162,
            163]
        k_il_max = 164
        k = len(c)
        if k > k_il_max:
            msg = "Input interleaver only defined for length of 164."
            raise ValueError(msg)

        c_apo = np.empty(k, 'int')
        i = 0
        for p_il_max in p_il_max_table:
            if p_il_max >= (k_il_max - k):
                c_apo[i] = c[p_il_max - (k_il_max - k)]
                i += 1
        return c_apo

    def _init_rate_match(self, k_target, n_target):
        """Implementing polar rate matching according to 3GPP TS 38.212.

        This part of the code only runs during the initialization and is not performance critical.
        For easier alignment and traceability with the standard document the implementation
        prefers `for loop`-based indexing.

        The relation of terminology between 3GPP TS 38.212 and this code is given as:
        `A`...`k_target`
        `E`...`n_target`
        `K`...`k_polar`
        `N`...`n_polar`
        `L`...`k_crc`.
        """

        # Check input for consistency (see Sec. 6.3.1.2.1 for UL)
        if n_target < k_target:
            msg = "n must be larger or equal k."
            raise ValueError(msg)
        if n_target < 18:
            msg = "n<18 is not supported by the 5G Polar coding scheme."
            raise ValueError(msg)
        if k_target > 1013:
            msg = "k too large - currently, no codeword segmentation supported."
            raise ValueError(msg)
        if n_target > 1088:
            msg = "n too large - currently, no codeword segmentation supported."

        # Select CRC polynomials (see Sec. 6.3.1.2.1 for UL)
        if self._channel_type == "uplink":
            if 12 <= k_target <= 19:
                crc_pol = "CRC6"
                k_crc = 6
            elif k_target >= 20:
                crc_pol = "CRC11"
                k_crc = 11
            else:
                raise ValueError("k_target<12 is not supported in 5G NR for " \
                    "the uplink; please use 'channel coding of small block  " \
                    "lengths' scheme from Sec. 5.3.3 in 3GPP 38.212 instead.")

            # PC bit for k_target = 12-19 bits (see Sec. 6.3.1.3.1 for UL)
            n_pc = 0
            if k_target <= 19:
                # n_pc = 3  # Currently deactivated
                n_pc = 0

        else:  # downlink channel
            # for downlink CRC24 is used
            # remark: in PDCCH messages are limited to k=140
            # as the input interleaver does not support longer sequences
            if k_target > 140:
                msg = "k too large for downlink configuration."
                raise ValueError(msg)
            if n_target < 25:
                msg = "n too small for downlink configuration with 24 bit CRC."
                raise ValueError(msg)
            if n_target > 576:
                msg = "n too large for downlink configuration."
                raise ValueError(msg)
            crc_pol = "CRC24C"  # following 7.3.2
            k_crc = 24
            n_pc = 0

        # Calculate Polar payload length (CRC bits are treated as info bits)
        k_polar = k_target + k_crc + n_pc
        self._k_crc = k_crc
        self._k_polar = k_polar

        if k_polar > n_target:
            msg = "Device is not expected to be configured " \
                  "with k_polar + k_crc + n_pc > n_target."
            raise ValueError(msg)

        # Select polar mother code length n_polar
        n_min = 5
        n_max = 10  # For uplink; otherwise 9
        if self._channel_type == "downlink":
            n_max = 9

        # Select rate-matching scheme following Sec. 5.3.1
        if (n_target <= ((9/8) * 2**(np.ceil(np.log2(n_target))-1)) and
                k_polar/n_target < 9/16):
            n1 = np.ceil(np.log2(n_target))-1
        else:
            n1 = np.ceil(np.log2(n_target))
        n2 = np.ceil(np.log2(8*k_polar))  # Lower bound such that rate > 1/8
        n_polar = int(2**np.max((np.min([n1, n2, n_max]), n_min)))

        # Puncturing and shortening as defined in Sec. 5.4.1.1
        prefrozen_pos = []  # List containing the pre-frozen indices
        if n_target < n_polar:
            if k_polar/n_target <= 7/16:
                # Puncturing
                n_int = 32 * np.ceil((n_polar-n_target) / 32)
                int_pattern = self.subblock_interleaving(np.arange(n_int))
                for i in range(n_polar-n_target):
                    # Freeze additional bits
                    prefrozen_pos.append(int(int_pattern[i]))
                if n_target >= 3*n_polar/4:
                    t = int(np.ceil(3/4*n_polar - n_target/2) - 1)
                else:
                    t = int(np.ceil(9/16*n_polar - n_target/4) - 1)
                # Extra freezing
                for i in range(t):
                    prefrozen_pos.append(i)
            else:
                # Shortening ("through" sub-block interleaver)
                n_int = 32 * np.ceil((n_polar) / 32)
                int_pattern = self.subblock_interleaving(np.arange(n_int))
                for i in range(n_target, n_polar):
                    prefrozen_pos.append(int_pattern[i])
        
        # Remove duplicates
        prefrozen_pos = np.unique(prefrozen_pos)

        # Load full channel ranking - use 0 as placeholder for k since we want all positions
        ch_ranking, _ = generate_5g_ranking(0, n_polar, sort=False)
        # Remove positions that are already frozen by `pre-freezing` stage
        info_cand = np.setdiff1d(ch_ranking, prefrozen_pos, assume_unique=True)
        
        # Check if we have enough candidate positions
        if len(info_cand) < k_polar:
            msg = f"Not enough candidate positions ({len(info_cand)}) for information bits ({k_polar}). "
            msg += f"Configuration: k_target={k_target}, n_target={n_target}, n_polar={n_polar}, prefrozen_pos={len(prefrozen_pos)}"
            raise ValueError(msg)

        # Identify k_polar most reliable positions from candidate positions
        info_pos = []
        for i in range(k_polar):
            info_pos.append(info_cand[-i-1])

        # Sort and create frozen positions for n_polar indices (no shortening)
        info_pos = np.sort(info_pos).astype(int)
        frozen_pos = np.setdiff1d(np.arange(n_polar),
                                  info_pos,
                                  assume_unique=True)

        # For downlink only: generate input bit interleaver
        if self._channel_type == "downlink":
            ind_input_int = self.input_interleaver(np.arange(k_polar))
        else:
            ind_input_int = None

        # Generate tf.gather indices for sub-block interleaver
        ind_sub_int = self.subblock_interleaving(np.arange(n_polar))

        # Rate matching via circular buffer as defined in Sec. 5.4.1.2
        c_int = np.arange(n_polar)
        idx_c_matched = np.zeros([n_target])
        if n_target >= n_polar:
            # Repetition coding
            # if self._verbose:
            #     print("Using repetition coding for rate-matching")
            for ind in range(n_target):
                idx_c_matched[ind] = c_int[np.mod(ind, n_polar)]
        else:
            if k_polar/n_target <= 7/16:
                # Puncturing
                for ind in range(n_target):
                    idx_c_matched[ind] = c_int[ind+n_polar-n_target]
            else:
                # Shortening
                for ind in range(n_target):
                    idx_c_matched[ind] = c_int[ind]

        # For uplink only: generate input bit interleaver
        if self._channel_type == "uplink":
            # if self._verbose:
            #     print("Using channel interleaver for uplink.")
            ind_channel_int = self.channel_interleaver(np.arange(n_target))

            # Combine indices for single tf.gather operation
            ind_t = idx_c_matched[ind_channel_int].astype(int)
            idx_rate_matched = ind_sub_int[ind_t]
        else:  # no channel interleaver for downlink
            idx_rate_matched = ind_sub_int[idx_c_matched.astype(int)]

        # if self._verbose:
        #     print("Code parameters after rate-matching: " \
        #           f"k = {k_target}, n = {n_target}")
        #     print(f"Polar mother code: k_polar = {k_polar}, " \
        #           f"n_polar = {n_polar}")
        #     print("Using", crc_pol)
        #     print("Frozen positions: ", frozen_pos)
        #     print("Channel type: " + self._channel_type)

        # Convert idx_rate_matched to PyTorch tensor
        idx_rate_matched_tensor = torch.tensor(idx_rate_matched, dtype=torch.int64)

        return crc_pol, n_polar, frozen_pos, idx_rate_matched_tensor, ind_input_int

    def forward(self, bits):
        """Polar encoding function including rate-matching and CRC encoding.

        This function returns the polar encoded codewords for the given
        information bits ``bits`` including rate-matching.

        Args:
            bits (torch.Tensor): Tensor of shape `[...,k]` containing the
            information bits to be encoded.

        Returns:
            torch.Tensor: Tensor of shape `[...,n]`.
        """

        # Ensure input is a torch tensor
        if not isinstance(bits, torch.Tensor):
            bits = torch.tensor(bits, dtype=self._dtype)
            
        device = bits.device

        # Validate input shape
        input_shape = bits.shape
        if input_shape[-1] != self._k_target:
            raise ValueError(f"Input tensor last dimension ({input_shape[-1]}) doesn't match encoder k_target ({self._k_target})")
        
        # Reshape inputs to [...,k]
        batch_size = np.prod(input_shape[:-1]).astype(int) if len(input_shape) > 1 else 1
        u = bits.reshape(batch_size, self._k_target)
        # Convert to float for CRC encoding to avoid CUDA int64 matrix multiplication issue
        orig_dtype = u.dtype
        u_float = u.to(torch.float32)
        
        # CRC encode with float32 to avoid CUDA issues with int64
        u_crc = self._enc_crc(u_float)
        
        # Convert back to original dtype
        u_crc = u_crc.to(orig_dtype)

        # For downlink only: apply input bit interleaver
        if self._channel_type == "downlink" and self._ind_input_int is not None:
            ind_input_int_tensor = torch.tensor(self._ind_input_int, dtype=torch.int64, device=device)
            u_crc = torch.index_select(u_crc, 1, ind_input_int_tensor)

        # Encode bits (= channel allocation + Polar transform)
        c = super().forward(u_crc)

        # Make sure the rate matching indices are on the correct device
        if self.ind_rate_matching.device != device:
            self.ind_rate_matching = self.ind_rate_matching.to(device)
            
        # Sub-block interleaving with 32 sub-blocks as in Sec. 5.4.1.1
        # Rate matching via circular buffer as defined in Sec. 5.4.1.2
        # For uplink only: channel interleaving
        c_matched = torch.index_select(c, 1, self.ind_rate_matching)

        # Restore original shape
        output_shape = list(input_shape)
        output_shape[-1] = self._n_target
        c_reshaped = c_matched.reshape(output_shape)

        return c_reshaped

    def __call__(self, bits):
        """Alias for the forward method"""
        return self.forward(bits)


def generate_polar_transform_mat(n_lift):
    """Generate the polar transformation matrix (Kronecker product).

    Parameters
    ----------
    n_lift: int
        Defining the Kronecker power, i.e., how often is the kernel lifted.

    Returns
    -------
    ndarray
        Array of `0s` and `1s` of shape `[2^n_lift , 2^n_lift]` containing
        the Polar transformation matrix.
    """
    if int(n_lift) != n_lift:
        raise ValueError("n_lift must be integer.")
    if n_lift < 0:
        raise ValueError("n_lift must be positive.")
    if n_lift >= 20:
        msg = "Warning: the resulting code length is large (=2^n_lift)."
        raise ValueError(msg)

    gm = np.array([[1, 0], [1, 1]])

    gm_l = np.copy(gm)
    for _ in range(n_lift-1):
        gm_l_new = np.zeros([2*np.shape(gm_l)[0], 2*np.shape(gm_l)[1]])
        for j in range(np.shape(gm_l)[0]):
            for k in range(np.shape(gm_l)[1]):
                gm_l_new[2*j:2*j+2, 2*k:2*k+2] = gm_l[j, k]*gm
        gm_l = gm_l_new
    return gm_l


def generate_rm_code(r, m):
    """Generate frozen positions of the (r, m) Reed Muller (RM) code.

    Parameters
    ----------
    r: int
        The order of the RM code.

    m: int
        `log2` of the desired codeword length.

    Returns
    -------
    frozen_pos: ndarray
        An array of ints of shape `[n-k]` containing the frozen
        position indices.

    info_pos: ndarray
        An array of ints of shape `[k]` containing the information
        position indices.

    n: int
        Resulting codeword length

    k: int
        Number of information bits

    d_min: int
        Minimum distance of the code.
    """
    if not isinstance(r, int):
        raise TypeError("r must be int.")
    if not isinstance(m, int):
        raise TypeError("m must be int.")
    if r > m:
        raise ValueError("order r cannot be larger than m.")
    if r < 0:
        raise ValueError("r must be positive.")
    if m < 0:
        raise ValueError("m must be positive.")

    n = 2**m
    d_min = 2**(m-r)

    # Calculate k to verify results
    k = 0
    for i in range(r+1):
        k += int(comb(m, i))

    # Select positions to freeze
    # Freeze all rows that have weight < m-r
    w = np.zeros(n)
    for i in range(n):
        x_bin = np.binary_repr(i)
        for x_i in x_bin:
            w[i] += int(x_i)
    frozen_vec = w < m-r
    info_vec = np.invert(frozen_vec)
    k_res = np.sum(info_vec)
    frozen_pos = np.arange(n)[frozen_vec]
    info_pos = np.arange(n)[info_vec]

    # Verify results
    if k_res != k:
        raise ValueError("Error: resulting k is inconsistent.")

    return frozen_pos, info_pos, n, k, d_min