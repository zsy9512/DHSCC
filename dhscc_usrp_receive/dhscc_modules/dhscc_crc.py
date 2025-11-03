#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch implementation of CRC encoder and decoder.
Adapted from the Sionna TensorFlow implementation.
Independent copy for DHSCC project.
"""

import torch
import numpy as np
import numbers

class DHSCC_CRCEncoder(torch.nn.Module):
    """Adds a Cyclic Redundancy Check (CRC) to the input sequence.

    The CRC polynomials from Sec. 5.1 in 3GPP TS 38.212 are available:
    `{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}`.

    Parameters
    ----------
    crc_degree: str, 'CRC24A' | 'CRC24B' | 'CRC24C' | 'CRC16' | 'CRC11' | 'CRC6'
        Defines the CRC polynomial to be used. Can be any value from
        `{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}`.

    dtype: torch.dtype, default=torch.float32
        Data type to use for internal calculations and outputs.

    Input
    -----
    bits : [...,k], torch.Tensor
        Binary tensor of arbitrary shape where the last dimension is
        `[...,k]`.

    Output
    ------
    x_crc : [...,k+crc_degree], torch.Tensor
        Binary tensor containing CRC-encoded bits of the same shape as
        ``inputs`` except the last dimension changes to
        `[...,k+crc_degree]`.

    Note
    ----
    For performance enhancements, a generator-matrix-based
    implementation is used for fixed `k` instead of the more common shift
    register-based operations. Thus, the encoder must rebuild the generator
    matrix if `k` changes.
    """

    def __init__(self, crc_degree, dtype=torch.float32):
        super().__init__()

        self._dtype = dtype
        self._crc_degree = crc_degree

        # Init 5G CRC polynomial
        self._crc_pol, self._crc_length = self._select_crc_pol(self._crc_degree)

        self._k = None
        self._n = None
        self.register_buffer('g_mat_crc_tensor', None)

    #########################################
    # Public methods and properties
    #########################################

    @property
    def crc_degree(self):
        """CRC degree as string"""
        return self._crc_degree

    @property
    def crc_length(self):
        """Length of CRC. Equals number of CRC parity bits."""
        return self._crc_length

    @property
    def crc_pol(self):
        """CRC polynomial in binary representation"""
        return self._crc_pol

    @property
    def k(self):
        """Number of information bits per codeword"""
        if self._k is None:
            print("Warning: CRC encoder is not initialized yet."\
                  "Input dimensions are unknown.")
        return self._k

    @property
    def n(self):
        """Number of codeword bits after CRC encoding."""
        if self._n is None:
            print("Warning: CRC encoder is not initialized yet."\
                  "Output dimensions are unknown.")
        return self._n

    #########################
    # Utility methods
    #########################

    def _select_crc_pol(self, crc_degree):
        """Select 5G CRC polynomial according to Sec. 5.1 in 3GPP TS 38.212."""

        if crc_degree=="CRC24A":
            crc_length = 24
            crc_coeffs = [24, 23, 18, 17, 14, 11, 10, 7, 6, 5, 4, 3, 1, 0]
        elif crc_degree=="CRC24B":
            crc_length = 24
            crc_coeffs = [24, 23, 6, 5, 1, 0]
        elif crc_degree=="CRC24C":
            crc_length = 24
            crc_coeffs = [24, 23, 21, 20, 17, 15, 13, 12, 8, 4, 2, 1, 0]
        elif crc_degree=="CRC16":
            crc_length = 16
            crc_coeffs = [16, 12, 5, 0]
        elif crc_degree=="CRC11":
            crc_length = 11
            crc_coeffs = [11, 10, 9, 5, 0]
        elif crc_degree=="CRC6":
            crc_length = 6
            crc_coeffs = [6, 5, 0]
        else:
            raise ValueError("Invalid CRC Polynomial")

        # Invert array (MSB instead of LSB)
        crc_pol_inv = np.zeros(crc_length + 1)
        crc_pol_inv[[crc_length - c for c in crc_coeffs]] = 1

        return crc_pol_inv.astype(int), crc_length

    def _gen_crc_mat(self, k, pol_crc):
        """ Build (dense) generator matrix for CRC parity bits.

        The principle idea is to treat the CRC as systematic linear code, i.e.,
        the generator matrix can be composed out of ``k`` linear independent
        (valid) codewords. For this, we CRC encode all ``k`` unit-vectors
        `[0,...1,...,0]` and build the generator matrix.
        To avoid `O(k^2)` complexity, we start with the last unit vector
        given as `[0,...,0,1]` and can generate the result for next vector
        `[0,...,1,0]` via another polynomial division of the remainder from the
        previous result. This allows to successively build the generator matrix
        at linear complexity `O(k)`.
        """
        crc_length = len(pol_crc) - 1
        g_mat = np.zeros([k, crc_length])

        x_crc = np.zeros(crc_length, dtype=int)
        x_crc[0] = 1
        for i in range(k):
            # shift by one position
            x_crc = np.concatenate([x_crc, [0]])
            if x_crc[0]==1:
                x_crc = np.bitwise_xor(x_crc, pol_crc)
            x_crc = x_crc[1:]
            g_mat[k-i-1,:] = x_crc

        return g_mat

    def _build(self, input_shape, device):
        """Build the generator matrix.

        The CRC is always added to the last dimension of the input.
        """
        k = input_shape[-1]  # we perform the CRC check on the last dimension
        assert k is not None, "Shape of last dimension cannot be None."
        g_mat_crc = self._gen_crc_mat(k, self._crc_pol)
        g_mat_crc_tensor = torch.tensor(g_mat_crc, dtype=self._dtype, device=device)
        self.register_buffer('g_mat_crc_tensor', g_mat_crc_tensor)

        self._k = k
        self._n = k + g_mat_crc.shape[1]

    def forward(self, bits):
        """Cyclic Redundancy Check (CRC) function.

        This function adds the CRC parity bits to ``inputs`` and returns the
        result of the CRC validation.

        Args:
            bits (torch.Tensor): Tensor of arbitrary shape `[...,k]`.

        Returns:
            torch.Tensor: CRC encoded bits ``x_crc`` of shape
                `[...,k+crc_degree]`.
        """
        # Ensure input is a torch tensor
        if not isinstance(bits, torch.Tensor):
            bits = torch.tensor(bits, dtype=self._dtype)

        device = bits.device

        # Re-init if shape has changed or if g_mat_crc_tensor is on wrong device
        if (self.g_mat_crc_tensor is None or
            bits.shape[-1] != self.g_mat_crc_tensor.shape[0] or
            (self.g_mat_crc_tensor.device != device)):
            self._build(bits.shape, device)

        # Reshape for batch processing
        input_shape = bits.shape
        batch_size = np.prod(input_shape[:-1]).astype(int) if len(input_shape) > 1 else 1
        k = input_shape[-1]

        # Reshape bits to [batch_size, k]
        bits_flat = bits.reshape(batch_size, k)

        # Convert to int64 for stable XOR operations
        bits_int = bits_flat

        # Calculate CRC bits using matrix multiplication
        x_crc = torch.matmul(bits_int, self.g_mat_crc_tensor) % 2

        # Cast back to original dtype
        x_crc = x_crc.to(bits.dtype)

        # Concatenate original bits with CRC bits
        x_conc = torch.cat([bits_flat, x_crc], dim=-1)

        # Reshape back to match original shape
        output_shape = list(input_shape)
        output_shape[-1] = k + self._crc_length
        x_out = x_conc.reshape(output_shape)

        return x_out

class DHSCC_CRCDecoder(torch.nn.Module):
    """Allows Cyclic Redundancy Check (CRC) verification and removes parity bits.

    The CRC polynomials from Sec. 5.1 in 3GPP TS 38.212 are available:
    `{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}`.

    Parameters
    ----------
    crc_encoder: CRCEncoder
        An instance of CRCEncoder associated with the CRCDecoder.

    dtype: torch.dtype, default=torch.float32
        Data type to use for internal calculations and outputs.

    Input
    -----
    x_crc: [...,k+crc_degree], torch.Tensor
        Binary tensor containing the CRC-encoded bits (the last
        `crc_degree` bits are parity bits).

    Output
    ------
    bits : [...,k], torch.Tensor
        Binary tensor containing the information bit sequence without CRC
        parity bits.

    crc_valid : [...,1], torch.bool
        Boolean tensor containing the result of the CRC check per codeword.
    """

    def __init__(self, crc_encoder, dtype=torch.float32):
        super().__init__()

        self._encoder = crc_encoder
        self._dtype = dtype
        self._input_shape = None

    #########################################
    # Public methods and properties
    #########################################

    @property
    def crc_degree(self):
        """CRC degree as string."""
        return self._encoder.crc_degree

    @property
    def encoder(self):
        """CRC Encoder used for internal validation."""
        return self._encoder

    def _build(self, input_shape):
        """Nothing to build but check shapes."""
        self._input_shape = input_shape
        if input_shape[-1] < self._encoder.crc_length:
            msg ="Input length must be greater than or equal to the CRC length."
            raise ValueError(msg)

    def forward(self, x_crc):
        """Cyclic Redundancy Check (CRC) verification function.

        This function verifies the CRC of ``x_crc``. It returns the result of
        the CRC validation and removes parity bits from ``x_crc``.

        Args:
            x_crc (torch.Tensor): Tensor of arbitrary shape `[...,k+crc_degree]`.

        Returns:
            Tuple [torch.Tensor, torch.bool]: ``[x_info, crc_valid]`` list, where
            ``x_info`` contains the information bits without CRC parity bits,
            of shape `[...,k]`, and ``crc_valid`` contains the result of the CRC
            validation for each codeword, of shape `[...,1]`.
        """
        # Ensure input is a torch tensor
        if not isinstance(x_crc, torch.Tensor):
            x_crc = torch.tensor(x_crc, dtype=self._dtype)

        # Check shape
        if self._input_shape != x_crc.shape:
            self._build(x_crc.shape)

        # Reshape for batch processing
        input_shape = x_crc.shape
        batch_size = np.prod(input_shape[:-1]).astype(int) if len(input_shape) > 1 else 1
        n = input_shape[-1]

        # Split information bits and CRC bits
        crc_length = self._encoder.crc_length
        k = n - crc_length

        # Reshape to [batch_size, n]
        x_crc_flat = x_crc.reshape(batch_size, n)

        # Extract information bits
        x_info = x_crc_flat[:, :k]

        # Re-encode information bits and get CRC bits
        x_parity = self._encoder(x_info)[:, k:]

        # Check if CRCs match
        crc_check = torch.all(x_parity == x_crc_flat[:, k:], dim=1, keepdim=True)

        # Reshape information bits to match input shape
        output_shape = list(input_shape)
        output_shape[-1] = k
        x_info_reshaped = x_info.reshape(output_shape)

        return x_info_reshaped, crc_check