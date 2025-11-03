#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch implementation of Polar decoders including Successive Cancellation (SC) decoder.
Adapted from the Sionna TensorFlow implementation.
"""

import torch
import numpy as np
import numbers
import matplotlib.pyplot as plt
import traceback
import os

class CRCEncoder(torch.nn.Module):
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

class PolarSCDecoder(torch.nn.Module):
    """Successive cancellation (SC) decoder for Polar codes.

    Parameters
    ----------
    frozen_pos: ndarray
        Array of int defining the `n-k` indices of the frozen positions.

    n: int
        Defining the codeword length.

    dtype: torch.dtype, default=torch.float32
        Data type to use for internal calculations and outputs.

    Input
    -----
    llr_ch: [...,n], torch.Tensor
        Tensor containing the channel LLR values (as logits).

    Output
    ------
    : [...,k], torch.Tensor
        Tensor containing hard-decided estimations of all k information bits.

    Note
    ----
    This implementation follows the recursive tree terminology and combines nodes 
    for increased throughputs without changing the outcome of the algorithm.

    As commonly done, we assume frozen bits are set to 0.
    """

    def __init__(self, 
                 frozen_pos,
                 n,
                 dtype=torch.float32):
        super().__init__()
        
        # Check inputs
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

        # Store internal attributes
        self._n = n
        self._frozen_pos = frozen_pos
        self._k = self._n - len(self._frozen_pos)
        self._info_pos = np.setdiff1d(np.arange(self._n), frozen_pos)
        self._dtype = dtype

        if self._k != len(self._info_pos):
            raise ValueError("Internal error: invalid info_pos generated.")

        self._llr_max = 30.0  # Internal max LLR value (uncritical for SC dec)
        
        # Create a frozen bit vector for simpler encoding
        self._frozen_ind = np.zeros(self._n)
        self._frozen_ind[self._frozen_pos] = 1
        
        # Register buffers for moving to different devices
        self.register_buffer('frozen_ind_tensor', torch.tensor(self._frozen_ind, dtype=self._dtype))
        self.register_buffer('info_pos_tensor', torch.tensor(self._info_pos, dtype=torch.int64))

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

    def _cn_op(self, x, y):
        """Check-node update (boxplus) for LLR inputs.
        
        Operations are performed element-wise using the numerically stable min-sum
        approximation with a correction factor.
        """
        # Clamp values for numerical stability
        x_in = torch.clamp(x, min=-self._llr_max, max=self._llr_max)
        y_in = torch.clamp(y, min=-self._llr_max, max=self._llr_max)
        
        # Use sign-magnitude formulation for numerical stability
        x_abs = torch.abs(x_in)
        y_abs = torch.abs(y_in)
        sign = torch.sign(x_in) * torch.sign(y_in)
        
        # Min-sum approximation with correction factor (0.9375 = 15/16)
        min_val = torch.minimum(x_abs, y_abs)
        correction = 0.9375
        
        # Special case handling for very small values
        llr_out = sign * torch.min(
            self._llr_max * torch.ones_like(min_val),
            correction * min_val
        )
        
        return llr_out

    def _vn_op(self, x, y, u_hat):
        """Variable-node update for LLR inputs.
        
        This operation transforms the upper path LLRs based on decisions
        from the previous stage.
        
        Args:
            x: LLRs from the left branch
            y: LLRs from the right branch
            u_hat: Bit decisions from the previous stage
            
        Returns:
            Updated LLRs for the lower path
        """
        # Ensure proper numerical range
        x_clamped = torch.clamp(x, min=-self._llr_max, max=self._llr_max)
        
        # Calculate the variable node update
        # For u_hat=0: return y + x
        # For u_hat=1: return y - x
        return y + (1 - 2 * u_hat) * x_clamped

    def _polar_decode_sc(self, llr_ch, frozen_ind):
        """Recursive SC decoding function.
        
        Recursively branch decoding tree and split into decoding of upper
        and lower path until reaching a leaf node.
        
        The function returns the u_hat decisions at stage 0 and the bit
        decisions of the intermediate stage s.
        """
        # Calculate current codeword length
        n = len(frozen_ind)
        device = llr_ch.device
        
        # Branch if leaf is not reached yet
        if n > 1:
            # Split the LLRs and frozen bit indicators
            n_half = n // 2
            llr_ch1 = llr_ch[..., :n_half]
            llr_ch2 = llr_ch[..., n_half:]
            frozen_ind1 = frozen_ind[0:n_half]
            frozen_ind2 = frozen_ind[n_half:]
            
            # Upper path - check node operation
            x_llr1_in = self._cn_op(llr_ch1, llr_ch2)
            
            # Decode upper path
            u_hat1, u_hat1_up = self._polar_decode_sc(x_llr1_in, frozen_ind1)
            
            # Lower path - variable node operation using upper path decisions
            x_llr2_in = self._vn_op(llr_ch1, llr_ch2, u_hat1_up)
            
            # Decode lower path
            u_hat2, u_hat2_up = self._polar_decode_sc(x_llr2_in, frozen_ind2)
            
            # Combine decisions from both paths
            u_hat = torch.cat([u_hat1, u_hat2], dim=-1)
            
            # Calculate partial sums (re-encoded version at current stage)
            u_hat1_temp = u_hat1_up.clone().to(torch.int8)
            u_hat2_temp = u_hat2_up.clone().to(torch.int8)
            
            # Apply the XOR operation of polar transform
            u_hat1_xor = torch.bitwise_xor(u_hat1_temp, u_hat2_temp)
            u_hat1_up_out = u_hat1_xor.to(self._dtype)
            
            # Concatenate for the return path
            u_hat_up = torch.cat([u_hat1_up_out, u_hat2_up], dim=-1)
            
        else:  # Leaf node reached
            # At leaf nodes we make decisions based on LLRs and frozen indicator
            if frozen_ind.item() == 1:  # Frozen position
                u_hat = torch.zeros_like(llr_ch)
                u_hat_up = torch.zeros_like(llr_ch)
            else:  # Information position
                # Hard decision based on LLR sign
                # LLR < 0 -> u_hat = 1
                # LLR > 0 -> u_hat = 0
                u_hat = torch.where(llr_ch < 0, 
                                   torch.ones_like(llr_ch),
                                   torch.zeros_like(llr_ch))
                u_hat_up = u_hat.clone()
                
        return u_hat, u_hat_up

    def __call__(self, llr_ch):
        """Wrapper method for the forward call."""
        return self.forward(llr_ch)

    def forward(self, llr_ch):
        """Successive cancellation (SC) decoding function.
        
        Performs successive cancellation decoding and returns the estimated
        information bits.
        
        Args:
            llr_ch (torch.Tensor): Tensor of shape [...,n] containing the
                channel LLR values (as logits).
                
        Returns:
            torch.Tensor: Tensor of shape [...,k] containing
            hard-decided estimations of all k information bits.
        """
        # Ensure input is a torch tensor
        if not isinstance(llr_ch, torch.Tensor):
            llr_ch = torch.tensor(llr_ch, dtype=self._dtype)
        
        device = llr_ch.device
        
        # Reshape inputs while preserving the correct dimensions
        input_shape = llr_ch.shape
        
        # Check if the last dimension matches expected n
        if input_shape[-1] != self._n:
            raise ValueError(f"Input tensor last dimension ({input_shape[-1]}) doesn't match decoder n ({self._n})")
        
        # Calculate batch size properly by flattening all dimensions except the last one
        batch_dim = np.prod(input_shape[:-1]).astype(int) if len(input_shape) > 1 else 1
        
        # Reshape to 2D: [batch_dim, n]
        llr_ch_reshaped = llr_ch.reshape(batch_dim, self._n)

        # Clamp LLRs to avoid numerical issues
        llr_ch_reshaped = torch.clamp(llr_ch_reshaped, min=-self._llr_max, max=self._llr_max)
        
        # Decode
        u_hat_n, _ = self._polar_decode_sc(llr_ch_reshaped, self.frozen_ind_tensor)

        # Extract only the information bits
        u_hat = torch.index_select(u_hat_n, dim=-1, index=self.info_pos_tensor)
        
        # Reshape back to match input shape but with the last dimension as k
        output_shape = list(input_shape)
        output_shape[-1] = self._k
        u_hat_reshaped = u_hat.reshape(output_shape)

        return u_hat_reshaped

class CRCDecoder(torch.nn.Module):
    """CRC Decoder for error detection.
    
    Parameters
    ----------
    crc_encoder: CRCEncoder
        CRC encoder to use for checking the decoded data.
        
    dtype: torch.dtype, default=torch.float32
        Data type to use for internal calculations and outputs.
    """
    
    def __init__(self, crc_encoder, dtype=torch.float32):
        super().__init__()
        self._encoder = crc_encoder
        self._dtype = dtype
    
    @property
    def encoder(self):
        """The corresponding CRC encoder"""
        return self._encoder
    
    def forward(self, bits):
        """Verify the CRC checksum of the input bits.
        
        Args:
            bits (torch.Tensor): Input bits of shape [..., k+crc_degree]
            
        Returns:
            tuple(torch.Tensor, torch.Tensor): 
                - Data bits of shape [..., k]
                - Boolean indicating CRC validity
        """
        # Ensure input is a torch tensor
        if not isinstance(bits, torch.Tensor):
            bits = torch.tensor(bits, dtype=self._dtype)
            
        device = bits.device
        
        # Calculate dimensions
        input_shape = bits.shape
        batch_size = np.prod(input_shape[:-1]).astype(int) if len(input_shape) > 1 else 1
        total_length = input_shape[-1]
        crc_length = self._encoder._crc_length
        data_length = total_length - crc_length
        
        # Reshape for batch processing
        bits_flat = bits.reshape(batch_size, total_length)
        
        # Extract data and received CRC
        data_bits = bits_flat[:, :data_length]
        received_crc = bits_flat[:, data_length:]
        
        # Calculate expected CRC
        encoded = self._encoder(data_bits)
        expected_crc = encoded[:, data_length:]
        
        # Check if CRCs match
        crc_valid = torch.all(expected_crc == received_crc, dim=1)
        
        # Reshape data bits to match input shape
        output_shape = list(input_shape)
        output_shape[-1] = data_length
        data_bits_reshaped = data_bits.reshape(output_shape)
        
        return data_bits_reshaped, crc_valid
    
    def __call__(self, bits):
        """Alias for the forward method"""
        return self.forward(bits)
class DHSCC_PolarSCLDecoder(torch.nn.Module):
    """Successive cancellation list (SCL) decoder for Polar codes and Polar-like codes.

    Parameters
    ----------
    frozen_pos: ndarray
        Array of int defining the ``n-k`` indices of the frozen positions.

    n: int
        Defining the codeword length.

    list_size: int, (default 8)
        Defines the list size of the decoder.

    crc_degree: str, "CRC24A" | "CRC24B" | "CRC24C" | "CRC16" | "CRC11" | "CRC6"
        Defining the CRC polynomial to be used. Can be any value from
        `{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}`.

    use_fast_scl: bool, (default True)
        If True, tree pruning is used to reduce the decoding complexity.
        The output is equivalent to the non-pruned version (besides numerical differences).

    ind_iil_inv: None or [k+k_crc], int or torch.int
        Default to None. If not None, the sequence is used as inverse
        input bit interleaver before evaluating the CRC.
        Note: This only affects the CRC evaluation but the output
        sequence is not permuted.

    return_crc_status: bool, (default False)
        If True, the decoder additionally returns the CRC status indicating
        if a codeword was (most likely) correctly recovered. This is only
        available if crc_degree is not None.

    dtype: torch.dtype, default=torch.float32
        Data type to use for internal calculations and outputs.

    Input
    -----
    llr_ch: [...,n], torch.Tensor
        Tensor containing the channel LLR values (as logits).

    Output
    ------
    b_hat : [...,k], torch.Tensor
        Binary tensor containing hard-decided estimations of all k
        information bits.

    crc_status : [...], torch.bool
        CRC status indicating if a codeword was (most likely) correctly
        recovered. This is only returned if return_crc_status is True.
        Note that false positives are possible.

    Note
    ----
    This implementation follows the notation from the Sionna version and uses
    LLR-based message updates. If option use_fast_scl is active, tree pruning
    is used and tree nodes are combined if possible.

    As commonly done, we assume frozen bits are set to 0.
    """

    def __init__(self,
                 frozen_pos,
                 n,
                 list_size=8,
                 crc_degree=None,
                 use_fast_scl=True,
                 ind_iil_inv=None,
                 return_crc_status=False,
                 dtype=torch.float32):

        super().__init__()

        # Check inputs
        if not isinstance(n, numbers.Number):
            raise TypeError("n must be a number.")
        n = int(n)  # n can be float (e.g. as result of n=k*r)

        if not isinstance(list_size, int):
            raise TypeError("list_size must be integer.")

        if not isinstance(use_fast_scl, bool):
            raise TypeError("use_fast_scl must be bool.")

        # Fix the data type check for frozen_pos
        if not isinstance(frozen_pos, np.ndarray):
            raise TypeError("frozen_pos must be a numpy.ndarray.")

        # Ensure the array has an integer dtype
        if not np.issubdtype(frozen_pos.dtype, np.integer):
            raise TypeError("frozen_pos contains non-integer values.")

        if len(frozen_pos) > n:
            raise ValueError("Number of elements in frozen_pos cannot be greater than n.")

        if np.log2(n) != int(np.log2(n)):
            raise ValueError("n must be a power of 2.")

        if np.log2(list_size) != int(np.log2(list_size)):
            raise ValueError("list_size must be a power of 2.")

        # Internal decoder parameters
        self._use_fast_scl = use_fast_scl

        # Store internal attributes
        self._n = n
        self._frozen_pos = frozen_pos
        self._k = self._n - len(self._frozen_pos)
        self._list_size = list_size
        self._info_pos = np.setdiff1d(np.arange(self._n), self._frozen_pos)
        self._dtype = dtype
        self._llr_max = 100.  # internal max LLR value (not very critical for SC)

        if self._k != len(self._info_pos):
            raise ArithmeticError("Internal error: invalid info_pos generated.")

        # Create a frozen bit vector
        self._frozen_ind = np.zeros(self._n)
        self._frozen_ind[self._frozen_pos] = 1
        self._cw_ind = np.arange(self._n)
        self._n_stages = int(np.log2(self._n))  # number of decoding stages

        # Init CRC check (if needed)
        if crc_degree is not None:
            self._use_crc = True
            # Import CRCEncoder and CRCDecoder implementation
            try:
                from .dhscc_crc import DHSCC_CRCEncoder, DHSCC_CRCDecoder
                self._crc_encoder = DHSCC_CRCEncoder(crc_degree, dtype=dtype)
                self._crc_decoder = DHSCC_CRCDecoder(self._crc_encoder, dtype=dtype)
                self._k_crc = self._crc_decoder.encoder.crc_length
            except ImportError:
                # If CRC module is not available, proceed without CRC functionality
                self._k_crc = 0
                self._use_crc = False
                if return_crc_status:
                    print("Warning: CRC module not found. CRC status will not be returned.")
                    return_crc_status = False
        else:
            self._use_crc = False
            self._k_crc = 0

        if self._k < self._k_crc:
            raise ValueError("Value of k is too small for given CRC_degree.")

        if (crc_degree is None) and return_crc_status:
            self._return_crc_status = False
            raise ValueError("Returning CRC status requires given crc_degree.")
        else:
            self._return_crc_status = return_crc_status

        # Store the inverse interleaver pattern
        if ind_iil_inv is not None:
            if len(ind_iil_inv) != self._k:
                raise ValueError("ind_iil_inv must have k elements.")
            self._ind_iil_inv = ind_iil_inv
            self._iil = True
        else:
            self._iil = False
            self._ind_iil_inv = None

        # Register buffers for moving to different devices
        self.register_buffer('frozen_ind_tensor', torch.tensor(self._frozen_ind, dtype=self._dtype))
        self.register_buffer('info_pos_tensor', torch.tensor(self._info_pos, dtype=torch.int64))

    #########################################
    # Public methods and properties
    #########################################

    @property
    def n(self):
        """Codeword length"""
        return self._n

    @property
    def k(self):
        """Number of information bits"""
        return self._k

    @property
    def k_crc(self):
        """Number of CRC bits"""
        return self._k_crc

    @property
    def frozen_pos(self):
        """Frozen positions for Polar decoding"""
        return self._frozen_pos

    @property
    def info_pos(self):
        """Information bit positions for Polar encoding"""
        return self._info_pos

    @property
    def llr_max(self):
        """Maximum LLR value for internal calculations"""
        return self._llr_max

    @property
    def list_size(self):
        """List size for SCL decoding"""
        return self._list_size

    #######################
    # Helper functions
    #######################

    def _update_rate0_code(self, msg_pm, msg_uhat, msg_llr, cw_ind):
        """Update rate-0 sub-code (i.e., all frozen) at pos ``cw_ind``.

        See eq. (26) in [Hashemi_SSCL].
        """
        n = len(cw_ind)
        stage_ind = int(np.log2(n))
        device = msg_llr.device

        # Get LLRs for the sub-code
        cw_ind_tensor = torch.tensor(cw_ind, dtype=torch.long, device=device)
        llr = torch.index_select(msg_llr[:, :, stage_ind, :], 2, cw_ind_tensor)
        llr_in = torch.clamp(llr, min=-self._llr_max, max=self._llr_max)

        # Update path metric for complete sub-block of length n
        pm_val = torch.nn.functional.softplus(-1.*llr_in)
        msg_pm += torch.sum(pm_val, dim=-1)

        return msg_pm, msg_uhat, msg_llr

    def _update_rep_code(self, msg_pm, msg_uhat, msg_llr, cw_ind):
        """Update rep. code (i.e., only rightmost bit is non-frozen)
        sub-code at position ``cw_ind``.

        See Eq. (31) in [Hashemi_SSCL].
        """
        n = len(cw_ind)
        stage_ind = int(np.log2(n))
        device = msg_llr.device

        # Convert indices to tensors for PyTorch indexing
        cw_ind_tensor = torch.tensor(cw_ind, dtype=torch.long, device=device)

        # Get LLRs for the sub-code
        llr = torch.index_select(msg_llr[:, :, stage_ind, :], 2, cw_ind_tensor)
        llr_in = torch.clamp(llr, min=-self._llr_max, max=self._llr_max)

        # Upper branch has negative llr values (bit is 1)
        llr_low = llr_in[:, :self._list_size, :]
        llr_up = -llr_in[:, self._list_size:, :]
        llr_pm = torch.cat([llr_low, llr_up], 1)
        pm_val = torch.nn.functional.softplus(-1.*llr_pm)
        msg_pm += torch.sum(pm_val, dim=-1)

        # Update bits for upper path (all 1s)
        msg_uhat1 = msg_uhat[:, :self._list_size, :, :]

        # Extract parts of the message
        if cw_ind[0] > 0:
            msg_uhat21 = msg_uhat[:, self._list_size:, stage_ind, :cw_ind[0]].unsqueeze(2)
        else:
            msg_uhat21 = torch.empty((msg_uhat.shape[0], self._list_size, 1, 0),
                                     dtype=self._dtype, device=device)

        if cw_ind[-1] < self._n - 1:
            msg_uhat22 = msg_uhat[:, self._list_size:, stage_ind, cw_ind[-1]+1:].unsqueeze(2)
        else:
            msg_uhat22 = torch.empty((msg_uhat.shape[0], self._list_size, 1, 0),
                                     dtype=self._dtype, device=device)

        # Create ones to insert
        msg_ones = torch.ones([msg_uhat.shape[0], self._list_size, 1, n],
                            dtype=self._dtype, device=device)

        # Combine parts with ones
        msg_uhat23 = torch.cat([msg_uhat21, msg_ones, msg_uhat22], 3)
        msg_uhat24_1 = msg_uhat[:, self._list_size:, :stage_ind, :]
        msg_uhat24_2 = msg_uhat[:, self._list_size:, stage_ind+1:, :]

        msg_uhat2 = torch.cat([msg_uhat24_1, msg_uhat23, msg_uhat24_2], 2)
        msg_uhat = torch.cat([msg_uhat1, msg_uhat2], 1)

        # Branch last bit and update pm at pos cw_ind[-1]
        msg_uhat = self._update_single_bit([cw_ind[-1]], msg_uhat)

        # Sort and duplicate paths
        msg_pm, msg_uhat, msg_llr = self._sort_decoders(msg_pm, msg_uhat, msg_llr)
        msg_uhat, msg_llr, msg_pm = self._duplicate_paths(msg_uhat, msg_llr, msg_pm)

        return msg_pm, msg_uhat, msg_llr

    def _update_single_bit(self, ind_u, msg_uhat):
        """Update single bit at position ``ind_u`` for all decoders.
        """
        # Position is non-frozen
        if self._frozen_ind[ind_u[0]] == 0:
            # Set upper half paths to have bit value 1 at this position
            msg_uhat[:, self._list_size:, 0, ind_u[0]] = 1

        return msg_uhat

    def _update_pm(self, ind_u, msg_uhat, msg_llr, msg_pm):
        """Update path metric of all decoders after updating bit_pos ``ind_u``.

        We implement (10) from [Stimming_LLR].
        """
        try:
            # Extract relevant tensors
            u_hat = msg_uhat[:, :, 0, ind_u[0]]
            llr = msg_llr[:, :, 0, ind_u[0]]

            # Check if dimensions match
            if u_hat.shape != llr.shape:
                print(f"Dimension mismatch in _update_pm: u_hat shape {u_hat.shape}, llr shape {llr.shape}")
                # Reshape to make compatible
                if llr.dim() > u_hat.dim():
                    u_hat = u_hat.unsqueeze(-1).expand_as(llr)
                elif u_hat.dim() > llr.dim():
                    llr = llr.unsqueeze(-1).expand_as(u_hat)

            llr_in = torch.clamp(llr, min=-self._llr_max, max=self._llr_max)

            # Ensure u_hat and llr_in have compatible dimensions
            if u_hat.dim() != llr_in.dim():
                print(f"Adjusting dimensions: u_hat {u_hat.shape}, llr_in {llr_in.shape}")
                if u_hat.dim() > llr_in.dim():
                    llr_in = llr_in.unsqueeze(-1).expand_as(u_hat)
                else:
                    u_hat = u_hat.unsqueeze(-1).expand_as(llr_in)

            # Numerically more stable implementation of log(1 + exp(-x))
            msg_pm += torch.nn.functional.softplus(-torch.mul((1 - 2*u_hat), llr_in))
            return msg_pm

        except Exception as e:
            print(f"Error in _update_pm: {e}")
            print(f"Shapes - u_hat: {u_hat.shape if 'u_hat' in locals() else 'unknown'}, llr: {llr.shape if 'llr' in locals() else 'unknown'}")
            # Return unchanged path metrics to allow continued decoding
            return msg_pm

    def _sort_decoders(self, msg_pm, msg_uhat, msg_llr):
        """Sort decoders according to their path metric."""
        try:
            # Get sorting indices
            _, ind = torch.sort(msg_pm, dim=1)

            # Prepare batch indices for gathering
            batch_size = msg_pm.shape[0]
            batch_indices = torch.arange(batch_size, device=msg_pm.device).unsqueeze(1)

            # Sort path metrics
            msg_pm_sorted = torch.gather(msg_pm, 1, ind)

            # Try advanced indexing for sorting decoder states, but with error handling
            try:
                # Create expanded indices safely
                try:
                    batch_indices_expanded = batch_indices.unsqueeze(2).unsqueeze(3)
                    ind_expanded = ind.unsqueeze(2).unsqueeze(3)

                    # Check dimensions before expansion
                    if batch_indices_expanded.dim() != 4 or ind_expanded.dim() != 4:
                        raise ValueError(f"Expanded indices have incorrect dimensions: batch_indices_expanded {batch_indices_expanded.shape}, ind_expanded {ind_expanded.shape}")

                    # Expand safely to match msg_uhat dimensions
                    batch_indices_expanded = batch_indices_expanded.expand(-1, -1, msg_uhat.shape[2], msg_uhat.shape[3])
                    ind_expanded = ind_expanded.expand(-1, -1, msg_uhat.shape[2], msg_uhat.shape[3])

                except Exception as e:
                    print(f"Error creating expanded indices: {e}")
                    # Create flat index versions as fallback
                    batch_indices_flat = batch_indices.repeat(1, msg_uhat.shape[2] * msg_uhat.shape[3]).flatten()
                    ind_flat = ind.repeat(1, msg_uhat.shape[2] * msg_uhat.shape[3]).flatten()

                # Use the expanded indices to gather elements
                try:
                    # Use proper advanced indexing
                    msg_uhat_sorted = msg_uhat[
                        batch_indices_expanded, ind_expanded,
                        torch.arange(msg_uhat.shape[2], device=msg_uhat.device).view(1, 1, -1, 1),
                        torch.arange(msg_uhat.shape[3], device=msg_uhat.device).view(1, 1, 1, -1)
                    ]

                    msg_llr_sorted = msg_llr[
                        batch_indices_expanded, ind_expanded,
                        torch.arange(msg_llr.shape[2], device=msg_llr.device).view(1, 1, -1, 1),
                        torch.arange(msg_llr.shape[3], device=msg_llr.device).view(1, 1, 1, -1)
                    ]

                except Exception as e:
                    print(f"Error in advanced indexing: {e}")
                    # Fallback to simpler but slower loop-based gathering
                    msg_uhat_sorted = torch.zeros_like(msg_uhat)
                    msg_llr_sorted = torch.zeros_like(msg_llr)

                    # Manual gather using loops
                    for b in range(batch_size):
                        for i, idx in enumerate(ind[b]):
                            msg_uhat_sorted[b, i] = msg_uhat[b, idx]
                            msg_llr_sorted[b, i] = msg_llr[b, idx]

            except Exception as e:
                print(f"Error in tensor sorting: {e}")
                # As a last resort, don't sort at all
                msg_uhat_sorted = msg_uhat
                msg_llr_sorted = msg_llr

            return msg_pm_sorted, msg_uhat_sorted, msg_llr_sorted

        except Exception as e:
            print(f"Critical error in _sort_decoders: {e}")
            print(traceback.format_exc())
            # Return original tensors if sorting fails completely
            return msg_pm, msg_uhat, msg_llr

    def _cn_op(self, x, y):
        """Check-node update (boxplus) for LLR inputs.

        Operations are performed element-wise using the numerically stable min-sum
        approximation with a correction factor.
        """
        # Clamp values for numerical stability
        x_in = torch.clamp(x, min=-self._llr_max, max=self._llr_max)
        y_in = torch.clamp(y, min=-self._llr_max, max=self._llr_max)
        
        # Use sign-magnitude formulation for numerical stability
        x_abs = torch.abs(x_in)
        y_abs = torch.abs(y_in)
        sign = torch.sign(x_in) * torch.sign(y_in)
        
        # Min-sum approximation with correction factor (0.9375 = 15/16)
        min_val = torch.minimum(x_abs, y_abs)
        correction = 0.9375
        
        # Special case handling for very small values
        llr_out = sign * torch.min(
            self._llr_max * torch.ones_like(min_val),
            correction * min_val
        )
        
        return llr_out

    def _vn_op(self, x, y, u_hat):
        """Variable node update for LLR inputs.

        Operations are performed element-wise.
        """
        return torch.mul((1 - 2*u_hat), x) + y

    def _duplicate_paths(self, msg_uhat, msg_llr, msg_pm):
        """Duplicate paths by copying the upper branch into the lower one.
        """
        msg_uhat = torch.cat([msg_uhat[:, :self._list_size, :, :],
                            msg_uhat[:, :self._list_size, :, :]], dim=1)
        msg_llr = torch.cat([msg_llr[:, :self._list_size, :, :],
                           msg_llr[:, :self._list_size, :, :]], dim=1)
        msg_pm = torch.cat([msg_pm[:, :self._list_size],
                          msg_pm[:, :self._list_size]], dim=1)

        return msg_uhat, msg_llr, msg_pm

    def _update_left_branch(self, msg_llr, stage_ind, cw_ind_left, cw_ind_right):
        """Update messages of left branch.
        """
        # Get device from input tensor
        device = msg_llr.device

        # First, handle the case when stage_ind is 0, which would cause issues
        if stage_ind <= 0:
            return msg_llr

        # Convert indices to tensors for PyTorch indexing
        cw_ind_left_tensor = torch.tensor(cw_ind_left, dtype=torch.long, device=device)
        cw_ind_right_tensor = torch.tensor(cw_ind_right, dtype=torch.long, device=device)

        # Get LLRs for left and right branches
        llr_left_in = torch.index_select(msg_llr[:, :, stage_ind, :],
                                        dim=2, index=cw_ind_left_tensor)

        llr_right_in = torch.index_select(msg_llr[:, :, stage_ind, :],
                                        dim=2, index=cw_ind_right_tensor)

        # Apply check node operation
        llr_left_out = self._cn_op(llr_left_in, llr_right_in)

        # Make a full copy of the stage_ind-1 layer to avoid dimension issues
        llr_s = msg_llr[:, :, stage_ind-1, :].clone()

        # Update only the left branch section with the calculated values
        for i, idx in enumerate(cw_ind_left):
            llr_s[:, :, idx] = llr_left_out[:, :, i]

        # Replace the stage_ind-1 layer with our updated version
        msg_llr = msg_llr.clone()
        msg_llr[:, :, stage_ind-1, :] = llr_s

        return msg_llr

    def _update_right_branch(self, msg_llr, msg_uhat, stage_ind, cw_ind_left, cw_ind_right):
        """Update messages for right branch.
        """
        # Get device from input tensor
        device = msg_llr.device

        # Convert indices to tensors for PyTorch indexing
        cw_ind_left_tensor = torch.tensor(cw_ind_left, dtype=torch.long, device=device)
        cw_ind_right_tensor = torch.tensor(cw_ind_right, dtype=torch.long, device=device)

        # Get bit decisions from left branch and LLRs for both branches
        u_hat_left_up = torch.index_select(
            msg_uhat[:, :, stage_ind-1, :], dim=2, index=cw_ind_left_tensor
        )

        llr_left_in = torch.index_select(
            msg_llr[:, :, stage_ind, :], dim=2, index=cw_ind_left_tensor
        )

        llr_right = torch.index_select(
            msg_llr[:, :, stage_ind, :], dim=2, index=cw_ind_right_tensor
        )

        # Apply variable node operation
        llr_right_out = self._vn_op(llr_left_in, llr_right, u_hat_left_up)

        # Get parts of the message before the update
        if cw_ind_left[0] > 0:
            indices_pre_left = torch.arange(0, cw_ind_left[0],
                                          dtype=torch.long, device=device)
            llr_left0 = torch.index_select(
                msg_llr[:, :, stage_ind-1, :], dim=2, index=indices_pre_left
            )
        else:
            llr_left0 = torch.empty((msg_llr.shape[0], msg_llr.shape[1], 0),
                                   dtype=msg_llr.dtype, device=device)

        # Get left branch values
        llr_left = torch.index_select(
            msg_llr[:, :, stage_ind-1, :], dim=2, index=cw_ind_left_tensor
        )

        # Get values after right branch
        if cw_ind_right[-1] < self._n - 1:
            indices_post_right = torch.arange(cw_ind_right[-1] + 1, self._n,
                                            dtype=torch.long, device=device)
            llr_right1 = torch.index_select(
                msg_llr[:, :, stage_ind-1, :], dim=2, index=indices_post_right
            )
        else:
            llr_right1 = torch.empty((msg_llr.shape[0], msg_llr.shape[1], 0),
                                    dtype=msg_llr.dtype, device=device)

        # Concatenate all parts with the new right branch LLRs
        llr_s = torch.cat([llr_left0, llr_left, llr_right_out, llr_right1], dim=2)
        llr_s = llr_s.unsqueeze(2)  # Add dimension

        # Update message LLRs
        msg_llr1 = msg_llr[:, :, 0:stage_ind-1, :]
        msg_llr2 = msg_llr[:, :, stage_ind:, :]
        msg_llr = torch.cat([msg_llr1, llr_s, msg_llr2], dim=2)

        return msg_llr

    def _update_branch_u(self, msg_uhat, stage_ind, cw_ind_left, cw_ind_right):
        """Update ``u_hat`` messages after executing both branches.
        """
        # Get device from input tensor
        device = msg_uhat.device

        # Convert indices to tensors for PyTorch indexing
        cw_ind_left_tensor = torch.tensor(cw_ind_left, dtype=torch.long, device=device)
        cw_ind_right_tensor = torch.tensor(cw_ind_right, dtype=torch.long, device=device)

        # Get bit decisions from both branches
        u_hat_left_up = torch.index_select(
            msg_uhat[:, :, stage_ind-1, :], dim=2, index=cw_ind_left_tensor
        )

        u_hat_right_up = torch.index_select(
            msg_uhat[:, :, stage_ind-1, :], dim=2, index=cw_ind_right_tensor
        )

        # Combine u_hat with XOR operation
        u_hat_left = torch.logical_xor(
            u_hat_left_up.to(torch.bool),
            u_hat_right_up.to(torch.bool)
        ).to(self._dtype)

        # Get parts of the message before the update
        if cw_ind_left[0] > 0:
            indices_pre_left = torch.arange(0, cw_ind_left[0],
                                          dtype=torch.long, device=device)
            u_hat_left_0 = torch.index_select(
                msg_uhat[:, :, stage_ind, :], dim=2, index=indices_pre_left
            )
        else:
            u_hat_left_0 = torch.empty((msg_uhat.shape[0], msg_uhat.shape[1], 0),
                                      dtype=msg_uhat.dtype, device=device)

        # Get parts of the message after the update
        if cw_ind_right[-1] < self._n - 1:
            indices_post_right = torch.arange(cw_ind_right[-1]+1, self._n,
                                            dtype=torch.long, device=device)
            u_hat_right_1 = torch.index_select(
                msg_uhat[:, :, stage_ind, :], dim=2, index=indices_post_right
            )
        else:
            u_hat_right_1 = torch.empty((msg_uhat.shape[0], msg_uhat.shape[1], 0),
                                       dtype=msg_uhat.dtype, device=device)

        # Concatenate all parts with the new bit decisions
        u_hat = torch.cat([u_hat_left_0, u_hat_left, u_hat_right_up, u_hat_right_1], dim=2)
        u_hat = u_hat.unsqueeze(2)  # Add dimension

        # Update message bit decisions
        msg_uhat1 = msg_uhat[:, :, 0:stage_ind, :]
        msg_uhat2 = msg_uhat[:, :, stage_ind+1:, :]
        msg_uhat = torch.cat([msg_uhat1, u_hat, msg_uhat2], dim=2)

        return msg_uhat

    def _polar_decode_scl(self, cw_ind, msg_uhat, msg_llr, msg_pm):
        """Recursive decoding function for SCL decoding.
        """
        try:
            # Current sub-code length and stage index (= tree depth)
            n = len(cw_ind)
            stage_ind = int(np.log2(n))

            # Safety check for tensor shapes
            if msg_uhat.ndim != 4:
                print(f"ERROR: Unexpected msg_uhat dimensions: {msg_uhat.shape}")
                # Try to recover by reshaping if possible
                batch_size = msg_uhat.shape[0]
                if isinstance(batch_size, int) and batch_size > 0:
                    print(f"Attempting to fix msg_uhat shape")
                    # Create a tensor with the expected shape
                    device = msg_uhat.device
                    dtype = msg_uhat.dtype
                    msg_uhat = torch.zeros([batch_size, 2*self._list_size, self._n_stages+1, self._n],
                                          dtype=dtype, device=device)

            # Recursively branch through decoding tree
            if n > 1:
                # Prune tree if rate-0 or rep code subcode is detected
                if self._use_fast_scl:
                    # Check if this is a rate-0 subcode (all frozen bits)
                    is_rate0 = all(self._frozen_ind[i] == 1 for i in cw_ind)
                    if is_rate0:
                        return self._update_rate0_code(msg_pm, msg_uhat, msg_llr, cw_ind)

                    # Check if this is a repetition code (only rightmost bit is info)
                    is_rep_code = (sum(self._frozen_ind[i] for i in cw_ind) == n-1 and
                                  self._frozen_ind[cw_ind[-1]] == 0)
                    if is_rep_code:
                        return self._update_rep_code(msg_pm, msg_uhat, msg_llr, cw_ind)

                # Split index into left and right part
                cw_ind_left = cw_ind[0:int(n/2)]
                cw_ind_right = cw_ind[int(n/2):]

                # ----- left branch -----
                try:
                    msg_llr = self._update_left_branch(msg_llr, stage_ind, cw_ind_left, cw_ind_right)
                except Exception as e:
                    print(f"Error in left branch update: {e}")
                    print(traceback.format_exc())
                    # If failed, try to continue with the original tensors

                # Call sub-graph decoder of left branch
                try:
                    msg_uhat, msg_llr, msg_pm = self._polar_decode_scl(
                        cw_ind_left, msg_uhat, msg_llr, msg_pm
                    )
                except Exception as e:
                    print(f"Error in left branch recursion: {e}")
                    print(traceback.format_exc())
                    # If left branch fails, we may need to bail out
                    raise

                # ----- right branch -----
                try:
                    msg_llr = self._update_right_branch(msg_llr, msg_uhat, stage_ind,
                                                   cw_ind_left, cw_ind_right)
                except Exception as e:
                    print(f"Error in right branch update: {e}")
                    print(traceback.format_exc())
                    # If failed, try to continue with the original tensors

                # Call sub-graph decoder of right branch
                try:
                    msg_uhat, msg_llr, msg_pm = self._polar_decode_scl(
                        cw_ind_right, msg_uhat, msg_llr, msg_pm
                    )
                except Exception as e:
                    print(f"Error in right branch recursion: {e}")
                    print(traceback.format_exc())
                    # If right branch fails, we may need to bail out
                    raise

                # Update uhat at current stage
                try:
                    msg_uhat = self._update_branch_u(msg_uhat, stage_ind, cw_ind_left, cw_ind_right)
                except Exception as e:
                    print(f"Error in branch_u update: {e}")
                    print(traceback.format_exc())
                    # If failed, try to continue with the original msg_uhat

            # If leaf is reached perform basic decoding op (=decision)
            else:
                try:
                    # Update bit value at current position
                    msg_uhat = self._update_single_bit(cw_ind, msg_uhat)

                    # Update PM
                    msg_pm = self._update_pm(cw_ind, msg_uhat, msg_llr, msg_pm)

                    # If non-frozen bit, sort and duplicate paths
                    if self._frozen_ind[cw_ind[0]] == 0:
                        msg_pm, msg_uhat, msg_llr = self._sort_decoders(msg_pm, msg_uhat, msg_llr)
                        msg_uhat, msg_llr, msg_pm = self._duplicate_paths(msg_uhat, msg_llr, msg_pm)
                except Exception as e:
                    print(f"Error in leaf node processing: {e}")
                    print(traceback.format_exc())
                    # If failed at leaf node, try to continue with original tensors

            # Check tensor dimensions before returning
            if msg_uhat.ndim != 4:
                print(f"WARNING: msg_uhat has incorrect dimensions {msg_uhat.shape} after stage {stage_ind}")
                # Try to restore expected dimensions if needed

            return msg_uhat, msg_llr, msg_pm

        except Exception as e:
            print(f"CRITICAL ERROR in _polar_decode_scl at stage {stage_ind if 'stage_ind' in locals() else 'unknown'}: {e}")
            print(traceback.format_exc())
            print(f"Input shapes - cw_ind: {len(cw_ind)}, msg_uhat: {msg_uhat.shape}, msg_llr: {msg_llr.shape}, msg_pm: {msg_pm.shape}")

            # Try to recover from the error by returning the input tensors
            # This may allow decoding to continue in some cases
            return msg_uhat, msg_llr, msg_pm

    def _decode_tf(self, llr_ch):
        """Main decoding function.

        Initializes memory and calls recursive decoding function.

        Args:
            llr_ch: Tensor containing the channel LLR values (as logits).

        Returns:
            List containing [msg_uhat, msg_pm]
        """
        try:
            # Get batch size and device
            batch_size = llr_ch.shape[0]
            device = llr_ch.device

            # Allocate memory for all 2*list_size decoders with correct dimensions
            msg_uhat = torch.zeros(
                [batch_size, 2*self._list_size, self._n_stages+1, self._n],
                dtype=self._dtype, device=device
            )

            msg_llr = torch.zeros(
                [batch_size, 2*self._list_size, self._n_stages+1, self._n],
                dtype=self._dtype, device=device
            )

            # Init all 2*L decoders with same llr_ch - ensure proper dimensions
            llr_ch_expanded = llr_ch.reshape(batch_size, 1, 1, self._n)
            llr_ch_expanded = llr_ch_expanded.expand(batch_size, 2*self._list_size, 1, self._n)

            # Init last stage with llr_ch
            msg_llr[:, :, self._n_stages, :] = llr_ch_expanded.squeeze(2)

            # Init path metrics - first decoder in each group has zero cost, others have high penalty
            pm0 = torch.zeros([batch_size, 1], dtype=self._dtype, device=device)
            pm1 = self._llr_max * torch.ones([batch_size, self._list_size-1],
                                            dtype=self._dtype, device=device)
            msg_pm = torch.cat([pm0, pm1, pm0, pm1], dim=1)

            # Verify all tensors are on the same device
            if msg_uhat.device != msg_llr.device or msg_llr.device != msg_pm.device:
                print(f"Warning: Device mismatch - moving all tensors to {device}")
                msg_uhat = msg_uhat.to(device)
                msg_llr = msg_llr.to(device)
                msg_pm = msg_pm.to(device)

            # Verify shapes to detect potential problems early

            # Call recursive graph function with error handling
            try:
                msg_uhat, msg_llr, msg_pm = self._polar_decode_scl(
                    self._cw_ind, msg_uhat, msg_llr, msg_pm
                )
            except Exception as e:
                print(f"Error in recursive decoding: {e}")
                print(traceback.format_exc())
                # Try to recover and continue - may need to reshape tensors
                if msg_uhat.ndim != 4:
                    print(f"Fixing msg_uhat shape from {msg_uhat.shape}")
                    msg_uhat = torch.zeros(
                        [batch_size, 2*self._list_size, self._n_stages+1, self._n],
                        dtype=self._dtype, device=device
                    )

            # Sort output with graceful fallback
            try:
                msg_pm, msg_uhat, msg_llr = self._sort_decoders(msg_pm, msg_uhat, msg_llr)
            except Exception as e:
                print(f"Sorting error: {e} - continuing with unsorted paths")

            return [msg_uhat, msg_pm]

        except Exception as e:
            # If everything fails, create minimal valid output to avoid crashes
            print(f"Critical error in _decode_tf: {e}")
            print(traceback.format_exc())
            batch_size = llr_ch.shape[0]
            device = llr_ch.device

            # Create minimal valid tensors
            msg_uhat = torch.zeros(
                [batch_size, 2*self._list_size, self._n_stages+1, self._n],
                dtype=self._dtype, device=device
            )
            msg_pm = torch.zeros([batch_size, 2*self._list_size], dtype=self._dtype, device=device)

            return [msg_uhat, msg_pm]

    def forward(self, llr_ch):
        """Successive cancellation list (SCL) decoding function.

        This function performs successive cancellation list decoding
        and returns the estimated information bits.

        An outer CRC can be applied optionally by setting ``crc_degree``.

        Args:
            llr_ch (torch.Tensor): Tensor of shape `[...,n]` containing the
                channel LLR values (as logits).

        Returns:
            torch.Tensor: Tensor of shape `[...,k]` containing
            hard-decided estimations of all k information bits.
            torch.Tensor (optional): CRC status if return_crc_status is True.
        """
        # Save original input shape
        input_shape = llr_ch.shape

        # Reshape inputs to [-1, n]
        llr_ch = llr_ch.reshape(-1, self._n)
        batch_size = llr_ch.shape[0]

        # Convert logits to LLRs
        llr_ch = -1.0 * llr_ch

        # Run main decoder
        try:
            # Call TF-style decoder (returns [msg_uhat, msg_pm])
            decoder_outputs = self._decode_tf(llr_ch)
            msg_uhat, msg_pm = decoder_outputs

            # Check CRC (and remove CRC parity bits)
            if self._use_crc:
                # Extract info bits with CRC for each path
                u_hat_list = torch.index_select(
                    msg_uhat[:, :, 0, :], dim=2,
                    index=torch.tensor(self._info_pos, device=llr_ch.device)
                )

                # Undo input bit interleaving if needed
                if self._iil and self._ind_iil_inv is not None:
                    ind_iil_inv_tensor = torch.tensor(self._ind_iil_inv,
                                                     dtype=torch.long,
                                                     device=llr_ch.device)
                    u_hat_list_crc = torch.index_select(u_hat_list, dim=2, index=ind_iil_inv_tensor)
                else:
                    u_hat_list_crc = u_hat_list

                # Verify CRC for each path
                crc_valid = torch.zeros((batch_size, 2*self._list_size),
                                       dtype=torch.bool, device=llr_ch.device)

                if hasattr(self, '_crc_decoder'):
                    for i in range(batch_size):
                        for j in range(2*self._list_size):
                            # Check CRC for each candidate
                            _, crc_status = self._crc_decoder(u_hat_list_crc[i, j:j+1])
                            crc_valid[i, j] = crc_status[0]

                    # Add penalty to pm if CRC fails
                    pm_penalty = ((1.0 - crc_valid.to(self._dtype)) * self._llr_max * self.k)
                    msg_pm += pm_penalty

            # Select most likely candidate
            cand_ind = torch.argmin(msg_pm, dim=1)

            # Gather the selected candidate for each batch item
            batch_indices = torch.arange(batch_size, device=llr_ch.device)
            c_hat = msg_uhat[batch_indices, cand_ind, 0, :]

            # Extract information bits
            u_hat = c_hat[:, self._info_pos]

            # Reshape output to match input shape
            output_shape = list(input_shape)
            output_shape[-1] = self.k
            u_hat = u_hat.reshape(output_shape)

            if self._return_crc_status and self._use_crc and hasattr(self, '_crc_decoder'):
                # Return CRC status alongside the decoded bits
                crc_status = torch.gather(crc_valid, dim=1, index=cand_ind.unsqueeze(1)).squeeze(1)
                status_shape = output_shape[:-1]  # Remove last dimension
                crc_status_reshaped = crc_status.reshape(status_shape)
                return u_hat, crc_status_reshaped
            else:
                return u_hat

        except Exception as e:
            # Fall back to SC decoding if SCL fails
            print(f"Error in SCL decoding: {str(e)}")
            print("Falling back to SC decoding")
            sc_decoder = PolarSCDecoder(self._frozen_pos, self._n, dtype=self._dtype)
            return sc_decoder(llr_ch.reshape(input_shape))

    def __call__(self, llr_ch):
        """Alias for the forward method."""
        return self.forward(llr_ch)

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


# =============================================================================
# 5G Polar Code Support - Based on polar_encoding.py and polar_decoding.py
# =============================================================================

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
    csv_path = os.path.join(os.path.dirname(__file__), "polar_5G.csv")
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

class Polar5GDecoder(torch.nn.Module):
    """Wrapper for 5G compliant decoding including rate-recovery and CRC removal.

    Parameters
    ----------
    enc_polar: Polar5GEncoder
        Instance of the Polar5GEncoder used for encoding including rate-matching.

    dec_type: str, default="SC"
        Defining the decoder to be used. Must be one of {"SC", "SCL"}.

    list_size: int, default=8
        Defining the list size if list-decoding is used.
        Only required for dec_type "SCL".

    return_crc_status: bool, default=False
        If True, the decoder additionally returns the CRC status indicating
        if a codeword was (most likely) correctly recovered.

    dtype: torch.dtype, default=torch.float32
        Data type to use for internal calculations and outputs.

    Input
    -----
    llr_ch: [...,n], torch.Tensor
        Tensor containing the channel LLRs.

    Output
    ------
    b_hat : [...,k], torch.Tensor
        Binary tensor containing hard-decided estimations of all k
        information bits.

    crc_status : [...], torch.bool
        CRC status indicating if a codeword was (most likely) correctly
        recovered. This is only returned if return_crc_status is True.
        Note that false positives are possible.

    Note
    ----
    This block supports the uplink and downlink Polar rate-matching scheme
    without codeword segmentation.

    Although the decoding list size is not provided by 3GPP, the consortium
    has agreed on a list size of 8 for the 5G decoding reference curves.

    All list-decoders apply CRC-aided decoding, however, the non-list
    decoders (SC) cannot materialize the CRC leading to an effective rate-loss.
    """

    def __init__(self,
                 enc_polar,
                 dec_type="SC",
                 list_size=8,
                 return_crc_status=False,
                 dtype=torch.float32):
        super().__init__()

        try:
            # Try to import from the same module since we have it here
            pass  # Polar5GEncoder is defined in this same file
        except ImportError:
            # Fallback for relative import
            pass  # Polar5GEncoder is defined in this same file

        # Check if enc_polar is a Polar5GEncoder instance
        if not hasattr(enc_polar, 'n_target') or not hasattr(enc_polar, 'k_target'):
            raise TypeError("enc_polar must be Polar5GEncoder.")
        if not isinstance(dec_type, str):
            raise TypeError("dec_type must be str.")
        if dec_type not in ["SC", "SCL"]:
            raise ValueError("dec_type must be one of {'SC', 'SCL'}")

        # Store internal attributes
        self._n_target = enc_polar.n_target
        self._k_target = enc_polar.k_target
        self._n_polar = enc_polar.n_polar
        self._k_polar = enc_polar.k_polar
        self._k_crc = enc_polar.enc_crc.crc_length
        self._bil = enc_polar._channel_type == "uplink"
        self._iil = enc_polar._channel_type == "downlink"
        self._llr_max = 100  # Internal max LLR value (for punctured positions)
        self._enc_polar = enc_polar
        self._dec_type = dec_type
        self._dtype = dtype

        # Initialize the de-interleaver patterns
        self._init_interleavers()

        # Initialize decoder based on type
        if dec_type == "SC":
            if return_crc_status:
                print("Warning: 5G Polar codes use an integrated CRC that "
                    "cannot be materialized with SC decoding and, thus, "
                    "causes a degraded performance. Please consider SCL "
                    "decoding instead.")
            self._polar_dec = PolarSCDecoder(self._enc_polar.frozen_pos,
                                             self._n_polar,
                                             dtype=dtype)
        elif dec_type == "SCL":
            self._polar_dec = DHSCC_PolarSCLDecoder(
                self._enc_polar.frozen_pos,
                self._n_polar,
                crc_degree=self._enc_polar.enc_crc.crc_degree,
                list_size=list_size,
                ind_iil_inv=self.ind_iil_inv,
                return_crc_status=return_crc_status,
                dtype=dtype)

        if not isinstance(return_crc_status, bool):
            raise TypeError("return_crc_status must be bool.")

        self._return_crc_status = return_crc_status

        # Init CRC decoder - we'll need this if SCL doesn't return CRC status
        if self._return_crc_status and dec_type != "SCL":
            self._dec_crc = CRCDecoder(self._enc_polar.enc_crc, dtype=dtype)

        # Register tensors as buffers
        self.register_buffer('ind_ch_int_inv_tensor', torch.tensor(self.ind_ch_int_inv, dtype=torch.int64))
        self.register_buffer('ind_sub_int_inv_tensor', torch.tensor(self.ind_sub_int_inv, dtype=torch.int64))
        if self._iil and self.ind_iil_inv is not None:
            self.register_buffer('ind_iil_inv_tensor', torch.tensor(self.ind_iil_inv, dtype=torch.int64))

    ###############################
    # Public methods and properties
    ###############################

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
        """Number of information bits of mother Polar code"""
        return self._k_polar

    @property
    def n_polar(self):
        """Codeword length of mother Polar code"""
        return self._n_polar

    @property
    def frozen_pos(self):
        """Frozen positions for Polar decoding"""
        return self._enc_polar.frozen_pos

    @property
    def info_pos(self):
        """Information bit positions for Polar encoding"""
        return self._enc_polar.info_pos

    @property
    def llr_max(self):
        """Maximum LLR value for internal calculations"""
        return self._llr_max

    @property
    def dec_type(self):
        """Decoder type used for decoding as str"""
        return self._dec_type

    @property
    def polar_dec(self):
        """Decoder instance used for decoding"""
        return self._polar_dec

    #################
    # Utility methods
    #################

    def _init_interleavers(self):
        """Initialize inverse interleaver patterns for rate-recovery."""

        # Channel interleaver
        ind_ch_int = self._enc_polar.channel_interleaver(
                                                np.arange(self._n_target))
        self.ind_ch_int_inv = np.argsort(ind_ch_int)  # Find inverse perm

        # Sub-block interleaver
        ind_sub_int = self._enc_polar.subblock_interleaving(
                                                np.arange(self._n_polar))
        self.ind_sub_int_inv = np.argsort(ind_sub_int)  # Find inverse perm

        # input bit interleaver
        if self._iil:
            self.ind_iil_inv = np.argsort(self._enc_polar.input_interleaver(
                                                np.arange(self._k_polar)))
        else:
            self.ind_iil_inv = None

    def forward(self, llr_ch):
        """Polar decoding and rate-recovery for 5G Polar codes.

        Args:
            llr_ch (torch.Tensor): Tensor of shape [...,n] containing the
                channel LLR values.

        Returns:
            torch.Tensor: Tensor of shape [...,k] containing
                hard-decided estimates of all k information bits.
            torch.Tensor (optional): CRC status if return_crc_status=True
        """
        # Ensure input is a torch tensor
        if not isinstance(llr_ch, torch.Tensor):
            llr_ch = torch.tensor(llr_ch, dtype=self._dtype)

        device = llr_ch.device

        # Reshape inputs to [batch_size, n_target]
        input_shape = llr_ch.shape
        batch_size = np.prod(input_shape[:-1]).astype(int) if len(input_shape) > 1 else 1
        llr_ch_reshaped = llr_ch.reshape(batch_size, self._n_target)

        # 1.) Undo channel interleaving
        if self._bil:
            ind_ch_int_inv_tensor = self.ind_ch_int_inv_tensor.to(device)
            llr_deint = torch.index_select(llr_ch_reshaped, dim=1, index=ind_ch_int_inv_tensor)
        else:
            llr_deint = llr_ch_reshaped

        # 2.) Remove puncturing, shortening, repetition (see Sec. 5.4.1.2)
        # a) Puncturing: set LLRs to 0
        # b) Shortening: set LLRs to infinity
        # c) Repetition: combine LLRs
        if self._n_target >= self._n_polar:
            # Repetition coding
            n_rep = self._n_target - self._n_polar
            llr_1 = llr_deint[:, :n_rep]
            llr_2 = llr_deint[:, n_rep:self._n_polar]
            llr_3 = llr_deint[:, self._n_polar:]
            llr_dematched = torch.cat([llr_1 + llr_3, llr_2], dim=1)
        else:
            if self._k_polar / self._n_target <= 7/16:
                # Puncturing
                llr_zero = torch.zeros([batch_size, self._n_polar - self._n_target],
                                      device=device, dtype=self._dtype)
                llr_dematched = torch.cat([llr_zero, llr_deint], dim=1)
            else:
                # Shortening
                llr_infty = -self._llr_max * torch.ones([batch_size, self._n_polar - self._n_target],
                                                      device=device, dtype=self._dtype)
                llr_dematched = torch.cat([llr_deint, llr_infty], dim=1)

        # 3.) Remove subblock interleaving
        ind_sub_int_inv_tensor = self.ind_sub_int_inv_tensor.to(device)
        llr_dec = torch.index_select(llr_dematched, dim=1, index=ind_sub_int_inv_tensor)

        # 4.) Run main decoder
        if self._dec_type == "SCL" and self._return_crc_status:
            # SCL decoder can directly return CRC status if requested
            u_hat_crc, crc_status = self._polar_dec(llr_dec)
        else:
            # SC decoder or SCL without CRC status
            u_hat_crc = self._polar_dec(llr_dec)

        # 5.) Shortening should be implicitly recovered by decoder

        # 6.) Remove input bit interleaving for downlink channels only
        if self._iil:
            ind_iil_inv_tensor = self.ind_iil_inv_tensor.to(device)
            u_hat_crc = torch.index_select(u_hat_crc, dim=1, index=ind_iil_inv_tensor)

        # 7.) Evaluate or remove CRC (and PC)
        if self._return_crc_status and self._dec_type != "SCL":
            # Use CRC decoder to verify and remove CRC bits
            u_hat, crc_status = self._dec_crc(u_hat_crc)
        else:
            # Just remove CRC bits (for SC decoder or if CRC status not needed)
            u_hat = u_hat_crc[:, :-self._k_crc] if self._k_crc > 0 else u_hat_crc

        # And reconstruct input shape
        output_shape = list(input_shape)
        output_shape[-1] = self._k_target
        u_hat_reshape = u_hat.reshape(output_shape)

        if self._return_crc_status:
            # Reconstruct CRC shape
            status_shape = output_shape[:-1]  # Remove last dimension
            crc_status_reshaped = crc_status.reshape(status_shape)
            return u_hat_reshape, crc_status_reshaped
        else:
            return u_hat_reshape

    def __call__(self, llr_ch):
        """Alias for the forward method"""
        return self.forward(llr_ch)
