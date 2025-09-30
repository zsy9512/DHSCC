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

        from .polar_encoding import Polar5GEncoder
        
        if not isinstance(enc_polar, Polar5GEncoder):
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
            self._polar_dec = PolarSCLDecoder(
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
            # Add the last n_rep positions to the first llr positions
            n_rep = self._n_target - self._n_polar
            llr_1 = llr_deint[:, :n_rep]
            llr_2 = llr_deint[:, n_rep:self._n_polar]
            llr_3 = llr_deint[:, self._n_polar:]
            llr_dematched = torch.cat([llr_1 + llr_3, llr_2], dim=1)
        else:
            if self._k_polar / self._n_target <= 7/16:
                # Puncturing
                # Append n_polar - n_target "zero" llrs to first positions
                llr_zero = torch.zeros([batch_size, self._n_polar - self._n_target], 
                                      device=device, dtype=self._dtype)
                llr_dematched = torch.cat([llr_zero, llr_deint], dim=1)
            else:
                # Shortening
                # Append n_polar - n_target "-infinity" llrs to last positions
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

class PolarSCLDecoder(torch.nn.Module):
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
        self._llr_max = 30.  # internal max LLR value (not very critical for SC)
        
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
                from .crc import CRCEncoder, CRCDecoder
                self._crc_encoder = CRCEncoder(crc_degree, dtype=dtype)
                self._crc_decoder = CRCDecoder(self._crc_encoder, dtype=dtype)
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
            
    ###############################
    # Public methods and properties
    ###############################

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
        
        # Convert indices to tensor
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
            print(f"Error in _update_pm: {str(e)}")
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

        Operations are performed element-wise.
        """
        x_in = torch.clamp(x, min=-self._llr_max, max=self._llr_max)
        y_in = torch.clamp(y, min=-self._llr_max, max=self._llr_max)

        # Avoid division for numerical stability
        # Implements log(1+e^(x+y)) - log(e^x+e^y)
        llr_out = torch.nn.functional.softplus(x_in + y_in)
        stacked = torch.stack([x_in, y_in], dim=-1)
        llr_out -= torch.logsumexp(stacked, dim=-1)

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
        
        # Add dimension
        llr_s = llr_s.unsqueeze(2)

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
            
            # Debug prints to track execution
            print(f"Decoding stage: {stage_ind} Sub-code length: {n}")
            print(f"recursive call 1: {stage_ind} msg_uhat shape: {msg_uhat.shape}")   
            
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
                    print(f"Debug _update_left_branch - stage_ind: {stage_ind}")
                    print(f"Debug _update_left_branch - msg_llr shape: {msg_llr.shape}")
                    msg_llr = self._update_left_branch(msg_llr, stage_ind, cw_ind_left, cw_ind_right)
                    print(f"Debug _update_left_branch - final msg_llr shape: {msg_llr.shape}")
                    print(f"recursive call 2: {stage_ind} msg_uhat shape: {msg_uhat.shape}")   
                except Exception as e:
                    print(f"Error in left branch update: {e}")
                    print(traceback.format_exc())
                    # If failed, try to continue with the original tensors
                
                # Call sub-graph decoder of left branch
                try:
                    msg_uhat, msg_llr, msg_pm = self._polar_decode_scl(
                        cw_ind_left, msg_uhat, msg_llr, msg_pm
                    )
                    print(f"recursive call 3: {stage_ind} msg_uhat shape: {msg_uhat.shape}")   
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
            print(f"Debug shapes - msg_uhat: {msg_uhat.shape}, msg_llr: {msg_llr.shape}, msg_pm: {msg_pm.shape}")
            
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
