"""
USRP TX Module

This module handles USRP hardware initialization and transmission operations.
Ported from dqpsk_system.py and tx_program.py.
"""

import uhd
import time
import threading
import numpy as np
from collections import deque
from .services import SenderService
from .dqpsk_modem import modulate_bits
def init_usrp(center_freq=900e6, samp_rate=1e6, tx_gain=40, clock_source="internal", time_source="internal", usrp_args="type=b200"):
    """
    Initialize USRP hardware for transmission with comprehensive setup.
    
    Args:
        center_freq (float): Center frequency in Hz
        samp_rate (float): Sampling rate in Hz
        tx_gain (float): TX gain in dB
        clock_source (str): Clock source ("internal" or "external")
        time_source (str): Time source ("internal" or "external")
        usrp_args (str): USRP device arguments
        
    Returns:
        tuple: (usrp, tx_streamer) - Initialized USRP object and TX streamer
    """
    try:
        # Initialize USRP
        usrp = uhd.usrp.MultiUSRP(usrp_args)

        # Configure parameters
        usrp.set_tx_freq(uhd.types.TuneRequest(center_freq))
        usrp.set_tx_gain(tx_gain)
        usrp.set_tx_rate(samp_rate)
        
        # Configure clock and timing
        usrp.set_clock_source(clock_source)
        usrp.set_time_source(time_source)

        pc_time_ns = time.time_ns()
        full_secs = pc_time_ns // 1000000000
        frac_secs = (pc_time_ns % 1000000000) / 1000000000.0
        uhd_time = uhd.types.TimeSpec(full_secs, frac_secs)
        usrp.set_time_now(uhd_time)
            
        print(f"USRP TX initialization complete: freq={center_freq/1e6:.1f}MHz, gain={tx_gain}dB")

        # Create TX stream
        tx_st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        tx_st_args.channels = [0]
        tx_streamer = usrp.get_tx_stream(tx_st_args)

        print("TX stream created successfully")
        
        return usrp, tx_streamer

    except Exception as e:
        print(f"USRP initialization failed: {e}")
        raise


class TXProgram:
    """
    Multi-threaded USRP transmission program with triple-array mechanism.
    
    Uses three arrays for thread-safe data flow:
    - data_frames: Accumulates generated data frames (producer thread)
    - qpsk_frames: Transmission buffer (timed copying from data_frames)
    - tx_frames: Hardware transmission buffer (consumer thread)
    """
    
    def __init__(self, usrp_args="type=b200", center_freq=900e6, samp_rate=1e6, tx_gain=40,
                 clock_source="internal", time_source="internal", 
                 model_path="checkpoints/direct_polar_training_20251009_210057",
                 copy_interval=1.0, batch_size=10, args=None):
        """
        Initialize TXProgram with USRP parameters and threading setup.
        
        Args:
            usrp_args (str): USRP device arguments
            center_freq (float): Center frequency in Hz
            samp_rate (float): Sampling rate in Hz
            tx_gain (float): TX gain in dB
            clock_source (str): Clock source
            time_source (str): Time source
            model_path (str): Path to trained models
            copy_interval (float): Time interval for copying data_frames to qpsk_frames (seconds)
            batch_size (int): Number of frames to process in each batch
            args: Command line arguments object
        """
        self.usrp_args = usrp_args
        self.center_freq = center_freq
        self.samp_rate = samp_rate
        self.tx_gain = tx_gain
        self.clock_source = clock_source
        self.time_source = time_source
        self.model_path = model_path
        self.copy_interval = copy_interval
        self.batch_size = batch_size
        self.args = args
        
        # Initialize sender service
        self.sender_service = SenderService(args) if args else None
        self.data_iterator = iter(self.sender_service.get_data_loader()) if self.sender_service else None
        
        # Threading components
        self.data_lock = threading.Lock()
        self.qpsk_lock = threading.Lock()
        self.tx_lock = threading.Lock()
        
        # Triple-array mechanism
        self.data_frames = deque()  # Accumulates generated data frames
        self.qpsk_frames = deque()  # Transmission buffer
        self.tx_frames = deque()    # Hardware transmission buffer
        
        # Control flags
        self.running = False
        self.data_thread = None
        self.tx_thread = None
        
        # Frame counter for frame index encoding
        self.frame_counter = 0
        
        # USRP components
        self.usrp = None
        self.tx_streamer = None
        
        print("TXProgram initialized with triple-array mechanism")
    
    def start(self):
        """Start the multi-threaded transmission program."""
        try:
            # Initialize USRP
            self.usrp, self.tx_streamer = init_usrp(
                center_freq=self.center_freq,
                samp_rate=self.samp_rate,
                tx_gain=self.tx_gain,
                clock_source=self.clock_source,
                time_source=self.time_source,
                usrp_args=self.usrp_args
            )
            
            # Set running flag
            self.running = True
            
            # Start threads
            self.data_thread = threading.Thread(target=self._data_generation_thread_func, daemon=True)
            self.tx_thread = threading.Thread(target=self._tx_hardware_thread_func, daemon=True)
            
            self.data_thread.start()
            self.tx_thread.start()
            
            print("TXProgram started - data generation and transmission threads active")
            
            # Keep main thread alive
            while self.running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("TXProgram interrupted by user")
            self.stop()
        except Exception as e:
            print(f"TXProgram failed to start: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop the transmission program and cleanup resources."""
        print("Stopping TXProgram...")
        self.running = False
        
        # Wait for threads to finish
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=2.0)
        if self.tx_thread and self.tx_thread.is_alive():
            self.tx_thread.join(timeout=2.0)
        
        # Cleanup USRP
        if self.tx_streamer:
            self.tx_streamer = None
        if self.usrp:
            self.usrp = None
            
        print("TXProgram stopped and resources cleaned up")
    
    def _data_generation_thread_func(self):
        """
        Data generation thread function.
        
        Sequentially generates data frames using SenderService.process_batch(),
        accumulates them in data_frames, and periodically copies to qpsk_frames.
        """
        print("Data generation thread started")
        
        try:
            while self.running:
                # Generate batch of data frames
                batch_frames = []
                for _ in range(self.batch_size):
                    if not self.running:
                        break
                    
                    # Get next data batch from data loader
                    try:
                        data, target = next(self.data_iterator)
                
                        #print(f"target classes in batch: {target.unique()}")
                    except StopIteration:
                        # Reset data iterator if exhausted
                        self.data_iterator = iter(self.sender_service.get_data_loader())
                        data, target = next(self.data_iterator)
                    # Process one frame through the neural network pipeline

                # Accumulate in data_frames
                with self.data_lock:
                    self.data_frames.extend(batch_frames)
                    if len(self.data_frames) >= 5:
                        self.data_frames.popleft()
                
                # Atomic copy to qpsk_frames
                with self.qpsk_lock:
                    self.qpsk_frames = self.data_frames.copy()
                
                # Small sleep to prevent busy waiting
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Data generation thread error: {e}")
            self.running = False
    
    def _copy_data_to_qpsk(self):
        """Copy accumulated data_frames to qpsk_frames for transmission."""
        with self.data_lock:
            if not self.data_frames:
                return
                
            # Directly copy fully modulated frames to qpsk_frames
            qpsk_batch = []
            while self.data_frames:
                tx_signal = self.data_frames.popleft()
                qpsk_batch.append(tx_signal)
            with self.qpsk_lock:
                self.qpsk_frames.extend(qpsk_batch)
    def _tx_hardware_thread_func(self):
        """
        Hardware transmission thread function.
        
        Sequentially copies frames from qpsk_frames to tx_frames and transmits via USRP.
        """
        print("Hardware transmission thread started")
        
        try:
            while self.running:
                # Check if there are frames to transmit
                with self.qpsk_lock:
                    if not self.qpsk_frames:
                        time.sleep(0.01)  # Wait for frames
                        continue
                    
                    # Get first frame (don't pop, like tx_program)
                    qpsk_frame = self.qpsk_frames[0]
                    
                with self.tx_lock:
                    self.tx_frames.append(qpsk_frame)
                
                # Transmit the frame
                self._transmit_frame(qpsk_frame)
                
        except Exception as e:
            print(f"Hardware transmission thread error: {e}")
            self.running = False
    
    def _transmit_frame(self, qpsk_frame):
        """
        Transmit a single QPSK frame via USRP hardware.
        Sends the same frame multiple times consecutively, then waits 0.3s for new data.
        
        Args:
            qpsk_frame (numpy.ndarray): QPSK modulated symbols to transmit
        """
        try:
            # Ensure frame is numpy array with correct dtype
            if not isinstance(qpsk_frame, np.ndarray):
                qpsk_frame = np.array(qpsk_frame, dtype=np.complex64)
            elif qpsk_frame.dtype != np.complex64:
                qpsk_frame = qpsk_frame.astype(np.complex64)
            
            # Reshape to 1D: (num_samples,) for UHD compatibility
            qpsk_frame = qpsk_frame.flatten()  # Flatten to 1D array
            
            #print(f"Transmitting frame with shape: {qpsk_frame.shape}, dtype: {qpsk_frame.dtype}")
            
            # Debug: Check if data is valid
            if np.any(np.isnan(qpsk_frame)) or np.any(np.isinf(qpsk_frame)):
                print("Warning: Frame contains NaN or Inf values")
            
            # Send the same frame multiple times consecutively (like tx_program.py)
            repeat_count = getattr(self.args, 'repeat_count', 5) if self.args else 5
            
            # Create TX metadata once (like tx_program.py)
            metadata = uhd.types.TXMetadata()
            
            for i in range(repeat_count):
                # Set burst flags
                metadata.start_of_burst = (i == 0)
                metadata.end_of_burst = (i == repeat_count - 1)
                
                # Send the data
                try:
                    num_samps_sent = self.tx_streamer.send(qpsk_frame, metadata, timeout=0.1)
                    #print(f"Transmission {i+1}/{repeat_count}: sent {num_samps_sent}/{len(qpsk_frame)} samples")
                except Exception as e:
                    print(f"Send failed on transmission {i+1}: {e}")
                    break
                
                # Reset start_of_burst after first transmission (like tx_program.py)
                if i == 0:
                    metadata.start_of_burst = False
            
            #print(f"Completed transmission of frame (repeated {repeat_count} times)")
            
            # Wait 0.3 seconds before checking for new data (like tx_program.py)
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error transmitting frame: {e}")
            raise