"""
Wireless Communication System: A modular package for wireless communication simulation.

This package provides modular tools for simulating wireless communication systems,
including transmitters, channels, receivers, and performance evaluation.
"""

__version__ = "0.1.0"
__author__ = "Shengyu Zhang"

# Import main classes for easy access
from .transmitter import WirelessTransmitter
from .channel import WirelessChannel
from .receiver import WirelessReceiver
from .system_integration import WirelessCommunicationSystem
from .interface import WirelessInterface, WirelessSender, WirelessReceiver as WirelessReceiverInterface, create_wireless_interface

__all__ = [
    'WirelessTransmitter',
    'WirelessChannel', 
    'WirelessReceiver',
    'WirelessCommunicationSystem',
    'WirelessInterface',
    'WirelessSender',
    'WirelessReceiverInterface',
    'create_wireless_interface'
]