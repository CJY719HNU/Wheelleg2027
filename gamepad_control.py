"""Input adapters. XInput does not require pygame; all commands are in SI units."""
from dataclasses import dataclass
import ctypes
import time

@dataclass
class Command:
    speed: float = 0.
    yaw_rate: float = 0.
    height_rate: float = 0.
    jump: bool = False
    stop: bool = False
    recover: bool = False
    connected: bool = True

def deadzone(value, threshold=7849):
    x=max(-32767,min(32767,int(value)))
    return 0. if abs(x)<=threshold else (1 if x>0 else -1)*(abs(x)-threshold)/(32767-threshold)

class Gamepad(ctypes.Structure):
    _fields_=[('buttons',ctypes.c_uint16),('lt',ctypes.c_uint8),('rt',ctypes.c_uint8),
              ('lx',ctypes.c_int16),('ly',ctypes.c_int16),('rx',ctypes.c_int16),('ry',ctypes.c_int16)]
class State(ctypes.Structure):
    _fields_=[('packet',ctypes.c_uint32),('pad',Gamepad)]

class XInput:
    """Poll at GUI rate. Hold LB to drive; disconnect/release LB commands zero velocity."""
    def __init__(self,index=0):
        self.index=index;self.previous=0;self.connected=False
        self.dll=None
        for name in ['xinput1_4.dll','xinput1_3.dll','xinput9_1_0.dll']:
            try:self.dll=ctypes.WinDLL(name);break
            except (OSError,AttributeError):pass
        if self.dll is None:raise RuntimeError('XInput is available only on Windows with an XInput runtime')
        self.dll.XInputGetState.argtypes=[ctypes.c_uint32,ctypes.POINTER(State)]
        self.dll.XInputGetState.restype=ctypes.c_uint32
    def decode(self,pad):
        edge=pad.buttons & ~self.previous;self.previous=pad.buttons
        enabled=bool(pad.buttons & 0x0100) # LB, deadman
        return Command(speed=2.*deadzone(pad.ly) if enabled else 0.,
                       yaw_rate=-.5*deadzone(pad.lx) if enabled else 0.,
                       height_rate=.02*((bool(pad.buttons & 1))-(bool(pad.buttons & 2))) if enabled else 0.,
                       jump=enabled and bool(edge & 0x1000),stop=bool(pad.buttons & 0x2000),
                       recover=bool(edge & 0x0010)) # A jump, B stop, START recovery
    def poll(self):
        state=State();ok=self.dll.XInputGetState(self.index,ctypes.byref(state))==0
        self.connected=ok
        if not ok:self.previous=0;return Command(connected=False)
        return self.decode(state.pad)
