from enum import Flag, auto, Enum

class MouseButton(Flag):
    LEFT   = auto()
    RIGHT  = auto()
    MIDDLE = auto()
    DOUBLE = auto()
    IMAGE  = auto()
    COORD  = auto()
    
    SHIFT_LEFT  = auto()
    SHIFT_RIGHT = auto()
    
    VAR_ALL = auto()
    VAR_TOP = auto()
    
    SPATIAL_LEFT  = auto()
    SPATIAL_RIGHT = auto()
    SPATIAL_ABOVE = auto()
    SPATIAL_BELOW = auto()