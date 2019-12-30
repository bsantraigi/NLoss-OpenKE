import ctypes
import os

base_file = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "./release/RWR.so")
)
lib = ctypes.cdll.LoadLibrary(base_file)

lib.RUN()