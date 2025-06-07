import pyopencl as cl
import numpy as np

def create_context_and_queue():
    """
    Create an OpenCL context and command queue.
    
    Returns:
        tuple: (context, queue)
    """
    platforms = cl.get_platforms()
    if not platforms:
        raise RuntimeError("No OpenCL platforms found.")
    
    # Use the first platform and its first device
    platform = platforms[0]
    devices = platform.get_devices()
    if not devices:
        raise RuntimeError("No OpenCL devices found.")
    
    device = devices[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    
    return context, queue