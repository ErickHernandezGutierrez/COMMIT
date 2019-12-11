from ctypes import *

set_globals = None

set_ic_lut = None
set_ec_lut = None
set_iso_lut = None

set_ic_data = None
set_ec_data = None
#set_iso_data = None
set_ic_data_transpose = None

free_data = None
check_cuda = None

multiply_Ax = None
multiply_Aty = None

def init():
    global set_globals

    global set_ic_lut
    global set_ec_lut
    global set_iso_lut

    global set_ic_data
    global set_ec_data
    #global set_iso_data
    global set_ic_data_transpose

    global free_data
    global check_cuda

    global multiply_Ax
    global multiply_Aty

    # setup C callable functions
    dll = CDLL('operator_withCUDA.so', mode=RTLD_GLOBAL)
    set_globals = dll.set_globals
    set_globals.argtypes = [c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint]

    set_ic_lut  = dll.set_ic_lut
    set_ic_lut.argtypes = [POINTER(c_float)]

    set_ec_lut  = dll.set_ec_lut
    set_ec_lut.argtypes = [POINTER(c_float)]

    set_iso_lut = dll.set_iso_lut
    set_iso_lut.argtypes = [POINTER(c_float)]

    set_ic_data = dll.set_ic_data
    set_ic_data.argtypes = [POINTER(c_uint), POINTER(c_uint), POINTER(c_ushort), POINTER(c_float)]

    set_ec_data = dll.set_ec_data
    set_ec_data.argtypes = [POINTER(c_uint), POINTER(c_ushort)]

    set_ic_data_transpose = dll.set_ic_data_transpose
    set_ic_data_transpose.argtypes = [POINTER(c_uint), POINTER(c_uint), POINTER(c_ushort), POINTER(c_float)]

    """set_iso_data = dll.set_iso_data
    set_iso_data.argtypes = [] #"""

    free_data = dll.free_data

    check_cuda = dll.check_cuda
    check_cuda.restype = c_int

    multiply_Ax = dll.multiply_Ax
    multiply_Ax.argtypes = [POINTER(c_double), POINTER(c_double)]

    multiply_Aty = dll.multiply_Aty
    multiply_Aty.argtypes = [POINTER(c_double), POINTER(c_double)]