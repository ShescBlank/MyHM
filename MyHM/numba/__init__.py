from MyHM.numba.aca import ACAPP_numba, ACAPP_CCC_numba, ACAPlusPP_numba, ACAPlusPP_CCC_numba
from MyHM.numba.wrappers import wrapper_compression_numba, wrapper_compression_numba2
from MyHM.numba.wrappers import wrapper_numba_node3d
from MyHM.numba.utils import numba_dot

# Number of threads used in parallel implementation of dot product between Tree3D and vector:
N_THREADS_DOT = 8