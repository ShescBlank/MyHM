from collections import OrderedDict
from numba import float64 # , complex128, int32, int64
# from numba.types import unicode_type
from numba.experimental import jitclass

spec = OrderedDict()
spec['rho_0'] = float64
spec['kappa'] = float64
spec['int_grid'] = float64[:,:]
spec['surf_grid'] = float64[:,:]
spec['int_alpha'] = float64[:]
spec['int_beta'] = float64[:]
spec['int_w'] = float64[:]
spec['surf_w'] = float64[:]
spec['normals'] = float64[:,::1] # "C" ordered (contiguous arrays)
spec['alpha_g'] = float64[:]
spec['diff_alpha'] = float64[:]
spec['int_grad_alpha'] = float64[:,:]
# spec['dom'] = float64[:,:]
# spec['codom'] = float64[:,:]

# TODO: Cambiar algunos a complex128 cuando apliquemos atenuación!

@jitclass(spec)
class Info_VSIE:
    def __init__(
        self,
        rho_0,
        kappa,
        int_grid,
        surf_grid,
        int_alpha,
        int_beta,
        int_w,
        surf_w,
        normals,
        alpha_g,
        diff_alpha,
        int_grad_alpha
    ):
        self.rho_0 = rho_0
        self.kappa = kappa
        self.int_grid = int_grid
        self.surf_grid = surf_grid
        self.int_alpha = int_alpha
        self.int_beta = int_beta
        self.int_w = int_w
        self.surf_w = surf_w
        self.normals = normals
        self.alpha_g = alpha_g
        self.diff_alpha = diff_alpha
        self.int_grad_alpha = int_grad_alpha

        # self.dom = self.int_grid
        # self.codom = self.surf_grid

    # def set_dom(self, identifier):
    #     if identifier == "int":
    #         self.dom = self.int_grid
    #     elif identifier == "surf":
    #         self.dom = self.surf_grid
    #     else:
    #         raise TypeError("identifier must be 'int' or 'surf'")

    # def set_codom(self, identifier):
    #     if identifier == "int":
    #         self.codom = self.int_grid
    #     elif identifier == "surf":
    #         self.codom = self.surf_grid
    #     else:
    #         raise TypeError("identifier must be 'int' or 'surf'")