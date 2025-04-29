import math
import numpy as np
from numba import njit, prange

from numpy.typing import NDArray


class GaussianFilterOperator:
    """numpy / cupy Gaussian filter operator"""

    def __init__(self, sigma: float | NDArray, **kwargs):
        """init method

        Parameters
        ----------
        sigma: float | array
            standard deviation of the gaussian filter
        **kwargs : sometype
            passed to the ndimage gaussian_filter function
        """
        super().__init__()
        self._sigma = sigma
        self._kwargs = kwargs

    def __call__(self, x: NDArray) -> NDArray:

        if isinstance(x, np.ndarray):
            import scipy.ndimage as ndimage

            return ndimage.gaussian_filter(x, sigma=self._sigma, **self._kwargs)
        else:
            import cupyx.scipy.ndimage as ndimagex

            return ndimagex.gaussian_filter(x, sigma=self._sigma, **self._kwargs)

    def adjoint(self, y: NDArray) -> NDArray:
        return self.__call__(y)


@njit(parallel=True)
def nearest_neighbors_3d(img, s, ninds):
    """Calculate the n nearest neighbors for all voxels in a 3D array

    Parameters
    ----------
    img : 3d numpy array
      containing the image

    s : 3d binary (uint) numpy array
      containing the neighborhood definition.
      1 -> voxel is in neighborhood
      0 -> voxel is not in neighnorhood
      The dimensions of s have to be odd.

    ninds: 2d numpy array used for output
      of shape (xp.prod(img.shape), nnearest).
      ninds[i,:] contains the indicies of the nearest neighbors of voxel i

    Note
    ----
    All voxel indices are "flattened". It is assumed that the numpy arrays
    are in 'C' order.
    """

    nnearest = ninds.shape[1]

    offsets = np.array(s.shape) // 2
    maxdiff = img.max() - img.min() + 1

    d12 = img.shape[1] * img.shape[2]
    d2 = img.shape[2]

    for i0 in prange(img.shape[0]):
        for i1 in range(img.shape[1]):
            for i2 in range(img.shape[2]):

                absdiff = np.zeros(s.shape)
                val = img[i0, i1, i2]

                i_flattened = np.zeros(s.shape, dtype=ninds.dtype)

                for j0 in range(s.shape[0]):
                    for j1 in range(s.shape[1]):
                        for j2 in range(s.shape[2]):
                            tmp0 = i0 + j0 - offsets[0]
                            tmp1 = i1 + j1 - offsets[1]
                            tmp2 = i2 + j2 - offsets[2]

                            i_flattened[j0, j1, j2] = tmp0 * d12 + tmp1 * d2 + tmp2

                            if (
                                (tmp0 >= 0)
                                and (tmp0 < img.shape[0])
                                and (tmp1 >= 0)
                                and (tmp1 < img.shape[1])
                                and (tmp2 >= 0)
                                and (tmp2 < img.shape[2])
                                and s[j0, j1, j2] == 1
                            ):
                                absdiff[j0, j1, j2] = np.abs(
                                    img[tmp0, tmp1, tmp2] - val
                                )
                            else:
                                absdiff[j0, j1, j2] = maxdiff

                vox = i_flattened[offsets[0], offsets[1], offsets[2]]
                ninds[vox, :] = i_flattened.flatten()[
                    np.argsort(absdiff.flatten())[:nnearest]
                ]


class BowsherGradient:
    """Bowsher gradient operator using n nearest neighbors"""

    def __init__(
        self,
        structural_image: NDArray,
        neighborhood: NDArray,
        num_nearest: int,
        nearest_neighbor_inds: None | NDArray = None,
    ) -> None:
        """init method

        Parameters
        ----------
        structural_image : NDArray
            the structural image
        neighborhood : NDArray
            the neighborhood definition image (1-> included, 0 -> excluded)
        num_nearest : int
            number of nearest neighbors used in gradient
        nearest_neighbor_inds : None | NDArray, optional
            lookup table of nearest neighbor indices, by default None
            if None, gets calculated from structural_image and neighborhood
        """

        self._structural_image = structural_image
        self._neighborhood = neighborhood
        self._num_nearest = num_nearest

        if isinstance(structural_image, np.ndarray):
            self._xp = np
            from scipy.sparse import csc_matrix

            self._csc_matrix = csc_matrix
        else:
            import cupy as cp

            self._xp = cp
            from cupyx.scipy.sparse import csc_matrix

            self._csc_matrix = csc_matrix

        if nearest_neighbor_inds is None:
            self._nearest_neighbor_inds = np.zeros(
                (math.prod(structural_image.shape), num_nearest), dtype=int
            )

            if isinstance(structural_image, np.ndarray):
                nearest_neighbors_3d(
                    self._structural_image,
                    self._neighborhood,
                    self._nearest_neighbor_inds,
                )
            else:
                nearest_neighbors_3d(
                    self._xp.asnumpy(self._structural_image),
                    self._xp.asnumpy(self._neighborhood),
                    self._xp.asnumpy(self._nearest_neighbor_inds),
                )
                self._nearest_neighbor_inds = self._xp.asarray(
                    self._nearest_neighbor_inds
                )

        else:
            self._nearest_neighbor_inds = nearest_neighbor_inds

        num_voxels = structural_image.size
        tmp = self._xp.arange(num_voxels, dtype=float)
        diag = self._csc_matrix(
            (self._xp.full(num_voxels, -1, dtype=float), (tmp, tmp)),
            shape=(num_voxels, num_voxels),
        )

        self._sparse_fwd_diff_matrices = []

        for i in range(num_nearest):
            off_diag = self._csc_matrix(
                (
                    self._xp.full(num_voxels, 1, dtype=float),
                    (tmp, self._nearest_neighbor_inds[:, i]),
                ),
                shape=(num_voxels, num_voxels),
            )

            self._sparse_fwd_diff_matrices.append(diag + off_diag)

        self._in_shape = self._structural_image.shape
        self._out_shape = (self._num_nearest,) + self._structural_image.shape

    @property
    def in_shape(self) -> tuple:
        return self._in_shape

    @property
    def out_shape(self) -> tuple:
        return self._out_shape

    @property
    def structural_image(self) -> NDArray:
        return self._structural_image

    @property
    def neighborhood(self) -> NDArray:
        return self._neighborhood

    @property
    def num_nearest(self) -> int:
        return self._num_nearest

    @property
    def nearest_neighbor_inds(self) -> NDArray:
        return self._nearest_neighbor_inds

    @property
    def sparse_fwd_diff_matrices(self) -> list:
        return self._sparse_fwd_diff_matrices

    def __call__(self, x: NDArray) -> NDArray:
        y = self._xp.zeros(tuple(self._out_shape), dtype=x.dtype)

        for i in range(self.num_nearest):
            y[i, ...] = self._xp.reshape(
                self.sparse_fwd_diff_matrices[i] @ x.ravel(), self._in_shape
            )

        return y

    def adjoint(self, y: NDArray) -> NDArray:
        x = self._xp.zeros(tuple(self._in_shape), dtype=y.dtype)

        for i in range(self.num_nearest):
            x += self._xp.reshape(
                self.sparse_fwd_diff_matrices[i].T @ y[i, ...].ravel(), self._in_shape
            )

        return x


if __name__ == "__main__":

    import cupy as xp

    xp.random.seed(0)
    st_img = xp.random.rand(101, 102, 102)

    neighb = xp.ones((3, 3, 3), dtype=int)
    neighb[1, 1, 1] = 0

    bow_grad = BowsherGradient(st_img, neighb, num_nearest=5)

    x = xp.random.rand(*st_img.shape)
    x_fwd = bow_grad(x)

    y = xp.random.rand(*x_fwd.shape)
    y_back = bow_grad.adjoint(y)

    assert xp.isclose(float((x * y_back).sum()), float((x_fwd * y).sum()))

    # power iterations to estimate norm of (bow_grad)^T * bow_grad
    for i in range(50):
        x = bow_grad.adjoint(bow_grad(x))
        norm = float(xp.linalg.norm(x))
        x /= norm
        if i % 10 == 0:
            print(i, norm)

    sm_op = GaussianFilterOperator(sigma=2.5)

    x_fwd_sm = sm_op(x)
    y_sm = xp.random.rand(*x_fwd_sm.shape)
    y_back_sm = sm_op.adjoint(y_sm)
    assert xp.isclose(float((x * y_back_sm).sum()), float((x_fwd_sm * y_sm).sum()))

    print()
    x = xp.random.rand(*st_img.shape)
    beta = 0.1
    for i in range(50):
        x = beta * bow_grad.adjoint(bow_grad(x)) + sm_op.adjoint(sm_op(x))
        norm = float(xp.linalg.norm(x))
        x /= norm
        if i % 10 == 0:
            print(i, norm)
