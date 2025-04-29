import numpy as np
import matplotlib.pyplot as plt

# if we have a CUDA GPU, use cupy (fast), else use numpy (slow)
try:
    import cupy as xp
except ImportError:
    print("cupy not found, using numpy instead")
    xp = np

from operators import BowsherGradient, GaussianFilterOperator

# %%
# input parameters
sm_fwhm_mm = 5.0  # resolution of our reconstructed input PET image in mm
num_iter = 100  # number of iterations for the optimization (100 is a good start)
beta = 3e-2  # regularization parameter (to be tuned based on noise level)
num_nearest = 5  # number of nearest neighbors for the Bowsher gradient operator (to be tunes, sth between 3 - 15)

track_cost = True

# simulation parameters
seed = 1
noise_level = 0.5


# %%
# load the brainweb phantom

with xp.load("brainweb54.npz") as data:
    phantom = data["volume"]
    voxelsize = data["voxelsize"]

sm_op = GaussianFilterOperator(sigma=sm_fwhm_mm / (2.35 * voxelsize))

# %% generate PET ground truth image based on phantom

# LUT for pet GT values (1 CSF, 2 GM, 3 WM)
lut = xp.zeros(12, dtype=float)
lut[1] = 0.0
lut[2] = 4.0
lut[3] = 1.0

pet_gt = lut[phantom]

# %%
# blur PET and add noise (unrealistic uncorrelated Gaussian noise for testing)
pet_gt_sm = sm_op(pet_gt)
xp.random.seed(seed)
pet = pet_gt_sm + noise_level * xp.random.randn(*pet_gt_sm.shape)

# %%
mr = phantom**0.5

##################################################################################
##################################################################################
# now we have a 3D test PET image in "pet", test MR image in "mr", and a
# gaussian conv. operator with known sigma (must be matched to scanner resolution)
# normally we would load the images from file ...
##################################################################################
##################################################################################

# %%
# setup the Bowsher gradient operator we need for structural guided denoising
# and deblurring

neigh = xp.ones((5, 5, 5), dtype=int)
neigh[2, 2, 2] = 0

bow_grad_op = BowsherGradient(mr, neighborhood=neigh, num_nearest=num_nearest)

# %%
# estimate the Lipschitz constant of the gradient of the cost function
# this is ||G^T G + beta B^T B|| <= ||G^T G|| + beta ||B^T B||
# the norm of the Gaussian smoothing operator is 1, norm of B^T B is ca 4*num_nearest

L = 1.5 * float(1 + beta * 4 * num_nearest)


def cost_function(x):
    r = sm_op(x) - pet
    bg = bow_grad_op(x)
    return 0.5 * float(xp.sum(r * r) + 0.5 * beta * xp.sum(bg * bg))


def cost_function_gradient(x):
    return sm_op.adjoint(sm_op(x) - pet) + beta * bow_grad_op.adjoint(bow_grad_op(x))


# %%
# run Nesterov's accelerated gradient descent to minimize the cost function

x = pet.copy()
y = x.copy()
t = 1.0

step_size = 1.0 / L

cost = np.zeros(num_iter, dtype=float)

for i in range(num_iter):
    if track_cost:
        cost[i] = cost_function(x)
        print(f"{i:03} {cost[i]:.4E}", end="\r")
    xnew = x - step_size * cost_function_gradient(x)
    xnew = xp.clip(xnew, 0, None)

    tnew = 0.5 * (1 + xp.sqrt(1 + 4 * t * t))
    ynew = xnew + (t - 1) / tnew * (xnew - x)

    x = xnew
    y = ynew
    t = tnew

print()

# %%
# show results

if not isinstance(x, np.ndarray):
    # move cuda arrays to CPU for matplotlib plots
    x = xp.asnumpy(x)
    pet = xp.asnumpy(pet)
    pet_gt = xp.asnumpy(pet_gt)
    mr = xp.asnumpy(mr)

# loglog plot of cost function
if track_cost:
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 3), layout="constrained")
    ax1.loglog(cost, "r-", label="cost function")
    ax1.set_xlim(1, None)
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("cost function")
    ax1.set_title("Cost function")
    fig1.show()


vmax = 1.2 * float(pet_gt.max())

fig2, ax2 = plt.subplots(2, 4, figsize=(12, 6), layout="constrained")
ax2[0, 0].imshow(pet_gt[80, :, :], cmap="Greys", origin="lower", vmin=0, vmax=vmax)
ax2[0, 1].imshow(pet[80, :, :], cmap="Greys", origin="lower", vmin=0, vmax=vmax)
ax2[0, 2].imshow(x[80, :, :], cmap="Greys", origin="lower", vmin=0, vmax=vmax)
ax2[0, 3].imshow(mr[80, :, :], cmap="Greys_r", origin="lower")

ax2[1, 0].imshow(pet_gt[:, 100, :], cmap="Greys", origin="lower", vmin=0, vmax=vmax)
ax2[1, 1].imshow(pet[:, 100, :], cmap="Greys", origin="lower", vmin=0, vmax=vmax)
ax2[1, 2].imshow(x[:, 100, :], cmap="Greys", origin="lower", vmin=0, vmax=vmax)
ax2[1, 3].imshow(mr[:, 100, :], cmap="Greys_r", origin="lower")

for axx in ax2.ravel():
    axx.set_xticks([])
    axx.set_yticks([])

ax2[0, 0].set_title("sim. PET GT", fontsize="medium")
ax2[0, 1].set_title("sim. blurry & noisy PET", fontsize="medium")
ax2[0, 2].set_title("denosied & deblurred PET", fontsize="medium")
ax2[0, 3].set_title("sim. MR", fontsize="medium")

fig2.show()

plt.show()
