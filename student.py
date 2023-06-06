import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
import util_sweep


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo (or mean albedo for RGB input images) for a pixel is less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights. All N images have the same dimension.
                  The images can either be all RGB (height x width x 3), or all grayscale (height x width x 1).

    Output:
        albedo -- float32 image. When the input 'images' are RGB, it should be of dimension height x width x 3,
                  while in the case of grayscale 'images', the dimension should be height x width x 1.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    stacked_images = np.stack(images, axis=0)  # Stack images along the first dimension

    inv_lights_product = np.linalg.inv(np.matmul(lights.T, lights))
    lights_images_product = np.matmul(lights.T, stacked_images.reshape(len(images), -1))

    G = np.matmul(inv_lights_product, lights_images_product).reshape(lights.T.shape[0], *stacked_images.shape[1:])
    diffuse_coeff = np.linalg.norm(G, axis=0)
    albedo = np.where(diffuse_coeff < 1e-7, 0, diffuse_coeff)

    avg_diffuse_coeff = np.linalg.norm(np.mean(G, axis=3), axis=0, keepdims=True)
    avg_diffuse_coeff = np.maximum(avg_diffuse_coeff, 1e-7)
    avg_diffuse_coeff_broadcasted = np.broadcast_to(avg_diffuse_coeff, G.shape[:-1])

    normals = np.mean(G, axis=3) / avg_diffuse_coeff_broadcasted
    normals[avg_diffuse_coeff_broadcasted < 1e-7] = 0
    normals = np.transpose(normals, (1, 2, 0))

    return albedo, normals


def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    projection_matrix = K.dot(Rt)
    height, width = points.shape[:2]

    points_4d = np.concatenate((points, np.ones((height, width, 1))), axis=-1)

    homogenous_pts = np.einsum('ij,hwj->hwi', projection_matrix, points_4d)

    projections = homogenous_pts[..., :2] / homogenous_pts[..., 2, np.newaxis]

    return projections


def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc_impl' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc_impl' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region; assumed to be odd
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    height, width, channels = image.shape
    mid_kernel = ncc_size // 2 #int div to get midpoint of kernel

    # use strides for faster computation
    strides = image.strides
    window_shape = (ncc_size, ncc_size, channels)
    window_strides = (strides[0], strides[1], strides[2])
    strided_windows = np.lib.stride_tricks.as_strided(
        image,
        shape=(height - 2 * mid_kernel, width - 2 * mid_kernel) + window_shape,
        strides=window_strides[:2] * 2 + window_strides[2:],
    )

    # calc mean, subtract it, and flatten the windows
    mean_windows = np.mean(strided_windows, axis=(2, 3), keepdims=True)
    windows_centered = strided_windows - mean_windows
    windows_flattened = windows_centered.transpose(0, 1, 4, 2, 3).reshape(height - 2 * mid_kernel, width - 2 * mid_kernel, -1)

    # normalize the windows
    norm = np.linalg.norm(windows_flattened, axis=2, keepdims=True)
    norm[norm < 1e-6] = 1
    normalized = windows_flattened / norm

    # make sure patches that extend out of bounds have vectors set to 0
    output = np.zeros((height, width, ncc_size**2 * channels), dtype=np.float32)
    output[mid_kernel:height - mid_kernel, mid_kernel:width - mid_kernel] = normalized

    return output

def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc_impl.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    ncc = np.sum(image1 * image2, axis=-1)
    return ncc
