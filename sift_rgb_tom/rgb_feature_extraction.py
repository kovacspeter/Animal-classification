from PIL import Image
import hyper_pars
import numpy as np


def get_rgb_features_from_image(image_path, points, patch_size=hyper_pars.RGB_PATCH_SIZE):
    assert (all([dim % 2 != 0 for dim in patch_size]))  # assert all patch dimensions are odd
    pixels = Image.open(image_path).load()
    features = np.ndarray(patch_size[0]*patch_size[1]*3)  # store RGB values of patch
    for x, y in points:
        idx = 0
        for x_offset in range(-(patch_size[0]-1)/2, (patch_size[0]-1)/2+1):
            for y_offset in range(-(patch_size[0]-1)/2, (patch_size[0]-1)/2+1):
                try:
                    features[idx:idx+3] = pixels[x+x_offset, y+y_offset]
                except IndexError:  # edge cases
                    features[idx:idx+3] = (0, 0, 0)
                idx += 3

        # normalize to [0, 1] interva;
        features /= 255
    return features

