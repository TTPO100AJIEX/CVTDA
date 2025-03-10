import numpy
import joblib

import skimage.color

import cvtda.logging


def rgb2gray(images: numpy.ndarray, n_jobs: int = -1) -> numpy.ndarray:
    return numpy.stack(
        joblib.Parallel(n_jobs = n_jobs)(
            joblib.delayed(skimage.color.rgb2gray)(img)
            for img in cvtda.logging.logger().pbar(images, desc = "rgb2gray")
        )
    )
