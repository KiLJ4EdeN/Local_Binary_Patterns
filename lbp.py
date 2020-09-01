# lbp calculation for an image.
# example usage: 
# gray = cv2.imread('image.jpg', cv2.IMREAD_GRAY)
# descriptor = LocalBinaryPatterns(24, 8)
# lbp = descriptor.describe(gray)


from skimage.feature import local_binary_pattern
import numpy as np

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius size.
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = local_binary_pattern(image, self.numPoints,
                                   self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist
    
def calc_lbp(image):
    """
    Calculate a local binary pattern image for a given photo.
    :param image: image to find the hist on
    """

    # convert image to gray.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # tensor allocation.
    lbp_image = np.zeros_like(gray_image)
    kernel_size = 3
    for ih in range(0, image.shape[0] - kernel_size):
        for iw in range(0, image.shape[1] - kernel_size):

            # move the kernel along the image.
            img = gray_image[ih:ih + kernel_size, iw:iw + kernel_size]

            # simple filter that only leaves out the ones bigger than the center pixel.
            center = img[1, 1]
            # 3 * 3, processed kernel.
            filtered_kernel = (img >= center) * 1.0

            flat_kernel = filtered_kernel.T.flatten()
            # it is ok to order counterclock manner
            # img01_vector = img01.flatten()

            # remove the center, e.g the 5th element of a size 9 kernel (3 * 3).
            flat_kernel = np.delete(flat_kernel, 4)
            # example: [1. 0. 0. 1. 0. 1. 1. 1.]

            non_zero_locations = np.where(flat_kernel)[0]
            if len(non_zero_locations) >= 1:
                num = np.sum(2 ** non_zero_locations)
            else:
                num = 0
            # adjust the center value.
            lbp_image[ih + 1, iw + 1] = num
    return lbp_image
