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
    # allocation.
    lbp_image = np.zeros_like(gray_image)
    kernel_size = 3
    for ih in range(0, image.shape[0] - kernel_size):
        for iw in range(0, image.shape[1] - kernel_size):

            # move the kernel along the image.
            img = gray_image[ih:ih + kernel_size, iw:iw + kernel_size]
            center = img[1, 1]
            img01 = (img >= center) * 1.0
            img01_vector = img01.T.flatten()
            # it is ok to order counterclock manner
            # img01_vector = img01.flatten()

            # what?
            img01_vector = np.delete(img01_vector, 4)

            # what?
            where_img01_vector = np.where(img01_vector)[0]
            if len(where_img01_vector) >= 1:
                num = np.sum(2 ** where_img01_vector)
            else:
                num = 0
            lbp_image[ih + 1, iw + 1] = num
    return lbp_image
