import numpy as np
import viridis_colormap as viridis_map

"""
class for mapping values to colormap,
For now it can handle HSV and VIRIDIS
"""

class CustomImg:

    # Init funtion colormap should be str HSV OR VIRIDIS
    def __init__(self):
        # Pre-convert colormap to uint8 (source is int64, values 0-254)
        self.colors = np.array(viridis_map.var, dtype=np.uint8)
    # Map colors - returns uint8 directly
    def map_colors(self,img) :
        return self.colors[img.astype(np.intp)]
     # Prepare custome image
    def get_new_img(self,img) :
        return self.map_colors(img)
"""
Class for making power values to custom indexes
like normalization
"""

class NormalizePowerValue:
    def __init__(self,step_size=.5):
        self.step_size = step_size
    def get_normalized_values(self,img):
        # for now lets make range from -120 to -20 dbm with .5 dbm step
        # Use single output array to minimize memory copies
        out = np.clip(img, -130, -3)
        np.round(out / self.step_size, out=out)
        out *= self.step_size
        out += 130
        out /= self.step_size
        return out
