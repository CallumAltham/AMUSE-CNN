from collections import OrderedDict
import matplotlib.patches as mpatches
from tqdm import tqdm
import numpy as np
from abc import ABC

lc_class_properties = {
    1: OrderedDict({'Classname': 'Asphalt', 'Classvalue': 1, 'RED': 0, 'GREEN': 0, 'BLUE': 0}),
    2: OrderedDict({'Classname': 'Water', 'Classvalue': 2, 'RED': 0, 'GREEN': 0, 'BLUE': 255}),
    3: OrderedDict({'Classname': 'Woodland', 'Classvalue': 3, 'RED': 0, 'GREEN': 112, 'BLUE': 19}),
    4: OrderedDict({'Classname': 'Grassland', 'Classvalue': 4, 'RED': 18, 'GREEN': 227, 'BLUE': 53}),
    5: OrderedDict({'Classname': 'Metal Roof', 'Classvalue': 5, 'RED': 130, 'GREEN': 148, 'BLUE': 130}),
    6: OrderedDict({'Classname': 'Clay Roof', 'Classvalue': 6, 'RED': 176, 'GREEN': 94, 'BLUE': 35}),
    7: OrderedDict({'Classname': 'Concrete Roof', 'Classvalue': 7, 'RED': 212, 'GREEN': 205, 'BLUE': 152}),
    8: OrderedDict({'Classname': 'Bare Soil', 'Classvalue': 8, 'RED': 232, 'GREEN': 232, 'BLUE': 90})
}

lu_class_properties = {
    1: OrderedDict({'Classname': 'Road', 'Classvalue': 1, 'RED': 50, 'GREEN': 50, 'BLUE': 50}),
    2: OrderedDict({'Classname': 'Bridges', 'Classvalue': 2, 'RED': 130, 'GREEN': 130, 'BLUE': 130}),
    3: OrderedDict({'Classname': 'Urban Areas', 'Classvalue': 3, 'RED': 255, 'GREEN': 0, 'BLUE': 0}),
    4: OrderedDict({'Classname': 'Urban Vegetation', 'Classvalue': 4, 'RED': 0, 'GREEN': 255, 'BLUE': 0}),
    5: OrderedDict({'Classname': 'Bare Land', 'Classvalue': 5, 'RED': 255, 'GREEN': 137, 'BLUE': 102}),
    6: OrderedDict({'Classname': 'Water Body', 'Classvalue': 6, 'RED': 0, 'GREEN': 197, 'BLUE': 255}),
    7: OrderedDict({'Classname': 'Installations', 'Classvalue': 7, 'RED': 0, 'GREEN': 0, 'BLUE': 0}),
    8: OrderedDict({'Classname': 'Hydrophytes/Debris', 'Classvalue': 8, 'RED': 42, 'GREEN': 131, 'BLUE': 65})
}

lu_classes_used = [1, 3, 4, 5, 6]


class LandClasses(ABC):
    def classes_array_to_rgb(self, array):
        output = np.zeros(array.shape + (3,))
        for class_idx in self.class_properties:
            output[array == int(class_idx)] = self.get_class_rgb(class_idx)
        return output.astype(int)

    def get_class_rgb(self, class_idx, no_data=255, norm=False):
        if class_idx == -1:
            if norm:
                return no_data / 255, no_data / 255, no_data / 255
            else:
                return no_data, no_data, no_data
        else:
            class_info = self.class_properties[str(class_idx)]
            if norm:
                return [class_info["RED"] / 255, class_info["GREEN"] / 255, class_info["BLUE"] / 255]
            else:
                return [class_info["RED"], class_info["GREEN"], class_info["BLUE"]]

    def get_legend(self):
        return [mpatches.Patch(color=self.get_class_rgb(class_idx, norm=True),
                               label=self.class_properties[class_idx]['Classname'].replace('_', ' '))
                for class_idx in self.class_properties]


class LandCoverClasses(LandClasses):
    def __init__(self, class_data=None):
        if class_data is None:
            class_data = lc_class_properties
        self.class_properties = class_data

    @staticmethod
    def niceify_string(string):
        classname = string.replace('_', ' ')
        return ''.join(
            [word + '\n' if i < len(classname.split(' ')) - 1 and len(
                word) > 3 else word + ' ' if i < len(
                classname.split(' ')) - 1 else word for i, word in enumerate(classname.split(' '))]
        )


class LandUseClasses(LandClasses):
    def __init__(self, class_data=None):
        if class_data is None:
            class_data = lu_class_properties
        # self.class_properties = {a: class_data[a] for a in lu_classes_used}
        self.class_properties = class_data

    @staticmethod
    def niceify_string(string):
        return ''.join(
            [word + '\n' if i < len(string.split(' ')) - 1 and len(
                word) > 3 else word + ' ' if i < len(
                string.split(' ')) - 1 else word for i, word in enumerate(string.split(' '))]
        )
