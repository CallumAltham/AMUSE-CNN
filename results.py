from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from sklearn.metrics import roc_curve, auc
import numpy as np
from rasterio.mask import mask
from itertools import cycle

lines = ["-", "--", "-.", ":"]
line_cycle = cycle(lines)


class Figure:
    def show(self):
        self.fig.show()

    def save(self, path, dpi=300):
        self.fig.savefig(path, bbox_inches='tight', dpi=dpi)


class FigureArrayVisualise(Figure):
    def __init__(self, tif_array, inset=None, get_legend=None):
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(np.flipud(tif_array), origin='lower')
        self.ax.set_axis_off()
        if inset is not None:
            self.inset = inset
            self.inset.add_inset(self.ax)
            self.inset.axins.imshow(np.flipud(tif_array))
        plt.xticks([], [])
        plt.yticks([], [])
        if get_legend is not None:
            self.ax.legend(handles=get_legend(), loc='center left', bbox_to_anchor=(1, 0.5))


class FigurePolygonInputs(Figure):
    def __init__(self, tif, training_polygons, lc_object, numb_rows=3, seed=4):
        self.fig, self.ax = plt.subplots(numb_rows, len(lc_object.class_properties))
        np.random.seed(seed)

        for class_idx in lc_object.class_properties:
            class_polygons = [polygon for polygon in training_polygons if polygon.class_idx == int(class_idx)]
            np.random.shuffle(class_polygons)
            for i in range(numb_rows):
                cropped_array = np.rollaxis(
                    mask(tif, [class_polygons[i].geometry], crop=True, nodata=255)[0], 0, 3)
                self.ax[i, int(class_idx) - 1].imshow(cropped_array)
                self.ax[i, int(class_idx) - 1].set_axis_off()
                self.ax[i, int(class_idx) - 1].add_patch(
                    plt.Rectangle((-.5, -.5),
                                  cropped_array.shape[1], cropped_array.shape[0], fill=False,
                                  color=lc_object.get_class_rgb(class_idx, norm=True),
                                  linewidth=3))
                classname = lc_object.class_properties[class_idx]['Classname'].replace('_', ' ')
                classname = ''.join(
                    [word + '\n' if i < len(classname.split(' ')) - 1 and len(
                        word) > 3 else word + ' ' if i < len(
                        classname.split(' ')) - 1 else word for i, word in enumerate(classname.split(' '))]
                )
                self.ax[i, int(class_idx) - 1].set_title(classname, fontsize=6, rotation=10)


class FigurePolygonOutputs(Figure):
    def __init__(self, tif, mlp_object, validation_polygons, lc_object, random_seed=6):
        self.fig = plt.figure(figsize=(8, 6))
        gs1 = gridspec.GridSpec(2, len(lc_object.class_properties))
        gs1.update(wspace=0.3, hspace=0.0, right=0.8, top=0.8, bottom=0.3)  # set the spacing between axes.
        np.random.seed(random_seed)
        keys = list(validation_polygons.keys())
        np.random.shuffle(keys)
        for class_idx in lc_object.class_properties:
            for polygon_idx in keys:
                if validation_polygons[polygon_idx].class_idx == int(class_idx):
                    ax1 = plt.subplot(gs1[int(class_idx) - 1])
                    ax1.set_frame_on(False)
                    ax1.set_xticks([])
                    ax1.set_yticks([])
                    cropped_array = np.rollaxis(
                        mask(tif, [validation_polygons[polygon_idx].geometry], crop=True, nodata=255)[0], 0, 3)
                    ax1.imshow(cropped_array)
                    ax1.add_patch(
                        plt.Rectangle((-.5, -.5),
                                      cropped_array.shape[1], cropped_array.shape[0], fill=False,
                                      color=lc_object.get_class_rgb(class_idx, norm=True),
                                      linewidth=3))
                    ax1.set_title(lc_object.niceify_string(lc_object.class_properties[class_idx]['Classname']),
                                  fontsize=8, rotation=60)
                    ax2 = plt.subplot(gs1[int(class_idx) - 1 + len(lc_object.class_properties)])
                    ax2.imshow(lc_object.classes_array_to_rgb(mlp_object.predict(cropped_array, no_data=255)))
                    ax2.get_xaxis().tick_bottom()
                    ax2.get_yaxis().tick_left()
                    ax2.set_xticks([])
                    ax2.set_yticks([])
                    ax2.set_frame_on(True)
                    if int(class_idx) == 1:
                        ax1.set_ylabel(f'Labelled \n validation set\n polygon pixels', fontsize=12)
                        ax2.set_ylabel(f'Corresponding\n pixel-wise\n MLP classifications', fontsize=12)
                    if int(class_idx) == len(lc_object.class_properties):
                        ax2.legend(handles=lc_object.get_legend(), loc='center left', bbox_to_anchor=(1, 0.5))
                    break


class FigureTileOutputs(Figure):
    def __init__(self, tif_array, mlp_object, lc_object, rows=5, cols=2, padding=64, random_seed=4):
        np.random.seed(random_seed)
        self.fig, ax = plt.subplots(rows, 2 * cols)
        for col in range(cols):
            ax[0, col * 2].set_title('Image Region')
            ax[0, col * 2 + 1].set_title(f'MLP\n Classification')
            for row in range(rows):
                image_i = np.random.randint(tif_array.shape[0] * 0.4, tif_array.shape[0] - padding)
                image_j = np.random.randint(padding, tif_array.shape[1] - padding)
                image_window = tif_array[image_i - padding:image_i + padding, image_j - padding:image_j + padding, :]
                ax[row, col * 2].imshow(image_window)
                ax[row, col * 2].set_xticks([])
                ax[row, col * 2].set_yticks([])
                ax[row, col * 2 + 1].imshow(lc_object.classes_array_to_rgb(mlp_object.predict(image_window)))
                ax[row, col * 2 + 1].set_xticks([])
                ax[row, col * 2 + 1].set_yticks([])
        ax[int((rows - 1) / 2), cols * 2 - 1].legend(handles=lc_object.get_legend(),
                                                     loc='center left',
                                                     bbox_to_anchor=(1, 0.5))


class FigureInset:
    def __init__(self, x, y, width, height, magnification):
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)
        self.magnification = int(magnification)
        self.axins = None

    def add_inset(self, ax):
        self.axins = zoomed_inset_axes(ax, self.magnification, loc=2)  # zoom = 10
        self.axins.set_xlim(self.x, self.x + self.width)
        self.axins.set_ylim(self.y, self.y + self.height)
        mark_inset(ax, self.axins, loc1=1, loc2=3, fc="none", ec="0.2")


class FigureReceiverOperatingCharacteristic(Figure):
    def __init__(self, Y_predict_prob, Y_validate, class_properties, niceify_fn, get_rgb):
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_aspect('equal')
        self.ax.grid(which='both')
        ml = MultipleLocator(0.1)
        self.ax.xaxis.set_minor_locator(ml)
        self.ax.yaxis.set_minor_locator(ml)
        self.ax.set_aspect('equal')
        for i, class_idx in enumerate(class_properties):
            Y_true_binary = Y_validate.copy()
            Y_true_binary[np.where(Y_validate != int(class_idx))] = 0
            Y_true_binary[np.where(Y_validate == int(class_idx))] = 1
            Y_pred_binary = Y_predict_prob[:, i]
            fpr, tpr, thresholds = roc_curve(Y_true_binary, Y_pred_binary)
            classname = class_properties[class_idx]['Classname']
            classname = niceify_fn(classname)
            self.ax.plot(fpr, tpr, color=get_rgb(class_idx, norm=True), linestyle=next(line_cycle),
                         label=classname + ' AUC: ' + str(np.round(auc(fpr, tpr), 3)))
        self.ax.set_xlabel('False positive rate')
        self.ax.set_ylabel('True positive rate', rotation=90)
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        self.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


class FigureJDL:
    def __init__(self, geometry):

        probs = lc_p_container.get_many_p(get_shape_pixel_idxs(geometry))
        probs = probs.reshape(int(probs.shape[0] ** 0.5), -1, NUMB_LC_CLASSES)
        fig, ax = plt.subplots(NUMB_LC_CLASSES + 1)
        print(len(ax))
        for i in range(NUMB_LC_CLASSES + 1):
            ax[i].set_axis_off()
            ax[i].set_position(
                [0.4, 1 - (i + 1) / (1 + NUMB_LC_CLASSES), 1 / (1 + NUMB_LC_CLASSES), 1 / (1 + NUMB_LC_CLASSES)])
            if i == 0:
                ax[i].imshow(np.rollaxis(TIF.read(), 0, 3)[imin:imax, jmin:jmax])
            else:
                ax[i].imshow(probs[..., i - 1], vmin=0, vmax=1)
        plt.show()


def confusion_matrix_to_latex(confusion_matrix, lc_class_object, normalised=False):
    class_properties = lc_class_object.class_properties
    if normalised:
        confusion_matrix = np.round(100 * confusion_matrix, 1)
    lines = []
    lines.append("\\begin{table}[!h]")
    lines.append("\\begin{center}")
    numb_cols = len(class_properties) + 2 if normalised else len(class_properties) + 3
    lines.append("\\begin{tabular}{" + "c" * (len(class_properties) + 3) + "}")
    lines.append(
        " &  & \\multicolumn{" + str(len(class_properties)) + "}{c}{MLP Class Prediction} &  \\\\ \\cline{3-" +
        str(len(class_properties) + 2) + "}")

    # column headings
    if normalised:
        lines.append(" & \\multicolumn{1}{c|}{(\\%)}")
    else:
        lines.append(" & \\multicolumn{1}{c|}{}")
    for i in class_properties:
        class_name = (class_properties[i]['Classname']).replace('_', ' ')
        lines[-1] = lines[-1] + " & \\multicolumn{1}{c|}{\\rotatebox[origin=c]{90}{" + class_name + "}}"
    if not normalised:
        lines[-1] = lines[-1] + " & Total"
    lines[-1] = lines[-1] + " \\\\ \\cline{2-" + str(len(class_properties) + 2) + "}"

    # line 1
    for i, class_idx in enumerate(class_properties):
        class_name = (class_properties[class_idx]['Classname']).replace('_', ' ')
        lines.append("\\multicolumn{1}{c|}")
        if i == 0:
            lines[-1] = lines[-1] + "{\\multirow{" + str(len(
                class_properties)) + "}{*}{\\rotatebox[origin=c]{90}{\\shortstack{Ground Truth\\\\Label}}}}"
        else:
            lines[-1] = lines[-1] + "{}"
        lines[-1] = lines[-1] + " & \\multicolumn{1}{c|}{" + class_name + "}"
        for j, class_idx_2 in enumerate(class_properties):
            lines[-1] = lines[-1] + " & \\multicolumn{1}{c|}{" + str(confusion_matrix[i, j]) + "}"
        if not normalised:
            lines[-1] = lines[-1] + " & " + str(confusion_matrix.sum(axis=1)[i])
        lines[-1] = lines[-1] + " \\\\ \\cline{2-" + str(len(class_properties) + 2) + "}"

    if not normalised:
        lines.append(" & Total &")
        for i in range(len(class_properties)):
            lines[-1] = lines[-1] + " " + str(confusion_matrix.sum(axis=0)[i]) + " &"
        lines[-1] = lines[-1] + " " + str(confusion_matrix.sum())
    lines.append("\\end{tabular}")
    lines.append("\\end{center}")
    lines.append("\\caption{Caption here.}")
    lines.append("\\label{Label_here}")
    lines.append("\\end{table}")
    print(f"\n".join(lines))


def kappa_from_confusion_matrix(confusion_array):
    N = confusion_array.sum()
    true_positives = np.diag(confusion_array).sum()
    errors = (confusion_array.sum(axis=0) * confusion_array.sum(axis=1)).sum()
    return (N * true_positives - errors) / (N ** 2 - errors)


def oa_from_confusion_matrix(confusion_array):
    N = confusion_array.sum()
    true_positives = np.diag(confusion_array).sum()
    return true_positives / N


def close_fig(numb_rows, numb_cols, boundary=0.1, width=9, height=6, keep_square=True):
    fig, ax = plt.subplots(numb_rows, numb_cols, figsize=(width, height))
    if isinstance(ax, np.ndarray):
        if ax.ndim == 2:
            ax_list = [b for a in ax for b in a]
        else:
            ax_list = [a for a in ax]
    else:
        ax_list = [ax]
    if keep_square:
        if width / height >= numb_cols / numb_rows:  # if figure wider ratio than grid
            topmost = 1 - boundary
            vertical_step = (1 - 2 * boundary) / numb_rows
            horizontal_step = vertical_step * height / width
            leftmost = (1 - numb_cols * horizontal_step) / 2
        else:  # if figure taller ratio than grid
            leftmost = boundary
            horizontal_step = (1 - 2 * boundary) / numb_cols
            vertical_step = horizontal_step * width / height
            topmost = 0.5 + (numb_rows * vertical_step) / 2
    else:
        topmost, leftmost = 1 - boundary, boundary
        horizontal_step, vertical_step = (1 - 2 * boundary) / numb_cols, (1 - 2 * boundary) / numb_rows
    for i in range(numb_rows):
        for j in range(numb_cols):
            ax_list[i * numb_cols + j].set_yticks([])
            ax_list[i * numb_cols + j].set_xticks([])
            ax_list[i * numb_cols + j].set_position(
                [leftmost + j * horizontal_step, topmost - (i + 1) * vertical_step, horizontal_step, vertical_step])
    if isinstance(ax, np.ndarray):
        return fig, np.array(ax_list).reshape(ax.shape)
    else:
        return fig, ax_list[0]
