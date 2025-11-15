import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib.lines as mlines

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)

                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

data = [['arc_ar', 'arc_zh', 'belebele_tha_Thai', 'hellaswag_ar', 'hellaswag_fr', 'm_mmlu_ar', 'm_mmlu_fr', 'paws_fr',
         'piqa_ar', 'xcopa_th', 'xcopa_zh', 'xnli_zh', 'xstorycloze_ar'],
        ('Multilingual Benchmarks', [
            [0.2344, 0.259, 0.2756, 0.3279, 0.37, 0.2454, 0.2525, 0.5325, 0.5789, 0.59, 0.61, 0.3337, 0.5632],
            [0.2173, 0.1957, 0.2267, 0.2818, 0.3085, 0.2288, 0.2458, 0.548, 0.5381, 0.56, 0.55, 0.3378, 0.5023],
            [0.2284, 0.2803, 0.2367, 0.3296, 0.4064, 0.2434, 0.2602, 0.539, 0.5653, 0.586, 0.666, 0.3739, 0.5606]])]

N = len(data[0])
theta = radar_factory(N, frame='polygon')

spoke_labels = data.pop(0)
title, case_data = data[0]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))
fig.subplots_adjust(top=0.85, bottom=0.05)

ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
ax.set_title(title, position=(0.5, 1.1), ha='center')

colors = ['b', 'r', 'g']
labels = ['Xmodel-1.5', 'PolyLM-1.7B', 'PolyLM-13B']

for d, color, label in zip(case_data, colors, labels):
    line = ax.plot(theta, d, color=color, label=label)
    ax.fill(theta, d, alpha=0.1, facecolor=color)
ax.set_varlabels(spoke_labels)

legend_elements = [mlines.Line2D([], [], color=colors[i], ls='-', lw=2, label=labels[i]) for i in range(len(colors))]
legend = ax.legend(handles=legend_elements, loc=(0.9, .95), labelspacing=0.1, fontsize='small',frameon=False)

plt.show()
