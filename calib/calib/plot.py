#! /usr/bin/env python

import numpy as np

import matplotlib as mpl
# mpl.use('Agg') # For usage without x-server

import matplotlib.pyplot as plots
from mpl_toolkits.mplot3d import Axes3D

import calib.data as data
from calib.misc import default


class Plot1d(data.JsonData):

    VALID_STYLES = ("plot", "scatter", "hist",)

    def _to_json_dict(self, path_context, memo=None):
        dct = {attr:getattr(self, attr, None)
            for attr in ("x", "axes_labels", "title", "subplot", "plot_kwargs", "style") }

        dct["data"] = data.at_least_numpydata(self.data).save(path_context, memo=memo)

        if self.x is not None:
            dct["x"] = data.at_least_numpydata(self.x).save(path_context, memo=memo)

        return dct

    @classmethod
    def _JsonDictToArgs(cls, path_context, data_location, dct, memo=None):
        _, args, kwargs = super()._JsonDictToArgs(path_context, data_location, dct, memo=memo)
        if data_location is not None:
            path_context = data_location.as_path_context()
        dct["data"] = data.NumpyData.Load(path_context, dct["data"], memo=memo)
        if dct["x"] is not None:
            dct["x"] = data.NumpyData.Load(path_context, dct["x"], memo)
        dct["subplot"] = tuple(dct["subplot"])
        kwargs.update(dct)
        return cls, args, kwargs

    def __init__(self, data, x=None, axes_labels=None, style=None, title=None, subplot=(1, 1, 1,), plot_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        if style is None:
            style = "plot"

        assert style in self.VALID_STYLES, "Invalid style recieved ({}).".format(style)

        self.plot_kwargs = default(plot_kwargs, {})
        self.data = data
        self.x = x
        self.axes_labels = axes_labels
        self.title = title
        self.subplot = subplot
        self.style=style

    def plot(self, figure_or_axes=None, show=None, save=None, **plot_kwargs):
        if figure_or_axes is None:
            figure_or_axes = plots.figure()

        if show is None:
            show = save is not True

        if save is None:
            save = not show

        args = (tuple() if self.x is None else (self.x,)) + (self.data,)
        kwargs = dict(self.plot_kwargs)
        kwargs.update(plot_kwargs)

        if not isinstance(figure_or_axes, plots.Figure):
            ax = figure_or_axes
        else:
            ax = figure_or_axes.add_subplot(*self.subplot)

        if self.style == "plot":
            ax.plot(*args, **kwargs)
        elif self.style == "hist":
            ax.hist(*args, **kwargs)
        elif self.style == "scatter":
            ax.scatter(*args, **kwargs)

        figure = ax.get_figure()

        if self.axes_labels is not None:
            try:
                ax.set_xlabel(self.axes_labels[0])
                ax.set_ylabel(self.axes_labels[1])
            except IndexError:
                pass

        if self.title is not None:
            ax.set_title(self.title)

        if save:
            dl = self.data_location
            assert dl is not None
            name = data.DataLocation(dl.path_context, dl.directory, dl.name, "png")
            name = name.computed_path
            figure.savefig(name)

        if show:
            figure.show()

        return figure, ax


class Plot2d(data.JsonData):

    VALID_STYLES = ("image", "colormap", "wire",)

    def _to_json_dict(self, path_context, memo=None):
        dct = {attr:getattr(self, attr, None)
            for attr in ("axes", "axes_labels", "title", "subplot", "style", "plot_kwargs") }

        dct["data"] = data.at_least_numpydata(self.data).save(path_context, memo=memo)
        if self.axes is not None:
            dct["axes"] = tuple(data.at_least_numpydata(dct["axes"][ii]).save(path_context, memo=memo) for ii in range(2))

        return dct

    @classmethod
    def _JsonDictToArgs(cls, path_context, data_location, dct, memo=None):
        _, args, kwargs = super()._JsonDictToArgs(path_context, data_location, dct, memo)
        if data_location is not None:
            path_context = data_location.as_path_context()
        dct["data"] = data.NumpyData.Load(path_context, dct["data"], memo)
        if dct["axes"] is not None:
            dct["axes"] = data.NumpyData.Load(path_context, dct["axes"], memo)
        dct["subplot"] = tuple(dct["subplot"])
        kwargs.update(dct)
        return cls, args, kwargs

    def __init__(self, data, axes=None, axes_labels=None, title=None, style=None, subplot=(1, 1, 1,), plot_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        if style is None:
            style = "colormap"

        assert style in self.VALID_STYLES, "Invalid style recieved ({}).".format(style)

        self.plot_kwargs = default(plot_kwargs, {})
        self.data = data
        self.axes = axes
        self.axes_labels = axes_labels
        self.title = title
        self.style = style
        self.subplot = subplot

    def plot(self, figure_or_axes=None, show=None, save=None, **plot_kwargs):
        if figure_or_axes is None:
            figure_or_axes = plots.figure()

        if not isinstance(figure_or_axes, plots.Figure):
            ax = figure_or_axes

        else:
            kwargs = {}
            if self.style == "wire":
                kwargs["projection"] = "3d"
            ax = figure_or_axes.add_subplot(*self.subplot, **kwargs)

        figure = ax.get_figure()

        if show is None:
            show = save is not True

        if save is None:
            save = not show

        args = (self.data,)
        kwargs = dict(self.plot_kwargs)
        kwargs.update(plot_kwargs)

        sweeps = self.axes
        if sweeps is None:
            sweeps = tuple(list(range(np.size(self.data, ii))) for ii in range(2))

        if self.style == "wire":
            XX, YY = np.meshgrid(sweeps[0], sweeps[1], indexing='ij')
            args = (XX, YY,) + args
            flip_labels = False

        else:
            kwargs["interpolation"] = kwargs.get("interpolation", 'none')
            extent = [sweeps[0][0], sweeps[0][-1], sweeps[1][0], sweeps[1][-1]]
            kwargs["extent"] = kwargs.get('extent', extent)
            kwargs["origin"] = kwargs.get('origin', 'lower')
            aspect = (extent[1] - extent[0])/(extent[3] - extent[2])
            kwargs["aspect"] = kwargs.get('aspect', aspect)
            flip_labels = True

        if self.style == "wire":
            sm = ax.plot_wireframe(*args, **kwargs)
        else:
            sm = ax.imshow(*args, **kwargs)

        if self.axes_labels is not None:
            try:
                if flip_labels:
                    ax.set_xlabel(self.axes_labels[1])
                    ax.set_ylabel(self.axes_labels[0])
                else:
                    ax.set_xlabel(self.axes_labels[0])
                    ax.set_ylabel(self.axes_labels[1])

            except IndexError:
                pass

        if self.title is not None:
            ax.set_title(self.title)

        if self.style == "colormap":
            figure.colorbar(sm, ax=ax)

        if save:
            dl = self.data_location
            assert dl is not None
            name = data.DataLocation(dl.path_context, dl.directory, dl.name, "png")
            name = name.computed_path
            figure.savefig(name)

        if show:
            figure.show()

        return figure, ax


def canvas(cls):
    ListType = data.list_of(cls)

    class Canvas(data.JsonData):

        def __init__(self, children=None, labels=None, data_location=None):
            super().__init__(data_location)
            self._children = ListType() if children is None else ListType.EnsureIsInstance(children)
            self._labels = [] if labels is None else labels

            assert len(self._labels) == len(self._children)

        def add(self, plot, label=None):
            self._children.append(plot)
            self._labels.append(label)

        def _to_json_dict(self, path_context, memo=None):
            dct = {}
            dct["children"] = self._children.save(path_context, memo=memo)
            dct["labels"] = self._labels
            return dct

        @classmethod
        def _JsonDictToArgs(cls, path_context, data_location, dct, memo=None):
            _, args, kwargs = super()._JsonDictToArgs(path_context, data_location, dct, memo=memo)
            if data_location is not None:
                path_context = data_location.as_path_context()
            path_context = data_location.as_path_context()
            children = ListType.Load(path_context, dct["children"], memo=memo)
            labels = dct["labels"]
            args = [children, labels] + args
            return cls, args, kwargs

        def plot(self, figure_or_axes=None, show=None, save=None):
            if figure_or_axes is None:
                figure_or_axes = plots.figure()

            if show is None:
                show = save is not True

            if save is None:
                save = not show

            if not isinstance(figure_or_axes, plots.Figure):
                ax = figure_or_axes
            else:
                ax = figure_or_axes.add_subplot(1, 1, 1)

            fig = ax.get_figure()

            for child, label in zip(self._children, self._labels):
                plot_kwargs = {}
                if label is not None:
                    plot_kwargs["label"] = label
                fig, ax = child.plot(ax, show=False, save=False, **plot_kwargs)

            if show:
                plots.legend()
                fig.show()

                plots.show()

            if save:
                dl = self.data_location
                assert dl is not None
                name = data.DataLocation(dl.path_context, dl.directory, dl.name, "png")
                name = name.computed_path
                fig.savefig(name)

            return fig, ax

    return Canvas


Canvas1d = canvas(Plot1d)
Canvas2d = canvas(Plot2d)
