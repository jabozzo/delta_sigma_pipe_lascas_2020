#! /usr/bin/env python

from argparse import ArgumentParser

import matplotlib.pyplot as plots

import calib.data as data
import calib.plot as cplot


def run(args):
    path_context = data.PathContext.relative()
    plot_location = data.DataLocation(path_context, args.location, args.plot, "json")

    PlotType = {  "canvas1d": cplot.Canvas1d, "canvas2d": cplot.Canvas2d,
                  "plot1d": cplot.Plot1d, "plot2d": cplot.Plot2d }
    PlotType = PlotType[getattr(args, "type")]

    print(" loading {}...".format(plot_location.computed_path))
    plot = data.load(PlotType, plot_location)

    figsize = [6.4, 4.8] if args.figsize is None else args.figsize
    assert len(figsize) == 2

    fig = plots.figure(figsize=figsize)
    dest = args.dest

    show = (dest is None) if args.show is None else bool(int(args.show))
    fig, ax = plot.plot(fig, show=show, save=False)

    if dest is not None:
        dest_location = data.DataLocation(path_context, args.location, dest, args.extension)
        print(" saving {}...".format(dest_location.computed_path))
        fig.savefig(dest_location.computed_path)


if __name__ == "__main__":
    parser = ArgumentParser(description='Execute a plot object.')

    parser.add_argument('location', help="Path to the input/output LOCATION.",  metavar="LOCATION")
    parser.add_argument('plot', help="Name of the PLOT file (without extension).",  metavar="ADCS")
    parser.add_argument('dest', nargs='?', default=None, help="Name of the DESTination file (without extension).",  metavar="DEST")
    parser.add_argument('--extension', default="png", choices=("png",), help="EXTENSION of the image file.",  metavar="EXTENSION")
    parser.add_argument('--type', default="canvas1d", choices=("canvas1d, canvas2d, plot1d, plot2d",), help="TYPE of the plot",  metavar="TYPE")
    parser.add_argument('--show', default=None, help="SHOW the plot",  metavar="SHOW")
    parser.add_argument('--figsize', nargs='+', default=None, help="SIZE (in inches) of the plot",  metavar="SIZE")

    args = parser.parse_args()

    run(args)
    exit(0)
