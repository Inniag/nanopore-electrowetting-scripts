#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This script calculates 1D and 2D representations of 3D grid data by slicing.

Its primary function is to calculate the grid value along a predefined spatial
curve. This curve needs to be given in terms of B-spline knots and control
points and will usually be calculated by CHAP. The script samples points along
the spline curve and calculates the value of the grid at these points by
trilinear interpolation of the values at the nearest grid points.

In addition to this, the script also calculates the grid values in xy-plane
slices. The z-coordinate of these slices is taken corresponding to a subset
of the points samples from the spline curve, so that each slice will
correspond to a value of the curve's s-coordinate. This enables visualising
the xy-plane grid value distribution alongside a 1D profile.

Both curve and slice values are written to a joint JSON file and can be loaded
for visualisation in e.g. R or Python. Curve sample points are also written to
a XYZ and a PDB file, where in the PDB file the temeprature factors and
occupancies are overwritten with the grid value at the given point. This
permits visualising the curve coloured by grid value in e.g. VMD.
"""

import argparse
import json as json
import warnings

import numpy as np
from scipy.interpolate import BSpline as bspl
from scipy.interpolate import RegularGridInterpolator

import MDAnalysis as mda
from gridData import Grid


def main(argdict):
    """ Main function for entry point checking.

    Expects a dictionary of command line arguments.
    """

    # load spline curve data:
    with open(argdict["curve_file"], "r") as f:
        chap_data = json.load(f)

    # extract spline knots:
    knots = np.array(chap_data["molPathCentreLineSpline"]["knots"])

    # create knot vector with repeated ends:
    degree = argdict["curve_spline_degree"]
    t = np.concatenate(
        (np.repeat(knots[0], degree-1), knots, np.repeat(knots[-1], degree-1))
    )

    # create nd-array representing control point coordinates:
    ctrlX = np.array(chap_data["molPathCentreLineSpline"]["ctrlX"])
    ctrlY = np.array(chap_data["molPathCentreLineSpline"]["ctrlY"])
    ctrlZ = np.array(chap_data["molPathCentreLineSpline"]["ctrlZ"])

    # perform b-spline interpolation along each axis:
    bsplX = bspl(t, ctrlX, degree, extrapolate=False)
    bsplY = bspl(t, ctrlY, degree, extrapolate=False)
    bsplZ = bspl(t, ctrlZ, degree, extrapolate=False)

    # load grid data:
    g = Grid(argdict["grid_file"])

    # prepare trilinear interpolation of grid:
    interp_grid = RegularGridInterpolator(
        g.midpoints,
        g.grid,
        method="linear",
        bounds_error=False)

    # calculate grid with derivative in each direction:
    # (uses 2nd order central difference for internal and first order accurate
    # one sided differences for boundary points)
    g_deriv = np.gradient(g.grid)
    g_dx = g_deriv[0]
    g_dy = g_deriv[1]
    g_dz = g_deriv[2]

    # trilinear interpolation of derivative grids:
    interp_grid_dx = RegularGridInterpolator(
        g.midpoints,
        g_dx,
        method="linear",
        bounds_error=False)
    interp_grid_dy = RegularGridInterpolator(
        g.midpoints,
        g_dy,
        method="linear",
        bounds_error=False)
    interp_grid_dz = RegularGridInterpolator(
        g.midpoints,
        g_dz,
        method="linear",
        bounds_error=False)

    # define centre-line coordinate at which to sample:
    curve_eval = np.linspace(min(t), max(t), argdict["num_eval_points"])

    # sample points on the spline curve (note conversion to Angstrom):
    NM2ANG = 10
    curve_points = NM2ANG*np.array(
        [np.array([bsplX(s), bsplY(s), bsplZ(s)]) for s in curve_eval]
    )

    # obtain grid values along curve:
    curve_values = interp_grid(curve_points)

    # obtain grid derivative values along curve:
    curve_values_dx = interp_grid_dx(curve_points)
    curve_values_dy = interp_grid_dy(curve_points)
    curve_values_dz = interp_grid_dz(curve_points)

    # create sample points for grid slices in xy-plane (cover grid range):
    x_min = np.min(g.midpoints[0])
    x_max = np.max(g.midpoints[0])
    y_min = np.min(g.midpoints[1])
    y_max = np.max(g.midpoints[1])
    x_eval = np.linspace(x_min, x_max, 100)
    y_eval = np.linspace(y_min, y_max, 100)

    # sample point z-coordinate from curve z-coordinate:
    num_slices = argdict["num_slices"]
    slice_step = int(np.ceil(np.size(curve_eval)/num_slices))
    z_eval = curve_points[0:-1:10, 2]
    s_eval = curve_eval[0:-1:slice_step]
    slice_eval = np.tile(s_eval, np.size(x_eval)*np.size(y_eval))

    # create meshgrid of xy-coordinates:
    mg_x, mg_y, mg_z = np.meshgrid(x_eval, y_eval, z_eval)

    # get array of three-vectors for grid interpolation:
    slice_points = np.transpose([mg_x.ravel(), mg_y.ravel(), mg_z.ravel()])

    # evaluate grid at slice points:
    slice_values = interp_grid(slice_points)

    # handle NaNs which can't go into JSON files:
    # NOTE: this has not been tested extensively
    if np.sum(np.isnan(curve_values)) > 0:

        # inform user:
        warnings.warn(
            "The provided spline curve leaves the volume covered by the grid at"
            " one or more points. No output data will be written for these"
            " points!"
        )

        # drop NaN values from curve arrays:
        curve_eval = curve_eval[~np.isnan(curve_values)]
        curve_points = np.array(curve_points)[~np.isnan(curve_values)]
        curve_values_dx = curve_values_dx[~np.isnan(curve_values)]
        curve_values_dy = curve_values_dy[~np.isnan(curve_values)]
        curve_values_dz = curve_values_dz[~np.isnan(curve_values)]
        curve_values = curve_values[~np.isnan(curve_values)]

    if np.sum(np.isnan(slice_values)):

        # inform user:
        warnings.warn(
            "The slice samples leave the volume covered by the grid at one or"
            " more points. No output data will be written for these points!"
        )

        # drop NaN values from slice arrays:
        slice_points = slice_points[~np.isnan(slice_values), :]
        slice_eval = slice_eval[~np.isnan(slice_values)]
        slice_values = slice_values[~np.isnan(slice_values)]

    # create output structure:
    output = {
        "curve": {
            "s": list(curve_eval),
            "x": list(curve_points[:, 0]),
            "y": list(curve_points[:, 1]),
            "z": list(curve_points[:, 2]),
            "val": list(curve_values),
            "val_dx": list(curve_values_dx),
            "val_dy": list(curve_values_dy),
            "val_dz": list(curve_values_dz)
        },
        "slices": {
            "s": list(slice_eval),
            "x": list(slice_points[:, 0]),
            "y": list(slice_points[:, 1]),
            "z": list(slice_points[:, 2]),
            "val": list(slice_values)
        },
        "residuePositions": chap_data["residuePositions"]
    }

    # write to JSON:
    with open(argdict["out_basename"] + ".json", "w") as f:
        json.dump(output, f)
        f.write("\n")

    # NOTE: writing the data to an XYZ file only to create a MDA universe is a
    # pretty dirty hack, but seems the fasted way to do it.

    # write data to xyz file format:
    with open(argdict["out_basename"] + ".xyz", "w") as f:

        f.write(str(np.size(curve_eval)) + "\n")
        f.write("testmolecule\n")

        for i in range(0, np.size(curve_eval)):

            # mind conversion to Angstrom:
            f.write(
                "Cl" + " "
                + str(curve_points[i, 0]) + " "
                + str(curve_points[i, 1]) + " "
                + str(curve_points[i, 2]) + " "
                + str(i) + "\n"
            )

    # read XYZ file:
    u = mda.Universe(argdict["out_basename"] + ".xyz")

    # write the curve values into the temperature factor and occupancy fields:
    u.add_TopologyAttr("tempfactors")
    u.add_TopologyAttr("occupancies")
    u.atoms.tempfactors = curve_values
    u.atoms.occupancies = curve_values

    # write all atoms to PDB file:
    a = u.select_atoms("all")
    a.write(argdict["out_basename"] + ".pdb")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-grid_file",
        type=str,
        nargs=None,
        default="potential.dx",
        help="Name of the grid file."
    )
    parser.add_argument(
        "-curve_file",
        type=str,
        nargs=None,
        default="stream_output.json",
        help="Name of JSON file containing the curve definition as set of"
        " B-spline knots and control points. Can be generated by CHAP with the"
        " -out-detailed flag."
    )
    parser.add_argument(
        "-out_basename",
        type=str,
        nargs=None,
        default="gridcurve",
        help="Base name of output files without file extension."
    )
    parser.add_argument(
        "-curve_spline_degree",
        type=int,
        nargs=None,
        default=3,
        help="Degree of curve interpolating spline."
    )
    parser.add_argument(
        "-num_eval_points",
        type=int,
        nargs=None,
        default=1000,
        help="Number of evaluation points along the centre line."
    )
    parser.add_argument(
        "-num_slices",
        type=int,
        nargs=None,
        default=100,
        help="Number of xy-plane slices taken."
    )

    args = parser.parse_args()
    argdict = vars(args)

    main(argdict)
