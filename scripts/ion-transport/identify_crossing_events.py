#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Identifies events of small particles crossing through an ion channel.

This script parses a trajectory and of an ion-channel-in-membrane simulation
and identifies events of small target particles (e.g. ions, water molecules)
crossing the ion channel. A crossing event is defined as an ion moving from the
bulk domain on one side of the membrane through the channel pore domain to the
bulk domain on the opposite side of the membrane. Cases where the target
particle moves through the periodic boundary or through the membrane itself are
not classified as crossing events.

Particles are assigned to the two bulk domains based on whether their
z-coordinate is smaller or larger than the mean z-coordinate of the membrane.
However, if a particle is located within a cylindrical region around the centre
of geometry of the channel protein, it is assigned to the pore domain instead.
Crossing events are then identified based on the particles moving between these
three domains.

The user can specify an MDAnalysis selection that defines which particles will
be considered. It should be noted that the script operates on a per-atom basis.
Therefore, if the user selected e.g. water molecules on a per-residue bases, the
crossing of one water molecule will be recorded as the three individual
crossings of the constituent atoms.

Output is written to a JSON file and corresponds to a data frame structure in
which each row represents one crossing event. The columns will contain the
(res)ID, (res)name, charge, and mass of the particle that crossed the pore
alongside the times at which the particle enters and exits the pore. A crossing
number is given which indicates the direction in which the particle crossed
the pore (i.e. +1 for crossing in positive z-direction and -1 for crossing in
negative z-direction).

The overall number of particles that cross the pore can be readily calculated
from this data as the cumulative sum over the crossing number. Similarly, the
total amount of charge or mass transported can be evaluated as the cumulative
sum over the product of the crossing number with charge and mass respectively.
"""


import argparse

import numpy as np
import pandas as pd

import MDAnalysis as mda


def determine_domain_indices(
    topology_file,
    trajectory_file,
    sel_pore,
    sel_bilayer,
    sel_target,
    r_margin,
    z_margin,
    b,
    e,
    dt
):
    """Passes through trajectory and assigns domain indices.

    The return value is a data frame containing three columns for particle ID,
    time stamp, and the domain index of that particle at the given time. The
    domain index gives an indication of whether a particle is located below
    the membrane (-1), above the membrane (+1), or inside the channel pore (0).
    The user can set a start time, end time, and time step based on which frames
    will be analysed or skipped.

    Particles are assigned to the -1 and +1 domains based on whether their
    z-coordinate is larger or smaller than the mean z-coordinate of the
    membrane. If a particle is located within a cylindrical region around the
    center of geometry of the protein, it is assigned the domain index 0
    instead. Radius and z-extend of the cylindrical region are determined from
    the bounding box of the protein, but the user may add a margin to either
    of these parameters.
    """

    # create MDA universe:
    u = mda.Universe(topology_file, trajectory_file)

    # determine atom groups for analysis:
    protein = u.select_atoms(sel_pore)
    membrane = u.select_atoms(sel_bilayer)
    target = u.select_atoms(sel_target)

    # find pore radius:
    prot_radius = max(np.diff(protein.bbox(), axis=0)[0][0:2]) / 2.0
    radius = prot_radius + r_margin

    # create data frame of target positions over time:
    dat = []
    for ts in u.trajectory:

        # skip unwanted frames:
        if ts.time < b or ts.time > e or ts.time % dt != 0.0:
            continue

        # inform user:
        print(
            "  ~> identifying domain indices in frame: " + str(ts.frame)
            + " at time: " + str(ts.time)
        )

        # create data frame from target particle positions:
        tmp = pd.DataFrame(
            target.positions,
            columns=["x", "y", "z"]
        )

        # add particle properties:
        tmp["particle_id"] = target.ids
        tmp["name"] = [str(x) for x in target.names]
        tmp["resname"] = [str(x) for x in target.resnames]
        tmp["resid"] = [int(x) for x in target.resids]
        tmp["charge"] = target.charges
        tmp["mass"] = target.masses

        # add time stamp:
        tmp["t"] = ts.time

        # position of protein COG in x/y-plane:
        x_cen = protein.center_of_geometry()[0]
        y_cen = protein.center_of_geometry()[1]

        # z-extent of proteina nd middle of membrane:
        z_min = protein.bbox()[0, 2] - z_margin
        z_max = protein.bbox()[1, 2] + z_margin
        z_mid = np.mean(membrane.bbox(), axis=0)[2]

        # assign domain indices:
        tmp.loc[
            (tmp["z"] < z_mid),
            "domain_idx"
        ] = int(-1)
        tmp.loc[
            (tmp["z"] >= z_mid),
            "domain_idx"
        ] = int(1)
        tmp.loc[
            (tmp["z"] >= z_min)
            & (tmp["z"] <= z_max)
            & ((tmp["x"] - x_cen)**2 + (tmp["y"] - y_cen)**2 < radius**2),
            "domain_idx"
        ] = int(0)

        # sanity check:
        num_nan = np.sum(tmp["domain_idx"].isna())
        if num_nan > 0:
            raise Exception("Could not unambiguously assign domain indices.")

        # drop unneccessary columns:
        tmp = tmp.drop(columns=["x", "y", "z"])

        # add to data frame list:
        dat.append(tmp)

    # combine into overall data frame:
    df = pd.concat(dat)

    # return the overall data frame of domain indices:
    return(df)


def aggregate_event_pairs(df):
    """Aggregates pairs of subsequent events.

    The resulting data frame will contain as many rows as the number of
    crossing events (i.e. a particle going through the channel) occuring in the
    given data frame. This means that the output data frame will be empty if no
    crossing event occurs. Each row will contain the time at which the particle
    entered and left the pore region as well as the mean of those two times. In
    addition it will contain a crossing number, which indicates whether the
    particle crossed the pore in positive or negative z-direction (its absolute
    value is always one).

    The input data frame must contain the domain index difference for a single
    particle only and may not contain any jump events, i.e. events where the
    particle moved from domain +1 to domain -1 (or vice versa) without passing
    through domain 0 (the pore). This can happen if the particle jumps through
    the periodic boundary or if it moves through the membrane.

    The crossing number is calculated as the mean of two subsequent domain index
    differences. If they have opposite sign, the crossing number will be zero
    and the row will be dropped. If they have the same sign, this indicates that
    the particle moved either from domain -1 to domain 0 to domain +1 or from
    domain +1 to domain 0 to domain -1, corresponding to a true crossing event.
    The sign of the crossing number will reflect the direction of the crossing.
    """

    # make copy of input data frame to avoid overwriting contents:
    df = df.copy()

    # make sure data frame has even number of rows:
    if df.shape[0] % 2 is not 0:
        df = df.iloc[:-1]

    # introduce pairing index:
    df["pairing"] = np.repeat(range(0, int(df.shape[0]/2)), 2)

    # calculate summry statistics over pairings:
    df["crossing_number"] = df.groupby("pairing").domain_diff.transform("mean")
    df["t_enter"] = df.groupby("pairing").t.transform("min")
    df["t_mean"] = df.groupby("pairing").t.transform("mean")
    df["t_exit"] = df.groupby("pairing").t.transform("max")
    df = df.groupby(["pairing", "name"]).mean()

    # remove all rows not corresponding to a crossing event:
    df = df.loc[df.domain_diff != 0.0, ]

    # return data frame to caller:
    return(df)


def identify_crossing_events(df):
    """Identifies pore crossing events in data frame of domain indices.

    Given a data frame of domain indices alongside time stamps and particle IDs,
    this function identifies crossing events, i.e. a particle moving from one
    side of the membrane to the other THROUGH THE PORE REGION. Cases where a
    particle moves through the periodic boundary or through the membrane itself
    (i.e. outside the cylindrical protein region) are not counted as crossing
    events.

    The return value is a data frame in which each row corresponds to a
    crossing event and the columns identify times at which the particle entered
    and exited the pore, the mean of these times, and the ID of the particle
    which crosses the pore. Note that the input data frame should contain only
    a single unique particle ID, i.e. this function should by run on the domain
    index data frame grouped by particle indices. In addition, a further column
    contains a crossing number indicating the direction in which the particle
    passed the pore.

    This is done by first calculating the difference between subsequent domain
    indices. Where this difference is zero, the particle stayed in the given
    domain and the corresponding row can be ignored. Where this difference is
    +1/-1, the particle moved into or out of the pore. Where the difference is
    +2/-2, the particle moved across the periodic boundary or through the
    membrane.

    The time series of domain index differences is then partitioned into
    individual series between periodic boundary jump events. In each such
    subdivision the target particle is known to start outside the pore domain.
    A crossing event can then be identified by averaging over the sum of
    pairs of subsequent domain index differences. Note that for a particle that
    starts out inside the pore domain at the beginning of the trajectory no
    clear crossing direction can be established (i.e. is it going back to the
    domain it came from or is it moving across the pore to the opposite domain).
    By convention, these events will therefore not be counted as pore crossing
    events.
    """

    # make copy of input data frame to avoid overwriting contents:
    df = df.copy()

    # calculate difference between subsequent indices:
    df["domain_diff"] = df["domain_idx"].shift(0) - df["domain_idx"].shift(1)
    df["t"] = 0.5*(df["t"].shift(0) + df["t"].shift(1))
    df = df.dropna()
    df = df.drop(columns=["domain_idx"])

    # drop cases where particle did not change domain at all:
    df = df.loc[np.abs(df["domain_diff"]) != 0.0, ]

    # create index for number of jumps across period boundary (or through the
    # membrane)
    df["pbcjump"] = 0
    df.pbcjump[(df.domain_diff == -2) | (df.domain_diff == 2)] = 1
    df.pbcjump = df.pbcjump.cumsum()

    # drop all pbcjump groupings with less then two non-pbcjump events:
    df = df.loc[(df.domain_diff != -2) & (df.domain_diff != 2), ]
    df["pbccount"] = df.groupby("pbcjump").domain_diff.transform("count")
    df = df.loc[df.pbccount > 1, ]
    df = df.drop(columns=["pbccount"])

    # handle special case of empty data frame:
    if df.empty is True:

        # manually add columns:
        df["crossing_number"] = None
        df["t_enter"] = None
        df["t_mean"] = None
        df["t_exit"] = None

    else:

        # aggregate pairwise domain index differences:
        df = df.groupby("pbcjump").apply(aggregate_event_pairs)

    # remove unneccessary columns and indices:
    df = df.drop(columns=["t", "domain_diff", "pbcjump"])
    df = df.reset_index(drop=True)

    # return data frame:
    return(df)


def main(argdict):
    """Main function for entry point checking.

    Expects to be given a dictionary of command line arguments.
    """

    # parse trajectory and assign domain indices:
    print("--> identifying domain indices...")
    nm2ang = 10.0
    domain_indices = determine_domain_indices(
        argdict["s"],
        argdict["f"],
        argdict["sel_pore"],
        argdict["sel_bilayer"],
        argdict["sel_target"],
        argdict["r_margin"] * nm2ang,
        argdict["z_margin"] * nm2ang,
        argdict["b"],
        argdict["e"],
        argdict["dt"]
    )

    # find crossing events by particle:
    print("--> identifying crossing events...")
    crossing_events = (
        domain_indices.groupby(["particle_id"]).apply(identify_crossing_events)
    )
    crossing_events = crossing_events.reset_index(drop=True)

    # restore atom and residue names:
    name = domain_indices.groupby("particle_id").name.first()
    crossing_events["name"] = np.array(name.loc[crossing_events.particle_id])
    crossing_events["resname"] = np.array(name.loc[crossing_events.particle_id])

    # serialize results:
    crossing_events.to_json(argdict["o"], orient="records")
    with open(argdict["o"], "a") as f:
        f.write("\n")


# entry point check:
if __name__ == "__main__":

    # turn off this warning:
    pd.options.mode.chained_assignment = None

    # parse command line arguments:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-s",
        type=str,
        nargs=None,
        default="production.tpr",
        help="""Topology file. Should contain charges and masses!"""
    )
    parser.add_argument(
        "-f",
        type=str,
        nargs=None,
        default="production.xtc",
        help="""Trajectory file name."""
    )
    parser.add_argument(
        "-o",
        type=str,
        nargs=None,
        default="crossing_events.json",
        help="""Name of output JSON file containing all crossing events."""
    )
    parser.add_argument(
        "-sel_target",
        type=str,
        nargs=None,
        default="resname NA CL",
        help="""MDAnalysis selection identifying the mobile particles
        crossing the pore."""
    )
    parser.add_argument(
        "-sel_pore",
        type=str,
        nargs=None,
        default="protein",
        help="""MDAnalysis selection identifying the channel domain."""
    )
    parser.add_argument(
        "-sel_bilayer",
        type=str,
        nargs=None,
        default="resname DOPC POPC",
        help="""MDAnalysis selection identifying the lipid bilayer."""
    )
    parser.add_argument(
        "-r_margin",
        type=float,
        nargs=None,
        default=0.0,
        help="""Radius margin around protein for identification of pore
        domain in nanometers. The pore domain is taken to be a cylinder whose
        radius is the radius of the protein (in the x/y-plane) plus this
        margin."""
    )
    parser.add_argument(
        "-z_margin",
        type=float,
        nargs=None,
        default=-1.0,
        help="""Margin for extension of pore domain around protein in
        nanometers. The pore domain is taken to be a cylinder whose the
        extension is determined from the z-extension of the protein plus this
        margin. Should never exceed half the length of the pore."""
    )
    parser.add_argument(
        "-b",
        type=float,
        nargs=None,
        default=0.0,
        help="""Start time for trajectory analysis in picoseconds. Frames
        before this time will be ignored."""
    )
    parser.add_argument(
        "-e",
        type=float,
        nargs=None,
        default=float("inf"),
        help="""End time for trajectory analysis in picoseconds. Frames
        after this time will be ignored."""
    )
    parser.add_argument(
        "-dt",
        type=float,
        nargs=None,
        default=100.0,
        help="""Time step for trajectroy analysis. Frames not that are not
        an integer multiple of this time step will be ignored. A time step of
        100 picoseconds seems to be sufficiently small for ions."""
    )

    # parse arguments and convert to dictionary:
    args = parser.parse_args()
    argdict = vars(args)

    # pass arguments to main function:
    main(argdict)
