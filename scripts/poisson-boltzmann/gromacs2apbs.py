#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TODO
"""

"""

import argparse
import json

import numpy as np

import apbs as apbs
import MDAnalysis as mda


def main(argdict):
    """ Main function for entry point checking.

    Expects a dictionary of command line arguments.
    """

    # make sure ionic strength arrays have same length:
    if len(argdict["ionic_radii"]) is not len(argdict["ionic_charges"]):
        raise ValueError("need as many ionic_radii as ionic_charges")
    if (
        len(argdict["ionic_concentrations"])
        is not len(argdict["ionic_charges"])
    ):
        raise ValueError("need as many ionic_concentrations as ionic_charges")

    # make ionic parameters a list of tuples:
    ionic_params = zip(
        argdict["ionic_charges"],
        argdict["ionic_concentrations"],
        argdict["ionic_radii"]
    )

    # make sure there are no signs in filename that apbs dislikes:
    if "=" in argdict["out_basename"]:
        raise RuntimeError("out_basename may not contain '=' character")

    # write log of the parameters used to call this script:
    logfile = argdict["out_basename"] + "_gromacs2apbs_logfile_.json"
    with open(logfile, "w") as f:
        print(json.dumps(argdict, sort_keys=True, indent=4),
              file=f)

    # load input PQR file:
    u = mda.Universe(argdict["in_pqr"])

    # select atoms for coarse and fine regions:
    sel_coarse = u.select_atoms(argdict["sel_coarse"])
    sel_fine = u.select_atoms(argdict["sel_fine"])

    # determine center of coarse and fine grid:
    cog_coarse = sel_coarse.center_of_geometry()
    cog_fine = sel_fine.center_of_geometry()

    # determine length of coarse and fine grid:
    # (scaling factor is used to ensure the boundary is far from the atoms)
    bb_coarse = sel_coarse.bbox()
    bb_fine = sel_fine.bbox()
    len_coarse = (bb_coarse[1] - bb_coarse[0]) * argdict["scale_coarse"]
    len_fine = (bb_fine[1] - bb_fine[0]) * argdict["scale_fine"]

    # estimate minimal integer constant that will ensure specified grid spacing:
    nlev = 4    # for mg-auto this is always 4!
    c = np.ceil((len_fine/argdict["grid_spacing"] - 1)/(pow(2, nlev + 1)))

    # calculate number of grid points according to this:
    dime = c*pow(2, nlev + 1) + 1

    # write coarse selection to a PQR file:
    out_pqr_name = argdict["out_basename"] + "_.pqr"
    sel_coarse.write(out_pqr_name)

    # will read data from the modified PQR file set up above:
    read_block = apbs.ApbsInputReadBlock()
    read_block.add_mol_input("pqr", argdict["out_basename"] + "_.pqr")

    # set up focusing finite difference calculation:
    elec_block = apbs.ApbsInputElecBlock()

    # which outputs to calculate and write out:
    elec_block.set_calcenergy(argdict["calc_energy"])
    elec_block.set_calcforce(argdict["calc_force"])  # requires spline surfaces
    elec_block.add_output(
        "charge", "dx", (argdict["out_basename"] + "_gridval-charge_")
    )
    elec_block.add_output(
        "pot", "dx", (argdict["out_basename"] + "_gridval-potential_")
    )
    elec_block.add_output(
        "lap", "dx", (argdict["out_basename"] + "_gridval-laplacian_")
    )

    # grid size:
    elec_block.set_cgcent(list(cog_coarse))
    elec_block.set_cglen(list(len_coarse))
    elec_block.set_fgcent(list(cog_fine))
    elec_block.set_fglen(list(len_fine))
    elec_block.set_dime(dime[0], dime[1], dime[2])

    # methods to use and algorithmic parameters:
    elec_block.set_chgm(argdict["chgm"])
    elec_block.set_pbetype(argdict["pbetype"])
    elec_block.set_sdens(argdict["sdens"])

    # physical parameters:
    elec_block.set_pdie(argdict["pdie"])
    elec_block.set_sdie(argdict["sdie"])
    elec_block.set_srad(argdict["srad"])
    elec_block.set_temp(argdict["temp"])

    # ionic strength:
    for charge, conc, radius in ionic_params:
        elec_block.add_ionic_species(charge, conc, radius)

    # molecules to consider:
    elec_block.set_mol(1)

    # will print energy to screen if it is calculated:
    if argdict["calc_energy"] is "total":
        print_block = apbs.ApbsInputPrintBlock("elecEnergy", 1)

    # write APBS input file:
    apbs_input = apbs.ApbsInput()
    apbs_input.add_block(read_block)
    apbs_input.add_block(elec_block)
    apbs_input.add_block(print_block)
    apbs_input.write(argdict["out_basename"] + "_.in")


if __name__ == "__main__":

    # parse command line arguments:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-sel_coarse",
        type=str,
        nargs=None,
        default="all and not resname SOL WAT HOH NA CL",
        help="Selection of the system defining the coarse grid."
    )
    parser.add_argument(
        "-sel_fine",
        type=str,
        nargs=None,
        default="protein",
        help="Selection of system defining the fine grid."
    )
    parser.add_argument(
        "-scale_coarse",
        type=float,
        nargs=None,
        default=1.75,
        help="Length coarse of grid will be this parameter times the bounding "
        "box of the coarse selection."
    )
    parser.add_argument(
        "-scale_fine",
        type=float,
        nargs=None,
        default=1.30,
        help="Length of fine grid will be this parameter times the bounding box"
        " of the fine selection."
    )
    parser.add_argument(
        "-in_pqr",
        type=str,
        nargs=None,
        default="production.pqr",
        help="Input PQR file generated from a TPR file via editconf."
    )
    parser.add_argument(
        "-out_basename",
        type=str,
        nargs=None,
        default="apbs",
        help="Base name for output files, to with .pqr or .in will be added as"
        " appropriate."
    )
    parser.add_argument(
        "-calc_energy",
        type=str,
        choices=["no", "total", "comps"],
        nargs=1,
        default="total",
        help="This optional keyword controls energy output from an apolar "
        "solvation calculation."
    )
    parser.add_argument(
        "-calc_force",
        type=str,
        choices=["no", "total", "comps"],
        nargs=1,
        default="no",
        help="This optional keyword controls force output from an apolar "
        "solvation calculation."
    )
    parser.add_argument(
        "-pdie",
        type=float,
        nargs=None,
        default=2.0,
        help="Specify the dielectric constant of the solute molecule. This is"
        " usually a value between 2 to 20, where lower values consider only"
        " electronic polarization and higher values consider additional"
        " polarization due to intramolecular motion."
    )
    parser.add_argument(
        "-sdie",
        type=float,
        nargs=None,
        default=78.5,
        help="Specify the dielectric constant of the solvent. Bulk water at"
        " biologically-relevant temperatures is usually modeled with a"
        " dielectric constant of 78-80."
    )
    parser.add_argument(
        "-srad",
        type=float,
        nargs=None,
        default=1.4,
        help="This keyword specifies the radius (in Å) of the solvent"
        " molecules; this parameter is used to define various solvent-related"
        " surfaces and volumes (see srfm (elec)). This value is usually set to"
        " 1.4 Å for a water-like molecular surface and set to 0 Å for a van der"
        " Waals surface."
    )
    parser.add_argument(
        "-temp",
        type=float,
        nargs=None,
        default=310,
        help="This keyword specifies the temperature (in K) for the"
        " calculation."
    )
    parser.add_argument(
        "-ionic_charges",
        type=float,
        nargs="+",
        default=[+1, -1],
        help="Array of ionic species charges (in elementary charge). Must have"
        " same length as -ionic_concentrations and -ionic_radii. Default"
        " parameters are for physiological NaCl."
    )
    parser.add_argument(
        "-ionic_concentrations",
        type=float,
        nargs="+",
        default=[0.150, 0.150],
        help="Array of ionic species concentrations (in M). Must have same"
        "length as -ionic_charges and -ionic_radii. Default parameters are for"
        "physiological NaCl."
    )
    parser.add_argument(
        "-ionic_radii",
        type=float,
        nargs="+",
        default=[1.680, 1.937],
        help="Array of ionic species radii (in Å). Must have same length as"
        " -ionic_concentrations and -ionic_charges. Default parameters are for"
        " physiological NaCl."
    )
    parser.add_argument(
        "-grid_spacing",
        type=float,
        nargs=None,
        default=1.0,
        help="Desired spacing of points on the fine grid (in Å). The actual"
        " grid spacing is determined internally by APBS, which for numerical"
        " reasons needs a magical number of grid points in each dimension. This"
        " parameter will be used to ensure that there are at least as many grid"
        " points as is necessary for the grid spacing to be at least this fine"
        " (but it may be finer). The grid spacing is determined individually"
        " for each grid dimension. The coarse grid will use the same number of"
        " grid points, but with different edge lengths."
    )
    parser.add_argument(
        "-chgm",
        type=str,
        choices=["spl0", "spl2", "spl4"],
        nargs=None,
        default="spl2",
        help="Specify the method by which the biomolecular point charges (i.e.,"
        " Dirac delta functions) by which charges are mapped to the grid for a"
        " multigrid (mg-manual, mg-auto, mg-para) Poisson-Boltzmann"
        " calculation. As we are attempting to model delta functions, the"
        " support (domain) of these discretized charge distributions is always"
        " strongly dependent on the grid spacing."
    )
    parser.add_argument(
        "-pbetype",
        type=str,
        choices=["lpbe", "npbe", "lrpbe", "nrpbe"],
        nargs=None,
        default="lpbe",  # TODO: default sensible?
        help="Which equation to solve: linearised Poisson-Boltzmann (lpbe),"
        " full (non-linear) Poisson-Boltzmann (npbe), or the respective"
        " regularised variants (lrpbe, nrpbe)."
    )
    parser.add_argument(
        "-sdens",
        type=float,
        nargs=None,
        default=10.0,
        help="This keyword specifies the number of quadrature points per Å2 to"
        " use in calculation surface terms (e.g., molecular surface, solvent"
        " accessible surface). This keyword is ignored when srad is 0.0 (e.g.,"
        " for van der Waals surfaces) or when srfm (elec) is spl2 (e.g., for"
        " spline surfaces).  A typical value is 10.0."
    )

    # parse arguments:
    args = parser.parse_args()
    argdict = vars(args)

    main(argdict)
