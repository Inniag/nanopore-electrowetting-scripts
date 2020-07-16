#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Low level classes implementing a writer for APBS input files. Only a subset
of the overall APBS functionality is implemented. In particular, only the
mg-auto method (auto-calibrated focusing finite differences solver) can be
used and some of its parameters have (reasonable) hard-coded defaults. This
module is intended to be used in conjunction with MDAnalysis to set up APBS
calculations from Gromacs topology and trajectory files, so that the force
field used in the Poisson-Boltzmann solver is consistent with that used in MD
simulations.
"""


class AbstractApbsInputBlock:
    """
    Abstract class from which the various APBS input blocks are derivedself.

    This is not technically an abstract class because it does not use Python's
    abc module, but I felt that would be overkill.
    """

    def __init__(self):
        """
        Just an empty constructor.
        """
        pass

    def write(self, f):
        """
        Abstract method for writing block.
        """
        pass


class ApbsInputReadBlock(AbstractApbsInputBlock):
    """
    Representation of READ block in an APBS input file.

    Only supports mol type input.
    """

    def __init__(self):
        """
        Initialise an empty list of inputs.
        """

        self.__mol_inputs = []

    def add_mol_input(self, format, path):
        """
        Add input files to READ block. Currently only supports PQR files.
        """

        # sanity check - only implement this format at the moment
        if format is not "pqr":
            raise ValueError("input format must be pqr")

        # append to internal lists:
        self.__mol_inputs.append((str(format), str(path)))

    def write_mol_keyword(self, f, format, path):

        f.write("    mol " + str(format) + " \"" + str(path) + "\"\n")

    def write(self, f):
        """
        Write this block to the given file.
        """

        f.write("read\n")

        # write an entry for each input file:
        for format, path in self.__mol_inputs:
            self.write_mol_keyword(f, format, path)

        f.write("end\n")


class ApbsInputElecBlock(AbstractApbsInputBlock):
    """
    Representation of an ELEC block in an APBS input fileself. Implements a
    subset of the input parameters available in APBS.
    """

    def __init__(self):
        """
        Sets up some hard-coded default parameters. All other parameters are
        initialised as None and setter functions are provided for them.
        """

        # member variables with hardcoded defaults:
        self.__method = "mg-auto"
        self.__bcfl = "sdh"
        self.__etol = 1e-6
        self.__srfm = "smol"
        self.__swin = 0.3

        # member variables to be set after __init__:
        self.__calcenergy = None
        self.__calcforce = None
        self.__cgcent = None
        self.__cglen = None
        self.__chgm = None
        self.__dime = None
        self.__fgcent = None
        self.__fglen = None
        self.__pbetype = None
        self.__mol = None
        self.__pdie = None
        self.__sdens = None
        self.__sdie = None
        self.__srad = None
        self.__temp = None

        # member lists to which parameters may be appended:
        self.__ions = []
        self.__outputs = []

    def set_calcenergy(self, calcenergy):
        """
        Safeguarded method for setting calcenergy parameter.
        """

        if calcenergy is "no" or calcenergy is "total" or calcenergy is "comps":
            self.__calcenergy = calcenergy
        else:
            raise ValueError("calcenergy must be one of no | total | comps")

    def set_calcforce(self, calcforce):
        """
        Safeguarded method for setting calcforce parameter.
        """

        if calcforce is "no" or calcforce is "total" or calcforce is "comps":
            self.__calcforce = calcforce
        else:
            raise ValueError("calcforce must be one of no | total | comps")

    def set_cgcent(self, xyz):
        """
        Sets center of coarse grid (in Ang).
        """

        self.__cgcent = [float(xyz[0]), float(xyz[1]), float(xyz[2])]

    def set_cglen(self, xyzlen):
        """
        Sets size of coarse grid (in Ang).
        """

        self.__cglen = [float(xyzlen[0]), float(xyzlen[1]), float(xyzlen[2])]

    def set_chgm(self, chgm):
        """
        Safeguarded method for setting chgm parameter.
        """

        if (chgm == "spl0") or (chgm == "spl2") or (chgm == "spl4"):
            self.__chgm = chgm
        else:
            raise ValueError(
                "chgm must be one of spl0 | spl2 | spl4 but is " + str(chgm)
            )

    def set_dime(self, nx, ny, nz):
        """
        Sets number of grid points in each direction, casts to integer.
        """

        self.__dime = [int(nx), int(ny), int(nz)]

    def set_fgcent(self, xyz):
        """
        Sets center of fine grid (in Ang).
        """

        self.__fgcent = [float(xyz[0]), float(xyz[1]), float(xyz[2])]

    def set_fglen(self, xyzlen):
        """
        Sets size of fine grid (in Ang).
        """

        self.__fglen = [float(xyzlen[0]), float(xyzlen[1]), float(xyzlen[2])]

    def set_pbetype(self, pbetype):
        """
        Safeguarded method for setting pbetype parameter.
        """

        if (
            str(pbetype) == "lpbe"
            or str(pbetype) == "npbe"
            or str(pbetype) == "lrpbe"
            or str(pbetype) == "nrpbe"
        ):
            self.__pbetype = pbetype
        else:
            raise ValueError(
                "pbetype must be one of lpbe | npbe | lrpbe | nrpbe"
            )

    def set_mol(self, mol):
        """
        Sets id of molecule on which to perform calculation.
        """

        self.__mol = int(mol)

    def set_pdie(self, pdie):
        """
        Sets solute/protein dielectric constant. Checks that this is >= 1.
        """

        if pdie >= 1.0:
            self.__pdie = pdie
        else:
            raise ValueError("pdie must be >= 1.0")

    def set_sdens(self, sdens):
        """
        Sets number of quadrature points per Ang^2.
        """

        self.__sdens = sdens

    def set_sdie(self, sdie):
        """
        Sets solvent dielectric constant. Checks that this is >= 1.
        """

        if sdie >= 1.0:
            self.__sdie = sdie
        else:
            raise ValueError("sdie must be > = 1.0")

    def set_srad(self, srad):
        """
        Sets solvent molecule radius (in Ang).
        """

        self.__srad = srad

    def set_temp(self, temp):
        """
        Sets temperature and checks that this is positive.
        """

        if temp > 0.0:
            self.__temp = temp
        else:
            raise ValueError("temperature (in K) must be positive")

    def add_ionic_species(self, charge, conc, radius):
        """
        Add an ionic species to the solvent. Charge is in units of ec, conc in
        units of M (i.e. molar) and radius is in Ang.
        """

        self.__ions.append((charge, conc, radius))

    def add_output(self, type, format, stem):
        """
        Add another type of output data.
        """

        allowed_types = set(["charge", "pot", "lap"])

        if (type in allowed_types) is False:
            raise ValueError("")

        # only use this format:
        if format is not "dx":
            raise ValueError("output format must be dx")

        self.__outputs.append((type, format, stem))

    def write_flag_param(self, f, param):
        """
        Writes a flag-valued parameter (i.e. only parameter name).
        """

        if param is not None:
            f.write("    " + str(param) + " " + "\n")
        else:
            raise RuntimeError("flag valued parameter missing")

    def write_scalar_param(self, f, param, value):
        """
        Writes a named scalar values parameter (which can be number of string).
        """

        if value is not None:
            f.write("    " + str(param) + " " + str(value) + "\n")
        else:
            raise RuntimeError("parameter " + str(param) + " is missing")

    def write_vector_param(self, f, param, vector):
        """
        Writes a vector valued parameter.
        """

        if vector is not None:
            f.write("    " + str(param))
            for val in vector:
                f.write(" " + str(val))
            f.write("\n")
        else:
            raise RuntimeError("parameter " + str(param) + " is missing")

    def write_intvector_param(self, f, param, vector):
        """
        Writes a vector valued parameter, where values are cast to integer.
        """

        if vector is not None:
            f.write("    " + str(param))
            for val in vector:
                f.write(" " + str(int(val)))
            f.write("\n")
        else:
            raise RuntimeError("parameter " + str(param) + " is missing")

    def write_ions(self, f):
        """
        Writes all ionic species to file. No sanity checking is performed and
        user has responsibility to make sure solution is electroneutral.
        """

        for charge, conc, radius in self.__ions:
            f.write("    ion charge " + str(charge) + " conc " +
                    str(conc) + " radius " + str(radius) + "\n")

    def write_outputs(self, f):
        """
        Writes all output data statements.
        """

        for type, format, stem in self.__outputs:
            f.write(
                "    write "
                + str(type) + " "
                + str(format) + " \""
                + str(stem) + "\"\n"
            )

    def write(self, f):
        """
        Implements the base classes write method.
        """

        f.write("elec\n")

        # only use mg-auto method:
        f.write("    " + str(self.__method) + "\n")

        # bcfl - boundary condition
        self.write_scalar_param(f, "bcfl", self.__bcfl)

        # calcenergy - which energy value should be written to output
        self.write_scalar_param(f, "calcenergy", self.__calcenergy)

        # calcforce - which force value should be written to output
        self.write_scalar_param(f, "calcforce", self.__calcforce)

        # cgcent - box center for coarse grid
        self.write_vector_param(f, "cgcent", self.__cgcent)

        # cglen - box size in each direction for coarse grid
        self.write_vector_param(f, "cglen", self.__cglen)

        # chgm - charge mapping onto grid
        self.write_scalar_param(f, "chgm", self.__chgm)

        # dime - number of grid points in each direction
        self.write_intvector_param(f, "dime", self.__dime)

        # etol - error tolarance for solver
        self.write_scalar_param(f, "etol", self.__etol)

        # fgcent - center of the fine grid
        self.write_vector_param(f, "fgcent", self.__fgcent)

        # fglen - length of the fine grid in each direction
        self.write_vector_param(f, "fglen", self.__fglen)

        # ion - bulk concentration of mobile ions
        self.write_ions(f)

        # lpbe / lrpbe / npbe / nrpbe
        self.write_flag_param(f, self.__pbetype)

        # mol - id of molecule to do calculation on
        self.write_scalar_param(f, "mol", self.__mol)

        # pdie - dielectric of solute molecule
        self.write_scalar_param(f, "pdie", self.__pdie)

        # sdens - density of quadrature points on surfaces
        self.write_scalar_param(f, "sdens", self.__sdens)

        # sdie - solvent dielectric constant
        self.write_scalar_param(f, "sdie", self.__sdie)

        # srad - radius of solvent molecules
        self.write_scalar_param(f, "srad", self.__srad)

        # srfm - model for generating dielectric and ion-accessibility coefs
        self.write_scalar_param(f, "srfm", self.__srfm)

        # swin - size of support for spline-based surfaces
        self.write_scalar_param(f, "swin", self.__swin)

        # temp - temperature
        self.write_scalar_param(f, "temp", self.__temp)

        # usemap - use precalculated coefficient maps
        # NOTE: not implemented here, feature not needed

        # write - output data definitions
        self.write_outputs(f)

        # writemat - write operators to matrix file
        # NOTE: not implemented here, feature not needed

        f.write("end\n")


class ApbsInputPrintBlock(AbstractApbsInputBlock):
    """
    Representation of a PRINT block in APBS input files.
    """

    def __init__(self, what, idop):
        """
        The what field and id/operation array must be specified at construction.

        Performs only minimal sanity checking, id/operation must be
        user-checked.
        """

        if what is "elecEnergy" or what is "elecForce":
            self.__what = what
        else:
            raise ValueError(
                "PRINT block can only be one of elecEnergy or elecForce"
            )

        # should probably do sanity checking here
        self.__idop = idop

    def write(self, f):
        """
        Writes this block to the given file.
        """

        f.write("print " + str(self.__what) + " " + str(self.__idop) + " end\n")


class ApbsInput:
    """
    Representation of an APBS input file. Can add blocks to this which will be
    written to a file ins equential order.
    """

    def __init__(self):
        """
        Constructor just creates an empty list of blocks.
        """

        self.__blocks = []

    def add_block(self, block):
        """
        Adds given block to the internal list.
        """

        self.__blocks.append(block)

    def write(self, filename):
        """
        Writes all blocks to file in the order they were added to the internal
        list.
        """

        with open(filename, "w") as f:

            # write all blocks in sequential order:
            for block in self.__blocks:
                block.write(f)

            # write quit statment:
            f.write("quit\n")
