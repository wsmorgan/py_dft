#!/usr/bin/python
from pydft import msg
import numpy as np
import csv

def RepresentInt(s):
    """Determines if a string can be represented as an integer.  code take
    from
    http://stackoverflow.com/questions/1265665/python-check-if-a-string-represents-an-int-without-using-try-except
    and was contributed by Trpitych.

    Args:
        s (str): The string to be checked.
    
    Returns:
       result (bool): True if s can be represented as an int.
    """

    try:
        int(s)
        return True
    except ValueError:
        raise ValueError("The number of divisions to take along each axes must be integers.")

def examples():

    """Prints examples of using the script to the console using colored output.
    """
    script = "DFT: Performs a Density Functional Theory ."
    explain = ("This code produces a numerical solution for the energy of a system of "
               "atoms.")
    contents = [(("Find the charge density for a sc cell with lattice parameter of 6 "
                  "with 4 divisions along each lattice vector."), 
                 "dft.py 6 -s [4,4,4]",
                 "This prints the value of phi evaluated at all the sample points "
                 "to the screen")]
    required = ("REQUIRED: The lattice parameter to be specified `a`.")
    output = ("RETURNS: A potential file named `potential.csv` if -poisson is specified.")
    details = (".")
    outputfmt = ("")

    msg.example(script, explain, contents, required, output, outputfmt, details)

script_options = {
    "a": dict(default=1., type=float,
              help=("The lattice parameter for the crystal structure.")),
    "-crystal": dict(default="sc", type=str,
                     help=("The type of primitive cell to use options are (sc)")),
    "-s": dict(default=[3,3,3], nargs="+",
              help=("The number of sampling points along each basis vector.")),
    "-poisson": dict(action="store_true",
                     help=("Prints the solution to the poisson equation to the screen.")),
     }
"""dict: default command-line arguments and their
    :meth:`argparse.ArgumentParser.add_argument` keyword arguments.
"""

def _parser_options():
    """Parses the options and arguments from the command line."""
    #We have two options: get some of the details from the config file,
    import argparse
    from pydft import base
    pdescr = "Numerical DFT code."
    parser = argparse.ArgumentParser(parents=[base.bparser], description=pdescr)
    for arg, options in script_options.items():
        parser.add_argument(arg, **options)
        
    args = base.exhandler(examples, parser)
    if args is None:
        return

    return args # pragma: no cover

def run(args):
    """Runs the code to find the energy of the crystal using DFT.
    """

    if args["s"]:
        args["s"] = [int(i) for i in args["s"] if RepresentInt(i)]
        
    if args["crystal"] != "sc":
        raise ValueError("Only sc cubic lattice are supported at this time.")

    if args["poisson"]:
        from pydft.poisson import poisson, charge_dist
        R = args["a"]*np.array([[1,0,0],[0,1,0],[0,0,1]])
        n = charge_dist(args["s"],R,[-1,1],[0.5,0.75])
        phi = poisson(args["s"],R,n)
        with open("potential.csv","w+") as out:
            phi_writer = csv.writer(out,delimiter='\n')
            phi_writer.writerow(phi)

    else:
        return 0
    
if __name__ == '__main__': # pragma: no cover
    run(_parser_options())

