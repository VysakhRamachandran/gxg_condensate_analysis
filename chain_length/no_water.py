import MDAnalysis as mda

# Input parsing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--topology', nargs=2, required=True, type=str, help="path to topology file")
parser.add_argument('-t', '--trajectory', nargs=2, type=str, help="path to trajectory file")

args = parser.parse_args()
if(args.trajectory):
    univ = mda.Universe(args.topology[0], args.trajectory[0])
else:
    univ = mda.Universe(args.topology[0])

    
no_water_group = univ.select_atoms("not resname HOH")
        
no_water_group.write(args.topology[1])

if(args.trajectory):
    with mda.Writer(args.trajectory[1], no_water_group.n_atoms) as W:
        for ts in univ.trajectory:
            W.write(no_water_group)
