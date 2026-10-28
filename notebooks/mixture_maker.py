def make_pdbs(seq_num_list, terminal_type):
    
    '''make .pdb files for peptide sequences (1 Letter seqs)'''

    import PeptideBuilder.Geometry
    from PeptideBuilder import Geometry
    import Bio.PDB

    ratio_list = []
    seq_list = []
    for i in seq_num_list:
        ratio_list.append(i[1])
        seq_list.append(i[0])

    pdb_list = []

    for seq in seq_list:
        
        geo_list = []
        
        letter_list = [item for item in seq]
        
        if terminal_type == 'capped':
                letter_list.insert(0, 'ACE')
                letter_list = letter_list + ['NME']

        for res in letter_list:
            geo = Geometry.geometry(res)
            geo_list.append(geo)
            
        structure = PeptideBuilder.initialize_res(geo_list[0])
        
        for geo in geo_list[1:]:
            PeptideBuilder.add_residue(structure, geo)

        PeptideBuilder.add_terminal_OXT(structure)

        out = Bio.PDB.PDBIO()
        out.set_structure(structure)
        if terminal_type == 'capped':
            out.save(seq + f'{terminal_type}.pdb')
            pdb_list.append(seq + f'{terminal_type}.pdb')
        else:
            out.save(seq + '.pdb')   
            pdb_list.append(seq + '.pdb')

    for i in pdb_list:
        from pdbfixer import PDBFixer
        from simtk.openmm.app import PDBFile
        fixer = PDBFixer(filename = i)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        PDBFile.writeFile(fixer.topology, fixer.positions, open(i, 'w'))

    return pdb_list





def num_chains(d, l, seq_num_list):
    '''d = density (mg/mL)
       l = box size (angstroms)
       seq_num_list = [(seq1, ratio), (seq2, ratio), (seq3, ratio)] #List of tuples
       ratio = [seq1] / [seq2]
       calculates and returns number of 
       peptide/protein chains needed to pack
    '''
    from Bio.SeqUtils import molecular_weight
    import numpy as np 
    import math

    ratio_list = []
    seq_list = []
    for i in seq_num_list:
        ratio_list.append(i[1])
        seq_list.append(i[0])
    ratio_tot = sum(ratio_list)

    chain_length = len(seq_list[0])
    
    keys = []
    values = []
    for i in seq_list:
        keys.append(i)
        values.append(molecular_weight(i, "protein"))
        mass_dict = {key : value for key, value in zip(keys, values)}

    mass_list = list(mass_dict.values())
    avg_mass = sum(mass_list)/len(seq_list)

    l = l/10 #convert to nm
    nchains = d * (l**3 * 1e-21) / (avg_mass/6.022e23*1000)
    
    ratio_list = []
    for i in seq_num_list:
        ratio_list.append(i[1])

    ratio_tot = sum(ratio_list)

    keys2 = []
    values2 = []
    for i in seq_num_list:
        keys2.append(i[0])
        values2.append(nchains / ratio_tot * i[1])
        if chain_length <= 3:
            chain_dict = {key : value + 2 for key, value in zip(keys2, values2)}
        elif chain_length <= 7:
            chain_dict = {key : value + 1 for key, value in zip(keys2, values2)}
        else:
            chain_dict = {key : value for key, value in zip(keys2, values2)}

    return chain_dict



def run_packmol(seq_num_list, terminal_type, path, d, l, header='A mixture of peptides'):

    ratio_list = []
    seq_list = []
    fname_list = []
    for i in seq_num_list:
        ratio_list.append(i[1])
        seq_list.append(i[0])

    pdb_list = make_pdbs(seq_num_list, terminal_type)

#     for i in seq_list:
#         fname_list.append(i[1])
#         fname = ''.join(fname_list)

    chain_dict = num_chains(d, l, seq_num_list)

    ### prep packmol file
    str_a = f'''
#
# {header}
#

tolerance 2.0
filetype pdb
output {'packed.pdb'}
  
#add TER to after each monomer chain
add_amber_ter
        
'''

    for i in pdb_list:
        for key in chain_dict.keys():
            if key == i[:-4]:
                nchains = int(chain_dict[key])
                string = f'''
structure {i}
number {nchains}
inside box 0. 0. 0. {l} {l} {l}
end structure
                                
            ''' 
                str_a += string

    packmol_file = open(f"{path}/mixture.inp", 'w')
    packmol_file.write(str_a)
    packmol_file.close()


    
def density_check(pdb_file):
    
    from Bio.SeqUtils import seq1
    from Bio.SeqUtils import molecular_weight
    import MDAnalysis as mda

    def listtostring(s):
        str1 = ""
        for ele in s:
            str1 += ele
        return str1

    def rnalisttostring(s):
        str1 = ""
        for ele in s:
            str1 += ele[1]
        return str1

    u = mda.Universe(f'{pdb_file}')
    prot = u.select_atoms('protein')
    rna = u.select_atoms('nucleic')

    system_list = prot.residues.resnames.tolist()
    box_str = listtostring(system_list)
    new_box_str = seq1(box_str)
    box_mass_amu = molecular_weight(new_box_str,'protein')

    system_nucleic_list = rna.residues.resnames.tolist()
    box_nucleic_str = rnalisttostring(system_nucleic_list)
    box_nucleic_mass_amu = molecular_weight(box_nucleic_str,'DNA')

    box_mass_mg = (box_mass_amu + box_nucleic_mass_amu) / 6.023e20

    box_volume_ml = (u.dimensions[0]*u.dimensions[1]*u.dimensions[2]) / (1.0E24)

    density = box_mass_mg / box_volume_ml
    
    return density
    
    

