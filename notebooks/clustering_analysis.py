import MDAnalysis as mda
import numpy as np
import math
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from skopt import gp_minimize
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from collections import Counter
import skopt
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import MDAnalysis.analysis.rdf
from MDAnalysis.core.topology import TransTable
from MDAnalysis.analysis import contacts
from MDAnalysis.analysis import distances
import itertools
from itertools import *
import pylab as pl
from scipy.spatial.distance import pdist,squareform
import pytim
from Bio.SeqUtils import seq1
from Bio.SeqUtils import molecular_weight
from ovito.io import import_file
from ovito.modifiers import ConstructSurfaceModifier
from ovito.modifiers import SelectTypeModifier
import warnings
warnings.filterwarnings('ignore')

def listtostring(s):
    str1 = ""
    for ele in s:
        str1 += ele
    return str1
def calist_listtostring(s):
    str1 = ""
    for ele in s:
        str1 += 'GLY' + ele + 'GLY'
    return str1



class cluster_analysis:
    def __init__(self, universe):
        self.universe = universe
    

    def dbscan_params(self, terminal_type, sim_species, frame, chain_length):
        
        ### Nested function in all clustering analysis functions ###
        ###       Defines dbscan parameters based on system      ###
        ###       sim_species : 'prot', 'prot+nucleic', or 'nucleic'     ###
        
        metric = 'precomputed'
        
        param_dict = {}
        
        snapshot = frame.frame
        
        resid = math.ceil(chain_length/2)
        
        if terminal_type == 'capped':
            ca_sel = self.universe.select_atoms(f'protein and resnum {resid} and name CA')
        else:
            ca_sel = self.universe.select_atoms(f'protein and resnum {resid} and name CA')
        
        nucleic_sel = self.universe.select_atoms(f"nucleic and resnum {resid} and name P")
        
        n_atoms = len(nucleic_sel) + len(ca_sel)
        
        tot_sel = nucleic_sel + ca_sel
        
        atoms = [atom.index for atom in nucleic_sel.atoms] + [atom.index for atom in ca_sel.atoms]

        to_cluster = [atom for i, atom in enumerate(self.universe.trajectory[snapshot].positions) if i in atoms]

        self_distances = distances.self_distance_array(tot_sel.positions, box=self.universe.trajectory[snapshot].dimensions)

        sq_dist_arr = np.zeros((n_atoms, n_atoms))
        triu = np.triu_indices_from(sq_dist_arr, k=1)

        sq_dist_arr[triu] = self_distances
        sq_dist_arr.T[triu] = self_distances

        # Define your objective function
        def custom_silhouette_score(params):
            param1, param2 = params
            # Your clustering function and scoring here
            model = DBSCAN(eps=param1, min_samples=param2, metric='precomputed')
            labels = model.fit_predict(sq_dist_arr)
            if len(Counter(labels)) > 1:
                score = silhouette_score(sq_dist_arr, labels)
            else:
                score = 0
            return -score

        # Define parameter space with lists of options
        if chain_length < 12:
            param_space = [[9,10,11,12],   # Discrete parameter options
                           [2,3,4]]              # Discrete parameter options
        elif chain_length >= 12 and chain_length < 18:
            param_space = [[12,13,14,15,16,17,18],   # Discrete parameter options
                        [2,3,4]]                                 # Discrete parameter options
        elif chain_length >= 18 and chain_length < 24:
            param_space = [[16,17,18,19,20,21,22,23,24,25,26],   # Discrete parameter options
                           [2,3,4]]                              # Discrete parameter options
        else:
            param_space = [[20,21,22,23,24,25,26,27,28,29,30],   # Discrete parameter options
                           [2,3,4]] 

        # Perform optimization
        result = gp_minimize(
            func=custom_silhouette_score,
            dimensions=param_space,
            n_calls=20                      # Number of function evaluations
        )

        best_eps, best_min_samples = result.x

        return best_eps, best_min_samples, metric
                

    def clustering(self, sim_species, terminal_type, frame, chain_length):

        ### Nested function in all clustering analysis functions ###
        ###       sim_species : 'prot', 'prot+nucleic', or 'nucleic'     ###
        ###          terminal_type : 'capped', 'charged'         ###
                
        resid = math.ceil(chain_length/2)
        
        if terminal_type == 'capped':
            ca_sel = self.universe.select_atoms(f'protein and resnum {resid} and name CA')
        else:
            ca_sel = self.universe.select_atoms(f'protein and resnum {resid} and name CA')
        
        nucleic_sel = self.universe.select_atoms(f"nucleic and resnum {resid} and name P")
        
        n_atoms = len(nucleic_sel) + len(ca_sel)
        
        tot_sel = nucleic_sel + ca_sel
        
        atoms = [atom.index for atom in nucleic_sel.atoms] + [atom.index for atom in ca_sel.atoms]

        to_cluster = [atom for i, atom in enumerate(self.universe.trajectory[frame.frame].positions) if i in atoms]

        self_distances = distances.self_distance_array(tot_sel.positions, box=frame.dimensions)

        sq_dist_arr = np.zeros((n_atoms, n_atoms))
        triu = np.triu_indices_from(sq_dist_arr, k=1)

        sq_dist_arr[triu] = self_distances
        sq_dist_arr.T[triu] = self_distances

        eps, min_num, metric = self.dbscan_params(terminal_type, sim_species, frame, chain_length)

        model = DBSCAN(eps=eps, min_samples=min_num, metric=metric)
        
        labels = model.fit_predict(sq_dist_arr)
        b = Counter(labels)
        
        if b.most_common()[0][0] == -1 and len(b) > 1:
            clustered_list = [atom for j, atom in enumerate(to_cluster) 
                              for i, item in enumerate(labels) 
                              if item == b.most_common()[1][0] 
                              and j == i]
        else:
            clustered_list = [atom for j, atom in enumerate(to_cluster) 
                              for i, item in enumerate(labels) 
                              if item == b.most_common()[0][0] 
                              and j == i]

        centroid = np.mean(clustered_list, axis=0)

        ind_list = [atoms[index] for index, j in enumerate(to_cluster) 
                    for i in clustered_list 
                    if j[0]==i[0] and j[1]==i[1] and j[2]==i[2]]

        index_list = [x+1 for x in ind_list]

        res_list = [self.universe.atoms[i].resindex for i in index_list]
        
        name_list = [self.universe.atoms[i].resname for i in index_list]

        tot_sel_res = tot_sel.residues.resnames

        return centroid, to_cluster, clustered_list, res_list, name_list, eps, min_num, labels
    

#     def cluster_check(self, sim_species, terminal_type, start, stop, step):
    def cluster_check(self, sim_species, terminal_type, start, stop, step):

    ###  This function will (1)find the largest cluster for each    ###
    ###  frame in a trajectory, (2)make a universe containing only  ###
    ###  the residues inside that cluster (pdb file), (3)visualize  ###
    ###  the cluster in red with the rest of the residues in purple ###
    
    # sim_species : 'prot', 'prot+nucleic', or 'nucleic'
    # terminal_type : 'capped', 'charged'
    # dens = biomolecular density of system (50, 125, or 250 mg/mL)
    # start = frame to start the analysis (int)(should be after equilibration portion of simulation)
    # stop = frame to stop the analysis (int)
    # step = stride to use to get from start to stop (int)
    
        ca_sel = self.universe.select_atoms('resid 2 and name CA')
        cutoff_num_res = int(ca_sel.n_atoms * 0.25) # cluster needs to contain >15% of residues
        
        for frame in self.universe.trajectory[start:stop:step]:

            centroid, to_cluster, clustered_list, res_list, name_list, eps, min_num, labels = self.clustering(sim_species, terminal_type, frame)

#             if len(Counter(labels)) > 1:
            if len(clustered_list) > cutoff_num_res:

                columns = ['x','y','z']
                all_df = pd.DataFrame(to_cluster, columns=columns)
                clustered_df = pd.DataFrame(clustered_list, columns=columns)
                
                import time
                import pylab as pl
                from IPython import display
                from ipywidgets import interact
                display.clear_output(wait=True)
                fig = px.scatter_3d(all_df, x='x', y='y', z='z', color_discrete_sequence=['steelblue'], range_x=[0,100], range_y=[0,100], range_z=[0,100])
                fig.add_trace(px.scatter_3d(clustered_df, x='x', y='y', z='z', color_discrete_sequence=['red'], range_x=[0,100], range_y=[0,100], range_z=[0,100]).data[0])
                fig.show(renderer='notebook');
                display.display(plt.gcf())
                time.sleep(0)
                print(frame.frame)
                
            else:
                columns = ['x','y','z']
                all_df = pd.DataFrame(to_cluster, columns=columns)
                clustered_df = pd.DataFrame(clustered_list, columns=columns)

                import time
                import pylab as pl
                from IPython import display
                from ipywidgets import interact
                display.clear_output(wait=True)
                fig = px.scatter_3d(all_df, x='x', y='y', z='z', color_discrete_sequence=['steelblue'], range_x=[0,100], range_y=[0,100], range_z=[0,100])
                fig.show(renderer='notebook');
                display.display(plt.gcf())
                time.sleep(0)
                print('hello', frame.frame)



    def cluster_rdf(self, sim_species, terminal_type, rdf_type, start, stop, step, output, mode='all'):

        ###  This function will (1)find the largest cluster for each    ###
        ###  frame in a trajectory, (2)make a universe containing only  ###
        ###  the residues inside that cluster (pdb file), (3)calculate  ###
        ###  rdf of that universe, and (4)average rdf's over clusters   ###
        ###  from each frame analyzed.                                  ###
        
        # sim_species : 'prot', 'prot+nucleic', or 'nucleic'
        # terminal_type : 'capped', 'charged'
        # start = frame to start the analysis (int)(should be after equilibration portion of simulation)
        # stop = frame to stop the analysis (int)
        # step = stride to use to get from start to stop (int)
        
        frame_num = 0
        for frame in self.universe.trajectory[start:stop:step]:

            centroid, to_cluster, clustered_list, res_list, name_list, eps, min_num, labels = self.clustering(sim_species, terminal_type, frame)

            selection = 'resindex '
            for item in res_list:
                selection += str(item) + ' or resindex '
            selection = selection[0:-13]
            sel = self.universe.select_atoms(selection)

            with mda.Writer(f'cluster.pdb') as pdb:
                pdb.write(sel)

            univ = mda.Universe(f'cluster.pdb')

            x_resid = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
                       'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
            
            groups = {}
            groups['polar'] = ['SER','THR','CYS','ASN','GLN']
            groups['nonpolar'] = ['GLY','ALA','PRO','VAL','LEU','ILE','MET']
            groups['aromatic'] = ['PHE','TRP','TYR']
            groups['positive'] = ['HIS','LYS','ARG']
            groups['negative'] = ['ASP','GLU']

            df = pd.DataFrame()

            if sim_species == 'prot':
                if rdf_type == 'residue':
                    for i in range(20):
                        for j in range(i, 20):
                            if(mode == 'all'):
                                a_group = univ.select_atoms("resname " + x_resid[i])
                                b_group = univ.select_atoms("resname " + x_resid[j])
                            elif(mode == 'ca'):
                                a_group = univ.select_atoms("resname " + x_resid[i] + " and name CA")
                                b_group = univ.select_atoms("resname " + x_resid[j] + " and name CA")
                            rdf = mda.analysis.rdf.InterRDF(a_group, b_group, nbins = 100, range = [1,25])
                            rdf.run()                                
                            if not "Radius" in df:
                                df["Radius"] = rdf.bins
                            df[x_resid[i] + "-" +  x_resid[j]] = rdf.rdf
                            
                elif rdf_type == 'group':   
                    key_list = list(groups.keys())
                    for i in range(5):
                        for j in range(i, 5):
                            temp_df = pd.DataFrame()
                            iters = 0
                            iter_array = np.zeros(100)
                            for k in groups[key_list[i]]:
                                for l in groups[key_list[j]]:
                                    a_group = univ.select_atoms('resname ' + k)
                                    b_group = univ.select_atoms('resname ' + l)
    
                                    rdf = mda.analysis.rdf.InterRDF(a_group, b_group, nbins = 100, range = [1,25])
                                    rdf.run()
                                    array = rdf.rdf
                                    iter_array = np.add(iter_array, array)
                                    iters += 1
                            iter_array = iter_array / iters
                            if not "Radius" in df:
                                df["Radius"] = rdf.bins
                            df[key_list[i] + "-" +  key_list[j]] = iter_array

            elif sim_species == 'prot+nucleic':
                if rdf_type == 'residue':
                    for res in x_resid:
                        a_group = univ.select_atoms("resname " + res + " and resnum 2")
                        b_group = univ.select_atoms("nucleic and resnum 2")
                        rdf = mda.analysis.rdf.InterRDF(a_group, b_group, nbins = 100, range = [1,25])
                        rdf.run()                                
                        if not "Radius" in df:
                            df["Radius"] = rdf.bins
                        df[res + "-nucleic"] = rdf.rdf
            
            array = df.to_numpy()
            columns = list(df.columns.values)
            
            if frame_num == 0:
                final_array = array
                frame_num += 1
            else:
                final_array = np.dstack((final_array, array))
                frame_num += 1
                
        avg_array = np.nanmean(final_array, axis=2)

        cluster_rdf_df = pd.DataFrame(avg_array, columns=columns)
        
        cluster_rdf_df.to_csv(output)
        

    def find_comp(self, sim_species, terminal_type, start, stop, step, output):
        
        ###  This function will (1)find the largest cluster for each    ###
        ###  frame in a trajectory, (2)extract the residue composition  ###
        ###  of the cluster, (3)output the average composition          ###
        ###  percentages throughout the given portion of the simulation ###
        
        # sim_species : 'prot', 'prot+nucleic', or 'nucleic' (str)
        # terminal_type : 'capped', 'charged' (str)
        # start = frame to start the analysis on (should be after equilibration portion of simulation (int)
        # stop = frame to stop the analysis on (int)
        # step = stride to use to get from start to stop (int)
        # output = filename for output file ('str)

        prot = self.universe.select_atoms('protein')
        nucleic = self.universe.select_atoms('nucleic')
        
        num_prot = prot.n_segments
        num_nucleic = nucleic.n_segments
        
        if sim_species == 'prot+nucleic':
            res_dict = {'ALA': 0, 'ARG': 0, 'ASP': 0, 'ASN': 0, 'HIS': 0, 'LYS': 0, 'PRO': 0, 'GLY': 0, 
                        'GLU': 0, 'THR': 0, 'SER': 0, 'GLN': 0, 'CYS': 0, 'TYR': 0, 'TRP': 0, 'PHE': 0, 
                        'VAL': 0, 'MET': 0, 'ILE': 0, 'LEU': 0, 'U': 0, 'G': 0, 'T': 0, 'C': 0, 'A': 0}
        elif sim_species == 'prot':
            res_dict = {'ALA': 0, 'ARG': 0, 'ASP': 0, 'ASN': 0, 'HIS': 0, 'LYS': 0, 'PRO': 0, 'GLY': 0,
                        'GLU': 0, 'THR': 0, 'SER': 0, 'GLN': 0, 'CYS': 0, 'TYR': 0, 'TRP': 0, 'PHE': 0,
                        'VAL': 0, 'MET': 0, 'ILE': 0, 'LEU': 0}
        else:
            res_dict = {'U': 0, 'G': 0, 'T': 0, 'C': 0, 'A': 0}

        x_resid = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
                       'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']

        num_res_dict = {}
        for i in res_dict.keys():
            if i == 'GLY':
                sel = self.universe.select_atoms(f'resid 2 and resname {i}')
            else:
                sel = self.universe.select_atoms(f'resname {i}')
            num_res = sel.n_residues
            num_res_dict[i] = num_res

        frame_num = 0
        for frame in self.universe.trajectory[start:stop:step]:

            centroid, to_cluster, clustered_list, res_list, name_list, eps, min_num, labels = self.clustering(sim_species, terminal_type, frame)

            res_ids = self.universe.residues[res_list]
            atom_ids = res_ids.atoms.ids

            resnames = res_ids.resnames
            resnames.sort()
            counter = Counter(resnames)

            for i, j in counter.items():
                #for key in res_dict.keys():
                #    if i == key:
                res_dict[i] += (j/num_res_dict[i])
            
            frame_num += 1

        upd_res_dict = {}
        for i, j in res_dict.items():
            upd_res_dict[i] = res_dict[i]/(frame_num)

        res_dict_df = pd.DataFrame(upd_res_dict.items(), columns=['Residue','Frequency'])

        res_dict_df.to_csv(output)



    def cluster_size(self, sim_species, terminal_type, start, stop, step, output):
        
        ###   This method will output a dataframe containing information about the ratio   ###
        ###   of residues inside the cluster out of the total number of residues in the    ###
        ###                                system as a whole.                              ###
        
        prot_size_list = []
        nucleic_size_list = []
        frame_list = []      
        size_df = pd.DataFrame()
        
        for frame in self.universe.trajectory[start:stop:step]:

            prot = self.universe.select_atoms('protein')
            num_prot = int(prot.n_residues / 3)
            
            nucleic = self.universe.select_atoms('nucleic')
            num_nucleic = int(nucleic.n_residues / 3)

            centroid, to_cluster, clustered_list, res_list, name_list, eps, min_num, labels = self.clustering(sim_species, terminal_type, frame)
            
            ca_sel = self.universe.select_atoms('resid 2 and name CA')
            cutoff_num_res = int(ca_sel.n_atoms * 0.15) # cluster needs to contain >15% of residues
            
            if len(clustered_list) > cutoff_num_res:

                prot_list = []
                nucleic_list = []
                for i in name_list:
                    if len(i) == 3:
                        prot_list.append(i)
                    else:
                        nucleic_list.append(i)

                prot_size = len(prot_list)
                nucleic_size = len(nucleic_list)

                if sim_species == 'prot+nucleic':
                    prot_size_list.append(prot_size/num_prot)
                    nucleic_size_list.append(nucleic_size/num_nucleic)
                elif sim_species == 'prot':
                    prot_size_list.append(prot_size/num_prot)
                    nucleic_size_list.append(nucleic_size)
                else:
                    prot_size_list.append(prot_size)
                    nucleic_size_list.append(nucleic_size/num_nucleic)

                frame_list.append(frame.frame)

        size_df['frame'] = frame_list
        size_df['prot-size'] = prot_size_list
        size_df['nucleic-size'] = nucleic_size_list
        
        size_df.to_csv(output)


    def layer_comp(self, sim_species, terminal_type, layers, alpha, dens, start, stop, step, output_1, output_2):

        ###  This method will output a dataframe containing information about the average  ###
        ###   hydrophobicity and number of charged central residues in each layer of the   ###
        ###                  clusters on average throughout the simulation.                ###
        
        polar = ['SER','THR','CYS','ASN','GLN']
        nonpolar = ['GLY','ALA','PRO','VAL','LEU','ILE','MET']
        aromatic = ['PHE','TRP','TYR']
        charged = ['HIS','LYS','ARG','ASP','GLU']

        shell_dict = {'ALA': 0, 'ARG': 0, 'ASP': 0, 'ASN': 0, 'HIS': 0, 'LYS': 0, 
                      'PRO': 0, 'GLY': 0, 'GLU': 0, 'THR': 0,'SER': 0, 'GLN': 0, 
                      'CYS': 0, 'TYR': 0, 'TRP': 0, 'PHE': 0, 'VAL': 0, 'MET': 0, 
                      'ILE': 0, 'LEU': 0}
        middle_dict = {'ALA': 0, 'ARG': 0, 'ASP': 0, 'ASN': 0, 'HIS': 0, 'LYS': 0,
                       'PRO': 0, 'GLY': 0, 'GLU': 0, 'THR': 0,'SER': 0, 'GLN': 0, 
                       'CYS': 0, 'TYR': 0, 'TRP': 0, 'PHE': 0, 'VAL': 0, 'MET': 0, 
                       'ILE': 0, 'LEU': 0}
        core_dict = {'ALA': 0, 'ARG': 0, 'ASP': 0, 'ASN': 0, 'HIS': 0, 'LYS': 0, 
                     'PRO': 0, 'GLY': 0, 'GLU': 0, 'THR': 0,'SER': 0, 'GLN': 0, 
                     'CYS': 0, 'TYR': 0, 'TRP': 0, 'PHE': 0, 'VAL': 0, 'MET': 0, 
                     'ILE': 0, 'LEU': 0}

        core_charged_res = []

        middle_charged_res = []

        shell_charged_res = []

        dist_list_core = []

        dist_list_middle = []

        dist_list_shell = []

        hphob_dict = {'ILE':0.972973,'VAL':0.891892,'LEU':0.972973,'PHE':1.0,'CYS':0.594595,
                      'MET':0.837838,'ALA':0.72973,'GLY':0.648649,'THR':0.675676,
                      'SER':0.594595,'TRP':0.945946,'TYR':0.864865,'PRO':1.0,'HIS':0.513514,
                      'GLU':0.459459,'GLN':0.513514,'ASP':0.378378,'ASN':0.432432,
                      'LYS':0.513514,'ARG':0.0}

        frames = 0

        for frame in self.universe.trajectory[start:stop:step]:

            centroid, to_cluster, clustered_list, res_list, name_list, eps, min_num, labels = self.clustering(sim_species, terminal_type, frame, dens)

            selection = 'resindex '
            for item in res_list:
                selection += str(item)+ ' or resindex '
            selection = selection[0:-13]
            sel = u.select_atoms(selection)

            with mda.Writer('./spatial.pdb') as pdb:
                pdb.write(sel)

            univ = mda.Universe('./spatial.pdb')

            g = univ.select_atoms('name CA')

            interface = pytim.GITIM(univ, group=g, alpha=alpha, max_layers=layers, molecular=True, centered='middle')

            if len(interface.layers[2].residues) > 5:

                for i,j in Counter(interface.layers[2].resnames).items():
                    for k,l in core_dict.items():
                        if i == k:
                            core_dict[k] += j
                for i,j in Counter(interface.layers[1].resnames).items():
                    for k,l in middle_dict.items():
                        if i == k:
                            middle_dict[k] += j
                for i,j in Counter(interface.layers[0].resnames).items():
                    for k,l in shell_dict.items():
                        if i == k:
                            shell_dict[k] += j
                core_charge = 0
                for i,j in Counter(interface.layers[2].resnames).items():
                    if i in charged:
                        core_charge += j
                    elif i in polar:
                        core_charge += j
                core_charged_res.append(core_charge/(len(interface.layers[2].resnames)))

                middle_charge = 0
                for i,j in Counter(interface.layers[1].resnames).items():
                    if i in charged:
                        middle_charge += j
                    elif i in polar:
                        middle_charge += j
                middle_charged_res.append(middle_charge/(len(interface.layers[1].resnames)))

                shell_charge = 0
                for i,j in Counter(interface.layers[0].resnames).items():
                    if i in charged:
                        shell_charge += j
                    elif i in polar:
                        shell_charge += j
                shell_charged_res.append(shell_charge/(len(interface.layers[0].resnames)))

                centroid = np.mean(univ.atoms.positions, axis=0)

                dist_list_core.append(np.mean([np.sqrt(((i[0]-centroid[0])**2) + 
                                                       ((i[1]-centroid[1])**2) + 
                                                       ((i[2]-centroid[2])**2)) 
                                                        for i in interface.layers[2].positions])/10)

                dist_list_middle.append(np.mean([np.sqrt(((i[0]-centroid[0])**2) + 
                                                       ((i[1]-centroid[1])**2) + 
                                                       ((i[2]-centroid[2])**2)) 
                                                        for i in interface.layers[1].positions])/10)

                dist_list_shell.append(np.mean([np.sqrt(((i[0]-centroid[0])**2) + 
                                                       ((i[1]-centroid[1])**2) + 
                                                       ((i[2]-centroid[2])**2)) 
                                                        for i in interface.layers[2].positions])/10)

                frames += 1

        for layer in [core_dict, middle_dict, shell_dict]:
            for i,j in layer.items():
                layer[i] /= frames

        core_df = pd.DataFrame.from_dict(core_dict, orient='index', columns=['Avg_Count'])
        core_df.index.name = 'Residue'
        core_df = core_df.sort_values(by='Avg_Count')
        for i,j in hphob_dict.items():
            core_df.loc[i,'Hydrophobicity'] = j
        core_df['lambda'] = core_df['Avg_Count'] * core_df['Hydrophobicity']

        middle_df = pd.DataFrame.from_dict(middle_dict, orient='index', columns=['Avg_Count'])
        middle_df.index.name = 'Residue'
        middle_df = middle_df.sort_values(by='Avg_Count')
        for i,j in hphob_dict.items():
            middle_df.loc[i,'Hydrophobicity'] = j
        middle_df['lambda'] = middle_df['Avg_Count'] * middle_df['Hydrophobicity']

        shell_df = pd.DataFrame.from_dict(shell_dict, orient='index', columns=['Avg_Count'])
        shell_df.index.name = 'Residue'
        shell_df = shell_df.sort_values(by='Avg_Count')
        for i,j in hphob_dict.items():
            shell_df.loc[i,'Hydrophobicity'] = j
        shell_df['lambda'] = shell_df['Avg_Count'] * shell_df['Hydrophobicity']

        core_value = core_df['lambda'].sum() / sum(core_dict.values())

        middle_value = middle_df['lambda'].sum() / sum(middle_dict.values())

        shell_value = shell_df['lambda'].sum() / sum(shell_dict.values())

        core_dist_value = "%.1f" % (np.mean(dist_list_core))

        mid_dist_value = "%.1f" % (np.mean(dist_list_middle))

        shell_dist_value = "%.1f" % (np.mean(dist_list_shell))

        layer_dict = {core_dist_value:core_value, 
                      mid_dist_value:middle_value, 
                      shell_dist_value:shell_value}

        charge_dict = {core_dist_value:np.mean(core_charged_res), 
                       mid_dist_value:np.mean(middle_charged_res), 
                       shell_dist_value:np.mean(shell_charged_res)}
        
        
        hphob_dict_df = pd.DataFrame(layer_dict.items(), columns=['distance','hydrophobicity'])
        
        charge_dict_df = pd.DataFrame(charge_dict.items(), columns=['distance','polar/charged_fraction'])
        
        
        hphob_dict_df.to_csv(output_1)
        
        charge_dict_df.to_csv(output_2)
        

        return hphob_dict_df, charge_dict_df
    
    def phase_densities(self, sim_species, terminal_type, start, stop, step, output, chain_length):
    
        frame_num = 0

        phases_df = pd.DataFrame(columns=['dense_phase_density_mgml','dilute_phase_density_mgml','ratio'])
        
        

        for frame in self.universe.trajectory[start:stop:step]:

            centroid, to_cluster, clustered_list, res_list, name_list, eps, min_num, labels = self.clustering(sim_species, terminal_type, frame, chain_length)
            
            ca_sel = self.universe.select_atoms('(resid 2 and name CA) or (resid 2 and name P)')
            cutoff_num_res = int(ca_sel.n_atoms * 0.25) # cluster needs to contain >20% of residues
            
            if len(clustered_list) > cutoff_num_res:

                selection = 'resindex '
                for item in res_list:
                    selection += str(item)+ ' or resindex '
                selection = selection[0:-13]
                sel = self.universe.select_atoms(selection)
                all_sel = sel.residues.segments.residues.atoms

                with mda.Writer('./cluster.pdb') as pdb:
                    pdb.write(all_sel)

                univ = mda.Universe('./cluster.pdb', guess_bonds=True)

                box_volume_ml = (univ.dimensions[0]*univ.dimensions[1]*univ.dimensions[2])/ (1.0E24)

                prot = self.universe.select_atoms('protein')
                system_prot_list = prot.residues.resnames.tolist()
                box_prot_str = listtostring(system_prot_list)
                new_box_prot_str = seq1(box_prot_str)
                box_prot_mass_amu = molecular_weight(new_box_prot_str,'protein')
                
                nucleic = self.universe.select_atoms('nucleic')
                system_nucleic_list = nucleic.residues.resnames.tolist()
                box_nucleic_str = listtostring(system_nucleic_list)
                try:
                    box_nucleic_mass_amu = molecular_weight(box_nucleic_str,'DNA')
                except ValueError:
                    box_nucleic_mass_amu = molecular_weight(box_nucleic_str,'RNA')

                box_mass_mg = (box_prot_mass_amu + box_nucleic_mass_amu) / 6.023e20

                pipeline = import_file('./cluster.pdb')
                
                ratio = len(clustered_list) / ca_sel.n_atoms
                
                if len(clustered_list) > int(ca_sel.n_atoms * 0.85):
                    pipeline.modifiers.append(ConstructSurfaceModifier(radius = 15, smoothing_level = 10, only_selected = False, identify_regions = True, map_particles_to_regions = True))
                elif len(clustered_list) > int(ca_sel.n_atoms * 0.65):
                    pipeline.modifiers.append(ConstructSurfaceModifier(radius = 13, smoothing_level = 10, only_selected = False, identify_regions = True, map_particles_to_regions = True))
                elif len(clustered_list) > int(ca_sel.n_atoms * 0.45):
                    pipeline.modifiers.append(ConstructSurfaceModifier(radius = 11, smoothing_level = 10, only_selected = False, identify_regions = True, map_particles_to_regions = True))
                else:
                    pipeline.modifiers.append(ConstructSurfaceModifier(radius = 9, smoothing_level = 10, only_selected = False, identify_regions = True, map_particles_to_regions = True))
                
                data = pipeline.compute()

                cluster_volume_ml = (data.attributes['ConstructSurfaceMesh.filled_volume']) / (1.0E24)
                
                prot_list = []
                nucleic_list = []
                
                for i in name_list:
                    if len(i) == 3:
                        prot_list.extend(['GLY']*(math.ceil(chain_length/2)-1))
                        prot_list.append(i)
                        prot_list.extend('GLY'*(math.ceil(chain_length/2)-1))
                    else:
                        nucleic_list.extend('U'*(math.ceil(chain_length/2)-1))
                        nucleic_list.append(i)
                        nucleic_list.extend('U'*(math.ceil(chain_length/2)-1))
                
                cluster_prot_list = prot_list
                cluster_prot_str = listtostring(cluster_prot_list)
                new_cluster_prot_str = seq1(cluster_prot_str)
                cluster_prot_mass_amu = molecular_weight(new_cluster_prot_str,'protein')
                
                cluster_nucleic_list = nucleic_list
                cluster_nucleic_str = listtostring(cluster_nucleic_list)
                try:
                    cluster_nucleic_mass_amu = molecular_weight(cluster_nucleic_str,'DNA')
                except ValueError:
                    cluster_nucleic_mass_amu = molecular_weight(cluster_nucleic_str,'RNA')

                cluster_mass_amu = cluster_prot_mass_amu + cluster_nucleic_mass_amu
                    
                cluster_mass_mg = (cluster_prot_mass_amu + cluster_nucleic_mass_amu) / 6.023e20

                dilute_mass_mg = box_mass_mg - cluster_mass_mg

                dilute_volume_ml = box_volume_ml - cluster_volume_ml

                cluster_dens = cluster_mass_mg / cluster_volume_ml
                dilute_dens = dilute_mass_mg / dilute_volume_ml

                phases_df.loc[frame.frame, 'dense_phase_density_mgml'] = cluster_dens
                phases_df.loc[frame.frame, 'dilute_phase_density_mgml'] = dilute_dens
                phases_df.loc[frame.frame, 'ratio'] = ratio
                
                phases_df.to_csv(output)
            
            else:
                continue

#         return phases_df, cluster_mass_mg, dilute_mass_mg, dilute_volume_ml, cluster_volume_ml, clustered_list, ca_sel, name_list
        return phases_df


########################################
########################################
###   BELOW ARE PARTIALLY FINISHED   ###
###      OR OUTDATED FUNCTIONS       ###
###          NOT TO BE USED          ###
########################################
########################################

    def surface_comp(self, sim_species, terminal_type, start, stop, step, output):

        prot = self.universe.select_atoms('protein')
        nucleic = self.universe.select_atoms('nucleic')
        
        num_prot = prot.n_segments
        num_nucleic = nucleic.n_segments
 
        frame_num = 0

        if sim_species == 'prot+nucleic':
            shell_dict = {'ALA': 0, 'ARG': 0, 'ASP': 0, 'ASN': 0, 'HIS': 0, 'LYS': 0, 'PRO': 0, 'GLY': 0, 'GLU': 0, 'THR': 0, 
                          'SER': 0, 'GLN': 0, 'CYS': 0, 'TYR': 0, 'TRP': 0, 'PHE': 0, 'VAL': 0, 'MET': 0, 'ILE': 0, 'LEU': 0, 
                          'U': 0, 'G': 0, 'T': 0, 'C': 0, 'A': 0}
        elif sim_species == 'prot':
            shell_dict = {'ALA': 0, 'ARG': 0, 'ASP': 0, 'ASN': 0, 'HIS': 0, 'LYS': 0, 'PRO': 0, 'GLY': 0, 'GLU': 0, 'THR': 0,
                          'SER': 0, 'GLN': 0, 'CYS': 0, 'TYR': 0, 'TRP': 0, 'PHE': 0, 'VAL': 0, 'MET': 0, 'ILE': 0, 'LEU': 0}
        else:
            shell_dict = {'U': 0, 'G': 0, 'T': 0, 'C': 0, 'A': 0}
        
        x_resid = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
                       'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']

        #num_res_dict = {}
        #for i in shell_dict.keys():
        #    if i == 'GLY':
        #        sel = self.universe.select_atoms(f'resid 2 and resname {i}')
        #    else:
        #        sel = self.universe.select_atoms(f'resname {i}')
        #    num_res = sel.n_residues
        #    num_res_dict[i] = num_res

        for frame in self.universe.trajectory[start:stop:step]:

            centroid, to_cluster, clustered_list, res_list, name_list, eps, min_num, labels = self.clustering(sim_species, terminal_type, frame)

            selection = 'resindex '
            for item in res_list:
                selection += str(item)+ ' or resindex '
            selection = selection[0:-13]
            sel = self.universe.select_atoms(selection)

            clustered_res_dict = Counter(name_list)

            with mda.Writer(f'spatial.pdb') as pdb:
                pdb.write(sel)

            univ = mda.Universe(f'spatial.pdb')

            g = univ.select_atoms('name CA or name P')

            interface = pytim.GITIM(univ, group=g, alpha=9, max_layers=1, molecular=False, centered='middle')

            for key, value in Counter(interface.layers[0].resnames).items():
                shell_dict[key] += (value / clustered_res_dict[key])

            pdb.close()
            frame_num += 1

        for i, j in shell_dict.items():
            shell_dict[i] /= frame_num
#            if len(i) == 3:
#                shell_dict[i] /= num_prot
#            elif len(i) == 1:
#                shell_dict[i] /= num_nucleic
            
        shell_df = pd.DataFrame.from_dict(shell_dict, orient='index', columns=['Frequency'])
        shell_df.index.name = 'Residue'
        shell_df = shell_df.sort_values(by='Frequency')
        
        shell_df.to_csv(output)

        return shell_df
    
    
    def cluster_water(self, sim_species, terminal_type, output, start, stop, step, dens):
        
        percentage_df = pd.DataFrame()
        percentage_list = []
        frame_list = []
        dist_list = []
        num_water_list = []
        mass_ratio_list = []
        
        for frame in self.universe.trajectory[start:stop:step]:
            
            tot_water_sel = self.universe.select_atoms('resname HOH')
            tot_water = tot_water_sel.n_residues
            
            centroid, to_cluster, clustered_list, res_list, name_list, eps, min_num = self.clustering(terminal_type, frame)
            
            if len(clustered_list) > 25:
                
                aa_str = calist_listtostring(name_list)

                new_aa_str = seq1(aa_str)

                pep_mass_amu = molecular_weight(new_aa_str,'protein')
                        
                res_df = pd.DataFrame()
                res_df['x'] = [item[0] for item in clustered_list]
                res_df['y'] = [item[1] for item in clustered_list]
                res_df['z'] = [item[2] for item in clustered_list]

                res_df['diff'] = (((centroid[0]-res_df['x'])**2)+
                                  ((centroid[1]-res_df['y'])**2)+
                                  ((centroid[2]-res_df['z'])**2))**0.5

                diff_list = res_df['diff'].tolist()

                diff_list = [100 - x if x > 50 else x for x in diff_list]

                res_df['difference'] = diff_list

                res_df = res_df.drop(['diff'], axis=1)
                
                clust_dist = ((res_df['difference'].max()*2) + res_df['difference'].mean())/3

                selection = 'resindex '
                for item in res_list:
                    selection += str(item) + ' or resindex '
                selection = selection[0:-13]
                sel_prot = self.universe.select_atoms(selection + ' or resname HOH')

                with mda.Writer(f'cluster.pdb') as pdb:
                    pdb.write(sel_prot)

                temp_univ = mda.Universe('cluster.pdb')
                
                if dens == 50:
                    total_sel = temp_univ.select_atoms('protein or around 6 protein')
                else:
                    total_sel = temp_univ.select_atoms('protein or around 4 protein')
                    
                with mda.Writer('cluster.pdb') as pdb:
                    pdb.write(total_sel)

                univ = mda.Universe('cluster.pdb')
                
                prot = univ.select_atoms('protein')
                
                dists = distances.self_distance_array(prot.positions, box=frame.dimensions)
                
                avg_dist = (dists.mean() + dists.max())/2

                water = univ.select_atoms('resname HOH')

                n_water = water.n_residues

                percent = n_water / tot_water
                
                mass_ratio = ((n_water*18.015) / pep_mass_amu)
                
                percentage_list.append(percent)
                frame_list.append(frame.frame)
                dist_list.append(avg_dist)
                num_water_list.append(n_water)
                mass_ratio_list.append(mass_ratio)
                
            else:
                continue
        
        percentage_df['Frame'] = frame_list
        percentage_df['Num_water'] = num_water_list
        percentage_df['% Water inside Droplet'] = percentage_list
        percentage_df['Distance'] = dist_list
        percentage_df['Water/Peptide Mass Ratio'] = mass_ratio_list

        mean = percentage_df['% Water inside Droplet'].mean()

        avg_radius = percentage_df['Distance'].mean()

        volume_mL = ((4/3)*(3.14)*(avg_radius**3))*(1.0*(10**-24))
        
        volume_A = ((4/3)*(3.14)*(avg_radius**3))

        percentage_df['Density (mg/mL)'] = (percentage_df['Num_water'] * (2.99*(10**-29))) / volume_mL
        
        percentage_df['Density (#/A^3)'] = (percentage_df['Num_water'] / volume_A)
        
        percentage_df.to_csv(output)

        return percentage_df
    
    def cluster_hbonding(self, sim_species, terminal_type, dens, start, stop, step):
        
        res_list = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
                    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']

        hb_df = pd.DataFrame(index = res_list, columns = ['avg_hbonds', 'error'])

        if dens == 50:
            num_res = 5
        elif dens == 125:
            num_res = 15
        else:
            num_res = 30

        for res in res_list:

            if res == 'GLY':
                h_sel = f'resname {res} and resnum 2'
            else:
                h_sel = f'resname {res}'

        for frame in self.universe.trajectory[start:stop:step]:

            centroid, to_cluster, clustered_list, res_list, name_list, eps, min_num = self.clustering(terminal_type, frame)

            selection = 'resindex '
            for item in res_list:
                selection += str(item) + ' or resindex '
            selection = selection[0:-13]
            sel = self.universe.select_atoms(selection)

            with mda.Writer(f'cluster.pdb') as pdb:
                pdb.write(sel)

            univ = mda.Universe(f'cluster.pdb')

            hbonds = HydrogenBondAnalysis(
                universe=univ,
                donors_sel=None,
                hydrogens_sel=h_sel,
                acceptors_sel="not protein and not name CL and not name NA",
                d_a_cutoff=3.0,
                d_h_a_angle_cutoff=150,
                update_selections=False)

            hbonds.run(verbose=True)

            avg = (np.mean(hbonds.count_by_time()) / num_res)

            error = (np.std(hbonds.count_by_time()) / num_res)

            hb_df.loc[res, 'avg_hbonds'] += avg
            hb_df.loc[res, 'error'] += error

            

    def cluster_dists(self, sim_species, terminal_type, start, stop, step, output):
        
        ###  This function will (1)find the largest cluster for each        ###
        ###  frame in a trajectory, (2)make a universe containing only      ###
        ###  the residues inside that cluster (pdb file), (3)run a distance ###
        ###  calculation on that pdb file, and (4)ultimately average        ###
        ###  all the distances for a final df of ONLY the residue distances ###
        ###  inside the cluster.                                            ###
        
        # start = frame to start the analysis on (should be after equilibration portion of simulation
        # stop = frame to stop the analysis on
        # step = stride to use to get from start to stop
    
        frame_num = 0

        resid = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
                 'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']

        combos = list(itertools.combinations_with_replacement(resid, 2))

        combolist = [i[0]+'-'+i[1] for i in combos]

        for frame in self.universe.trajectory[start:stop:step]:

            centroid, to_cluster, clustered_list, res_list, name_list, eps, min_num = self.clustering(terminal_type, frame)

            selection = 'resindex '
            for item in res_list:
                selection += str(item)+ ' or resindex '
            selection = selection[0:-13]
            sel = self.universe.select_atoms(selection)

            with mda.Writer(f'cluster_dists.pdb') as pdb:
                pdb.write(sel)

            univ = mda.Universe(f'cluster_dists.pdb')
            
            dist_dict = {}
            for i in resid:
                dist_dict[i] = 0

            for i in range(20):
                for j in range(i, 20):
                    name = resid[i] + '-' + resid[j]
                    a_group = univ.select_atoms("resname " + resid[i] + " and name CA")
                    b_group = univ.select_atoms("resname " + resid[j] + " and name CA")

                    dist_arr = distances.distance_array(a_group.positions, # reference
                                                        b_group.positions, # configuration
                                                        box=univ.dimensions)

                    new_dist_arr = np.nan_to_num(dist_arr)
                    avg = np.nan_to_num(np.nanmean(new_dist_arr))

                    if avg != 0:
                        for key in dist_dict.keys():
                            if name == key:
                                dist_dict[key] = avg
                    else:
                        continue
                    
            df = pd.DataFrame.from_dict(dist_dict, orient='index', columns=['Average Distance'])
            df.index.name = 'Pair'

            df2 = (pd.DataFrame(df.index.str.split('-',1).tolist(), columns = ['Res-1','Res-2']))
            df2['Average Distance'] = df['Average Distance'].tolist()

            result = df2.pivot(index='Res-1',columns='Res-2',values='Average Distance')
            result2 = df2.pivot(index='Res-2',columns='Res-1',values='Average Distance')

            avg_dist_df = result.combine_first(result2)

            array = avg_dist_df.to_numpy()
            columns = list(avg_dist_df.columns.values)

            if frame_num == 0:
                final_array = array
            else:
                final_array = np.dstack((final_array, array))

            frame_num += 1

        avg_array = np.nanmean(final_array, axis=2)

        cluster_dists_df = pd.DataFrame(avg_array, columns=columns)
        
        cluster_dists_df.to_csv(output)    
    

