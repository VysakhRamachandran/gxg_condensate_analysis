residue_list = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','HIS','ILE','LEU',
                'LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GLY']
    
res_list = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU',
            'LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']

conts_df = pd.DataFrame(index=res_list, columns=res_list)
conts_df = conts_df.fillna(float(0))

top = 'good_output.top'
dcd = '5ms-traj.dcd'

gly_u = mda.Universe(load_file(top), dcd)

gly = gly_u.select_atoms('resname GLY')

selection = ''
for resid in gly.residues:
    if len(list(resid.atoms)) == 7:
        selection += 'GLY' + str(resid.resnum) + '|'

selection = selection[0:-1]

for residue in residue_list:
    
    now = datetime.now()
    print(residue, now.time())
    
    if residue == 'GLY':
        
        residue_df = pd.read_csv(f'./new_plf_files/{residue}_plf.csv', header = [0,1,2], index_col=0)
        
        residue_df = residue_df[residue_df.columns.levels[0][residue_df.columns.levels[0].str.fullmatch(selection)]]
        
        index_list = []

        for j,i in enumerate(residue_df):
            if i[0] != i[1]:
                index_list.append(j)
        
        residue_df = residue_df.iloc[:,index_list]

        residue_df.columns = residue_df.columns.droplevel(0)
        residue_df.columns = residue_df.columns.droplevel(1)
        
    else:
        
        residue_df = pd.read_csv(f'./new_plf_files/{residue}_plf.csv', header = [0,1,2], index_col=0)

        index_list = []

        for j,i in enumerate(residue_df):
            if i[0] != i[1]:
                index_list.append(j)
        
        residue_df = residue_df.iloc[:,index_list]
        
        residue_df.columns = residue_df.columns.droplevel(0)
        residue_df.columns = residue_df.columns.droplevel(1)
    
    for res in res_list:
        if res == 'GLY':

            res_df = residue_df[residue_df.columns[residue_df.columns.str.fullmatch(selection)]]

        else:

            res_df = residue_df[residue_df.filter(like=f'{res}').columns]

        frames = res_df.shape[0]
        
        if conts_df.loc[residue, res] == 0.0:
            conts_df.loc[residue, res] = res_df.values.sum()/frames
        else:
            continue
        
        if conts_df.loc[res,residue] == 0.0:  
            conts_df.loc[res, residue] = res_df.values.sum()/frames
        else:
            continue
            
conts_df.to_csv('./central_res_conts.csv')