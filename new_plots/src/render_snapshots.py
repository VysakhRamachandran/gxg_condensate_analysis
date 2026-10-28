"""Render publication-quality snapshots of peptide condensates with PyMOL.

For each requested system, opens the trajectory at a representative frame
and produces a snapshot showing:
  - non-glycine (sticker) residues highlighted by chemistry class
  - glycine spacer residues as a faint grey backbone
  - background white, ray-traced
"""
import os
import subprocess

OUT = "/Volumes/potoyan-lab/will/gxg-project/new_plots/manuscript_figs/snapshots"
GXG = "/Volumes/potoyan-lab/will/gxg-project/gxg_comps"
os.makedirs(OUT, exist_ok=True)

# representative systems and their structure files
SYSTEMS = {
    "3C":  f"{GXG}/3_res/charged/output-nw.pdb",
    "9C":  f"{GXG}/9_res/charged/output-nw.pdb",
    "X3":  f"{GXG}/9_res/equal/3_close/output-nw.pdb",
    "X3S": f"{GXG}/9_res/equal/3_spread/output-nw.pdb",
}

# pymol script template
def pml_script(in_pdb, out_png, sys_label):
    return f"""
load {in_pdb}, struct
bg_color white
hide everything
# remove ions and water from the visualization
remove resn NA+CL+SOD+CLA+HOH+TIP3+WAT+SOL
# peptides as sticks (looks cleaner than spheres for these short chains)
show sticks, struct
set stick_radius, 0.18
# also show heavy-atom spheres at a small scale to fill volume
show spheres, struct and not name C+N+O+CA
set sphere_scale, 0.20, struct
# glycine = faint grey
color grey80, struct and resn GLY
# stickers — color by chemistry class
color marine,    struct and resn ASP+GLU
color firebrick, struct and resn LYS+ARG+HIS
color salmon,    struct and resn PHE+TYR+TRP
color forest,    struct and resn SER+THR+ASN+GLN+CYS
color sand,      struct and resn ALA+VAL+LEU+ILE+MET+PRO
# camera setup
orient struct
zoom struct, 3
set ray_opaque_background, 0
ray 1800, 1800
png {out_png}, dpi=220
quit
"""

def render_one(label):
    in_pdb = SYSTEMS[label]
    if not os.path.exists(in_pdb):
        print(f"  missing input: {in_pdb}"); return
    out_png = os.path.join(OUT, f"snap_{label}.png")
    pml = pml_script(in_pdb, out_png, label)
    script_path = os.path.join(OUT, f"_render_{label}.pml")
    with open(script_path, "w") as fh:
        fh.write(pml)
    cmd = ["pymol", "-c", "-q", script_path]
    print(f"  rendering {label}…", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("STDERR:", res.stderr)
    else:
        print(f"    saved {out_png}")

if __name__ == "__main__":
    for s in SYSTEMS:
        render_one(s)
    print("Done.")
