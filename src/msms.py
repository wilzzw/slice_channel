import subprocess
import os

import numpy as np

from conf.conf_analysis import xyzt


def read_triangles(face_outfile):
    read_facefile = np.loadtxt(face_outfile, dtype=int, skiprows=3)
    return read_facefile[:,:3] - 1 # Convert to zero based indices

def read_vertices(vert_outfile):
    read_vertfile = np.loadtxt(vert_outfile, skiprows=3)
    return read_vertfile[:,:3]

def triangle_zcut(triangulation, zlevel):
    zproj = triangulation[:,:,2]

    above_zlevel = np.any(zproj > zlevel, axis=1)
    # above_zlevel = np.any(zproj >= zlevel, axis=1)
    below_zlevel = np.any(zproj < zlevel, axis=1)
    cross_zlevel = np.all([above_zlevel, below_zlevel], axis=0)

    crossing_trg_indices, = np.where(cross_zlevel)
    select_triangles = triangulation[crossing_trg_indices]
    return select_triangles

def trgs_zslice_vertices(triangulation, zlevel):
    # Indices of vertex pairs to make directed line segments of triangles
    lineseg_vertindex = np.array([[0,1],[1,2],[2,0]])

    r_ij = np.diff(triangulation[:, lineseg_vertindex, :], axis=2)
    r_ij = np.squeeze(r_ij, axis=2)

    f = (zlevel - triangulation[:,:,2]) / r_ij[:,:,2]

    # Taking equal sign for one of them is important
    # But not both
    # Address pathological case:
    # f_j = [-0.          0.74890255  1.        ]
    # Triangulation_j =
    # [[  5.244  40.565 121.   ]
    # [  4.432  40.282 120.147]
    # [  5.244  39.703 121.286]]
    through_zlevel = ((f >= 0) & (f < 1))
    if len(through_zlevel) == 0:
        return None
        
    ### problematic test = np.where(~(np.sum(through_zlevel, axis=1) == 2)) ###
    assert np.all(np.sum(through_zlevel, axis=1) == 2)

    f = np.expand_dims(f, axis=2)
    linseg_onzlevel = triangulation + f * r_ij

    indices_select = np.squeeze(np.apply_along_axis(np.argwhere, 1, through_zlevel))
    lineseg_xy = linseg_onzlevel[np.arange(len(linseg_onzlevel))[..., np.newaxis], indices_select, :2]

    return lineseg_xy

class prot_msms(xyzt):
    def __init__(self, traj_id, probe_radius=1.4, look_for=False):
        super().__init__(traj_id, close_after_init=False)
        # TODO: temporary
        self.load_refs()

        self.traj_id = traj_id
        self.probe_radius = probe_radius
        self.look_for = look_for
        print(self.grotop_ids)
        self.grotop_id = self.grotop_ids[0]
        self.ref = self.refs.get(self.grotop_id)
        self.prot_aindex = self.ref.top.select("protein")
        self.protein = self.ref.atom_slice(self.prot_aindex)

        # In Angstroms for both
        self.radii = np.array([a.element.radius for a in self.protein.top.atoms]) * 10
        self.xyz = self.prot_xyz.variables['coordinate'][self.traj_id, :self.nframes.get(self.traj_id), self.prot_aindex, :] * 10

        self.prot_xyz.close()

    def run(self, timestep, cmd="bin/msms", xyzr_name="msms", outname="MSMS"):
        self.write_xyzr(timestep, xyzr_name)
        self.run_msms(timestep, cmd, outname)
        self.triangles(timestep)

    def write_xyzr(self, timestep, xyzr_name):
        self.xyzr_name = "tmp/{}-{}".format(self.traj_id, xyzr_name)
        full_xyzr_name = "{}_{}.xyzr".format(self.xyzr_name, timestep)

        if self.look_for and os.path.exists(full_xyzr_name):
            return
            
        with open(full_xyzr_name, "w") as output:
            for xyz, r in zip(self.xyz[timestep], self.radii):
                x, y, z = xyz
                output.write(f"{x:6.3f}\t{y:6.3f}\t{z:6.3f}\t{r:1.2f}\n")

    def run_msms(self, timestep, cmd="bin/msms", outname="MSMS"):
        self.msms_outname = "results/data/{}-{}".format(self.traj_id, outname)
        msms_outname = "{}_{}".format(self.msms_outname, timestep)
        full_xyzr_name = "{}_{}.xyzr".format(self.xyzr_name, timestep)
        face_name = "{}_{}.face".format(self.msms_outname, timestep)
        vert_name = "{}_{}.vert".format(self.msms_outname, timestep)

        if self.look_for and os.path.exists(face_name) and os.path.exists(vert_name):
            return
        command = "{} -probe_radius {} -if {} -of {}".format(cmd, self.probe_radius, full_xyzr_name, msms_outname)
        subprocess.run(command.split())

    def triangles(self, timestep):
        face_name = "{}_{}.face".format(self.msms_outname, timestep)
        vert_name = "{}_{}.vert".format(self.msms_outname, timestep)

        self.faces = read_triangles(face_name)
        self.vertices = read_vertices(vert_name)

        triangulation = self.vertices[self.faces] # .shape = (num_triangles, num_vertices=3, num_dimensions_xyz=3)
        self.triangulation = triangulation

    def visual_msms(self, timestep):
        vis_file = "results/data/{}-MSMS_{}.tcl".format(self.traj_id, timestep)

        with open(vis_file, "w") as f:
            for triangle in self.triangulation:
                p1, p2, p3 = triangle
                f.write("graphics top triangle {{{} {} {}}} {{{} {} {}}} {{{} {} {}}}\n".format(*p1, *p2, *p3))
