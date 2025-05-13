import numpy as np
import pandas as pd
import mdtraj as md

from shapely.geometry import Point#, MultiPolygon
from multiprocess.pool import Pool

import os
from itertools import product
from multiprocessing import cpu_count

from circle_geoms import union_of_disks#, union_cvxhull
#from utils import overlapping_split

class pore_slices:
    """
    Class to analyze the pore/cavity of a protein (e.g. channels, transporters) 
    by slicing the region enclosed by the protein molecular surface
    """
    def __init__(self, protein_xyz: np.ndarray, protein_top: md.Topology=None, probe_radius: float=0.14):
        
        self.protein_xyz = protein_xyz
        self.protein_top = protein_top
        self.probe_radius = probe_radius

        if protein_xyz.ndim == 2:
            self.nframes = 1
            self.natoms = protein_xyz.shape[0]
        elif protein_xyz.ndim == 3:
            self.nframes, self.natoms, xyz = protein_xyz.shape
            if xyz != 3:
                raise ValueError("protein coordinates should be 3D, but the last dimension of protein_xyz is %d" % xyz)
        else:
            raise ValueError("protein coordinates should be 2D or 3D, but the protein_xyz has %d dimensions" % protein_xyz.ndim)

        # if self.from_nc:
        #     prot_xyz = Dataset("data/xyz/protein_xyz.nc", "r", format="NETCDF4", persist=True)
        #     self.nframes = int(prot_xyz.variables['nframes'][self.traj_id].data)
        #     self.protein_xyz = prot_xyz.variables["coordinate"][self.traj_id,:,:,:]
        #     self.natoms = np.sum(~self.protein_xyz[0,:,0].mask)
        #     ref = md.load(grotop_file4traj_id(self.traj_id))
        #     self.ref = ref.atom_slice(ref.top.select("protein"))
        #     prot_xyz.close()
        # else:
        #     traj = md.load(trajectory(traj_id), top=grotop_file4traj_id(traj_id))
        #     self.ref = traj.atom_slice(traj.top.select("protein"))
        #     self.protein_xyz = self.ref.xyz
        #     self.nframes = traj.n_frames
        #     self.natoms = self.ref.n_atoms

    def zslice(self, frame, zlevel, center, radius, color=False):
        """
        Slices the protein at a given zlevel and returns the information about the protein at that slice
        """
        return _zslice(xyz=self.protein_xyz[frame,:,:], 
                      prot_top=self.protein_top, zlevel=zlevel, 
                      center=center, radius=radius, probe_radius=self.probe_radius, 
                      color=color)

    def slice_run(self, lower, upper, incr, center=(2.5,5), radius=(300**0.5)/10, parallel=False):

        # Levels of z-values to take slices at
        zlevels = np.arange(lower, upper+incr, incr)
        # I think this is the parallel implementation
        if parallel:
            def proc(xyz, zlevel):
                zslice_shapes = _zslice(xyz, prot_top=self.ref, zlevel=zlevel, center=center, radius=radius, probe_radius=self.probe_radius)
                prot, _, accessible, _ = zslice_shapes
                return prot, accessible

            fz_pairs = [(xyz, z) for xyz, z in product(self.protein_xyz[:,:self.natoms,:], np.arange(lower,upper+incr,incr))]

            if 'SLURM_NPROCS' in os.environ:
                numprocs = int(os.environ['SLURM_NPROCS'])
            else:
                numprocs = cpu_count()        
            
            with Pool(numprocs) as p:
                shapes_collect = p.starmap(proc, fz_pairs)  
                shapes_collect = [fz+output for fz, output in zip(product(np.arange(self.nframes), np.arange(lower,upper+incr,incr)), shapes_collect)]      
        else:
            shapes_collect = []
            for f in range(0, self.nframes):
                print("Analyzing frame %d out of %d" % (f, self.nframes))
                for z in zlevels:
                    zslice_shapes = self.zslice(frame=f, zlevel=z, center=center, radius=radius)
                    prot, _, accessible, _ = zslice_shapes
                    shapes_collect.append((f, z, prot, accessible))
        
        # Turn into df
        slice_df = pd.DataFrame(shapes_collect, columns=["frame", "z", "prot", "void"])
        self.slice_df = slice_df
        return self.slice_df

    def plot_slice(self, axs, prot, cylinder, accessible):
        # Plot the protein slice boundaries
        _plot_boundary(axs, prot, color="black")
        _plot_boundary(axs, accessible, color="red")

        x, y = cylinder.boundary.xy
        axs.plot(x, y, color="cyan", ls="--")

        axs.set_aspect('equal', adjustable='box', anchor='C')
        axs.set_xlim(-2,6)
        axs.set_ylim(1,9)

    def view_slices(self, frame):
        """
        Plot the slices of the protein for inspection
        """
        # Get the frame
        slices_subdf = self.slice_df.query("frame == @frame")
        for _, row in slices_subdf.iterrows():
            prot = row['prot']
            accessible = row['void']
            z = row['z']
            area = row['area']
            fig, axs = plt.subplots()
            self.plot_slice(axs, prot, accessible)
            axs.set_title(f"Frame {frame}, z = {z}, area = {area:.2f}")
            axs.set_xlim(-6,6)
            axs.set_ylim(-6,6)
        return
    
    def area_vs_z(self, frame, axs: plt.Axes=None):
        """
        Plot the area of the slices as a function of z
        """
        # Get the frame
        slices_subdf = self.slice_df.query("frame == @frame")
        # Get the area
        area = slices_subdf['area']
        z = slices_subdf['z']
        if axs is None:
            fig, axs = plt.subplots()
        axs.scatter(z, area)
        axs.set_xlabel("z")
        axs.set_ylabel("Area")
        return axs
    
    def calc_volume(self, frame):
        """
        Calculate the volume of the enclosed void space by integrating the area of the voids in slices
        """
        # Get the frame
        slices_subdf = self.slice_df.query("frame == @frame")
        # Get the area
        area = slices_subdf['area']
        z = slices_subdf['z']
        # Calculate the volume by integrating the area
        volume = np.trapz(area, z)
        return volume

### Utility functions ###
def protein_section(prot_xyz: np.ndarray, prot_top: md.Topology, zlevel, padding, color=False):
    # xy coordinates of the protein
    protxy = prot_xyz[:,:2]
    # z coordinates of the protein
    protz = prot_xyz[:,2]
    atomic_radii = np.array([a.element.radius + padding for a in prot_top.atoms])
    # Some geometry
    at_level = np.less(np.abs(protz - zlevel), atomic_radii)
    # Radii projection on the given xy-plane at zlevel
    proj_radii = np.sqrt(atomic_radii[at_level]**2 - (protz[at_level] - zlevel)**2)

    level_protxy = protxy[at_level]
    # Union of the circles onto the plane
    region_shape = union_of_disks(level_protxy, proj_radii)

    if color:
        color_scheme = {'N': 'blue', 'O': 'red'}
        atom_types = np.array([a.element.symbol for a in prot_top.top.atoms])
        color_list = [color_scheme.get(e) for e in atom_types[at_level]]
        color_info = zip(level_protxy, proj_radii, color_list)
        return region_shape, color_info
    return region_shape, None

def _zslice(xyz, prot_top, zlevel, center, radius, probe_radius, color=False):
    sliced_protein, color_info = protein_section(prot_xyz=xyz, prot_top=prot_top, zlevel=zlevel, padding=probe_radius, color=color)
    # Area: used to measure pore accessible area
    cylinderxy = Point(*center).buffer(radius)
    access_area = cylinderxy.difference(sliced_protein)
    return sliced_protein, cylinderxy, access_area, color_info

# Utility function to draw the shapes (shapely objects) on the axes
def _plot_boundary(axs, shapeset, color="black", linestyle="-"):
    try:
        shapeset.boundary
    except:
        return
    else:
        try:
            shapeset.boundary.geoms
        except:
            geos = [shapeset.boundary]
        else:
            geos = shapeset.boundary.geoms

    for dA in geos:
        x, y = dA.xy
        axs.plot(x, y, color=color, ls=linestyle)
    return

# Unused
# def select_voids(centers, radii, precision=10e-8):
#     """
#     Select 2D void regions on a slice of the protein.
#     Proteins are represented as atoms, which are represented as circles on the slice.
#     The voids are the regions that are not covered by the circles.
#     """
#     disks = union_of_disks(centers, radii)
#     cvh = union_cvxhull(centers)

#     protein_closed = []
#     for void in cvh.difference(disks):
#         closed_by_protein = True
#         # Determining whether segments overlap is rather tricky complicated by floating point precision
#         for p1, p2 in overlapping_split(void.boundary.coords):
#             if Point(p1).distance(cvh.boundary) < precision and Point(p2).distance(cvh.boundary) < precision:
#                 closed_by_protein = False
#                 break
#         if closed_by_protein:
#             protein_closed.append(void)
#     protein_closed = MultiPolygon(protein_closed)
#     return protein_closed