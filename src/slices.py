import numpy as np
import pandas as pd
import mdtraj as md
from netCDF4 import Dataset
from shapely.geometry import Point, MultiPolygon
from multiprocess.pool import Pool

import os
from itertools import product
from multiprocessing import cpu_count

from circle_geoms import union_of_disks, union_cvxhull

class pore_slices:
    """
    Class to analyze the pore/cavity of a protein (e.g. channels, transporters) 
    by slicing the region enclosed by the protein molecular surface
    """
    def __init__(self, traj_id, from_nc=True, probe_radius=0.14):
        self.traj_id = traj_id
        self.from_nc = from_nc
        self.probe_radius = probe_radius

        if self.from_nc:
            prot_xyz = Dataset("data/xyz/protein_xyz.nc", "r", format="NETCDF4", persist=True)
            self.nframes = int(prot_xyz.variables['nframes'][self.traj_id].data)
            self.xyz = prot_xyz.variables["coordinate"][self.traj_id,:,:,:]
            self.natoms = np.sum(~self.xyz[0,:,0].mask)
            ref = md.load(grotop_file4traj_id(self.traj_id))
            self.ref = ref.atom_slice(ref.top.select("protein"))
            prot_xyz.close()
        else:
            traj = md.load(trajectory(traj_id), top=grotop_file4traj_id(traj_id))
            self.ref = traj.atom_slice(traj.top.select("protein"))
            self.xyz = self.ref.xyz
            self.nframes = traj.n_frames
            self.natoms = self.ref.n_atoms

    def _zslice(self, frame, zlevel, center, radius, color=False):
        return zslice(xyz=self.xyz[frame,:self.natoms,:], prot_top=self.ref, zlevel=zlevel, center=center, radius=radius, probe_radius=self.probe_radius, color=color)

    def slice_run(self, lower, upper, incr, center=(2.5,5), radius=(300**0.5)/10, parallel=False):
        zrange = np.arange(lower, upper+incr, incr)

        if parallel:
            def proc(xyz, zlevel):
                zslice_shapes = zslice(xyz, prot_top=self.ref, zlevel=zlevel, center=center, radius=radius, probe_radius=self.probe_radius)
                prot, _, accessible, _ = zslice_shapes
                return prot, accessible

            fz_pairs = [(xyz, z) for xyz, z in product(self.xyz[:,:self.natoms,:], np.arange(lower,upper+incr,incr))]

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
                for z in zrange:
                    zslice_shapes = self._zslice(zlevel=z, center=center, radius=radius, frame=f)
                    prot, _, accessible, _ = zslice_shapes
                    shapes_collect.append((f, z) + (prot, accessible))
        
        # Turn into df
        slice_df = pd.DataFrame(shapes_collect, columns=["frame", "z", "prot", "void"])
        self.slice_df = slice_df
        return self.slice_df

    def plot_slice(self, axs, prot, cylinder, accessible):
        # Plot the protein slice boundaries
        plot_boundary(axs, prot, color="black")
        plot_boundary(axs, accessible, color="red")

        x, y = cylinder.boundary.xy
        axs.plot(x, y, color="cyan", ls="--")

        axs.set_aspect('equal', adjustable='box', anchor='C')
        axs.set_xlim(-2,6)
        axs.set_ylim(1,9)


def select_voids(centers, radii, precision=10e-8):
    disks = union_of_disks(centers, radii)
    cvh = union_cvxhull(centers)

    protein_closed = []
    for void in cvh.difference(disks):
        closed_by_prot = True
        # Determining whether segments overlap is rather tricky complicated by floating point precision
        for p1, p2 in overlapping_split(void.boundary.coords):
            if Point(p1).distance(cvh.boundary) < precision and Point(p2).distance(cvh.boundary) < precision:
                closed_by_prot = False
                break
        if closed_by_prot:
            protein_closed.append(void)
    protein_closed = MultiPolygon(protein_closed)
    return protein_closed

### Protein specific ###
def prot_section(prot_xyz, prot_top, zlevel, padding, color=False):
    protxy = prot_xyz[:,:2]
    protz = prot_xyz[:,2]
    atomic_radii = np.array([a.element.radius + padding for a in prot_top.top.atoms])
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

def zslice(xyz, prot_top, zlevel, center, radius, probe_radius, color=False):
    sliced_protein, color_info = prot_section(prot_xyz=xyz, prot_top=prot_top, zlevel=zlevel, padding=probe_radius, color=color)
    # Area: used to measure pore accessible area
    cylinderxy = Point(*center).buffer(radius)
    access_area = cylinderxy.difference(sliced_protein)
    return sliced_protein, cylinderxy, access_area, color_info

def plot_boundary(axs, shapeset, color="black", linestyle="-"):
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