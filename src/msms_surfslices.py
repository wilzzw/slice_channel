from itertools import product
import pandas as pd
from shapely.ops import unary_union
from msms import prot_msms, trgs_zslice_vertices, triangle_zcut
from polygons_construct import linesegs2polygons


import networkx as nx
import numpy as np
from shapely.geometry import Point, Polygon

from slices import _plot_boundary


# from utils.core_utilities import overlapping_split

class dag_ends:
    def __init__(self, DiGraph):
        self.graph = DiGraph
        self.nodes = self.graph.nodes

        self.is_root = np.vectorize(self._is_root)
        self.is_leaf = np.vectorize(self._is_leaf)

        self.roots = np.array(self.nodes)[self.is_root(self.nodes)]
        self.leafs = np.array(self.nodes)[self.is_leaf(self.nodes)]

    # n is the node in
    def node_relations(self, n):
        num_ancestors = len(nx.ancestors(self.graph, n))
        num_descendants = len(nx.descendants(self.graph, n))
        return num_ancestors, num_descendants

    def _is_root(self, n):
        num_ancestors, num_descendants = self.node_relations(n)
        return (num_ancestors == 0 and num_descendants != 0)

    def _is_leaf(self, n):
        _, num_descendants = self.node_relations(n)
        return (num_descendants == 0)


class root2leaf_longest(dag_ends):
    def __init__(self, DiGraph):
        super().__init__(DiGraph)

        all_paths_set = [list(nx.all_simple_paths(self.graph, *root_leaf)) for root_leaf in product(self.roots, self.leafs)]
        self.rootleaf_longest = [max(paths_set, key=len) for paths_set in all_paths_set if len(paths_set) > 0]


def draw_molsection(enclosure_dag, polygons_list):
    rootleaf_analysis = root2leaf_longest(enclosure_dag)

    draw_instruction = rootleaf_analysis.rootleaf_longest

    # Reverse to start from the "outermost" layer
    draw_instruction = [path[::-1] for path in draw_instruction]

    leafs = rootleaf_analysis.leafs
    # All paths should start with one of the leaf nodes
    assert np.all([(path[0] in leafs) for path in draw_instruction])

    solid = unary_union(polygons_list[leafs])

    if len(draw_instruction) > 0:
        max_clave_chain = len(max(draw_instruction, key=len))

        draw_instruction_matrix = np.tile(-1, (len(draw_instruction), max_clave_chain))
        for i, instr in enumerate(draw_instruction):
            draw_instruction_matrix[i,:len(instr)] = instr

        for j in range(1, draw_instruction_matrix.shape[1]):
            draw_shapes = draw_instruction_matrix[:,j]
            draw_shapes = draw_shapes[draw_shapes >= 0]

            clave_level_shapes = unary_union(polygons_list[draw_shapes])

            if j % 2 == 0:
                solid = solid.union(clave_level_shapes)
            else:
                solid = solid.difference(clave_level_shapes)

    return solid


def enclosed_in(a, b):
    if a == b:
        return False
    return a.within(b)


class msms_surfslice:
    def __init__(self, msms_trgs, zlevel, rtol=1e-10):
        self.triangulation = msms_trgs
        self.zlevel = zlevel
        self.rtol = rtol

        self.triangles_zcut = triangle_zcut(self.triangulation, self.zlevel)
        # Break if select triangulation is empty
        self.linesegs_xy = trgs_zslice_vertices(self.triangles_zcut, zlevel)

        if self.linesegs_xy is None:
            self.slice = Polygon([])
        else:
            # Build the slice shapes with polygons
            polymake = linesegs2polygons(self.linesegs_xy)

            self.polygons_collect = polymake.build_polygons()
            # Changed from verts_xy to unique_vertices_xy in case of exclusion of vert_indices due to not forming lines
            # Index mismatch can result
            self.polygons = [Polygon(polymake.unique_vertices_xy[polygon_path]) for polygon_path in self.polygons_collect]
            # Check for invalid polygons; e.g. pathological case t=15, f=8, z=143
            # See also: https://shapely.readthedocs.io/en/stable/manual.html
            # Potentially test and check for area preservation?
            valid_polygons = [polygon.is_valid for polygon in self.polygons]
            fixed_polygons = [polygon.buffer(0) for polygon in self.polygons]
            # Edge case t=15, f=131, z=141.5
            # Strangely np.where() below would give 2d-array
            # Speculation: due to fixed_polygons end up as separate polygons, adding an extra dimension
            self.polygons = np.where(~np.array(valid_polygons), fixed_polygons, self.polygons).flatten()

            self.enclosed_in = np.vectorize(enclosed_in)
            i, j = np.indices((len(self.polygons), len(self.polygons)))
            self.enclose_matrix = self.enclosed_in(self.polygons[i], self.polygons[j])

            self.enclosures = self.enclosure_analysis()
            self.slice = draw_molsection(self.enclosures, self.polygons)

    def enclosure_analysis(self):
        enclose_relations = np.argwhere(self.enclose_matrix)
        enclosure_dag = nx.DiGraph()

        enclosure_dag.add_nodes_from(np.arange(len(self.polygons)))
        enclosure_dag.add_edges_from(enclose_relations)
        # connected_subgraphs = [enclosure_dag.subgraph(c) for c in nx.weakly_connected_components(enclosure_dag)]
        return enclosure_dag


class msms_slices(prot_msms):
    def __init__(self, traj_id, probe_radius=1.4, look_for=False):
        super().__init__(traj_id, probe_radius, look_for)
        self.nframes = self.nframes.get(self.traj_id)
        self.timestep = 0
        self.run(timestep=self.timestep)

    def zslice(self, timestep, zlevel, center, radius):
        if timestep != self.timestep:
            self.timestep = timestep
            self.run(timestep=self.timestep)
        surfslice = msms_surfslice(self.triangulation, zlevel)
        prot_slice = surfslice.slice
        cylinderxy = Point(*center).buffer(radius)
        access_area = cylinderxy.difference(prot_slice)
        return prot_slice, cylinderxy, access_area

    def slice_run(self, lower, upper, incr, center=(25,50), radius=10.5, parallel=False):
        zrange = np.arange(lower, upper+incr, incr)
        shapes_collect = []
        # if parallel:
        #     def proc(timestep, zlevel):
        #         prot_slice, _, access_area = self.zslice(timestep, zlevel, center, radius)
        #         return prot_slice, access_area

        #     fz_pairs = [(f, z) for f, z in product(np.arange(self.nframes), np.arange(lower,upper+incr,incr))]

        #     if 'SLURM_NPROCS' in os.environ:
        #         numprocs = int(os.environ['SLURM_NPROCS'])
        #     else:
        #         numprocs = cpu_count()        

        #     with Pool(numprocs) as p:
        #         shapes_collect = p.starmap(proc, fz_pairs)  
        #         shapes_collect = [fz+output for fz, output in zip(product(np.arange(self.nframes), np.arange(lower,upper+incr,incr)), shapes_collect)]      
        # else:
        for f in range(self.nframes):
        # for f in range(131, self.nframes):
            print("Analyzing frame %d out of %d" % (f, self.nframes))
            for z in zrange:
                print(f,z)
                prot_slice, _, access_area = self.zslice(f, z, center, radius)
                shapes_collect.append((f, z) + (prot_slice, access_area))

        # Turn into df
        slice_df = pd.DataFrame(shapes_collect, columns=["frame", "z", "prot", "void"])
        self.slice_df = slice_df

    def plot_slice(self, axs, prot, cylinder, accessible):
        # Plot the protein slice boundaries
        _plot_boundary(axs, prot, color="black")
        _plot_boundary(axs, accessible, color="red")

        x, y = cylinder.boundary.xy
        axs.plot(x, y, color="cyan", ls="--")

        axs.set_aspect('equal', adjustable='box', anchor='C')
        axs.set_xlim(-20,60)
        axs.set_ylim(10,90)