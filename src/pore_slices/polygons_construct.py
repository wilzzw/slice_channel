import numpy as np
import networkx as nx

class linesegs2polygons:
    def __init__(self, linesegs_xy):
        self.linesegs_xy = linesegs_xy.round(decimals=2)
        self.polygons_collect = []

        # concatenate() to combine first two dimensions (num_linesegs, num_vertices=2)
        # This gives all the xy coordinates of vertices, with repeats
        vertices_xy = np.concatenate(self.linesegs_xy)
        # vertices_xy.T consists of xvals and yvals, each as a 1d-array
        self.unique_xvals_yvals = [self.truly_unique(dim_vals) for dim_vals in vertices_xy.T]
        # Clean-up and replace to "standardize" float values calculated from separate calculations
        # i.e matching floats are supposed to be equal
        _ = np.apply_along_axis(self.closest_replace, 2, self.linesegs_xy)

        # After clean-up, re-concatenate() to combine first two dimensions (num_linesegs, num_vertices=2)
        # This time self.linesegs_xy have "truly unique" x and y values
        vertices_xy = np.concatenate(self.linesegs_xy)

        # Some line segments will share the same vertex
        self.unique_vertices_xy = np.unique(vertices_xy, axis=0)
        # Create a dictionary to map xval, yval of vertex to its vertex index
        self.xy2index = {tuple(xy): i for i, xy in enumerate(self.unique_vertices_xy)}

        # This is line segments but instead of xval and yval of the vertices we have indices of the vertices
        self.linesegs_index = np.apply_along_axis(self.assign_vertindex, 2, self.linesegs_xy)

        # Some vertices on either ends of a line segment might be so close that squeeze the line segment such that
        # both vertices have the same index
        squeezed_where, = np.where(np.equal(self.linesegs_index[:,0], self.linesegs_index[:,1]))
        # Remove these
        self.linesegs_xy = np.delete(self.linesegs_xy, squeezed_where, axis=0)
        self.linesegs_index = np.delete(self.linesegs_index, squeezed_where, axis=0)

        # Further clean-up: remove potential duplicate line segments
        self.linesegs_index, sort_unique_linesegs = np.unique(self.linesegs_index, axis=0, return_index=True)
        self.linesegs_xy = self.linesegs_xy[sort_unique_linesegs]

        # verts_index: array of indices of vertices involved in forming line segments 
        # (might be edge case surprises where vertices are captured in self.xy2index.values() but got trimmed out?)

        self.verts_index, self.verts_occur_counts = np.unique(self.linesegs_index.flatten(), return_counts=True)
        self.verts_xy = self.unique_vertices_xy[self.verts_index]

        # Indices of vertices that are found in only one line segment
        # Possible reasons: 1) stray points; 2) disconnection from a very close nearby points
        # Solution: attempt to connect to the closest nearby vertex
        disconn_vertindex = self.verts_index[self.verts_occur_counts == 1]
        if len(disconn_vertindex) > 0:
            print("Number of disconnected vertices: %d" % len(disconn_vertindex))
            if len(disconn_vertindex) == 2:
                dist_between_disconn = np.linalg.norm(np.diff(self.unique_vertices_xy[disconn_vertindex], axis=0))

                # Print out explicitly the connection details
                print("""Make an extra connection [%d, %d].\n
                            Distance between them: %f""" % (*disconn_vertindex, dist_between_disconn))
                # Add the line segment
                self.linesegs_index = np.vstack([self.linesegs_index, disconn_vertindex])
                # print(self.linesegs_index, self.linesegs_index.shape)
                # print(self.verts_xy[[i, vi]], self.verts_xy[[i, vi]].shape)
                self.linesegs_xy = np.vstack([self.linesegs_xy, [self.verts_xy[disconn_vertindex]]])        
            else:
                for i in disconn_vertindex:
                    # Trying to exclude the vertex to which this disconnected vertex i is already connected
                    conn2i = np.setdiff1d(self.linesegs_index[np.any(self.linesegs_index == i, axis=1)], i)[0]

                    xy = self.unique_vertices_xy[i]
                    dist_from_vert = np.linalg.norm(self.verts_xy - xy, axis=1)
                    sort_dist_from_vert = self.verts_index[np.argsort(dist_from_vert)]
                    # The first one should be i itself, with a distance of zero
                    assert sort_dist_from_vert[0] == i

                    # Connect to the closest vertex to which i is not already connected
                    for vi in sort_dist_from_vert[1:]:
                        if vi != conn2i:
                            break
                    # Print out explicitly the connection details
                    print("""Make an extra connection [%d, %d].\n
                            Distance between them: %f""" % (i, vi, dist_from_vert[vi]))
                    # Add the line segment
                    self.linesegs_index = np.vstack([self.linesegs_index, [i, vi]])
                    # print(self.linesegs_index, self.linesegs_index.shape)
                    # print(self.verts_xy[[i, vi]], self.verts_xy[[i, vi]].shape)
                    self.linesegs_xy = np.vstack([self.linesegs_xy, [self.verts_xy[[i, vi]]]])

        # Indices for line segments
        self.linesegs_jndex = np.arange(len(self.linesegs_index))

    def build_polygons2(self, debug=None):
        return

    def build_polygons(self):
        polygons_G = nx.Graph()
        polygons_G.add_nodes_from(self.verts_index)
        polygons_G.add_edges_from(self.linesegs_index)
        return nx.cycle_basis(polygons_G)

    # Truly unique floats
    # test: np.any(np.isclose(np.diff(truly_unique(a)), 0))
    def truly_unique(self, array1d):
        sorted_array = np.sort(array1d)
        repeated_atindex, = np.where(np.isclose(np.diff(sorted_array), 0))
        return np.delete(sorted_array, repeated_atindex)

    # Instead of using np.isclose()
    # This now replace with the closest to avoid matching with multiple values that "isclose"
    def closest_replace(self, float_array1d):
        # Iterate through num of dimensions
        # e.g. len(float_array1d) should be 2 for 2d-coordinates
        for i in range(len(float_array1d)):
            replace_with = np.argmin(np.abs(self.unique_xvals_yvals[i] - float_array1d[i]))
            float_array1d[i] = self.unique_xvals_yvals[i][replace_with]
        return float_array1d

    # Assign indices to xy points
    def assign_vertindex(self, xyval):
        return self.xy2index.get(tuple(xyval))


    # def build_polygons(self, axs=None):
    #     # Initialization
    #     lineseg_indices = np.arange(len(self.linesegs_xy), dtype=int)
    #     polygons_collect = []

    #     # Initialize polygon tracing
    #     # Take the first of the remaining line segment index to begin
    #     init_seg_j = lineseg_indices[0]
    #     # The indices of the vertices for the line segment
    #     init_seg = self.linesegs_vertindices[init_seg_j]
    #     # Remember the first vertex to see whether the walk has returned to the start
    #     init_vert = init_seg[0]
    #     print("init_vert {}".format(init_vert))
    #     # polygon_path is a sequence of vertex indices
    #     # Begin tracing by initiating the polygon by recording the two vertex indices of the current segment
    #     polygon_path = np.array(init_seg, dtype=int)
    #     # The line segment index of the current step: initialization
    #     step_seg_j = init_seg_j
    #     # Keeping track of the line segments that have been counted -- by their indices
    #     linesegs_current_cycle = [step_seg_j]

    #     # report_polypath = False
    #     for l in range(len(self.linesegs_vertindices)):
    #         # The vertex index of the current step: initialization
    #         step_vert = polygon_path[-1]
    #         # Get line segment jndices that contain the current step_vert index
    #         # There should be two and one is the current step_seg_j
    #         has_step_vert = np.any(self.linesegs_vertindices == step_vert, axis=1)
    #         # Edge case fix for t=15, f=63, z=148
    #         # has_not_walked = ~np.in1d(self.linesegs_vertindices, lineseg_indices)
    #         #  & has_not_walked
    #         linesegs_have_step_vert, = np.where(has_step_vert)
    #         print(linesegs_have_step_vert)

    #         assert len(linesegs_have_step_vert) > 0
    #         if len(linesegs_have_step_vert) > 2:
    #             print("""Warning: A vertex point was found in more than two segments;\n
    #                      suggesting that the vertex might be shared between two polygons.""")
    #             # print(step_vert, step_seg_j, linesegs_have_step_vert, len(linesegs_have_step_vert))
    #             # report_polypath = True
    #             # Address edge case t=15, f=18, z=132.5 where next line segment ends up in a different polygon but shared vertex
    #             # When it comes back to this vertex, it will find that has_step_vert has not changed
    #             # because setdiff1d() will not be called until step_seg[-1] == init_vert
    #             # Current solution: kick out the step_seg_j and the next one before setdiff1d() is called
    #             # Alternative solution not yet implemented (requires a bit of rewrite): recursive call this function build_polygons() again
    #             # And start at this branching crossover index
    #             # If not fixed, assert len(polygon_path) == len(np.unique(polygon_path)) + 1 will fail

    #             # Remove right now
    #             lineseg_indices = np.setdiff1d(lineseg_indices, step_seg_j)
                
    #         elif len(linesegs_have_step_vert) == 1:
    #             # Connect anyways; sometimes float numbers are still slightly off (traj_id=15, f=5, z=129.5)
    #             # But print out explicitly the treatment details
    #             step_vert_xy = self.unique_vertices_xy[step_vert]
    #             # Address edge case like t=15, f=30, z=136.5
    #             # Connecting should also connect weird cases like this one, in which there is an actual disconnection from triangulation
    #             # Weird that MSMS could even do that
    #             # Should have excluded the vertices that have already walked through
    #             # Unless it is the first init_vert of course
    #             # Currently only considering disconnection to another disconnected vertex as in the edge case mentioned

    #             ### Consider fixing this issue at the time in __init__() when self.linesegs_xy was defined, in the future
    #             dist_from_stepvert = np.linalg.norm(self.unique_vertices_xy - step_vert_xy, axis=1)
    #             sort_dist_from_stepvert = np.argsort(dist_from_stepvert)
    #             assert sort_dist_from_stepvert[0] == step_vert
    #             print(step_vert, step_seg_j)
    #             # Replace with the closest vertex
    #             for i in range(1, len(sort_dist_from_stepvert)):
    #                 step_vert = sort_dist_from_stepvert[i]
    #                 if step_vert not in polygon_path[1:]:
    #                     break
    #             print("""Warning: matching vertex not found in any other line segments;\n
    #                      Merge vertex with the closest next vertex\n
    #                      distance = %f""" % dist_from_stepvert[step_vert])

    #             # Get line segment jndices that contain the current step_vert index
    #             # There should be two and one is the current step_seg_j
    #             has_step_vert = np.any(self.linesegs_vertindices == step_vert, axis=1)
    #             linesegs_have_step_vert, = np.where(has_step_vert)




    #         if step_vert == init_vert:
    #             assert len(linesegs_current_cycle) == len(np.unique(linesegs_current_cycle))

    #             # Add to polygon collection
    #             # But first, check for crossover loops such as figure 8 shaped scenario
    #             # Temporary fix: assume one crossover
    #             unique_verts_polygon, counts = np.unique(polygon_path[:-1], return_counts=True)
    #             if len(unique_verts_polygon) < len(polygon_path[:-1]):
    #                 where_moreoccur, = np.where(counts > 1)
    #                 crossover_vert = unique_verts_polygon[where_moreoccur]
    #                 subpolygon_where, = np.where(polygon_path == crossover_vert)
    #                 start, end = subpolygon_where
    #                 polygons_collect.append(polygon_path[start:end+1])
    #                 polygon_path = np.delete(polygon_path, np.arange(start,end))
    #                 assert len(polygon_path) == len(np.unique(polygon_path)) + 1
    #                 polygons_collect.append(polygon_path)
    #             else:
    #                 # No more step_seg; needs to append now
    #                 polygon_path = np.append(polygon_path, step_vert)
    #                 assert len(polygon_path) == len(np.unique(polygon_path)) + 1
    #                 # Add to polygon collection
    #                 polygons_collect.append(polygon_path)
    #                 # print("drew a polygon!!!!!")
    #             # if report_polypath:
    #             #     print(polygon_path)
    #             #     print(len(np.unique(polygon_path)), len(polygon_path))
    #             #     report_polypath = False

    #             # Update and remove line segment jndices that has been walked through
    #             lineseg_indices = np.setdiff1d(lineseg_indices, linesegs_current_cycle)

    #             if len(lineseg_indices) == 0:
    #                 break

    #             # Re-nitialize polygon tracing
    #             # Take the first of the remaining line segment index to begin
    #             init_seg_j = lineseg_indices[0]
    #             # The indices of the vertices for the line segment
    #             init_seg = self.linesegs_vertindices[init_seg_j]
    #             # Remember the first vertex to see whether the walk has returned to the start
    #             init_vert = init_seg[0]
    #             # polygon_path is a sequence of vertex indices
    #             # Begin tracing by initiating the polygon by recording the two vertex indices of the current segment
    #             polygon_path = np.array(init_seg, dtype=int)
    #             # The line segment index of the current step: initialization
    #             step_seg_j = init_seg_j
    #             # Keeping track of the line segments that have been counted -- by their indices
    #             linesegs_current_cycle = [step_seg_j]

        




    #         # Update step_seg_j by jumping to the next segment
    #         # Also need to modify to reflect the assert len(polygon_path) == len(np.unique(polygon_path)) + 1 fail
    #         # After all, this line is how the next step segment will be selected
    #         # If not, removing from lineseg_indices fix will do nothing in the current walk cycle
    #         next_step_seg_j = linesegs_have_step_vert[(linesegs_have_step_vert != step_seg_j) & np.in1d(linesegs_have_step_vert, lineseg_indices)]
    #         print("next_step_seg_j: {}".format(next_step_seg_j))
    #         # Edge case t=15, f=63, z=148; 3 edges from crossover vertex but really it looks like 4
    #         # A very special case
    #         if len(next_step_seg_j) == 0:
    #             # Connect anyways to closest
    #             step_vert_xy = self.unique_vertices_xy[step_vert]
    #             ### Consider fixing this issue at the time in __init__() when self.linesegs_xy was defined, in the future
    #             dist_from_stepvert = np.linalg.norm(self.unique_vertices_xy - step_vert_xy, axis=1)
    #             sort_dist_from_stepvert = np.argsort(dist_from_stepvert)
    #             assert sort_dist_from_stepvert[0] == step_vert

    #             # Replace with the closest vertex
    #             for i in range(1, len(sort_dist_from_stepvert)):
    #                 step_vert = sort_dist_from_stepvert[i]
    #                 if step_vert not in polygon_path[1:]:
    #                     break
    #             print("""Warning: matching vertex not found in any other line segments;\n
    #                      Merge vertex with the closest next vertex\n
    #                      distance = %f""" % dist_from_stepvert[step_vert])
    #             # Get line segment jndices that contain the current step_vert index
    #             # There should be two and one is the current step_seg_j
    #             has_step_vert = np.any(self.linesegs_vertindices == step_vert, axis=1)
    #             linesegs_have_step_vert, = np.where(has_step_vert)
    #             # assert len(linesegs_have_step_vert) == 1 
    #             # next_step_seg_j = linesegs_have_step_vert[(linesegs_have_step_vert != step_seg_j) & np.in1d(linesegs_have_step_vert, lineseg_indices)]
    #             next_step_seg_j = linesegs_have_step_vert

    #         step_seg_j = next_step_seg_j[0]
    #         # Adding the index of the line segment jndex to record that it has been counted
    #         lineseg_indices = np.setdiff1d(lineseg_indices, step_seg_j)
    #         linesegs_current_cycle.append(step_seg_j)
            
    #         if axs is not None:
    #             axs.plot(*self.linesegs_xy[step_seg_j].T)

    #         # Reveal the indices of the vertices that make up this step_seg_j id line segment
    #         step_seg = self.linesegs_vertindices[step_seg_j]
    #         # The order of the vertex pair could be reversed
    #         # Reverse if needed
    #         if step_seg[0] != step_vert:
    #             step_seg = step_seg[::-1]
    #         assert step_seg[0] == step_vert

    #         # Add the new vertex index, which is step_seg[-1] to the polygon_path
    #         polygon_path = np.append(polygon_path, step_seg[-1])

    #         # If the index has returned to the starting point, restart a new polygon
    #         if step_seg[-1] == init_vert:
    #             assert len(linesegs_current_cycle) == len(np.unique(linesegs_current_cycle))

    #             # Add to polygon collection
    #             # But first, check for crossover loops such as figure 8 shaped scenario
    #             # Temporary fix: assume one crossover
    #             unique_verts_polygon, counts = np.unique(polygon_path[:-1], return_counts=True)
    #             if len(unique_verts_polygon) < len(polygon_path[:-1]):
    #                 where_moreoccur, = np.where(counts > 1)
    #                 crossover_vert = unique_verts_polygon[where_moreoccur]
    #                 subpolygon_where, = np.where(polygon_path == crossover_vert)
    #                 start, end = subpolygon_where
    #                 polygons_collect.append(polygon_path[start:end+1])
    #                 polygon_path = np.delete(polygon_path, np.arange(start,end))
    #                 assert len(polygon_path) == len(np.unique(polygon_path)) + 1
    #                 polygons_collect.append(polygon_path)
    #             else:
    #                 assert len(polygon_path) == len(np.unique(polygon_path)) + 1
    #                 # Add to polygon collection
    #                 polygons_collect.append(polygon_path)
    #                 # print("drew a polygon!!!!!")
    #             # if report_polypath:
    #             #     print(polygon_path)
    #             #     print(len(np.unique(polygon_path)), len(polygon_path))
    #             #     report_polypath = False

    #             # Update and remove line segment jndices that has been walked through
    #             lineseg_indices = np.setdiff1d(lineseg_indices, linesegs_current_cycle)

    #             if len(lineseg_indices) == 0:
    #                 break

    #             # Re-nitialize polygon tracing
    #             # Take the first of the remaining line segment index to begin
    #             init_seg_j = lineseg_indices[0]
    #             # The indices of the vertices for the line segment
    #             init_seg = self.linesegs_vertindices[init_seg_j]
    #             # Remember the first vertex to see whether the walk has returned to the start
    #             init_vert = init_seg[0]
    #             # polygon_path is a sequence of vertex indices
    #             # Begin tracing by initiating the polygon by recording the two vertex indices of the current segment
    #             polygon_path = np.array(init_seg, dtype=int)
    #             # The line segment index of the current step: initialization
    #             step_seg_j = init_seg_j
    #             # Keeping track of the line segments that have been counted -- by their indices
    #             linesegs_current_cycle = [step_seg_j]

    #     return polygons_collect
