from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union, voronoi_diagram, triangulate

def Or2Point(O, r):
    return Point(*O).buffer(r)

def union_of_disks(centers, radii):
    disks = [Point(*O).buffer(r) for O, r in zip(centers, radii)]
    protein_parts = unary_union(disks)
    return protein_parts

def union_cvxhull(centers):
    cvxhull = MultiPoint(centers).convex_hull
    return cvxhull

def union_voronoi(centers):
    voronoi = voronoi_diagram(MultiPoint(centers), envelope=union_cvxhull(centers))
    return voronoi

def union_dlytrig(centers):
    triangles = triangulate(MultiPoint(centers))
    return triangles