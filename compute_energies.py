#!/usr/bin/env python3

import numpy as jnp

def extract_dihedrals(faces):

    """
    Find dihedrals from face connectivity information.
    Assume a closed triangle mesh.
    The order of the indices are as follow:

    (b, c, a, d)

        a
       /|\
      / | \
     d  |  c
      \ | /
       \|/
        b

    Arguments:
        faces: Face connectivity of the triangle mesh.

    Returns:
        The dihedral informtions (one entry of 4 indices per dihedral)
    """

    edges_to_faces = {}

    for faceid, f in enumerate(faces):
        for i in range(3):
            edge = (f[i], f[(i+1)%3])
            edges_to_faces[edge] = faceid

    dihedrals = []

    for faceid, f0 in enumerate(faces):
        for i in range(3):
            a = f0[i]
            b = f0[(i+1)%3]
            c = f0[(i+2)%3]
            edge = (a, b)
            otherfaceid = edges_to_faces[edge]
            f1 = faces[otherfaceid]
            d = [v for v in f1 if v != a and v != b]
            assert len(d) == 1
            d = d[0]
            dihedrals.append([b, c, a, d])

    return jnp.array(dihedrals)

def _compute_dihedral_normals(dihedrals, vertices):
    b = vertices[dihedrals[:,0],:]
    c = vertices[dihedrals[:,1],:]
    a = vertices[dihedrals[:,2],:]
    d = vertices[dihedrals[:,3],:]
    ab = b-a
    n0 = jnp.cross(ab, c-a)
    n1 = jnp.cross(ab, a-d)
    n0 /= jnp.linalg.norm(n0, axis=1)[:,jnp.newaxis]
    n1 /= jnp.linalg.norm(n1, axis=1)[:,jnp.newaxis]
    return n0, n1

def _compute_triangle_areas(faces, vertices):
    a = vertices[faces[:,0],:]
    b = vertices[faces[:,1],:]
    c = vertices[faces[:,2],:]
    n = jnp.cross(b-a, c-a)
    return 0.5 * jnp.linalg.norm(n, axis=1)

def compute_vertex_mean_curvatures(vertices,
                                   faces,
                                   dihedrals,
                                   return_vertex_areas=False):
    """
    Compute the mean curvature of each vertex.

    Arguments:
        vertices: positions of the mesh.
        faces: connectivity of the mesh.
        dihedrals: return of extract_dihedrals()
        return_vertex_areas: if True, also return the area associated to each vertex.

    Return:
        The mean curvature of each vertex.
        Optionally, also returns the vertex areas.
    """
    nv = len(vertices)

    faces_areas = _compute_triangle_areas(faces, vertices)

    b = vertices[dihedrals[:,0],:]
    c = vertices[dihedrals[:,1],:]
    a = vertices[dihedrals[:,2],:]
    d = vertices[dihedrals[:,3],:]

    ab = b-a
    n0 = jnp.cross(ab, c-a)
    n1 = jnp.cross(ab, a-d)

    arg = jnp.sum(n0*n1, axis=1) / (jnp.linalg.norm(n0, axis=1) * jnp.linalg.norm(n1, axis=1))
    theta = jnp.arccos(jnp.maximum(-1, jnp.minimum(1, arg)))
    l = jnp.linalg.norm(ab, axis=1)
    ltheta = l * theta

    vertex_areas = jnp.zeros(nv)        \
                      .at[faces[:,0]]   \
                      .add(faces_areas) \
                      .at[faces[:,1]]   \
                      .add(faces_areas) \
                      .at[faces[:,2]]   \
                      .add(faces_areas) \
                      / 3

    vertex_mean_curvatures = jnp.zeros(nv) \
                                .at[dihedrals[:,2]] \
                                .add(ltheta) \
                                .at[dihedrals[:,0]] \
                                .add(ltheta) \
                                / (8 * vertex_areas)

    if return_vertex_areas:
        return vertex_mean_curvatures, vertex_areas
    else:
        return vertex_mean_curvatures


def compute_bending_energy(vertices,
                           faces,
                           dihedrals,
                           kb,
                           H0=0,
                           kade=0,
                           deltaA0=0):
    """
    Compute the bending energy using the Juelicher model.

    Arguments:
        vertices: positions of the mesh.
        faces: connectivity of the mesh.
        dihedrals: return of extract_dihedrals()
        kb: bending energy coefficient for the Helfrich term
        H0: spontaneous mean curvature
        kade: bending energy coefficient for the ADE term (alpha * kb * pi / D**2 in Juelicher1997 or Bian2020)
        deltaA0: equilibrium area difference
    Return:
        The total bending energy of the mesh.
    """

    vertex_mean_curvatures, vertex_areas = compute_vertex_mean_curvatures(vertices=vertices,
                                                                          faces=faces,
                                                                          dihedrals=dihedrals,
                                                                          return_vertex_areas=True)

    # Helfrich energy
    EH = 2 * kb * jnp.sum((vertex_mean_curvatures - H0)**2 * vertex_areas)

    # ADE energy
    deltaA = jnp.sum(vertex_mean_curvatures * vertex_areas)
    A = jnp.sum(vertex_areas)
    EADE = kade / (2 * A) * (deltaA - deltaA0)**2
    return EH + EADE

def compute_shear_energy(vertices,
                         vertices0,
                         faces,
                         Ka, mu,
                         a3=-2.0, a4=8.0, b1=0.7, b2=1.84):
    """
    Compute the shear energy with respect to the SFS using the Lim model.

    Arguments:
        vertices: positions of the mesh vertices.
        vertices0: positions of the unstressed mesh vertices.
        faces: connectivity of the mesh.
        Ka: area dilation coefficient.
        mu: shear modulus
        a3, a4, b1, b2: non linear coefficients.

    Return:
        The total shear energy of the mesh.
    """

    v1 = vertices[faces[:,0]]
    v2 = vertices[faces[:,1]]
    v3 = vertices[faces[:,2]]

    u1 = vertices0[faces[:,0]]
    u2 = vertices0[faces[:,1]]
    u3 = vertices0[faces[:,2]]

    y12 = u2 - u1
    y13 = u3 - u1
    eq_area = 0.5 * jnp.linalg.norm(jnp.cross(y12, y13, axis=1), axis=1)
    eq_dotp = jnp.sum(y12*y13, axis=1)

    x12 = v2 - v1
    x13 = v3 - v1

    area = 0.5 * jnp.linalg.norm(jnp.cross(x12, x13, axis=1), axis=1)
    area_inv = 1.0 / area
    area0_inv = 1.0 / eq_area

    alpha = area * area0_inv - 1.0

    e0sq_A = jnp.sum(x12*x12, axis=1) * area_inv
    e1sq_A = jnp.sum(x13*x13, axis=1) * area_inv

    e0sq_A0 = jnp.sum(y12*y12, axis=1) * area0_inv
    e1sq_A0 = jnp.sum(y13*y13, axis=1) * area0_inv

    dotp = jnp.sum(x12*x13, axis=1)

    dot_4A = 0.25 * eq_dotp * area0_inv
    mixed_v = 0.125 * (e0sq_A0 * e1sq_A + e1sq_A0 * e0sq_A)

    beta = mixed_v - dot_4A * dotp * area_inv - 1.0

    return 0.5 * Ka * jnp.sum((alpha**2 + a3 * alpha**3 + a4 * alpha**4) * eq_area) + \
        mu * jnp.sum((beta + b1*alpha*beta + b2*beta**2) * eq_area)

def compute_area(vertices, faces):
    areas = _compute_triangle_areas(faces, vertices)
    return jnp.sum(areas)

def compute_volume(vertices, faces):
    a = vertices[faces[:,0],:]
    b = vertices[faces[:,1],:]
    c = vertices[faces[:,2],:]
    n = jnp.cross(b-a, c-a, axis=1)
    return jnp.sum(a * n) / 6

def compute_helfrich(vertices, faces):
    """
    Compute the bending energy using the Juelicher model.

    Arguments:
        vertices: positions of the mesh.
        faces: connectivity of the mesh.
    Return:
        The total bending energy of the mesh.
    """
    kb=1
    H0=0

    dihedrals = extract_dihedrals(faces)

    vertex_mean_curvatures, vertex_areas = compute_vertex_mean_curvatures(vertices=vertices,
                                                                          faces=faces,
                                                                          dihedrals=dihedrals,
                                                                          return_vertex_areas=True)

    # Helfrich energy
    EH = 2 * kb * jnp.sum((vertex_mean_curvatures - H0)**2 * vertex_areas)

    return EH