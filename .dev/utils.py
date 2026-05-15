import warnings
import zipfile

import geojson
import numpy as np
import shapely
import tqdm


# from polygon_util import expand_polygon, align_expanded_polygon_to_original
import geojson

geojson.geometry.DEFAULT_PRECISION = 2


def write_zip_geojson(out_path, geojson_obj, name=None):
    assert out_path.endswith(".zip")

    if name is not None:
        if issubclass(
            type(geojson_obj), (shapely.Geometry, shapely.GeometryCollection)
        ):
            print(f"Convert to feature with name: {name}")
            geojson_obj = geojson.Feature(
                geometry=geojson_obj, properties=dict(name=name)
            )

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "data.geojson",
            geojson.dumps(geojson_obj),
        )


def calculate_simplex_circumradii(simplices_coords: np.ndarray) -> np.ndarray:
    """
    Calculates the circumradii of multiple n-simplices efficiently.

    Args:
        simplices_coords: A NumPy array of shape (N, n+1, d) where:
            N is the number of simplices.
            n+1 is the number of vertices per simplex (e.g., 3 for triangles,
                                                        4 for tetrahedra).
            d is the dimension of the embedding space (d >= n).
            Each element `simplices_coords[i, j, :]` contains the d-dimensional
            coordinates of the j-th vertex of the i-th simplex.

    Returns:
        A NumPy array of shape (N,) containing the circumradius for each simplex.
        Returns np.inf for degenerate simplices where the calculation fails
        (e.g., collinear points for a triangle).

    Raises:
        ValueError: If the input array shape is invalid.

    Notes:
        - This function is optimized for speed using NumPy vectorization.
        - It calculates the circumcenter relative to the first vertex (v0)
          and then finds its magnitude (the circumradius).
        - The method involves solving a linear system for each simplex.
          Degenerate simplices (where vertices are linearly dependent in a way
          that makes the system singular) will result in np.inf in the output
          and a RuntimeWarning.
    """
    if simplices_coords.ndim != 3:
        raise ValueError("Input array must have shape (N, n+1, d)")

    N, num_vertices, d = simplices_coords.shape
    n = num_vertices - 1  # Dimension of the simplex itself

    if n <= 0:
        # 0-simplex (point) has radius 0? Or undefined? Let's return 0.
        # 1-simplex (line segment) has radius half the length.
        if n == 0:
            return np.zeros(N)
        if n == 1:
            v0 = simplices_coords[:, 0, :]
            v1 = simplices_coords[:, 1, :]
            return 0.5 * np.sqrt(np.sum((v1 - v0) ** 2, axis=1))

    # Translate simplices so the first vertex (v0) is at the origin
    # v0 shape: (N, 1, d)
    v0 = simplices_coords[:, 0:1, :]
    # U contains vectors u_i = v_i - v0 for i = 1 to n
    # U shape: (N, n, d)
    U = simplices_coords[:, 1:, :] - v0

    # Build the matrix M where M_ij = u_i . u_j
    # We want a matrix of shape (N, n, n) for each simplex
    # einsum('Nik,Njk->Nij', U, U) computes the dot products efficiently
    # M shape: (N, n, n)
    M = np.einsum("Nik,Njk->Nij", U, U)

    # Build the vector B where B_i = |u_i|^2
    # B shape originally (N, n), add dim to make it (N, n, 1) for solving
    B = np.sum(U * U, axis=2, keepdims=True)  # Shape (N, n, 1)

    # --- Solve the linear system 2 * M * alpha = B for alpha ---
    # The circumcenter c' (relative to v0) is given by c' = sum(alpha_j * u_j)
    # We need to solve N independent systems of size n x n.
    # np.linalg.solve can handle stacks of matrices.

    # Pre-allocate output array with np.inf for potential failures
    radii = np.full(N, np.inf, dtype=np.float64)

    # Avoid division by zero or issues with tiny matrices if n=0 handled above
    # Calculate 2*M safely
    two_M = 2 * M

    # Use a loop with try-except for np.linalg.solve to handle singular matrices
    # Although np.linalg.solve handles stacks, catching errors per item is harder.
    # A potentially faster way for *some* singular matrices is pinv, but solve
    # is generally preferred for well-behaved systems.
    # Let's try direct solve and see performance. Add error handling if needed.

    # Check for near-zero determinant or high condition number?
    # For speed, let's try direct solve first.
    # Add a small value to the diagonal for numerical stability? Only if necessary.
    # regularizer = 1e-12
    # two_M += np.eye(n) * regularizer # Optional regularization

    valid_mask = np.ones(N, dtype=bool)
    alpha = np.zeros((N, n, 1), dtype=np.float64)

    try:
        # This might raise LinAlgError if any matrix in the stack is singular
        alpha = np.linalg.solve(two_M, B)
    except np.linalg.LinAlgError:
        warnings.warn(
            "One or more simplices may be degenerate (singular matrix encountered). "
            "Circumradius set to infinity for these cases.",
            RuntimeWarning,
        )
        # Need to identify *which* ones failed. This is tricky without looping.
        # Let's refine by checking condition numbers or determinants if needed.
        # For now, we can loop *if* the batch solve fails.
        # A simple approach: loop and solve individually if batch fails.
        alpha = np.full((N, n, 1), np.nan)  # Mark all as potentially failed initially
        for i in range(N):
            try:
                alpha[i] = np.linalg.solve(two_M[i], B[i])
                valid_mask[i] = True
            except np.linalg.LinAlgError:
                valid_mask[i] = False  # Keep radius as np.inf

        # Filter out simplices that caused the error
        if not np.any(valid_mask):  # If all failed
            return radii  # Return all infs

        # Recalculate for valid ones (if any succeeded in the loop)
        U = U[valid_mask]
        alpha = alpha[valid_mask]

    # Calculate the circumcenter c' relative to v0
    # c' = sum(alpha_j * u_j)
    # alpha shape: (N_valid, n, 1)
    # U shape: (N_valid, n, d)
    # We want to compute sum(alpha[:, j, 0] * U[:, j, :]) over j
    # Use einsum or direct broadcasting + sum
    # c_prime shape: (N_valid, d)
    c_prime = np.sum(alpha * U, axis=1)  # Broadcasting works: (N,n,1)*(N,n,d)->(N,n,d)

    # The circumradius R is the magnitude of c'
    # R^2 = |c'|^2 = c' . c'
    R_squared = np.sum(c_prime * c_prime, axis=1)  # Shape (N_valid,)

    # Handle potential negative values due to floating point errors near zero
    R_squared[R_squared < 0] = 0

    # Assign calculated radii back to the original positions
    radii[valid_mask] = np.sqrt(R_squared)

    return radii


def segmentize_scipy(coordinates, d_min, d_max):
    elength, edge_idxs = delaunay_edge_distances(coordinates)

    emask = (elength >= d_min) & (elength <= d_max)
    return shapely.unary_union(
        shapely.polygonize(
            [shapely.LineString(coordinates[ee]) for ee in tqdm.tqdm(edge_idxs[emask])]
        )
    )


def delaunay_edge_distances(coordinates):
    import scipy.spatial

    tri = scipy.spatial.Delaunay(coordinates)
    edge_idxs = np.tile(tri.simplices, (1, 2)).reshape(-1, 2)

    tri = None

    edge_idxs.sort(axis=1)
    edge_idxs = np.unique(edge_idxs, axis=0)

    elength = np.linalg.norm(
        coordinates[edge_idxs[:, 0]] - coordinates[edge_idxs[:, 1]], axis=1
    )
    return elength, edge_idxs


def alphashape_shapely(points, d_max):
    tri = shapely.delaunay_triangles(shapely.multipoints(points))
    # NOTE shapely.minimum_bounding_radius is not the same as circumradius
    # but might be more intuitive
    circumdiameters = [
        2 * shapely.minimum_bounding_radius(gg) for gg in tqdm.tqdm(tri.geoms)
    ]
    return shapely.unary_union(
        [gg for gg, dd in zip(tri.geoms, circumdiameters) if dd <= d_max]
    )


def alphashape(coordinates, d_max):
    import scipy.spatial

    tri = scipy.spatial.Delaunay(coordinates)
    diameter_lengths = 2 * calculate_simplex_circumradii(coordinates[tri.simplices[:]])
    triangles = shapely.multipolygons(
        coordinates[tri.simplices[diameter_lengths <= d_max]]
    )
    return shapely.unary_union(triangles)


def remove_small_holes(geometry, min_area):
    polygons = []
    for geom in shapely.get_parts(geometry):
        if not isinstance(geom, shapely.Polygon):
            # Skip non-polygon geometries
            print(f"Skipping non-polygon geometry: {geom}")
            continue
        if geom.interiors == 0:
            polygons.append(geom)
            continue
        ee, *ii = shapely.get_rings(geom)

        polygon = shapely.Polygon(
            ee, list(filter(lambda rr: shapely.polygons(rr).area >= min_area, ii))
        )
        polygons.append(polygon)
    return shapely.MultiPolygon(polygons)


def whole_tissue_alphashape(coordinates, max_coordinates=50_000, d_max_percentile=99.5):
    import scipy.spatial

    if max_coordinates is None:
        max_coordinates = len(coordinates)
    idxer = np.arange(len(coordinates))
    n_selected = np.min([len(coordinates), max_coordinates])
    idxer = np.random.choice(idxer, n_selected, replace=False)

    coordinates = coordinates[idxer]
    tri = scipy.spatial.Delaunay(coordinates)
    buffer_size, r_max = np.percentile(
        calculate_simplex_circumradii(coordinates[tri.simplices]),
        [50, d_max_percentile],
    )
    print(buffer_size, r_max)
    polygons = alphashape(coordinates, r_max * 2)
    polygons = remove_small_holes(polygons.buffer(buffer_size), np.pi * buffer_size**2)
    return polygons.buffer(buffer_size)
