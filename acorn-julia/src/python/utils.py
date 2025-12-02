import numpy as np
import random
from sklearn.neighbors import BallTree
import geopandas as gpd
import pandas as pd

#############
# Paths
#############
project_path = "/home/fs01/jl2966/acorn-julia2/acorn-julia"
tgw_path = "/home/shared/vs498_0001/im3_hyperfacets_tgw"

#############
# Names
#############
zone_names = {
    "A": "West",
    "B": "Genesee",
    "C": "Central",
    "D": "North",
    "E": "Mohawk Valley",
    "F": "Capital",
    "G": "Hudson Valley",
    "H": "Millwood",
    "I": "Dunwoodie",
    "J": "New York City",
    "K": "Long Island",
}

month_names = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

month_keys = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


###################
# Lat/Lon -> zone
###################
def merge_to_zones(
    df,
    nyiso_zone_shp_path: str = f"{project_path}/data/nyiso/gis/NYISO_Load_Zone_Dissolved.shp",
    lat_name: str = "lat",
    lon_name: str = "lon",
    join: str = "inner",
):
    """
    Merge a dataframe with lat/lon coordinates to the NYISO zones.
    """
    # Read NYISO zones
    nyiso_gdf = gpd.read_file(nyiso_zone_shp_path)

    # Merge
    df_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_name], df[lat_name]),
        crs="EPSG:4326",
    )

    # Merge
    df_gdf = gpd.sjoin(df_gdf, nyiso_gdf, how=join, predicate="within")

    # Return
    return df_gdf.drop(columns=["index_right"])


#################################
# Map GenX zones to NYISO zones
#################################
def map_genX_zones_to_nyiso(
    df_genX,
    genX_zone_col: str = "region",
    C_and_E_mapping="random",
    G_to_I_mapping="G",
):
    """
    Map GenX zones to NYISO zones.
    """
    # Tidy region column
    if genX_zone_col == "region":
        df_genX["genX_zone"] = df_genX["region"].apply(lambda x: x.split("_")[-1])
    else:
        df_genX["genX_zone"] = df_genX[genX_zone_col]

    # Define mapping functions
    if C_and_E_mapping == "random":
        C_and_E_mapping_func = lambda x: random.choice(["C", "E"])
    elif C_and_E_mapping == "C" or C_and_E_mapping == "E":
        C_and_E_mapping_func = lambda x: C_and_E_mapping
    else:
        raise ValueError(f"Invalid mapping for {C_and_E_mapping}")

    genX_to_NYSIO_zones_map = {
        "A": lambda x: "A",
        "B": lambda x: "B",
        "D": lambda x: "D",
        "F": lambda x: "F",
        "K": lambda x: "K",
        "J": lambda x: "J",
        "C&E": C_and_E_mapping_func,
        "G-I": lambda x: G_to_I_mapping,
    }

    # Apply map
    df_genX["genX_zone"] = df_genX["genX_zone"].apply(
        lambda x: genX_to_NYSIO_zones_map[x](x)
    )

    return df_genX


#########################
# Nearest bus functions
# Adapted from: https://github.com/boyuan276/NYgrid-python/blob/main/nygrid/allocate.py
#########################
def get_nearest(
    src_points: np.array,
    candidates: np.array,
    k_neighbors: int = 1,
    metric: str = "minkowski",
    leaf_size: int = 20,
):
    """
    Find nearest neighbors for all source points from a set of candidate points

    Parameters
    ----------
    src_points : np.ndarray
        A numpy array of shape (n, 2) representing the source points.
    candidates : np.ndarray
        A numpy array of shape (m, 2) representing the candidate points.
    k_neighbors : int
        Number of nearest neighbors to return.
    metric : str
        The distance metric to use. Default is 'minkowski'.
    leaf_size : int
        Leaf size passed to BallTree. Default is 20.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The indices of the k-nearest neighbors in the candidates array and the corresponding distances.
    """

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=leaf_size, metric=metric)

    # Find the closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    closest = indices[0]
    closest_dist = distances[0]

    # Return indices and distances
    return closest, closest_dist


def nearest_neighbor_lat_lon(
    origin_gdf: gpd.GeoDataFrame,
    match_zones: bool = True,
    return_dist: bool = False,
    leaf_size: int = 20,
    PV_bus_only: bool = True,
):
    """
    For each point in origin_gdf, find closest point in right GeoDataFrame and return them.
    If match_zones is True, only points with matching zone values will be considered.

    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).

    Parameters
    ----------
    origin_gdf : gpd.GeoDataFrame
        A GeoDataFrame containing the origin points.
    return_dist : bool
        If True, the distance between the nearest neighbors is returned.
    leaf_size : int
        Leaf size passed to BallTree. Default is 20.
    match_zones : bool
        If True, only points with matching zone values will be considered.
        Assumes 'zone' column exists in both origin_gdf and candidate_gdf.

    Returns
    -------
    closest_points: Union[Dict, Tuple]
        A dictionary or tuple containing the closest points and distances (if requested).
    """
    # Read byus GDF as candidate_gdf
    bus_gdf = gpd.read_file(f"{project_path}/data/grid/gis/Bus_clean.shp")
    if PV_bus_only:
        bus_gdf = bus_gdf[bus_gdf["BUS_TYPE"] == 2].copy()

    left_geom_col = origin_gdf.geometry.name
    right_geom_col = bus_gdf.geometry.name

    # Initialize the results container
    result_rows = []
    distances = []

    # If zone matching is requested, process each point individually by zone
    if match_zones:
        for idx, left_row in origin_gdf.iterrows():
            left_zone = left_row["zone"]

            # Filter right_gdf to only include points with matching zone
            matching_right = (
                bus_gdf[bus_gdf["zone"] == left_zone].copy().reset_index(drop=True)
            )

            # Skip if no matching zones found
            if len(matching_right) == 0:
                continue

            # Parse coordinates for the single left point and convert to radians
            left_point_radians = np.array(
                [(left_row.geometry.x * np.pi / 180, left_row.geometry.y * np.pi / 180)]
            )

            # Parse coordinates for all matching right points and convert to radians
            right_points_radians = np.array(
                matching_right[right_geom_col]
                .apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180))
                .to_list()
            )

            # Find nearest neighbor among zone-matching points
            closest_idx, dist = get_nearest(
                src_points=left_point_radians,
                candidates=right_points_radians,
                leaf_size=leaf_size,
            )

            # Get the closest point information
            closest_point = matching_right.iloc[closest_idx[0]]
            result = pd.concat([left_row, closest_point[["bus_id"]]])
            result_rows.append(result)

            if return_dist:
                distances.append(dist[0])
    # If no zone filtering, use the original efficient implementation
    else:
        # Ensure that index in right gdf is formed of sequential numbers
        right = bus_gdf.copy().reset_index(drop=True)

        # Parse coordinates from points and insert them into a numpy array as RADIANS
        left_radians = np.array(
            origin_gdf[left_geom_col]
            .apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180))
            .to_list()
        )
        right_radians = np.array(
            right[right_geom_col]
            .apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180))
            .to_list()
        )

        # Find the nearest points
        # -----------------------
        # closest ==> index in candidate_gdf that corresponds to the closest point
        # dist ==> distance between the nearest neighbors (in meters)
        closest, dist = get_nearest(
            src_points=left_radians,
            candidates=right_radians,
            metric="haversine",
            leaf_size=leaf_size,
        )

        # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
        result_bus_idds = right.loc[closest]["bus_id"].to_numpy()
        result_rows = origin_gdf.copy()
        result_rows["bus_id"] = result_bus_idds
        distances = dist

    # Create the final GeoDataFrame
    if match_zones:
        closest_points = gpd.GeoDataFrame(result_rows, geometry=right_geom_col)
    else:
        closest_points = result_rows.reset_index(drop=True)

    # Add distance if requested
    if return_dist:
        # Convert to meters from radians
        earth_radius = 6371000  # meters
        closest_points["distance"] = [d * earth_radius for d in distances]

    return closest_points