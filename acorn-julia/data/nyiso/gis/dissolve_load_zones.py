import geopandas as gpd

# Read disaggregated load zones
gdf = gpd.read_file("./NYISO_Load_Zone.shp")

# Map names to zones
name_mapping = {
    "West": "A",
    "Genesee": "B",
    "Central": "C",
    "North": "D",
    "Mohawk Valley": "E",
    "Capital": "F",
    "Hudson Valley": "G",
    "Millwood": "H",
    "Dunwoodie": "I",
    "New York City": "J",
    "Long Island": "K",
}

# Dissolve and save
gdf_dissolved = gdf.dissolve(by="ZONE_NAME").reset_index()
gdf_dissolved["ZONE"] = gdf_dissolved["ZONE_NAME"].map(name_mapping)
gdf_dissolved = gdf_dissolved[["ZONE", "ZONE_NAME", "geometry", "COLOR"]]

# Rename columns
gdf_dissolved.rename(
    columns={"ZONE_NAME": "zone_name", "ZONE": "zone", "COLOR": "color"},
    inplace=True,
)

gdf_dissolved.to_file("./NYISO_Load_Zone_Dissolved.shp")
