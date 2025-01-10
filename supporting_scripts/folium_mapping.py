    
import folium
from folium.features import DivIcon
from folium.plugins import MousePosition



def basin_mapping(basin, site_feature):
    # create map
    m = folium.Map(tiles='http://services.arcgisonline.com/arcgis/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}',
                                    attr="Sources: National Geographic",
                                    zoom_start=8, 
                        control_scale=True)
    _ = MousePosition().add_to(m)


    # watershed boundary
    watershed_json = basin.to_crs(epsg='4326').to_json()
    w = folium.features.GeoJson(data=watershed_json, style_function=lambda x: {'color':'darkblue', 'fillColor':'blue'})
    m.add_child(w)

    folium.GeoJson(site_feature, 
                tooltip=folium.GeoJsonTooltip(["identifier"]),
                marker=folium.Marker(icon=folium.Icon(color='blue', icon='water', prefix='fa'))).add_to(m)


    # Set the map extent (bounds) to the extent of the sites
    m.fit_bounds(m.get_bounds())
    return m


def snotel_mapping(gdf_in_bbox, basin, site_feature):
    # Calculate the bounds to set the map's initial view
    minx, miny, maxx, maxy = gdf_in_bbox.total_bounds

    # Calculate the center of the bounding box
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2

    # Convert GeoDataFrames to GeoJSON
    geojson1 = gdf_in_bbox.astype(dict(beginDate=str, endDate=str)).to_json()
    geojson2 = basin.to_json()

    # Create a folium map
    m = folium.Map([center_lat, center_lon], zoom_start=10,tiles='http://services.arcgisonline.com/arcgis/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}',
                                    attr="Sources: National Geographic",
                        control_scale=True)

    _ = MousePosition().add_to(m)

    folium.GeoJson(site_feature, 
                tooltip=folium.GeoJsonTooltip(["identifier"]),
                marker=folium.Marker(icon=folium.Icon(color='blue', icon='water', prefix='fa'))
                ).add_to(m)


    # Add GeoJSON layers for each GeoDataFrame to the map
    folium.GeoJson(geojson1, tooltip=folium.GeoJsonTooltip(["code"]), 
                name='GeoDataFrame 1',
                marker=folium.Marker(icon=folium.Icon(color='blue', icon='snowflake', prefix='fa'))).add_to(m)
    folium.GeoJson(geojson2, name='GeoDataFrame 2').add_to(m)

    # Add layer control to the map
    folium.LayerControl().add_to(m)

    # Display the map
    return m