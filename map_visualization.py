
import os
import math

import folium


def validate_points(points):
    valid_points = []

    for point in points or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue

        try:
            lat = float(point[0])
            lon = float(point[1])

            if not math.isfinite(lat) or not math.isfinite(lon):
                continue

            if -90 <= lat <= 90 and -180 <= lon <= 180:
                valid_points.append([lat, lon])

        except (TypeError, ValueError):
            continue

    return valid_points


def create_route_map(
    gps_points,
    predicted_route,
    road_network=None,
    output_path=None
):
    gps_points = validate_points(gps_points)
    predicted_route = validate_points(predicted_route)

    all_points = gps_points + predicted_route

    if not all_points:
        raise ValueError("No valid GPS or route coordinates available.")

    center_lat = sum(p[0] for p in all_points) / len(all_points)
    center_lon = sum(p[1] for p in all_points) / len(all_points)

    route_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles="OpenStreetMap"
    )

    if isinstance(road_network, dict) and road_network.get("features"):
        roads_layer = folium.FeatureGroup(
            name="Road Network",
            show=True
        )

        folium.GeoJson(
            road_network,
            style_function=lambda feature: {
                "color": "#F59E0B",
                "weight": 2,
                "opacity": 0.7
            },
            tooltip="Road Network"
        ).add_to(roads_layer)

        roads_layer.add_to(route_map)

    gps_layer = folium.FeatureGroup(
        name="Original GPS Observations",
        show=True
    )

    for index, point in enumerate(gps_points):
        folium.CircleMarker(
            location=point,
            radius=4,
            color="#2563EB",
            fill=True,
            fill_color="#2563EB",
            fill_opacity=0.8,
            tooltip=f"GPS Observation {index + 1}"
        ).add_to(gps_layer)

    gps_layer.add_to(route_map)

    if predicted_route:
        predicted_layer = folium.FeatureGroup(
            name="Predicted Route",
            show=True
        )

        folium.PolyLine(
            locations=predicted_route,
            color="#16A34A",
            weight=5,
            opacity=0.9,
            tooltip="Predicted Route"
        ).add_to(predicted_layer)

        folium.Marker(
            predicted_route[0],
            tooltip="Predicted Route Start",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(predicted_layer)

        folium.Marker(
            predicted_route[-1],
            tooltip="Predicted Route End",
            icon=folium.Icon(color="red", icon="stop")
        ).add_to(predicted_layer)

        predicted_layer.add_to(route_map)

    folium.LayerControl(
        collapsed=False
    ).add_to(route_map)

    route_map.fit_bounds(all_points)

    legend_html = """
    <div style="
        position: fixed;
        bottom: 35px;
        left: 35px;
        z-index: 9999;
        background: white;
        padding: 12px;
        border: 1px solid grey;
        border-radius: 6px;
        font-size: 13px;
        color: black;
    ">
        <b>PathSnap Map Legend</b><br>
        <span style="color:#2563EB;">●</span>
        Original GPS Observations<br>
        <span style="color:#F59E0B;">━</span>
        Road Network<br>
        <span style="color:#16A34A;">━</span>
        Predicted Route
    </div>
    """

    route_map.get_root().html.add_child(
        folium.Element(legend_html)
    )

    if output_path:
        os.makedirs(
            os.path.dirname(output_path) or ".",
            exist_ok=True
        )

        route_map.save(output_path)

    return route_map
