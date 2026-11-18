
import streamlit as st
from streamlit_folium import st_folium
from map_visualization import create_route_map

from api_client import (
    check_api_connection,
    get_evaluation_trips
)


st.set_page_config(
    page_title="PathSnap",
    page_icon="🗺️",
    layout="wide"
)


st.title("🗺️ PathSnap")
st.subheader(
    "Distributed Map-Matching & Route Reconstruction"
)

st.write(
    "Reconstructing realistic vehicle trajectories "
    "from noisy GPS observations."
)

st.divider()


st.header("Evaluation Dashboard")


st.subheader("Backend Connection")

connected, health_data = check_api_connection()

if connected:
    st.success("Laptop 1 API is connected.")

    with st.expander("API health information"):
        st.json(health_data)

else:
    st.error("Laptop 1 API is currently unavailable.")

    st.caption(
        "Start the FastAPI server on Laptop 1 "
        "and check the API address in the .env file."
    )


st.divider()


st.subheader("Evaluation Trip Selection")

if connected:

    trips, error = get_evaluation_trips()

    if error:
        st.error(f"Could not retrieve trips: {error}")

    else:
        st.write(
            f"Available evaluation trips: {len(trips)}"
        )

        selected_trip = st.selectbox(
            "Select an evaluation trip",
            trips
        )

        st.info(f"Selected trip: {selected_trip}")

        st.caption(
            "Route prediction will be enabled "
            "after the backend prediction interface "
            "is finalised."
        )

else:
    st.warning(
        "Trip selection will become available "
        "when Laptop 1's API is running."
    )


st.divider()

st.header("Route Visualisation")

if "prediction_result" in st.session_state:

    result = st.session_state["prediction_result"]

    try:
        route_map = create_route_map(
            gps_points=result.get("gps_points", []),
            predicted_route=result.get("predicted_route", []),
            road_network=result.get("road_network")
        )

        st_folium(
            route_map,
            width=None,
            height=550,
            key="pathsnap_map"
        )

    except ValueError as error:
        st.warning(str(error))

else:
    st.info(
        "Select an evaluation trip and run prediction "
        "to display its geographical map."
    )


st.header("Evaluation Results")

st.info(
    "Prediction metrics will appear here "
    "after receiving the actual API response."
)


st.divider()

st.caption(
    "PathSnap | Distributed Map-Matching System"
)
