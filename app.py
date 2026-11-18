
import streamlit as st

from streamlit_folium import st_folium

from api_client import (
    check_api_connection,
    get_evaluation_trips
)

from map_visualization import create_route_map


st.set_page_config(
    page_title="PathSnap",
    page_icon="🗺️",
    layout="wide"
)


st.title("🗺️ PathSnap")
st.caption(
    "DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION"
)

st.write(
    "Reconstruct realistic vehicle trajectories "
    "from noisy GPS observations."
)

st.divider()


with st.sidebar:
    st.title("🗺️ PathSnap")
    st.caption("Evaluation Control Panel")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Route Visualisation",
            "Evaluation Results"
        ]
    )

    st.divider()

    st.caption("Laptop 2 | User Interface")


connected, health_data = check_api_connection()


if page == "Dashboard":

    st.header("Evaluation Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Backend Status",
            "Connected" if connected else "Offline"
        )

    with col2:
        st.metric(
            "Evaluation Trips",
            "10 reserved",
            help="The actual trip list will be fetched from Laptop 1."
        )

    if connected:
        st.success("Laptop 1 API is responding.")
    else:
        st.warning(
            "Laptop 1 API is not connected. "
            "Trip selection will be available once the backend starts."
        )

    st.divider()

    st.subheader("Select Evaluation Trip")

    if connected:

        trips, error = get_evaluation_trips()

        if error:
            st.error(error)

        else:
            selected_trip = st.selectbox(
                "Choose a trip",
                trips
            )

            st.write(f"Selected Trip: **{selected_trip}**")

            st.button(
                "Predict Route",
                disabled=True,
                use_container_width=True,
                help="Prediction will be enabled after the API contract is confirmed."
            )

            st.info(
                "Prediction integration is being prepared. "
                "The button will be connected to Laptop 1's API."
            )

    else:
        st.info(
            "Waiting for the ten reserved evaluation trips "
            "from Laptop 1."
        )

    st.divider()

    st.subheader("System Architecture")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "**Laptop 1 – Backend**\n\n"
            "Model, HMM/Viterbi, road network and FastAPI."
        )

    with col2:
        st.info(
            "**Laptop 2 – Frontend**\n\n"
            "Streamlit interface, API communication "
            "and Folium visualisation."
        )


elif page == "Route Visualisation":

    st.header("Interactive Route Visualisation")

    result = st.session_state.get("prediction_result")

    if result:

        try:
            route_map = create_route_map(
                gps_points=result.get("gps_points", []),
                predicted_route=result.get("predicted_route", []),
                road_network=result.get("road_network"),
                reference_route=result.get("reference_route", [])
            )

            st_folium(
                route_map,
                width=None,
                height=600,
                key="pathsnap_map"
            )

        except ValueError as error:
            st.error(str(error))

    else:
        st.info(
            "The interactive map will appear here "
            "after a successful route prediction."
        )


elif page == "Evaluation Results":

    st.header("Evaluation Results")

    result = st.session_state.get("prediction_result")

    if result:

        metrics = result.get("metrics", {})

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Accuracy",
                metrics.get("accuracy", "N/A")
            )

        with col2:
            st.metric(
                "Prediction Time",
                metrics.get("prediction_time", "N/A")
            )

        with col3:
            st.metric(
                "Distance Error",
                metrics.get("distance_error", "N/A")
            )

        st.json(metrics)

    else:
        st.info(
            "Evaluation metrics will be displayed "
            "after receiving the actual prediction response."
        )


st.divider()

st.caption(
    "PathSnap | Distributed Map-Matching System"
)
