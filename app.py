import os
import streamlit as st
import streamlit.components.v1 as components

from api_client import (
    API_BASE_URL,
    check_api_connection,
    get_evaluation_trips,
    predict_route
)
from map_visualization import create_route_map


st.set_page_config(
    page_title="PathSnap",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ PathSnap")
st.caption("DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION")

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

    custom_api_url = st.text_input(
        "Laptop 1 API URL",
        value=API_BASE_URL,
        help="Set Laptop 1's IP address (e.g. http://192.168.1.15:8000 or http://127.0.0.1:8000)"
    )

    st.divider()
    st.caption("Laptop 2 | User Interface")


connected, health_data = check_api_connection(custom_api_url)


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
            help="The actual trip list is fetched live from Laptop 1."
        )

    if connected:
        model_arch = health_data.get("model_architecture", "Active") if isinstance(health_data, dict) else "Active"
        st.success(f"Laptop 1 API is responding (Loaded Model: {model_arch}).")
    else:
        st.warning(
            "Laptop 1 API is not connected. "
            "Trip selection will be available once the backend starts."
        )

    st.divider()

    st.subheader("Select Evaluation Trip")

    if connected:
        trips, error = get_evaluation_trips(custom_api_url)

        if error:
            st.error(error)
        else:
            selected_trip = st.selectbox(
                "Choose a trip",
                trips
            )

            st.write(f"Selected Trip: **{selected_trip}**")

            if st.button(
                "Predict Route",
                type="primary",
                use_container_width=True
            ):
                with st.spinner(f"Sending Trip {selected_trip} to Laptop 1 API and running ML + HMM/Viterbi pipeline..."):
                    result, pred_error = predict_route(selected_trip, custom_api_url)

                    if pred_error:
                        st.error(f"Prediction request failed: {pred_error}")
                    else:
                        st.session_state["prediction_result"] = result
                        st.success(
                            f"Route reconstructed for Trip **{selected_trip}**! "
                            f"Switch to **Route Visualisation** or **Evaluation Results** in the sidebar, or view the summary below."
                        )

            result = st.session_state.get("prediction_result")
            if result and result.get("trip_id") == selected_trip:
                metrics = result.get("metrics", {})
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Observation Accuracy", metrics.get("accuracy", "N/A"))
                m2.metric("Route Continuity", metrics.get("route_continuity", "N/A"))
                m3.metric("Prediction Time", metrics.get("prediction_time", "N/A"))
                m4.metric("Mean Distance Error", metrics.get("distance_error", "N/A"))

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
        trip_id = result.get("trip_id", "unknown")
        st.subheader(f"Trip `{trip_id}` — Reconstructed Trajectory")

        try:
            output_html_path = os.path.join("outputs", f"predicted_route_{trip_id}.html")
            route_map = create_route_map(
                gps_points=result.get("gps_points", []),
                predicted_route=result.get("predicted_route", []),
                road_network=result.get("road_network"),
                reference_route=result.get("reference_route", []),
                output_path=output_html_path
            )

            components.html(
                route_map._repr_html_(),
                height=620
            )
            st.caption(f"Interactive Folium HTML map also saved to `{output_html_path}`.")

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
        trip_id = result.get("trip_id", "unknown")
        st.subheader(f"Server-Side Evaluation Metrics — Trip `{trip_id}`")

        metrics = result.get("metrics", {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Accuracy",
                metrics.get("accuracy", "N/A")
            )

        with col2:
            st.metric(
                "Route Continuity",
                metrics.get("route_continuity", "N/A")
            )

        with col3:
            st.metric(
                "Prediction Time",
                metrics.get("prediction_time", "N/A")
            )

        with col4:
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

st.caption("PathSnap | Distributed Map-Matching System")
