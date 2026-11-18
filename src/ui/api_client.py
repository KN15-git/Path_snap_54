import os
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
).rstrip("/")


def get_base_url(override_url=None):
    if override_url and str(override_url).strip():
        return str(override_url).strip().rstrip("/")
    return API_BASE_URL


def check_api_connection(override_url=None):
    base_url = get_base_url(override_url)
    try:
        response = requests.get(
            f"{base_url}/health",
            timeout=5
        )
        response.raise_for_status()
        return True, response.json()

    except requests.RequestException as error:
        return False, str(error)


def get_evaluation_trips(override_url=None):
    base_url = get_base_url(override_url)
    try:
        response = requests.get(
            f"{base_url}/trips",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        trips = data.get("trips") if isinstance(data, dict) else data

        if not isinstance(trips, list):
            raise ValueError("Unexpected trip response format")

        trip_ids = []
        for trip in trips:
            if isinstance(trip, str):
                trip_ids.append(trip)
            elif isinstance(trip, dict):
                trip_id = trip.get("trip_id") or trip.get("id")
                if trip_id is not None:
                    trip_ids.append(str(trip_id))

        if not trip_ids:
            raise ValueError("No valid evaluation trips received")

        return trip_ids, None

    except (requests.RequestException, ValueError) as error:
        return [], str(error)


def predict_route(trip_id, override_url=None):
    """
    Sends the selected evaluation trip ID to Laptop 1's FastAPI backend
    to run preprocessing, candidate generation, ML classification,
    HMM/Viterbi route reconstruction, and server-side evaluation.
    """
    base_url = get_base_url(override_url)
    try:
        response = requests.post(
            f"{base_url}/predict/{trip_id}",
            timeout=60
        )
        response.raise_for_status()
        return response.json(), None

    except requests.RequestException as error:
        return None, str(error)
