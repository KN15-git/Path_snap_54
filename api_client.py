
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
).rstrip("/")


def check_api_connection():
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5
        )

        response.raise_for_status()

        return True, response.json()

    except requests.RequestException as error:
        return False, str(error)


def get_evaluation_trips():
    try:
        response = requests.get(
            f"{API_BASE_URL}/trips",
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
