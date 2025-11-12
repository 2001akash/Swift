import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from collections import Counter
import os
import warnings

# --- Configuration ---
INPUT_JSON_FILE = 'Swift Assignment 4 - Dataset (1).json'
OUTPUT_DETAILED_CSV = 'transit_performance_detailed.csv'
OUTPUT_SUMMARY_CSV = 'transit_performance_summary.csv'
IST = pytz.timezone('Asia/Kolkata')

# Ignore SettingWithCopyWarning for cleaner output during calculations
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


# --- Helper Functions ---

def safe_get(data, keys, default=None):
    """Safely retrieve nested dictionary keys."""
    if not isinstance(data, dict):
        return default
    _data = data
    for key in keys:
        if isinstance(_data, dict):
            _data = _data.get(key)
            if _data is None:
                return default
        else:
            return default
    return _data

def parse_timestamp(ts_data):
    """Parse timestamp from MongoDB $numberLong or ISO string format to UTC datetime."""
    if isinstance(ts_data, dict) and '$numberLong' in ts_data:
        try:
            # Timestamp is in milliseconds since epoch
            ms_epoch = int(ts_data['$numberLong'])
            # Convert milliseconds to seconds
            return pd.to_datetime(ms_epoch / 1000, unit='s', utc=True)
        except (ValueError, TypeError):
            return pd.NaT
    elif isinstance(ts_data, str):
        try:
            # Assume ISO 8601 format
            dt = pd.to_datetime(ts_data)
            # Ensure it's timezone-aware (assume UTC if naive)
            if dt.tzinfo is None:
                return dt.tz_localize('UTC')
            else:
                return dt.tz_convert('UTC')
        except (ValueError, TypeError):
             return pd.NaT # Handle potential parsing errors
    return pd.NaT # Return Not a Time for other types or errors

def get_event_location_key(event):
    """Creates a unique key for a facility location based on available address info."""
    city = safe_get(event, ['address', 'city'], '')
    state = safe_get(event, ['address', 'stateOrProvinceCode'], '')
    postal = safe_get(event, ['address', 'postalCode'], '')
    # Use a combination, handling potential missing parts
    key_parts = [part for part in [city, state, postal] if part]
    if not key_parts:
        return None # No usable location info
    return "_".join(key_parts).upper() # Normalize

def calculate_inter_facility_time(events_df):
    """Calculate total time spent between consecutive facility events."""
    facility_events = events_df[
        events_df['arrivalLocation'].str.contains("FACILITY", na=False)
    ].sort_values('timestamp').drop_duplicates(subset=['timestamp']) # Ensure sorted and unique timestamps

    if len(facility_events) < 2:
        return 0

    # Calculate time difference between consecutive facility events
    time_diffs = facility_events['timestamp'].diff()

    # Sum the time differences (first diff will be NaT, sum ignores it)
    total_inter_facility_timedelta = time_diffs.sum()

    # Convert timedelta to hours
    if pd.notna(total_inter_facility_timedelta):
        return total_inter_facility_timedelta.total_seconds() / 3600
    else:
        return 0


# Part 1: Load Data 
print(f"Loading data from {INPUT_JSON_FILE}...")
try:
    with open(INPUT_JSON_FILE, 'r') as f:
        data = json.load(f)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: Input JSON file not found at '{INPUT_JSON_FILE}'")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{INPUT_JSON_FILE}'")
    exit()

# Part 2, 3 & 4: Flatten Data and Calculate Metrics per Shipment
print("Processing shipments...")
all_shipment_metrics = []
all_event_types_found = set()
processed_count = 0

for shipment_record in data:
    track_details_list = safe_get(shipment_record, ['trackDetails'], [])
    if not track_details_list:
        # print("Skipping record with no trackDetails.")
        continue

    # Assuming one trackDetail per record as per sample data
    track_details = track_details_list[0]
    if not track_details:
        continue

    processed_count += 1
    # Extract static shipment info
    tracking_number = safe_get(track_details, ['trackingNumber']) # [cite: 15]
    if not tracking_number:
        # print("Skipping track detail with no tracking number.")
        continue # Essential identifier missing

    service_type = safe_get(track_details, ['service', 'type']) # [cite: 16]
    service_description = safe_get(track_details, ['service', 'description']) # [cite: 17]
    carrier_code = safe_get(track_details, ['carrierCode']) # [cite: 18]
    package_weight_value = safe_get(track_details, ['packageWeight', 'value']) # [cite: 20]
    package_weight_units = safe_get(track_details, ['packageWeight', 'units']) # [cite: 20]
    packaging_type = safe_get(track_details, ['packaging', 'description']) # [cite: 21]

    origin_city = safe_get(track_details, ['shipperAddress', 'city']) # [cite: 23]
    origin_state = safe_get(track_details, ['shipperAddress', 'stateOrProvinceCode']) # [cite: 23]
    origin_pincode = None # Not directly available for shipper, will try extracting from first PU event

    dest_city = safe_get(track_details, ['destinationAddress', 'city']) # [cite: 24]
    dest_state = safe_get(track_details, ['destinationAddress', 'stateOrProvinceCode']) # [cite: 24]
    dest_pincode = None # Not directly available for destination, will try extracting from last DL event

    delivery_location_type = safe_get(track_details, ['deliveryLocationType']) # [cite: 47]

    # Convert weight to KG 
    package_weight_kg = None
    if package_weight_value is not None and package_weight_units is not None:
        if package_weight_units.upper() == 'KG':
            package_weight_kg = package_weight_value
        elif package_weight_units.upper() == 'LB':
            package_weight_kg = package_weight_value * 0.453592 # Convert LB to KG
        # Add other unit conversions if needed

    # Extract events [cite: 26]
    events_data = safe_get(track_details, ['events'], [])
    if not events_data: # Handle shipments with no events 
        # Append minimal data if no events exist
        shipment_metrics = {
            'tracking_number': tracking_number,
            'service_type': service_type,
            'carrier_code': carrier_code,
            'package_weight_kg': package_weight_kg,
            'packaging_type': packaging_type,
            'origin_city': origin_city,
            'origin_state': origin_state,
            'origin_pincode': origin_pincode,
            'destination_city': dest_city,
            'destination_state': dest_state,
            'destination_pincode': dest_pincode,
            'pickup_datetime_ist': pd.NaT,
            'delivery_datetime_ist': pd.NaT,
            'total_transit_hours': np.nan,
            'num_facilities_visited': 0,
            'num_in_transit_events': 0,
            'time_in_inter_facility_transit_hours': 0,
            'avg_hours_per_facility': np.nan,
            'is_express_service': False, # Default based on sample data
            'delivery_location_type': delivery_location_type,
            'num_out_for_delivery_attempts': 0,
            'first_attempt_delivery': np.nan, # Cannot determine without events
            'total_events_count': 0
        }
        all_shipment_metrics.append(shipment_metrics)
        continue # Move to the next shipment

    # Flatten events data
    flattened_events = []
    for event in events_data:
        timestamp_utc = parse_timestamp(safe_get(event, ['timestamp'])) # [cite: 28, 53]
        if pd.isna(timestamp_utc):
            continue # Skip events with invalid timestamps

        event_type = safe_get(event, ['eventType']) # [cite: 27]
        if event_type:
             all_event_types_found.add(event_type)

        flattened_events.append({
            'tracking_number': tracking_number,
            'timestamp': timestamp_utc,
            'eventType': event_type,
            'eventDescription': safe_get(event, ['eventDescription']), # [cite: 29]
            'event_city': safe_get(event, ['address', 'city']), # [cite: 30]
            'event_state': safe_get(event, ['address', 'stateOrProvinceCode']), # [cite: 31]
            'event_postalCode': safe_get(event, ['address', 'postalCode']), # [cite: 32]
            'arrivalLocation': safe_get(event, ['arrivalLocation']) # [cite: 33]
        })

    if not flattened_events: # Handle case where all events had invalid timestamps
         # Append minimal data similar to the 'no events' case
        shipment_metrics = {
            'tracking_number': tracking_number, 'service_type': service_type, 'carrier_code': carrier_code,
            'package_weight_kg': package_weight_kg, 'packaging_type': packaging_type,
            'origin_city': origin_city, 'origin_state': origin_state, 'origin_pincode': origin_pincode,
            'destination_city': dest_city, 'destination_state': dest_state, 'destination_pincode': dest_pincode,
            'pickup_datetime_ist': pd.NaT, 'delivery_datetime_ist': pd.NaT, 'total_transit_hours': np.nan,
            'num_facilities_visited': 0, 'num_in_transit_events': 0,
            'time_in_inter_facility_transit_hours': 0, 'avg_hours_per_facility': np.nan,
            'is_express_service': False, 'delivery_location_type': delivery_location_type,
            'num_out_for_delivery_attempts': 0, 'first_attempt_delivery': np.nan,
            'total_events_count': 0
        }
        all_shipment_metrics.append(shipment_metrics)
        continue

    events_df = pd.DataFrame(flattened_events)
    # Handle duplicate events at the same timestamp 
    events_df = events_df.drop_duplicates(subset=['timestamp', 'eventType', 'eventDescription', 'event_city', 'event_state'])
    events_df = events_df.sort_values(by='timestamp') # Ensure chronological order
    events_df.reset_index(drop=True, inplace=True)

    # --- Calculate Metrics for the current shipment ---

    # Get total events count
    total_events_count = len(events_df) # 

    # 1. Facility Touchpoints [cite: 36]
    facility_events = events_df[events_df['arrivalLocation'].str.contains("FACILITY", na=False)]
    unique_facility_locations = set()
    for _, event_row in facility_events.iterrows():
        loc_key = get_event_location_key(event_row.to_dict())
        if loc_key:
            unique_facility_locations.add(loc_key)
    num_facilities_visited = len(unique_facility_locations) # 

    # Event type counts [cite: 38] - Define 'in transit' vs 'arrival' (simplistic approach here)
    event_counts = events_df['eventType'].value_counts()
    num_in_transit_events = event_counts.get('IT', 0) # Count 'IT' events
    # Could add counts for 'AR', 'DP', etc. if needed, but not required for output CSV

    # 2. Transit Time Analysis [cite: 40]
    pickup_event = events_df[events_df['eventType'] == 'PU'].head(1)
    delivery_event = events_df[events_df['eventType'] == 'DL'].tail(1)

    pickup_datetime = pickup_event['timestamp'].iloc[0] if not pickup_event.empty else pd.NaT
    delivery_datetime = delivery_event['timestamp'].iloc[0] if not delivery_event.empty else pd.NaT

    # Extract origin/destination pincodes from first PU/last DL event if available
    if origin_pincode is None and not pickup_event.empty:
        origin_pincode = pickup_event['event_postalCode'].iloc[0] #
    if dest_pincode is None and not delivery_event.empty:
        dest_pincode = delivery_event['event_postalCode'].iloc[0] #


    total_transit_hours = np.nan
    if pd.notna(pickup_datetime) and pd.notna(delivery_datetime) and delivery_datetime > pickup_datetime:
        total_transit_timedelta = delivery_datetime - pickup_datetime
        total_transit_hours = total_transit_timedelta.total_seconds() / 3600 # [cite: 41, 69]

    # Time in inter-facility transit [cite: 42, 72]
    time_in_inter_facility_hours = calculate_inter_facility_time(events_df)


    # 3. Transit Velocity [cite: 43]
    avg_hours_per_facility = np.nan
    if num_facilities_visited > 0 and not np.isnan(total_transit_hours):
        avg_hours_per_facility = total_transit_hours / num_facilities_visited # 

    # Service category classification 
    # Assuming 'SAVER' is economy/standard based on description 'FedEx Economy'
    is_express_service = False
    if service_type and 'EXPRESS' in service_type and 'SAVER' not in service_type:
         is_express_service = True # Simple logic, might need refinement

    # 4. Delivery Characteristics [cite: 46]
    num_out_for_delivery = event_counts.get('OD', 0) # 

    first_attempt_delivery = np.nan
    if num_out_for_delivery > 0:
        first_attempt_delivery = (num_out_for_delivery == 1) # 


    # Store calculated metrics
    shipment_metrics = {
        'tracking_number': tracking_number, # [cite: 61]
        'service_type': service_type, # [cite: 62]
        'carrier_code': carrier_code, # [cite: 63]
        'package_weight_kg': package_weight_kg, # 
        'packaging_type': packaging_type, # [cite: 65]
        'origin_city': origin_city, # [cite: 66]
        'origin_state': origin_state, # [cite: 66]
        'origin_pincode': origin_pincode, # [cite: 66]
        'destination_city': dest_city, # [cite: 67]
        'destination_state': dest_state, # [cite: 67]
        'destination_pincode': dest_pincode, # [cite: 67]
        'pickup_datetime_ist': pickup_datetime.tz_convert(IST) if pd.notna(pickup_datetime) else pd.NaT, # 
        'delivery_datetime_ist': delivery_datetime.tz_convert(IST) if pd.notna(delivery_datetime) else pd.NaT, # 
        'total_transit_hours': total_transit_hours, # 
        'num_facilities_visited': num_facilities_visited, # [cite: 70]
        'num_in_transit_events': num_in_transit_events, # [cite: 71]
        'time_in_inter_facility_transit_hours': time_in_inter_facility_hours, # 
        'avg_hours_per_facility': avg_hours_per_facility, # [cite: 73]
        'is_express_service': is_express_service, # [cite: 74]
        'delivery_location_type': delivery_location_type, # [cite: 75]
        'num_out_for_delivery_attempts': num_out_for_delivery, # [cite: 76]
        'first_attempt_delivery': first_attempt_delivery, # [cite: 77]
        'total_events_count': total_events_count # 
    }
    all_shipment_metrics.append(shipment_metrics)

print(f"Processed {processed_count} shipments.")

# List unique event types found [cite: 39]
unique_event_types_list = sorted(list(all_event_types_found))
print(f"\nUnique Event Types Found in Dataset: {unique_event_types_list}")

# Part 5: Create Detailed CSV 
print(f"\nCreating detailed CSV: {OUTPUT_DETAILED_CSV}...")
detailed_df = pd.DataFrame(all_shipment_metrics)

# Format datetime columns for CSV output 
detailed_df['pickup_datetime_ist'] = detailed_df['pickup_datetime_ist'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
detailed_df['delivery_datetime_ist'] = detailed_df['delivery_datetime_ist'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')

# Rename columns to match CSV specification
detailed_df = detailed_df.rename(columns={'destination_pincode': 'destination_pincode'}) # No actual rename needed if name matches

# Ensure all required columns exist, adding empty ones if necessary
required_detailed_cols = [
    'tracking_number', 'service_type', 'carrier_code', 'package_weight_kg',
    'packaging_type', 'origin_city', 'origin_state', 'origin_pincode',
    'destination_city', 'destination_state', 'destination_pincode',
    'pickup_datetime_ist', 'delivery_datetime_ist', 'total_transit_hours',
    'num_facilities_visited', 'num_in_transit_events',
    'time_in_inter_facility_transit_hours', 'avg_hours_per_facility',
    'is_express_service', 'delivery_location_type',
    'num_out_for_delivery_attempts', 'first_attempt_delivery',
    'total_events_count'
]
for col in required_detailed_cols:
    if col not in detailed_df.columns:
        detailed_df[col] = np.nan

# Reorder columns to match the specification
detailed_df = detailed_df[required_detailed_cols]

# Write to CSV
detailed_df.to_csv(OUTPUT_DETAILED_CSV, index=False, na_rep='NULL')
print(f"Detailed CSV '{OUTPUT_DETAILED_CSV}' created successfully.")


# Part 6: Create Summary CSV 
print(f"\nCreating summary CSV: {OUTPUT_SUMMARY_CSV}...")
summary_data = {}

# Overall Metrics [cite: 81]
summary_data['total_shipments_analyzed'] = len(detailed_df) # [cite: 82]
valid_transit_hours = detailed_df['total_transit_hours'].dropna()
summary_data['avg_transit_hours'] = valid_transit_hours.mean() # [cite: 83]
summary_data['median_transit_hours'] = valid_transit_hours.median() # [cite: 83]
summary_data['std_dev_transit_hours'] = valid_transit_hours.std() # [cite: 83]
summary_data['min_transit_hours'] = valid_transit_hours.min() # [cite: 84]
summary_data['max_transit_hours'] = valid_transit_hours.max() # [cite: 84]

# Facility Metrics [cite: 85]
valid_facilities = detailed_df['num_facilities_visited'].dropna()
summary_data['avg_facilities_per_shipment'] = valid_facilities.mean() # [cite: 86]
summary_data['median_facilities_per_shipment'] = valid_facilities.median() # [cite: 86]
# Mode can return multiple values, take the first one or handle appropriately
mode_facilities = valid_facilities.mode()
summary_data['mode_facilities_per_shipment'] = mode_facilities.iloc[0] if not mode_facilities.empty else np.nan # [cite: 87]

valid_avg_hours_per_facility = detailed_df['avg_hours_per_facility'].dropna()
summary_data['avg_hours_per_facility'] = valid_avg_hours_per_facility.mean() # [cite: 88]
summary_data['median_hours_per_facility'] = valid_avg_hours_per_facility.median() # [cite: 88]

# Service Type Comparison [cite: 89]
service_type_groups = detailed_df.groupby('service_type')
avg_transit_by_service = service_type_groups['total_transit_hours'].mean() # [cite: 91]
avg_facilities_by_service = service_type_groups['num_facilities_visited'].mean() # [cite: 92]
count_by_service = service_type_groups.size() # [cite: 93]

for service, avg_transit in avg_transit_by_service.items():
    summary_data[f'avg_transit_hours_by_service_type_{service}'] = avg_transit
for service, avg_facilities in avg_facilities_by_service.items():
    summary_data[f'avg_facilities_by_service_type_{service}'] = avg_facilities
for service, count in count_by_service.items():
    summary_data[f'count_shipments_by_service_type_{service}'] = count

# Delivery Performance [cite: 94]
valid_first_attempt = detailed_df['first_attempt_delivery'].dropna()
if len(valid_first_attempt) > 0:
    summary_data['pct_first_attempt_delivery'] = (valid_first_attempt.sum() / len(valid_first_attempt)) * 100 # [cite: 95]
else:
    summary_data['pct_first_attempt_delivery'] = np.nan

valid_attempts = detailed_df['num_out_for_delivery_attempts'].dropna()
summary_data['avg_out_for_delivery_attempts'] = valid_attempts.mean() # [cite: 96]


# Create Summary DataFrame
# Convert summary dict to DataFrame (one row)
summary_df = pd.DataFrame([summary_data])

# Write to CSV
summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, na_rep='NULL')
print(f"Summary CSV '{OUTPUT_SUMMARY_CSV}' created successfully.")

print("\nAnalysis complete.")