import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import folium
from streamlit_folium import folium_static 
from geopy.geocoders import Nominatim
import re
# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Streamlit support for Plotly charts."""

import json
import urllib.parse
from typing import TYPE_CHECKING, Any, Dict, List, Set, Union, cast

from typing_extensions import Final, Literal, TypeAlias

from streamlit import type_util
from streamlit.errors import StreamlitAPIException
from streamlit.logger import get_logger
from streamlit.proto.PlotlyChart_pb2 import PlotlyChart as PlotlyChartProto
from streamlit.runtime.legacy_caching import caching
from streamlit.runtime.metrics_util import gather_metrics

import matplotlib
import plotly.graph_objs as go
from plotly.basedatatypes import BaseFigure

from streamlit.delta_generator import DeltaGenerator



# Setzen Sie das Plotly-Standard-Design


# Standard coordinates for St. Gallen
default_lat, default_lon = 47.424482, 9.376717

def find_similar_properties_adjusted(input_rooms, input_size, data, threshold=5):
    # Using the 'Details' column to extract rooms and area, then comparing with input
    similar_properties = data[
        data.apply(lambda row: is_similar_property(row['Details'], input_rooms, input_size, threshold), axis=1)
    ]
    return similar_properties

def is_similar_property(details, input_rooms, input_size, threshold):
    # Extracting rooms and area from the 'Details' column
    rooms_match = re.search(r'(\d+(\.\d+)?) Zi\.', details)
    area_match = re.search(r'(\d+(\.\d+)?) m²', details)

    if rooms_match and area_match:
        rooms = float(rooms_match.group(1))
        area = float(area_match.group(1))
        return (rooms >= input_rooms - 1 and rooms <= input_rooms + 1) and (area >= input_size - threshold and area <= input_size + threshold)
    return False

def extract_rooms_and_size(details_str):
    rooms_match = re.search(r'(\d+(\.\d+)?) Zi\.', details_str)
    size_match = re.search(r'(\d+(\.\d+)?) m²', details_str)
    rooms = float(rooms_match.group(1)) if rooms_match else None
    size = float(size_match.group(1)) if size_match else None
    return rooms, size


# Initialize session state variables
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'address' not in st.session_state:
    st.session_state.address = ""

def go_to_next_step():
    st.session_state.current_step += 1

def go_to_previous_step():
    st.session_state.current_step -= 1

def preprocess_and_train():
    file_path = 'Immobilienliste_20231212-220234.xlsx'
    sorted_data = pd.read_excel('Immobilienliste_20231212-220234.xlsx')

    sorted_data.drop(columns=['Name', 'Description'], inplace=True)

    def extract_details(detail_str):
        rooms = re.search(r'(\d+(\.\d+)?) Zi\.', detail_str)
        area = re.search(r'(\d+(\.\d+)?) m²', detail_str)
        return float(rooms.group(1)) if rooms else None, float(area.group(1)) if area else None

    sorted_data['rooms'], sorted_data['area'] = zip(*sorted_data['Details'].apply(extract_details))

    def convert_price(price_str):
        price_str = re.sub(r'[^\d.]', '', price_str)
        return float(price_str) if price_str else None

    sorted_data['Price'] = sorted_data['Price'].apply(convert_price)
    sorted_data['area_code'] = sorted_data['zip'].str.extract(r'(\d{4})')

    sorted_data.dropna(inplace=True)

    X = sorted_data[['rooms', 'area', 'area_code']]
    y = sorted_data['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, sorted_data

def extract_zip_code(input_text):
    parts = input_text.replace(',', ' ').split()
    for part in parts:
        if part.isdigit() and len(part) == 4:
            return part
    return None

def predict_price(size_m2, extracted_zip_code, rooms, model):
    try:
        area_code = int(extracted_zip_code)
        size_m2 = float(size_m2)
        rooms = int(rooms)
    except ValueError as e:
        st.error(f"Invalid input: {e}")
        return None

    input_features = pd.DataFrame({
        'rooms': [rooms],
        'area': [size_m2],
        'area_code': [area_code]
    })

    predicted_price = model.predict(input_features)
    return predicted_price[0]

## Function to predict the price based on the model
#def predict_price(size_m2, area_code, rooms, model): OLD VERSION JUST KEPT IT FOR SECURITA REASONS
#    input_features = pd.DataFrame({
#        'Rooms': [rooms],
#        'Size_m2': [size_m2],
#        'area_code': [zip_code]
#    })
#    predicted_price = model.predict(input_features)
#    return predicted_price[0]

def extract_zip_from_address(address):
    valid_st_gallen_zip_codes = ['9000', '9001', '9004', '9006', '9007', '9008', '9010', '9011', '9012', '9013', '9014', '9015', '9016', '9020', '9021', '9023', '9024', '9026', '9027', '9028', '9029']
    non_specific_inputs = ['st. gallen', 'st gallen', 'sankt gallen']

    # Check for non-specific input
    if address.lower().strip() in non_specific_inputs:
        return "non-specific"

    # If the input is a specific zip code, use it as is
    if address.strip() in valid_st_gallen_zip_codes:
        return address.strip()

    # Otherwise, append ", St. Gallen" to localize the search
    address += ", St. Gallen"

    # Extract zip code from the full address
    geolocator = Nominatim(user_agent="http")
    location = geolocator.geocode(address, country_codes='CH')
    if location:
        address_components = location.raw.get('display_name', '').split(',')
        for component in address_components:
            if component.strip() in valid_st_gallen_zip_codes:
                return component.strip()
    return None

def get_lat_lon_from_address_or_zip(input_text):
    geolocator = Nominatim(user_agent="http")
    # Add 'St. Gallen' suffix for zip codes to narrow down the search
    if input_text.isdigit() and len(input_text) == 4:
        input_text += ", St. Gallen, Switzerland"

    location = geolocator.geocode(input_text)
    if location:
        return location.latitude, location.longitude
    else:
        return default_lat, default_lon  # Return default coordinates if no location is found


def update_step(new_step):
    st.session_state.current_step = new_step
    st.experimental_rerun()

# Function to process and localize the address input to St. Gallen
def process_address_input(input_address):
    # Define different variations of St. Gallen to check
    st_gallen_variants = ['st. gallen', 'st gallen', 'sankt gallen', 'saint gallen']
    
    # Check if any variant of St. Gallen is already in the address
    if any(variant in input_address.lower() for variant in st_gallen_variants):
        # Address already contains a variant of St. Gallen
        return input_address
    else:
        # Append "St. Gallen" to localize the search
        return input_address + ", St. Gallen"


def process_input(input_text):
    # Check if input is a 4-digit number (zip code)
    if input_text.isdigit() and len(input_text) == 4:
        return input_text + ", St. Gallen, Switzerland"
    else:
        return process_address_input(input_text)  # Process general address

# Initialize session state for current step
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

# Preprocess data and train the model
model, real_estate_data = preprocess_and_train()

st.title("Rental Price Prediction")

steps = ["Location", "Rooms", "Size", "My Current Rent", "Results"]

if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
step_content = st.empty()

def render_step(step, placeholder):
    with placeholder.container():
        if step == 0:
            # Step 1: Location
            address_input = st.text_input("Please enter an address or zip code in St. Gallen:", 
                                  value=st.session_state.get('address', ''), 
                                  key="address_input_step1")

            if address_input:
                processed_input = process_input(address_input)
                st.session_state.address = processed_input
                # Set the extracted_zip_code in session state
                extracted_zip_code = extract_zip_code(processed_input)
                st.session_state.extracted_zip_code = extracted_zip_code

                try:
                    lat, lon = get_lat_lon_from_address_or_zip(processed_input)
                    popup_message = f"Location: {processed_input}"
                except Exception:
                    lat, lon = default_lat, default_lon
                    popup_message = "Error in location retrieval, showing default location."

            else:
                lat, lon = default_lat, default_lon
                popup_message = "Default Location in St. Gallen"

            # Create and display the map
            map = folium.Map(location=[lat, lon], zoom_start=16)
            folium.Marker([lat, lon], popup=popup_message, icon=folium.Icon(color='red')).add_to(map)
            folium_static(map)
        
        
        elif step == 1:
            #step 2 rooms
                # Calculate the index for the select box
            rooms_index = st.session_state.get('rooms', 0)
            rooms_index = rooms_index - 1 if rooms_index > 0 else 0
            rooms_selection = st.selectbox("Select the number of rooms", 
                                        range(1, 7), 
                                        index=rooms_index, 
                                        key='rooms_step2')
            st.session_state.rooms = rooms_selection

            # Step 3: Size
        elif step == 2:
                size_input = st.number_input("Enter the size in square meters", 
                             min_value=0, 
                             value=st.session_state.get('size_m2', 0), 
                             key='size_m2_step3')
                st.session_state.size_m2 = size_input
            # Step 4: Current Rent
        elif step == 3:
                st.session_state.current_rent = st.number_input("Enter your current rent in CHF:", min_value=0, value=st.session_state.get('current_rent', 0), step=10, key = "current_rent_step4")

            # Step 5: Result
       # Step 5: Result
        elif step == 4:  # Results step
            if 'extracted_zip_code' in st.session_state and 'rooms' in st.session_state and 'size_m2' in st.session_state:
                if st.button('Predict Rental Price', key='predict_button'):
                    extracted_zip_code = st.session_state.extracted_zip_code
                    if extracted_zip_code is not None:
                        predicted_price = predict_price(st.session_state.size_m2, extracted_zip_code, st.session_state.rooms, model)
                        if predicted_price is not None:
                            st.session_state.predicted_price = predicted_price  # Speichern des berechneten Preises im session state
                            st.write(f"The predicted price for the apartment is CHF {predicted_price:.2f}")

                            # Ähnliche Immobilien finden und anzeigen
                            similar_properties = find_similar_properties_adjusted(st.session_state.rooms, st.session_state.size_m2, real_estate_data)
                            if not similar_properties.empty:
                                st.markdown("### Ähnliche Immobilien:")
                                col1, col2 = st.columns(2)
                                for index, row in similar_properties.head(6).iterrows():
                                    current_col = col1 if index % 2 == 0 else col2
                                    with current_col:
                                        rooms, size_m2 = extract_rooms_and_size(row.get('Details', ''))
                                        price_per_month = row.get('Price', 'N/A')
                                        area_code = row.get('zip', 'N/A')
                                        st.markdown(
                                            f"**Zimmer:** {rooms if rooms is not None else 'N/A'} \n"
                                            f"**Größe:** {size_m2 if size_m2 is not None else 'N/A'} m² \n"
                                            f"**Preis:** CHF {price_per_month} pro Monat \n"
                                            f"**Adresse:** {area_code}")
                            else:
                                st.write("Keine ähnlichen Immobilien gefunden.")

                            # Plotly-Diagramm
                            current_rent_step4 = st.session_state.current_rent
                            # Anpassung der Gauge-Werte
                            min_gauge_value = 0.9 * predicted_price
                            max_gauge_value = 1.5 * predicted_price
                            one_third_point = min_gauge_value + (1/3) * (max_gauge_value - min_gauge_value)
                            one_four_point_five_range = (1/4.5) * (max_gauge_value - min_gauge_value)

                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number+delta",
                                value = current_rent_step4,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Calculated rental price (red line) and your Rent (blue bar)", 'font': {'size': 21}},
                                delta = {'reference': predicted_price, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                                gauge = {
                                    'axis': {'range': [min_gauge_value, max_gauge_value], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                    'bar': {'color': "darkblue"},
                                    'bgcolor': "white",
                                    'borderwidth': 2,
                                    'bordercolor': "gray",
                                    'steps': [
                                        {'range': [min_gauge_value, one_third_point], 'color': 'green'},
                                        {'range': [one_third_point, one_third_point + one_four_point_five_range], 'color': 'yellow'},
                                        {'range': [one_third_point + one_four_point_five_range, one_third_point + 2 * one_four_point_five_range], 'color': 'orange'},
                                        {'range': [one_third_point + 2 * one_four_point_five_range, max_gauge_value], 'color': 'red'}],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': predicted_price}
                                }))

                            fig.update_layout(paper_bgcolor = "white", font = {'color': "darkblue", 'family': "Arial"})
                            st.plotly_chart(fig)
                        else:
                            st.error("Unable to predict price. Please check your inputs.")
                    else:
                        st.error("Invalid or missing zip code. Please enter a valid address or zip code.")
            else:
                st.error("Please enter all required information in the previous steps.")


# Function to render navigation buttons
def render_navigation_buttons(placeholder):
    col1, col2 = st.columns([1, 1])
  
    with col1:
        if st.session_state.current_step > 0:
            if st.button('Previous'):
                st.session_state.current_step -= 1
                placeholder.empty()  # Clear the previous content
                render_step(st.session_state.current_step, placeholder)
    
    with col2:
        if st.session_state.current_step < len(steps) - 1:
            if st.button('Next'):
                st.session_state.current_step += 1
                placeholder.empty()  # Clear the previous content
                render_step(st.session_state.current_step, placeholder)

# Call the render_step function with the current step and the placeholder
render_step(st.session_state.current_step, step_content)
render_navigation_buttons(step_content)