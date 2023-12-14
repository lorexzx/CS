"""
This code is part of a Streamlit application designed to estimate fair rental values for properties, 
particularly useful for students assessing the fairness of their current or prospective rental accommodations. 
The app is focused on the St. Gallen area and uses a linear regression model to make its predictions.

Simplified overview of how the application works:

Data Handling: The app processes real estate data, focusing on properties in St. Gallen. This includes gathering details like room count, size, and price.
Model Training: Using this data, a linear regression model is trained to understand the relationship between these property features and their rental prices.
User Interaction: Students can interact with the app by entering details about a property they are interested in or currently renting. This includes specifics like the number of rooms, size, and location.
Fair Price Estimation: The app then uses the trained model to predict a fair rental price for the given property, based on the inputted features.
Visualization and Comparison: Additionally, the app may offer visualizations like maps and compare the predicted fair price with the actual price to help users determine if they are paying a reasonable amount.


In summary, this code uses a linear regression model to provide valuable insights into rental prices in St. Gallen.

Note: The effectiveness of the application depends on the quality of the data used to train the model.
"""

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import folium
from streamlit_folium import folium_static 
from geopy.geocoders import Nominatim
import re

import matplotlib as mp
import plotly.graph_objs as go
from plotly.basedatatypes import BaseFigure

# Standard coordinates for St. Gallen
default_lat, default_lon = 47.424482, 9.376717

def find_similar_properties_adjusted(input_rooms, input_size, data, threshold=5):
    # Filters properties similar to the user's input based on room count and size
    similar_properties = data[
        data.apply(lambda row: is_similar_property(row['Details'], input_rooms, input_size, threshold), axis=1)
    ]
    return similar_properties

def is_similar_property(details, input_rooms, input_size, threshold):
    # Extracts room count and size from property details and compares with user input
    rooms_match = re.search(r'(\d+(\.\d+)?) Zi\.', details)
    area_match = re.search(r'(\d+(\.\d+)?) m²', details)

    if rooms_match and area_match:
        rooms = float(rooms_match.group(1))
        area = float(area_match.group(1))
        return (rooms >= input_rooms - 1 and rooms <= input_rooms + 1) and (area >= input_size - threshold and area <= input_size + threshold)
    return False

def extract_rooms_and_size(details_str):
    # Extracts room count and size from a string using regular expressions
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

# Advances the Streamlit app to the next step
def go_to_next_step():
    st.session_state.current_step += 1

# Returns the Streamlit app to the previous step
def go_to_previous_step():
    st.session_state.current_step -= 1

# Preprocesses the data and trains the Linear Regression model
def preprocess_and_train(): 
    file_path = 'Immobilienliste.xlsx'
    sorted_data = pd.read_excel('Immobilienliste.xlsx')

    sorted_data.drop(columns=['Name', 'Description'], inplace=True)
    coords_path = 'gallen_coord.csv'
    coords_data = pd.read_csv(coords_path)

    # Extracts room count and size details from a string
    def extract_details(detail_str):
        rooms = re.search(r'(\d+(\.\d+)?) Zi\.', detail_str)
        area = re.search(r'(\d+(\.\d+)?) m²', detail_str)
        return float(rooms.group(1)) if rooms else None, float(area.group(1)) if area else None

    sorted_data['rooms'], sorted_data['area'] = zip(*sorted_data['Details'].apply(extract_details))

    # Converts price string to a float, removing non-numeric characters
    def convert_price(price_str):
        price_str = re.sub(r'[^\d.]', '', price_str)
        return float(price_str) if price_str else None

    sorted_data['Price'] = sorted_data['Price'].apply(convert_price)
    sorted_data['area_code'] = sorted_data['zip'].str.extract(r'(\d{4})')

    sorted_data.dropna(inplace=True)
    sorted_data['area_code'] = sorted_data['area_code'].astype(int)
    sorted_data = sorted_data.merge(coords_data[['area_code', 'latitude', 'longitude']], on ='area_code', how='left')
    X = sorted_data[['rooms', 'area', 'latitude', 'longitude']]
    y = sorted_data['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, sorted_data

# Extracts a zip code from the input text
def extract_zip_code(input_text):
    parts = input_text.replace(',', ' ').split()
    for part in parts:
        if part.isdigit() and len(part) == 4:
            return part
    return None

# Predicts the price based on the given features using the trained model
def predict_price(size_m2, extracted_zip_code, rooms, model):

    # Load and preprocess data from Excel file
    sorted_data = pd.read_excel('Immobilienliste.xlsx') 
    sorted_data.drop(columns=['Name', 'Description'], inplace=True)

    # Load coordinate data from CSV file    
    coords_path = 'gallen_coord.csv'
    coords_data = pd.read_csv(coords_path)

    # Function to extract room and area details from a string
    def extract_details(detail_str):
        rooms = re.search(r'(\d+(\.\d+)?) Zi\.', detail_str)
        area = re.search(r'(\d+(\.\d+)?) m²', detail_str)
        return float(rooms.group(1)) if rooms else None, float(area.group(1)) if area else None

    # Apply the extract_details function to the data
    sorted_data['rooms'], sorted_data['area'] = zip(*sorted_data['Details'].apply(extract_details))

    # Function to convert price string to float
    def convert_price(price_str):
        price_str = re.sub(r'[^\d.]', '', price_str)
        return float(price_str) if price_str else None

    # Apply convert_price function and extract area codes
    sorted_data['Price'] = sorted_data['Price'].apply(convert_price)
    sorted_data['area_code'] = sorted_data['zip'].str.extract(r'(\d{4})')

    # Clean and merge data
    sorted_data.dropna(inplace=True)
    sorted_data['area_code'] = sorted_data['area_code'].astype(int)
    sorted_data = sorted_data.merge(coords_data[['area_code', 'latitude', 'longitude']], on ='area_code', how='left')

    # Convert inputs to correct types and check for validity
    try:
        area_code = int(extracted_zip_code)
        size_m2 = float(size_m2)
        rooms = int(rooms)

        # Check if area code is in the data and get corresponding latitude and longitude
        if area_code in sorted_data['area_code'].values:
            area_data = sorted_data[sorted_data['area_code'] == area_code]
            longitude = area_data.iloc[0]['longitude']
            latitude = area_data.iloc[0]['latitude']
        else:
            st.error("Area code not found in mapping.")
            return None
    
    except ValueError as e:
        st.error(f"Invalid input: {e}")
        return None

    # Prepare the input features for the model
    input_features = pd.DataFrame({
        'rooms': [rooms],
        'area': [size_m2],
        #'area_code': [area_code]
        'latitude': [latitude],
        'longitude': [longitude],
        
    })
    # Use the model to predict the price based on input features
    predicted_price = model.predict(input_features)
    return predicted_price[0] # Return the first (and only) prediction from the model

# List of valid St. Gallen zip codes for validation
def extract_zip_from_address(address):
    # List of non-specific inputs that don't provide exact locations
    valid_st_gallen_zip_codes = ['9000', '9001', '9004', '9006', '9007', '9008', '9010', '9011', '9012', '9013', '9014', '9015', '9016', '9020', '9021', '9023', '9024', '9026', '9027', '9028', '9029']
    non_specific_inputs = ['st. gallen', 'st gallen', 'sankt gallen']

    # Handling non-specific inputs by returning default coordinates
    if address.lower().strip() in non_specific_inputs:
        return default_lat, default_lon  

    # Use the input as is if it's a valid specific zip code
    if address.strip() in valid_st_gallen_zip_codes:
        return get_lat_lon_from_address_or_zip(address.strip())

    # For other inputs, try to geocode the address to get coordinates
    geolocator = Nominatim(user_agent="http")
    location = geolocator.geocode(address + ", St. Gallen", country_codes='CH')
    if location:
        return location.latitude, location.longitude
    else:
        st.error("Invalid or missing zip code. Please enter a valid address or zip code in St. Gallen.")
        return None, None  # Handling cases where geocoding fails

# Geolocate a given text input to get latitude and longitude
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

 # Update the current step in the Streamlit session state and rerun the app
def update_step(new_step):
    st.session_state.current_step = new_step
    st.experimental_rerun()

# Function to process and localize the address input to St. Gallen
def process_address_input(input_address):
    # Variants of St. Gallen to check in the address
    st_gallen_variants = ['st. gallen', 'st gallen', 'sankt gallen', 'saint gallen']
    
    # Checks if the input address already contains a variant of St. Gallen
    if any(variant in input_address.lower() for variant in st_gallen_variants):
        # Address already contains a variant of St. Gallen
        return input_address
    else:
        # Append "St. Gallen" to localize the search
        return input_address + ", St. Gallen"

# Process the input text based on whether it's a zip code or a general address
def process_input(input_text):
    if input_text.isdigit() and len(input_text) == 4:
        return input_text + ", St. Gallen, Switzerland"# Process as zip code
    else:
        return process_address_input(input_text)  # Process as ageneral address

# Initialize session state for current step
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

# Load data and train the model
model, real_estate_data = preprocess_and_train()

# Streamlit UI setup
st.title("Rental Price Prediction")

# Steps in the UI for navigating through the application
steps = ["Location", "Rooms", "Size", "My Current Rent", "Results"]

# Ensure the current step is set in the session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
step_content = st.empty()

# Embedding the CSS style for the property details display
def display_property_details(row):
    frame_style = """
    <style>
    .frame {
        border: 2px solid #f0f0f0;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
    """

    
    st.markdown(frame_style, unsafe_allow_html=True)

   
    with st.container():
        st.markdown('<div class="frame">', unsafe_allow_html=True)

         # Extracting room and size details from the property data
        rooms, size_m2 = extract_rooms_and_size(row.get('Details', ''))
        price_per_month = row.get('Price', 'N/A')
        area_code = row.get('zip', 'N/A')
        websites = row.get('websites', '')  # Neues Feld für Website hinzufügen

        # Displaying the extracted property details in English
        st.write(f"**Number of rooms:** {rooms if rooms is not None else 'N/A'}")
        st.write(f"**Size:** {size_m2 if size_m2 is not None else 'N/A'} m²")
        st.write(f"**Price:** CHF {price_per_month} per month")
        st.write(f"**Location:** {area_code}")

        # Displaying the website if available
        if websites:
            st.markdown(f"**Websites:** [ {websites} ](https://{websites})", unsafe_allow_html=True)
        else:
            st.write("**Websites:** N/A")

        st.markdown('</div>', unsafe_allow_html=True)

# Function to render different steps in the Streamlit app
def render_step(step, placeholder):
    with placeholder.container():
        if step == 0:
            # Step 1: Location
            address_input = st.text_input("Please enter an address AND your zip code in St. Gallen:", 
                                  value=st.session_state.get('address', ''), 
                                  key="address_input_step1")
            # Process and store the entered address
            if address_input:
                processed_input = process_input(address_input)
                st.session_state.address = processed_input
                # Extract and store the zip code from the address
                extracted_zip_code = extract_zip_code(processed_input)
                st.session_state.extracted_zip_code = extracted_zip_code

                # Geocode the address and display on map
                try:
                    lat, lon = get_lat_lon_from_address_or_zip(processed_input)
                    popup_message = f"Location: {processed_input}"
                except Exception:
                    lat, lon = default_lat, default_lon
                    popup_message = "Error in location retrieval, showing default location."

            else:
                lat, lon = default_lat, default_lon
                popup_message = "Default Location in St. Gallen"

            # Display the map with the location marker
            map = folium.Map(location=[lat, lon], zoom_start=16)
            folium.Marker([lat, lon], popup=popup_message, icon=folium.Icon(color='red')).add_to(map)
            folium_static(map)
        
        # Step 2: Room Selection
        elif step == 1:
    
            rooms_index = st.session_state.get('rooms', 0)
             # Adjust the range of selectable room options
            adjusted_rooms_list = [float(f"{i/2:.1f}") for i in range(2, 15)]  # Creates a list [1, 1.5, 2, ..., 7]
            rooms_index = adjusted_rooms_list.index(rooms_index) if rooms_index in adjusted_rooms_list else 0

            rooms_selection = st.selectbox("Select the number of rooms", 
                                        adjusted_rooms_list, 
                                        index=rooms_index, 
                                        key='rooms_step2')
            st.session_state.rooms = rooms_selection
            
        # Step 3: Size Input
        elif step == 2:
                size_input = st.number_input("Enter the size in square meters", 
                             min_value=0, 
                             value=st.session_state.get('size_m2', 0), 
                             key='size_m2_step3')
                st.session_state.size_m2 = size_input
        # Step 4: Current Rent Input
        elif step == 3:
                st.session_state.current_rent = st.number_input("Enter your current rent in CHF:", min_value=0, value=st.session_state.get('current_rent', 0), step=10, key = "current_rent_step4")

        # Step 5: Result Display
        elif step == 4: 
            if 'extracted_zip_code' in st.session_state and 'rooms' in st.session_state and 'size_m2' in st.session_state:
                if st.button('Predict Rental Price', key='predict_button'):
                    extracted_zip_code = st.session_state.extracted_zip_code
                    if extracted_zip_code is not None:
                        #predicted_price = predict_price(st.session_state.size_m2, extracted_zip_code, st.session_state.rooms, model) krish
                        predicted_price = predict_price(st.session_state.size_m2, extracted_zip_code, st.session_state.rooms, model)
                        if predicted_price is not None:
                            st.session_state.predicted_price = predicted_price  # Speichern des berechneten Preises im session state
                            st.markdown(f"**The predicted price for the apartment is CHF {predicted_price:.2f}**", unsafe_allow_html=True)

                            # Display user input for confirmation
                            st.markdown(f"### Your Input:")
                            st.write(f"**Address or zipcode:** {st.session_state.address}")
                            st.write(f"**Rooms:** {st.session_state.rooms}")
                            st.write(f"**Size:** {st.session_state.size_m2} m²")
                            st.write(f"**Current rent:** CHF {st.session_state.current_rent}")

                           # Visualization with Plotly gauge chart
                            current_rent_step4 = st.session_state.current_rent
                            # Gauge value adjustments
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

                            fig.update_layout(paper_bgcolor = "white", font = {'color': "black", 'family': "Arial"})
                            st.plotly_chart(fig)

                            # Find and display similar properties
                            similar_properties = find_similar_properties_adjusted(st.session_state.rooms, st.session_state.size_m2, real_estate_data)
                            if not similar_properties.empty:
                                st.markdown("### Find similar properties:")
                                # Iterate over each property and display it in two columns
                                for index in range(0, len(similar_properties), 2):
                                    col1, col2 = st.columns(2)

                                    # Show property in the first column
                                    if index < len(similar_properties):
                                        row = similar_properties.iloc[index]
                                        with col1:
                                            display_property_details(row)

                                    # Show property in the second column, if available
                                    if index + 1 < len(similar_properties):
                                        row = similar_properties.iloc[index + 1]
                                        with col2:
                                            display_property_details(row)
                            else:
                                st.write("No similar properties found.")

                        else:
                            st.error("Unable to predict price. Please check your inputs.")
                    else:
                        st.error("Invalid or missing zip code. Please enter a valid address or zip code.")
            else:
                st.error("Please enter all required information in the previous steps.")


# Render navigation buttons
def render_navigation_buttons(placeholder):
    col1, col2 = st.columns([1, 1])
    
    # Previous step button
    with col1:
        if st.session_state.current_step > 0:
            if st.button('Previous'):
                st.session_state.current_step -= 1
                placeholder.empty()  # Clear the previous content
                render_step(st.session_state.current_step, placeholder)

    # Next step button
    with col2:
        if st.session_state.current_step < len(steps) - 1:
            if st.button('Next'):
                st.session_state.current_step += 1
                placeholder.empty()  # Clear the previous content
                render_step(st.session_state.current_step, placeholder)

# Calls the render_step function with the current step and the placeholder
render_step(st.session_state.current_step, step_content)
render_navigation_buttons(step_content)

