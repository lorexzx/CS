import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim

# Standard coordinates for St. Gallen
default_lat, default_lon = 47.424482, 9.376717

# Initialize session state variables
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'address' not in st.session_state:
    st.session_state.address = ""

# Define functions to handle navigation
def go_to_next_step():
    st.session_state.current_step += 1

def go_to_previous_step():
    st.session_state.current_step -= 1

# Backend Code: Data Preprocessing and Model Training
def preprocess_and_train():
    # Load the dataset (replace with your actual file path)
    real_estate_data = pd.read_excel('real-estate-scraped-data.xlsx')

    # Data Preprocessing
    # Define the function to split 'Col3'
    def split_column(row):
        parts = row.split(' • ')
        property_type = parts[0]
        rooms = parts[1] if len(parts) > 1 else None
        size_m2 = parts[2] if len(parts) > 2 else None
        return {'Property_Type': property_type, 'Rooms': rooms, 'Size_m2': size_m2}

    # Apply the function to each row in 'Col3' and create a new DataFrame
    split_data = real_estate_data['Col3'].apply(split_column).apply(pd.Series)

    # Cleaning and renaming columns
    real_estate_data['area_code'] = real_estate_data['Col4'].str.extract(r'\b(\d{4})\b')

    # Extracting numeric values from 'Col5' and 'Col6'
    real_estate_data['price_per_month'] = real_estate_data['Col5'].str.extract(r'(\d+[\’\']?\d*)')[0].str.replace("’", "").str.replace("'", "").str.strip()
    real_estate_data['price_per_m2_per_year'] = real_estate_data['Col6'].str.extract(r'(\d+[\’\']?\d*)')[0].str.replace("’", "").str.replace("'", "").str.strip()

    # Remove 'Zi.' from 'Rooms' and 'm²' from 'Size_m2', with checks for non-string data
    split_data['Rooms'] = split_data['Rooms'].str.replace(' Zi.', '').str.strip() if split_data['Rooms'].dtype == "object" else split_data['Rooms']
    split_data['Size_m2'] = split_data['Size_m2'].str.replace(' m²', '').str.strip() if split_data['Size_m2'].dtype == "object" else split_data['Size_m2']

    # Concatenate the new DataFrame with the original one, now including cleaned columns
    real_estate_data = pd.concat([split_data, real_estate_data.drop(columns=['Col3', 'Col4', 'Col5', 'Col6'])], axis=1)

    # Rearrange columns
    new_columns = ['Property_Type', 'Rooms', 'Size_m2', 'area_code', 'price_per_month', 'price_per_m2_per_year']
    real_estate_data = real_estate_data[new_columns]

    real_estate_data.dropna(inplace=True)

    # Convert columns to numeric as necessary
    real_estate_data['Rooms'] = pd.to_numeric(real_estate_data['Rooms'], errors='coerce')
    real_estate_data['Size_m2'] = pd.to_numeric(real_estate_data['Size_m2'], errors='coerce')
    real_estate_data['area_code'] = pd.to_numeric(real_estate_data['area_code'], errors='coerce')
    real_estate_data['price_per_month'] = pd.to_numeric(real_estate_data['price_per_month'], errors='coerce')

    # Drop any rows with NaN values
    real_estate_data.dropna(inplace=True)

    # Selecting features and target for the model
    X = real_estate_data[['Rooms', 'Size_m2', 'area_code']]  # Example features
    y = real_estate_data['price_per_month']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

def extract_zip_code(input_text):
    # Zerlegen des Strings anhand von Kommata oder Leerzeichen
    parts = input_text.replace(',', ' ').split()

    # Durchsuchen der Teile nach einer Zahlenfolge
    for part in parts:
        if part.isdigit() and len(part) == 4:  # Schweizer Postleitzahlen haben 4 Ziffern
            return part
    return None  # Keine gültige Postleitzahl gefunden

#NEWER VERSION PRICE PREDICT NOT MY AEREA SO NOT SURE
def predict_price(size_m2, extracted_zip_code, rooms, model):
    # Ensure that size_m2 and rooms are numeric and non-negative
    try:
        size_m2 = float(size_m2)
        rooms = int(rooms)
        if size_m2 < 0 or rooms < 0:
            raise ValueError("Size and rooms must be non-negative.")
    except ValueError as e:
        st.error(f"Invalid input for size or rooms: {e}")
        return None

    # Ensure that extracted_zip_code is a valid Swiss zip code (4 digits)
    try:
        area_code = int(extracted_zip_code)
        if not (1000 <= area_code <= 9999):
            raise ValueError("Invalid Swiss zip code.")
    except ValueError as e:
        st.error(f"Invalid zip code: {e}")
        return None

    # Create the input features DataFrame
    input_features = pd.DataFrame({
        'Rooms': [rooms],
        'Size_m2': [size_m2],
        'area_code': [area_code]
    })

    # Predict the price using the model
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
model = preprocess_and_train()

# Streamlit UI
st.title("Rental Price Prediction")

# Define the steps
steps = ["Location", "Rooms", "Size", "My Current Rent", "Results"]

# Initialize session state for current step and create a placeholder for content
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
step_content = st.empty()

# Function to render the content of each step
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
        elif step == 4:
                    if all(key in st.session_state for key in ['extracted_zip_code', 'rooms', 'size_m2']):
                        # Use st.session_state variables for prediction
                        if st.button('Predict Rental Price', key='predict_button'):
                            predicted_price = predict_price(st.session_state.size_m2, st.session_state.extracted_zip_code, st.session_state.rooms, model)
                            if predicted_price is not None:
                                st.write(f"The predicted price for the apartment is CHF {predicted_price:.2f}")
                            else:
                                st.error("Unable to predict price. Please check your inputs.")
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

# Render the navigation buttons, pass the placeholder as well
render_navigation_buttons(step_content)