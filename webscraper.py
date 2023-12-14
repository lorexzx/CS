"""
Web Scraper for Real Estate Data from RealAdvisor.ch

This script is a web scraper designed to extract real estate listings from the website realadvisor.ch. 
It specifically targets properties in the St. Gallen area. The scraper navigates through the site, 
collects vital information such as broker names, descriptions, locations, prices, price per square meter, zip codes, 
and website links. 

The collected data is cleaned and structured using custom functions. This structured data is then compiled 
into a pandas DataFrame. Each page's data from the website is appended to a collective DataFrame, 
which is finally saved as an Excel file. This file serves as a dataset for further analysis, such as 
building a linear regression model to understand and predict real estate price trends.

Usage:
- The script is currently set to scrape a limited number of pages (adjustable in the for loop).
- The required libraries (pandas, BeautifulSoup, requests) should be installed before running the script.
- The final dataset is saved as an Excel file on the specified path.

We used a Youtube tutorial to help us build the web scraper: https://www.youtube.com/watch?v=XVv6mJpFOb0
"""

import re
import pandas as pd
from bs4 import BeautifulSoup
import requests
from datetime import datetime

def clean_price(price): 
    # Replace non-breaking spaces and remove hyphens from the price
    return price.replace('\xa0', ' ').replace('-', '').strip()

def clean_pricem2(pricem2):
    # Remove non-breaking spaces and anything after / m² in the price per square meter
    pricem2 = pricem2.replace('\xa0', ' ')
    return re.split(r' / m²', pricem2)[0].strip()

# URL the scraper accessess when its running and 
base_url = 'https://realadvisor.ch/de/mieten/stadt-st-gallen/haus-wohnung'
params = '?east=9.651524100294182&north=47.78651906423726&south=47.05433631754212&west=9.091221365919182'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

all_data = []

# Loop for the URL for the individual 18 pages
for page in range(1, 19): 
    if page == 1:
        url = base_url + params  
    else:
        url = f'{base_url}/seite-{page}{params}'  

    # Requests the page
    response = requests.get(url, headers=headers)
    
    # We resend the second page, this allows us to bypass the cookies window and keep scraping
    if page == 2:
        response = requests.get(url, headers=headers)

    # If the response is successful, parse the page
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'lxml')
        
        # The individual data that gets scraped and their element
        names_brooker_elements = soup.find_all('div', class_='css-1mtsvd6-AggregatesListingCard')
        names_brooker = [element.text.strip() for element in names_brooker_elements]

        zip_elements = soup.find_all('div', class_='css-1wo7mki-AggregatesListingCard')
        zips = [element.text.strip() for element in zip_elements]

        description_elements = soup.find_all('div', class_='css-qb670f-AggregatesListingCard')
        descriptions = [element.text.strip() for element in description_elements]

        description2_elements = soup.find_all('div', class_='css-1lelbas-AggregatesListingCard')
        descriptions2 = [element.text.strip() for element in description2_elements]

        location_elements = soup.find_all('div', class_='css-1lelbas-AggregatesListingCard')
        locations = [element.text.strip() for element in location_elements]

        price_elements = soup.find_all('span', class_='css-1r801wc')
        prices = [clean_price(element.text) for element in price_elements]

        pricem2_elements = soup.find_all('div', class_='css-1eo6i6u-AggregatesListingCard')
        pricem2 = [clean_pricem2(element.text) for element in pricem2_elements]

        website_elements = soup.find_all('div', class_='css-vc4s6w-AggregatesListingCard')
        websites = [element.text.strip() for element in website_elements]

        # Dictionary for the page data
        page_data = {
            'Name': names_brooker,
            'Description': descriptions,
            'Details': descriptions2,
            'Place': locations,
            'Price': prices,
            'Price_per_m2_per_year': pricem2,
            'zip': zips,
            'Websites': websites 
        }

        # Assigning the data to page_data
        all_data.append(pd.DataFrame(page_data))

    else:
        print(f"Error loading page {page}")

# Merging the data into one DataFrame
df = pd.concat(all_data, ignore_index=True)

#Saving the data file
file_path = 'Immobilienliste.xlsx'  
df.to_excel(file_path, index=False)