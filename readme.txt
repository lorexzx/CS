This is short explanation of how to get our code to run.
The App is live on streamlit under the public link: https://csgruppe6-1.streamlit.app/ .
The webscraper that collects the real estate data gets executed automatically at 00:00 UTC+1 every day and updates the "Immobilienliste.xlsx" database in our repository. This is achieved via a GitHub actions workflow. This ensures that the data is always up to date and that the applications works completely standalone and requires no more manual inputs to stay up and running.
