import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from collections import Counter
from streamlit_option_menu import option_menu
from PIL import Image
import nltk

# Download NLTK stopwords and punkt tokenizer
nltk.download('stopwords')
nltk.download('punkt')
# Setting up page configuration
# Setting up page configuration
icon = Image.open("IHR2.png")
st.set_page_config(page_title="Industrial Human Resource  | By Ponishadevi",
                   page_icon=icon,
                   layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items={'About': """# This dashboard app is created by *Ponishadevi*!"""})

# Creating option menu in the sidebar
with st.sidebar:
    selected = option_menu("Menu", ["Home", "Overview", "Explore"],
                           icons=["house", "graph-up-arrow", "bar-chart-line"],
                           menu_icon="menu-button-wide",
                           default_index=0,
                           styles={"nav-link": {"font-size": "20px", "text-align": "left", "margin": "-2px",
                                               "--hover-color": "#FF5A5F"},
                                   "nav-link-selected": {"background-color": "#FF5A5F"}}
                           )




# Based on the selected option, display the appropriate page
# Function to merge CSV files in a folder
def merge_csv_files(folder_path):
    try:
        # List all CSV files in the folder
        csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

        # Initialize an empty list to store DataFrame objects
        dfs = []

        # Read each CSV file and append its DataFrame to the list
        for file in csv_files:
            file_path = os.path.join(folder_path, file)
            try:
                # Try reading the CSV file with different encodings
                df = pd.read_csv(file_path, encoding='utf-8-sig')  # Try utf-8-sig encoding
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='latin-1')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='ISO-8859-1')
            dfs.append(df)

        # Concatenate all DataFrames into a single DataFrame
        merged_df = pd.concat(dfs, ignore_index=True)
        
        return merged_df
    except Exception as e:
        print("An error occurred:", e)

# Specify the folder path containing CSV files
folder_path = 'C:/Users/hp/Downloads/zen class 1/project/DataSets-20240430T111853Z-001/DataSets'

# Merge CSV files in the folder
merged_df = merge_csv_files(folder_path)

# Print merged DataFrame for debugging
print("Merged DataFrame:", merged_df)

# Print column names for debugging
print("Column names:", merged_df.columns)

# Print number of rows for debugging
print("Number of rows:", len(merged_df))

# Further processing...


#-------------------------------------------------------------------------------------------------------------------------------------------------------#
# Separate state and district names
merged_df[['STATE', 'District']] = merged_df['India/States'].str.split(' - ', expand=True)

# Function to separate state and district names
def separate_state_district(row):
    # Split the string based on the separator '-'
    parts = row.split(' - ')
    
    # If the first part is in uppercase (assumed to be state name), return it
    if parts[0].isupper():
        return parts[0]
    else:
        return None

# Apply the function to create a new column for state names
merged_df['State Name'] = merged_df['District'].apply(separate_state_district)

# Filter out None values and then print unique state names with commas
state_names = merged_df['State Name'].dropna().unique()
print(", ".join(state_names))



# Create a mapping dictionary for state names
state_name_mapping = {
    'ANDHRA PRADESH': 'Andhra Pradesh',
    'ARUNACHAL PRADESH': 'Arunachal Pradesh',
    'ASSAM': 'Assam',
    'BIHAR': 'Bihar',
    'CHHATTISGARH': 'Chhattisgarh',
    'GOA': 'Goa',
    'GUJARAT': 'Gujarat',
    'HARYANA': 'Haryana',
    'HIMACHAL PRADESH': 'Himachal Pradesh',
    'JAMMU AND KASHMIR': 'Jammu & Kashmir',
    'JHARKHAND': 'Jharkhand',
    'KARNATAKA': 'Karnataka',
    'KERALA': 'Kerala',
    'MADHYA PRADESH': 'Madhya Pradesh',
    'MAHARASHTRA': 'Maharashtra',
    'MANIPUR': 'Manipur',
    'MEGHALAYA': 'Meghalaya',
    'MIZORAM': 'Mizoram',
    'NAGALAND': 'Nagaland',
    'ODISHA': 'Orissa',
    'PUNJAB': 'Punjab',
    'RAJASTHAN': 'Rajasthan',
    'SIKKIM': 'Sikkim',
    'TAMIL NADU': 'Tamil Nadu',
    'TELANGANA': 'Telangana',
    'TRIPURA': 'Tripura',
    'UTTAR PRADESH': 'Uttar Pradesh',
    'UTTARAKHAND': 'Uttaranchal',
    'WEST BENGAL': 'West Bengal',
    'ANDAMAN AND NICOBAR ISLANDS': 'Andaman & Nicobar Island',
    'CHANDIGARH': 'Chandigarh',
    'DADRA AND NAGAR HAVELI AND DAMAN AND DIU': 'Dadra & Nagar Haveli & Daman & Diu',
    'LAKSHADWEEP': 'Lakshadweep',
    'NCT OF DELHI': 'Delhi',
    'PUDUCHERRY': 'Puducherry'
}

# Apply the mapping to normalize state names
merged_df['State Name'] = merged_df['State Name'].apply(lambda x: state_name_mapping.get(x, x))

# Check and print normalized state names
print(merged_df['State Name'].unique())


#---------------------------------------------------------------------------------------------------------------------------------------------------#

if selected == "Home":
    # Set the title and image for the home page
    st.title("Industrial Human Resource Geo-Visualization")
    image = Image.open("IHR.png")
    st.image(image, use_column_width=True)

    # Dataset
    st.subheader("About the Dataset:")
    st.write("Our dataset comprises state-wise counts of main and marginal workers across diverse industries, including manufacturing, construction, retail, and more.")

  

    # Introduction
    st.write("Explore the dynamic landscape of India's workforce with our Industrial Human Resource Geo-Visualization project.")
    st.write("Gain insights into employment trends, industry distributions, and economic patterns to drive informed decision-making and policy formulation.")

    # Key Features
    st.subheader("Key Features:")
    st.markdown("""
    - **Data Exploration:** Dive deep into state-wise industrial classification data.
    - **Visualization:** Interactive charts and maps for intuitive data exploration.
    - **Natural Language Processing:** Analyze and categorize core industries using NLP techniques.
    - **Insights and Analysis:** Extract actionable insights to support policy-making and resource management.
    """)

    # About the Project
    st.subheader("About the Project:")
    st.write("Our project aims to:")
    st.markdown("""
    - Update and refine the industrial classification data of main and marginal workers.
    - Provide accurate and relevant information for policy-making and employment planning.
    - Empower stakeholders with actionable insights to foster economic growth and development.
    """)



#-------------------------------------------------------------------------------------------------------------------------------------------------------#


if selected == "Overview":
    
    # Dataset
    st.subheader("Dataset Overview:")
    st.write("Our dataset includes:")
    st.markdown("""
    - State-wise counts of main and marginal workers across various industries.
    - Gender-based distribution of workforce in different sectors.
    - Historical data for trend analysis and forecasting.
    """)

    # Technologies Used
    st.subheader("Technologies Utilized:")
    st.write("We leverage cutting-edge technologies such as:")
    st.markdown("""
    - Python for data processing and analysis.
    - Streamlit for interactive visualization.
    - Plotly and Matplotlib for creating insightful charts.
    - NLTK for Natural Language Processing tasks.
    """)

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X_tfidf = tfidf_vectorizer.fit_transform(merged_df['NIC Name'])

    # KMeans Clustering
    num_clusters = 5  # Adjust the number of clusters as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    merged_df['Cluster'] = kmeans.fit_predict(X_tfidf)

    # Selectbox for choosing the cluster
    selected_cluster = st.selectbox('Select Cluster', range(num_clusters))

    # Filter text data for the selected cluster
    text_for_cluster = merged_df[merged_df['Cluster'] == selected_cluster]['NIC Name']

    # Tokenize and clean text data
    tokens = word_tokenize(' '.join(text_for_cluster))
    tokens = [word.lower() for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Count word frequency
    word_freq = Counter(tokens)
   

    # Generate word cloud for the selected cluster
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(word_freq))

    # Display the word cloud in Streamlit
    st.subheader(f'Word Cloud for Cluster {selected_cluster}')
    st.image(wordcloud.to_array(), caption='Word Cloud', use_column_width=True)
    

        # Streamlit app
    st.title('Cluster Distribution')

    # Visualize the clustering results
    st.subheader('Distribution of Clusters (Pie Chart)')

    # Count the occurrences of each cluster
    cluster_counts = merged_df['Cluster'].value_counts()

    # Convert counts to a pie chart
    fig, ax = plt.subplots()
    ax.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

    
    # Filter options for work type
    work_type_options = ['Main Workers - Total -  Persons', 'Marginal Workers - Total -  Persons']
    selected_work_type = st.selectbox("Select Work Type:", work_type_options)

    # Filter for top 10 NIC Names based on the selected work type
    top_10_nic_names = merged_df.groupby('NIC Name')[selected_work_type].sum().nlargest(10).index
    top_10_merged_df = merged_df[merged_df['NIC Name'].isin(top_10_nic_names)]

    # Plotting the box plot using Seaborn and Matplotlib
    st.subheader(f'Box Plot of {selected_work_type} by Top 10 NIC Name')

   

    # Calculate total values for each NIC Name
    top_10_nic_names_totals = top_10_merged_df.groupby('NIC Name')[selected_work_type].sum().reset_index()

    # Create the treemap
    fig = px.treemap(top_10_nic_names_totals, path=['NIC Name'], values=selected_work_type, title=f'Treemap of {selected_work_type} by Top 10 NIC Name')
    st.plotly_chart(fig)
   # ---------------------------------------------------------------------------------------------------------------------------------------------------#



    
elif selected == "Explore":
    # Tokenize and clean text data
    text = ' '.join(merged_df['NIC Name'])
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    

        # Count word frequency
    word_freq = Counter(tokens)
    top_words = word_freq.most_common(10)

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X_tfidf = tfidf_vectorizer.fit_transform(merged_df['NIC Name'])

    # KMeans Clustering
    num_clusters = 5  # Adjust the number of clusters as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    merged_df['Cluster'] = kmeans.fit_predict(X_tfidf)

    # Streamlit App
    st.title("Industrial Human Resource  Dashboard")

    # Select box for type of worker
    worker_type = st.selectbox('Select Worker Type', ['Main Workers', 'Marginal Workers'])

    # Column mapping
    if worker_type == 'Main Workers':
        column_total = 'Main Workers - Total -  Persons'
        column_rural = 'Main Workers - Rural -  Persons'
        column_urban = 'Main Workers - Urban -  Persons'
    else:
        column_total = 'Marginal Workers - Total -  Persons'
        column_rural = 'Marginal Workers - Rural -  Persons'
        column_urban = 'Marginal Workers - Urban -  Persons'

    # Strip any extra spaces from column names
    merged_df.columns = [col.strip() for col in merged_df.columns]

    # Print DataFrame columns for debugging
    print("DataFrame Columns:", merged_df.columns)

    # Scatter Plot
    fig1 = px.scatter(merged_df, x=column_total, y=column_rural, color='Cluster', title=f'{worker_type} - Total vs Rural')
    st.plotly_chart(fig1)


    fig2 = px.scatter(merged_df, x=column_total, y=column_urban, color='Cluster', title=f'{worker_type} - Total vs Urban')
    st.plotly_chart(fig2)

    # Box Plot for Top 10 NIC Names
    top_10_nic_names = merged_df['NIC Name'].value_counts().head(10).index
    top_10_df = merged_df[merged_df['NIC Name'].isin(top_10_nic_names)]

    fig3 = px.box(top_10_df, x='NIC Name', y=column_total, title=f'{worker_type} by Top 10 NIC Names')
    st.plotly_chart(fig3)

    # Cluster Distribution
    fig4 = px.histogram(merged_df, x='Cluster', title='Cluster Distribution')
    st.plotly_chart(fig4)



    # Count plot for a categorical column
    st.subheader(f"Distribution of {worker_type} by State")

    # Set the color palette
    sns.set_palette("bright")  # You can choose different palettes like "pastel", "deep", "bright", etc.

    # Create the plot
    fig, ax = plt.subplots()
    sns.countplot(x='State Name', data=merged_df, ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)


    # Plot
    st.subheader(f'Relationship between {worker_type} - Rural/Urban - Persons and {worker_type} - Total - Persons')

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(merged_df[f'{worker_type} - Rural -  Persons'], merged_df[f'{worker_type} - Total -  Persons'], label='Rural', alpha=0.5)
    ax.scatter(merged_df[f'{worker_type} - Urban -  Persons'], merged_df[f'{worker_type} - Total -  Persons'], label='Urban', alpha=0.5)
    ax.set_xlabel(f'{worker_type} - Rural - Persons / {worker_type} - Urban - Persons')
    ax.set_ylabel(f'{worker_type} - Total - Persons')
    ax.set_title(f'Relationship between {worker_type} - Rural/Urban - Persons and {worker_type} - Total - Persons')
    ax.legend()
    ax.grid(True)

    # Display the plot
    st.pyplot(fig)


  


    # Create a word cloud using the top words
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(top_words))

    # Display the word cloud in Streamlit
    st.subheader('Word Cloud for Top 10 NIC Names')
    st.image(wordcloud.to_array(), caption='Word Cloud', use_column_width=True)



    merged_df = merged_df.dropna(subset=['State Name'])

    # Fetch GeoJSON data for India's states
    @st.cache_resource
    def fetch_geojson():
        geojson_url = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
        response = requests.get(geojson_url)
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Failed to fetch GeoJSON data")

    # Main Streamlit App
    def main():
        st.title("India Map Visualization")

        # Fetch GeoJSON data
        geojson_data = fetch_geojson()

        # Extract state names from GeoJSON data
        geojson_state_names = set(feature['properties']['NAME_1'] for feature in geojson_data['features'])

        # State names from DataFrame
        dataframe_state_names = set(merged_df['State Name'])

        # Select box for type of worker
        worker_type = st.selectbox('Select Worker Type', ['Main Workers', 'Marginal Workers'], key="worker_type_selectbox")

        # Select box for sex
        sex_type = st.selectbox('Select Sex', ['Males', 'Females'], key="sex_type_selectbox")

        # Select box for area
        area_type = st.selectbox('Select Area', ['Rural', 'Urban'], key="area_type_selectbox")

        # Determine the column based on selected worker type, sex, and area
        column_name = f'{worker_type} - {area_type} - {sex_type}'

        # Plotly Choropleth map
        fig = go.Figure(go.Choroplethmapbox(
            geojson=geojson_data,
            locations=merged_df['State Name'],  # Use the column with state names
            featureidkey="properties.NAME_1",  # Key in geojson to match with DataFrame
            z=merged_df[column_name],  # Use the column for analysis
            colorscale='Viridis',
            zmin=merged_df[column_name].min(),
            zmax=merged_df[column_name].max(),
            marker_opacity=0.7,
            marker_line_width=0,
        ))

        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=3,
            mapbox_center={"lat": 20.5937, "lon": 78.9629},
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            title=f"{worker_type} ({sex_type}, {area_type}) Population Across Indian States",
            title_x=0.5
        )

        # Display the map
        st.plotly_chart(fig)

        # Top NIC Names State-wise
        st.title("Top NIC Names State-wise")
        for state in merged_df['State Name'].unique():
            top_nic_name = merged_df[merged_df['State Name'] == state]['NIC Name'].mode()[0]
            st.write(f"Top NIC Name in {state}: {top_nic_name}")


      

    # Call the main function
    if __name__ == "__main__":
        main()
