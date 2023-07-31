import streamlit as st
import pandas as pd

# Set up the Streamlit layout
st.set_page_config(layout="wide")



# Define the function for the homepage
def homepage():
    st.title("Welcome to the Spazil Urban Mobility!")
    
    # Heading
    description = """
    At SPAZIL Consultancy we're changing the way cities move.
    """
    # Add the heading to the page
    st.markdown(description)

    # Adding the image
    from PIL import Image
    image = Image.open("bicycle-3.png")
    st.image(image, width=300)
    
    # Subheading
    second_description = """
    You have landed on SPAZIL's platform for Urban Mobility. Here, you can find visualizations that will help you optimize your fleet, and make Washington D.C more sustainable.
    """
    #Add subheading
    st.markdown(second_description)
    
    # Context
    third_description = """
    SPAZIL’s products use sophisticated mathematical models to estimate demand and predict fleet dynamics.

     - Visualize data on customers' trends and behaviors 
     - Forecast demand for our clients
     - Understand the impact of weather on our clients
    """
    #Add context
    st.markdown(third_description)
    
    # Context 2
    fourth_description = """
    At SPAZIL, our Machine Learning engineers and coding enthusiasts use 🐱 cat-boosting techniques 🐱 to convert data into meaningful, interpretable insights for our clients. For this fleet optimization project, we worked on a dataset from 2011 to 2012 that showed us the bike usage for every day, every week, every month, and every season. Even all the temperatures. 

Our engineers used this data to predict the future of ridership so it can help you make better decisions about the provision of bikes in the city.
    """
    #Add context 2
    st.markdown(fourth_description)
    
    
    #Instructions for the usage 
    fifth_description = """
    On your left, you will find a drop-down list with the following options:
    
     - EDA: where you will find some graphs useful to draw interesting insights on the historical data provided 
     - Predictions: here the output prediction of our Machine Learning model will be presented, the result refers to the last week of December 2012, treated in our case as unseen data
    """
    #Add instructions
    st.markdown(fifth_description)

    
    
# Define the function for the EDA page
def eda():
    import streamlit as st 
    st.title("Exploratory Data Analysis")
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.express as px
    import numpy as np 
    import seaborn as sns
    from PIL import Image
    
    # Here import an image 
    import altair as alt
    import streamlit as st
    data = pd.read_csv("bike-sharing_hourly.csv")
    data['dteday'] = pd.to_datetime(data['dteday'])
    

    # Group data by date and sum the bike counts
    daily_counts = data.groupby('dteday')['cnt'].sum().reset_index()

    
    # BIKE COUNT OVER TIME 
    chart = alt.Chart(daily_counts).mark_line().encode(
        x='dteday:T',
        y='cnt:Q',
        tooltip=['dteday:T', 'cnt:Q']
    ).properties(
        width=600,
        height=400,
        title='Bike Count Over Time'
    )

    # Set axis titles
    chart = chart.configure_axis(
        titleFontSize=14,
        labelFontSize=12
    ).configure_title(
        fontSize=16
    ).encode(
        x=alt.X('dteday:T', axis=alt.Axis(title='Months [2011 - 2012]')),
        y=alt.Y('cnt:Q', axis=alt.Axis(title='Bike Count'))
    )

    # Display the plot in Streamlit
    st.altair_chart(chart, use_container_width=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    
    # COUNT OF BIKES BY MONTH
    # Create plotly express chart
    fig = px.bar(
        data_frame=data,
        x="mnth",
        y="cnt",
        labels={
            "mnth": "Month",
            "cnt": "Bike Count"
        },
        title="Bike Count by Month"
    )

    st.plotly_chart(fig)

    # BIKE COUNTS BY WEATHER CONDITION

    # Define labels for weather conditions
    labels = {
        1: "Clear",
        2: "Cloudy",
        3: "Light Rain",
        4: "Heavy Rain"
    }

    # Replace weathersit values with labels
    data["weathersit"] = data["weathersit"].replace(labels)

    # Compute mean bike counts by weather condition
    weather_counts = data.groupby("weathersit")["cnt"].mean().reset_index()

    # Create Altair chart
    chart = alt.Chart(weather_counts).mark_bar().encode(
        x=alt.X("weathersit:N", axis=alt.Axis(title="Weather Condition")),
        y=alt.Y("cnt:Q", axis=alt.Axis(title="Average Bike Count")),
        tooltip=["weathersit:N", "cnt:Q"]
    ).properties(
        width=600,
        height=400,
        title="Average bike count by Weather Condition"
    )

    st.altair_chart(chart, use_container_width=True)
    
    
    # BIKE COUNTS BY SEASON

    # Define labels for seasons
    season_labels = {
        1: "Spring",
        2: "Summer",
        3: "Fall",
        4: "Winter"
    }

    # Replace season values with labels
    data["season"] = data["season"].replace(season_labels)

    # Compute mean bike counts by season
    season_counts = data.groupby("season")["cnt"].mean().reset_index()

    # Create Altair chart
    chart = alt.Chart(season_counts).mark_bar().encode(
        x=alt.X("season:N", axis=alt.Axis(title="Season")),
        y=alt.Y("cnt:Q", axis=alt.Axis(title="Average Bike Count")),
        tooltip=["season:N", "cnt:Q"]
    ).properties(
        width=600,
        height=400,
        title="Average bike count by Season"
    )

    st.altair_chart(chart, use_container_width=True)
    
    
    # BIKE COUNTS BY HOUR AND TIME OF DAY

    # Compute mean bike counts by time of day
    time_counts = data.groupby("hr")["cnt"].mean().reset_index()

    # Create Altair chart
    chart = alt.Chart(time_counts).mark_bar().encode(
        x=alt.X("hr:N", axis=alt.Axis(title="Hour")),
        y=alt.Y("cnt:Q", axis=alt.Axis(title="Average Bike Count")),
        tooltip=["hr:N", "cnt:Q"]
    ).properties(
        width=600,
        height=400,
        title="Average bike count by Hour"
    )

    st.altair_chart(chart, use_container_width=True)

    # Compute mean bike counts by time of day
    data["day_time"] = pd.cut(data["hr"], bins=[0, 6, 12, 18, 24], labels=["Night", "Morning", "Afternoon", "Evening"])
    time_counts = data.groupby("day_time")["cnt"].mean().reset_index()

    # Create Altair chart
    chart = alt.Chart(time_counts).mark_bar().encode(
        x=alt.X("day_time:N", axis=alt.Axis(title="Time of Day")),
        y=alt.Y("cnt:Q", axis=alt.Axis(title="Average Bike Count")),
        tooltip=["day_time:N", "cnt:Q"]
    ).properties(
        width=600,
        height=400,
        title="Average bike count by Time of Day"
    )

    st.altair_chart(chart, use_container_width=True)

    

# Define the function for the Predictions page
def predictions():
    st.title("Bike Rental Prediction Results")
    # Load the "values" DataFrame
    values = pd.read_csv("24_31_Predictions.csv")

    # Define some CSS to set the background to white and text to black
    st.markdown(
        """
        <style>
        body {
            background-color: #FFFFFF;
            color: #000000;
        }

        .st-ba {
            margin: 0 auto;
            max-width: 800px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Define the description for the page
    description = """
    This page displays the predicted bike rental count for unseen data, referring to the last week of the year (24/12 to 31/12), by hour, using the CatBoostRegressor algorithm. The model was trained on a dataset that was split into training and testing sets, and evaluated on different models using various metrics such as root mean squared error (RMSE), mean squared error (MSE), mean absolute error (MAE), and R2 score. A version of CatBoostRegressor optimized through a grid search offers us the best performance, with an R-squared of ~0.97 on our test set.
    """

    # Add the description to the page
    st.markdown(description)

    #Define instructions
    second_description = """
    By modifying the following filters, the data will be filtered and displayed accordingly. Keep in mind that a single value can be specified for the mutually exclusive filters related to Cold, Mild and Hot temperature and for the wind-related ones (i.e. if Calm Wind is set to 1.0, Moderate wind must be set to 0.0).
    Pay attention also to the line chart presented below the table.
    """

    #Add instructions
    st.markdown(second_description)

    # Add some vertical spacing
    st.markdown("<br/>", unsafe_allow_html=True)

    # Create a list of unique values in the "workingday" column
    workingdays = values["Working Day"].unique().tolist()
    # Add a dropdown to filter by workingday
    selected_workingday = st.selectbox("Select Working Day [1 = working day, 0 = weekend]", workingdays)
    filtered_values = values[values["Working Day"] == selected_workingday]

    # Create a list of unique values in the "temp_cat_cold" column
    temp_cat_colds = values["Cold Temperature"].unique().tolist()
    # Add a dropdown to filter by temp_cat_cold
    selected_temp_cat_cold = st.selectbox("Select Cold Temperature [1 = yes, 0 = no]", temp_cat_colds)
    filtered_values = filtered_values[filtered_values["Cold Temperature"] == selected_temp_cat_cold]

    # Create a list of unique values in the "temp_cat_mild" column
    temp_cat_milds = values["Mild Temperature"].unique().tolist()
    # Add a dropdown to filter by temp_cat_mild
    selected_temp_cat_mild = st.selectbox("Select Mild Temperature [1 = yes, 0 = no]", temp_cat_milds)
    filtered_values = filtered_values[filtered_values["Mild Temperature"] == selected_temp_cat_mild]

    # Create a list of unique values in the "temp_cat_hot" column
    temp_cat_hots = values["Hot Temperature"].unique().tolist()
    # Add a dropdown to filter by temp_cat_hot
    selected_temp_cat_hot = st.selectbox("Select Hot Temperature [1 = yes, 0 = no]", temp_cat_hots)
    filtered_values = filtered_values[filtered_values["Hot Temperature"] == selected_temp_cat_hot]

    # Create a list of unique values in the "hum_cat_high" column
    windspeed_cat_calms = values["Calm wind"].unique().tolist()
    # Add a dropdown to filter by hum_cat_high
    selected_windspeed_cat_calm = st.selectbox("Select Calm Wind [1 = yes, 0 = no]", windspeed_cat_calms)
    filtered_values = filtered_values[filtered_values["Calm wind"] == selected_windspeed_cat_calm]

    # Create a list of unique values in the "hum_cat_high" column
    windspeed_cat_moderates = values["Moderate wind"].unique().tolist()
    # Add a dropdown to filter by hum_cat_high
    selected_windspeed_cat_moderate = st.selectbox("Select Moderate Wind [1 = yes, 0 = no]", windspeed_cat_moderates)
    filtered_values = filtered_values[filtered_values["Moderate wind"] == selected_windspeed_cat_moderate]

    # Show the filtered dataframe
    st.markdown("<h2 style='text-align: center;'>Filtered Data</h2>", unsafe_allow_html=True)
    centered_data = f"<div style='display: flex; justify-content: center;'>{filtered_values.to_html(index=False)}</div>"

    st.markdown(
        """
       <style>
        div.stDataFrame {
            height: 10px;
            overflow-y: scroll;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(centered_data, unsafe_allow_html=True)

    # Add some vertical spacing
    st.markdown("<br/><br/>", unsafe_allow_html=True)

    # Add a header to the page
    st.title("Predictions by Day and Hour")

    #Define instructions
    third_description = """
    The following line chart offers an overview of the Predictions by hour of the day, given the data filtered above. 
    A drop-down list under the graph offers the opportunity to select the days, given the filters set above.
    Keep in mind that some values will appear as truncated, because the fact that the metereological conditions can change during the day.
    """

    #Add instructions
    st.markdown(third_description)

    # Create a line chart showing the prediction over time
    import altair as alt

    # Define chart
    base = alt.Chart(filtered_values).mark_line().encode(
        x='Hour:T',
        y='Prediction:Q',
        color='Day:N'
    ).properties(width=700)

    # Define day selection dropdown
    day_dropdown = alt.binding_select(options=sorted(filtered_values['Day'].unique().tolist()))
    day_select = alt.selection_single(fields=['Day'], bind=day_dropdown, name='Select')

    # Add selection filter to chart
    filtered_chart = base.add_selection(
        day_select
    ).transform_filter(
        day_select
    ).properties(title='Prediction by Hour and Day')

    # Show chart
    st.altair_chart(filtered_chart, use_container_width=True)

    

# Define the sidebar options
options = ["Home", "EDA", "Predictions"]

# Define the function to display the selected page based on the option chosen in the sidebar
def display_page(option):
    if option == "Home":
        homepage()
    elif option == "EDA":
        eda()
    elif option == "Predictions":
        predictions()

# Add a sidebar to select the page to display
st.sidebar.title("Select a Page")
selected_page = st.sidebar.selectbox("", options)
  
# Call the display_page function with the selected option
display_page(selected_page)
