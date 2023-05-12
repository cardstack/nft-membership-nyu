import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pyecharts import options as opts
from pyecharts.charts import Pie
from PIL import Image
from pyecharts import options as opts
from pyecharts.charts import Radar
from pyecharts.render import make_snapshot
from matplotlib.gridspec import GridSpec
from pathlib import Path
AWS_ACCESS_KEY_ID = "<access_key>"
AWS_SECRET_ACCESS_KEY = "<secret_key>"

s3_path = "s3a://nft-membership-nyu-dune/Bins"
parquet_files = fs.glob(f"s3a://nft-membership-nyu-dune/Bins/*.parquet")

import s3fs
import fastparquet as fp


def Ingestion():
    # data_dir3 = Path("/Users/abhishek/Documents/Bins")
    # full_df3 = pd.concat(
    #     pd.read_parquet(parquet_file)
    #     for parquet_file in data_dir3.glob('*.parquet')
    # )
    s3 = s3fs.S3FileSystem()
    # fs = s3fs.core.S3FileSystem()
    fs = s3fs.core.S3FileSystem(anon=False, key=AWS_ACCESS_KEY_ID, secret=AWS_SECRET_ACCESS_KEY)

    all_paths_from_s3 = fs.glob(path=s3_path)

    myopen = s3.open
    #use s3fs as the filesystem
    fp_obj = fp.ParquetFile(all_paths_from_s3,open_with=myopen)
    #convert to pandas dataframe
    full_df3 = fp_obj.to_pandas()
    return full_df3

def Tiering(full_df3, search_text):
    st.write("<h1 style='text-align: center; font-size: 48px;'>Membership Status</h1>", unsafe_allow_html=True)
    filtered_df = full_df3[full_df3['buyer'].str.contains(search_text)]

    font1_size = 0
    font2_size = 0
    font3_size = 0
    tier1_size = 0.2
    tier2_size = 0.2
    tier3_size = 0.2

    if filtered_df.iloc[0]['tier'] == 'tier1':
        tier1_size = 0.4
        font1_size = 8
    #        st.metric(label="Membership Status", value="Bronze")
    elif filtered_df.iloc[0]['tier'] == 'tier2':
        tier2_size = 0.4
        font2_size = 8
    #        st.metric(label="Membership Status", value="Silver")
    elif filtered_df.iloc[0]['tier'] == 'tier3':
        tier3_size = 0.4
        font3_size = 8

    # # Define circle properties
    # circle_radius = 0.5
    # circle_color = 'none'
    # circle_edge_color = 'black'
    # circle_edge_width = 5
    # text = 'Hello World'
    # text_color = 'black'
    # text_size = 30

    fig = plt.figure(figsize=(8, 2))
    gs = GridSpec(2, 4, figure=fig, wspace=0.0, hspace=0.0)

    # Circle 1
    ax3 = fig.add_subplot(gs[1, 0])
    circle3 = plt.Circle((0.5, 0.5), tier1_size, color=(0.804, 0.498, 0.196))
    ax3.add_artist(circle3)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.text(0.5, -0.2, 'Bronze', ha='center', va='center', fontsize=10 + font1_size)

    # Circle 2
    ax2 = fig.add_subplot(gs[1, 1])
    circle2 = plt.Circle((0.5, 0.5), tier2_size, color=(0.753, 0.753, 0.753))  # Gold
    ax2.add_artist(circle2)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.text(0.5, -0.2, 'Silver', ha='center', va='center', fontsize=10 + font2_size)

    # Circle 3
    ax1 = fig.add_subplot(gs[1, 2])
    circle1 = plt.Circle((0.5, 0.5), tier3_size, color=(1.0, 0.843, 0.0))  # Platinum
    ax1.add_artist(circle1)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.text(0.5, -0.2, 'Gold', ha='center', va='center', fontsize=10 + font3_size)

    st.pyplot(fig)


def Radar_Chart(full_df3, search_text):
    filtered_df = full_df3[full_df3['buyer'].str.contains(search_text)]
    # Load sample data
    df = pd.DataFrame({
        'Category': ['Amount Spent', 'Average Duration', 'Number of Currency', 'Number of Transactions', 'Number of NFT Contracts', 'Total Earnings'],
        'Value': [filtered_df.iloc[0, 2], filtered_df.iloc[0, 4], filtered_df.iloc[0, 6], filtered_df.iloc[0, 8],
                  filtered_df.iloc[0, 10], filtered_df.iloc[0, 12]],
        'Color': ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                  'rgb(148, 103, 189)', 'rgb(121, 18, 150)']
    })

    # Create a polar subplot with 1 radial axis and 1 angular axis
    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=df['Value'],
        theta=df['Category'],
        marker=dict(
            color=df['Color'],
            line=dict(
                color='Black',
                width=2
            )
        ),
        opacity=1,
        hoverinfo='text',
        hovertext=df['Category'] + ': ' + 'Tier ' + df['Value'].astype(str)
    ))

    # Set layout and plot title
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5],  # Set the range for radial axis here
                showticklabels=False,
                ticks=''
            ),
            angularaxis=dict(
                visible=True,
                direction='clockwise'
            )
        ),
        title={
            'text': 'Principle Measures',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        plot_bgcolor='black',
        paper_bgcolor='white',
    )

    # Render the chart in Streamlit
    st.plotly_chart(fig)

def histograms(full_df3, search_text):
    #  Bar Graphs - Histograms
    ###########################################################################################
    # Plot 1 and Plot 2
    bin_labels = ['Range 1', 'Range 2', 'Range 3', 'Range 4', 'Range 5']
    custom_bins1 = [0, 100, 1000, 10000, 100000, 1000000]
    custom_bins2 = [0, 86400, 604800, 2628000, 31540000, 1000000000]
    # Bin the data using pandas cut function
    full_df3["buckets_amount_spent"] = pd.cut(full_df3["amount_spent"], bins=custom_bins1, labels=bin_labels)
    full_df3["buckets_avg_duration"] = pd.cut(full_df3["avg_duration"], bins=custom_bins2, labels=bin_labels)

    # Convert the Interval object to a string representation of the bin interval
    full_df3["buckets_amount_spent"] = full_df3["buckets_amount_spent"].apply(lambda x: str(x))
    full_df3["buckets_avg_duration"] = full_df3["buckets_avg_duration"].apply(lambda x: str(x))

    # Calculate the histograms of the binned columns
    histogram_amount_spent = full_df3.groupby("buckets_amount_spent").size().reset_index(name="count_amount_spent")
    histogram_amount_spent = histogram_amount_spent.sort_values("buckets_amount_spent")

    histogram_avg_duration = full_df3.groupby("buckets_avg_duration").size().reset_index(name="count_avg_duration")
    histogram_avg_duration = histogram_avg_duration.sort_values("buckets_avg_duration")

    # Set the color for the highlighted row
    highlight_color = 'red'

    filtered_df1 = full_df3[full_df3['buyer'].str.contains(search_text)]

    colors = ['gray', 'gray', 'gray', 'gray', 'gray']

    # Create the plots side by side
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4), gridspec_kw={'wspace': 0.8})

    # Plot the amount_spent histogram
    for i, row in histogram_amount_spent.iterrows():
        # Set the color of the bar
        if row['buckets_amount_spent'] == filtered_df1.iloc[0]['buckets_amount_spent']:
            color = highlight_color
        else:
            color = colors[i]
        # Plot the bar
        ax1.bar(row['buckets_amount_spent'], row['count_amount_spent'], width=1, edgecolor="black", color=color)
    ax1.set_xlabel("Amount Spent Buckets")
    ax1.set_ylabel("Count")
    ax1.set_title("Average Spend Position Indicator")

    # Plot the avg_duration histogram
    for i, row in histogram_avg_duration.iterrows():
        # Set the color of the bar
        if row['buckets_avg_duration'] == filtered_df1.iloc[0]['buckets_avg_duration']:
            color = highlight_color
        else:
            color = colors[i]
        # Plot the bar
        ax2.bar(row['buckets_avg_duration'], row['count_avg_duration'], width=1, edgecolor="black", color=color)
    ax2.set_xlabel("Avg Duration Buckets")
    ax2.set_ylabel("Count")
    ax2.set_title("Average Duration Position Indicator")

    # Add a text box outside each plot
    fig.subplots_adjust(right=0.85)

    ax1.text(1.12, 0.5,
             "Ranges:\n\nR1: $0-$100\nR2: $100-$1,000\nR3: $1,000-$10,000\nR4: $10,000-$100,000\nR5: $100,000+",
             transform=ax1.transAxes, fontsize=8, verticalalignment='center',
             bbox=dict(facecolor='none', edgecolor='black', pad=5.0))

    ax2.text(1.12, 0.5,
             "Ranges:\n\nR1: 0-1 days\nR2: 1-7 days\nR3: 7-30 days\nR4: 30-365 days\nR5: >1 year",
             transform=ax2.transAxes, fontsize=8, verticalalignment='center',
             bbox=dict(facecolor='none', edgecolor='black', pad=5.0))

    # Display the plots
    st.pyplot(fig)

    ##################################################################################
    ### Plot 3 and Plot 4

    bin_labels = ['Range 1', 'Range 2', 'Range 3', 'Range 4', 'Range 5']
    custom_bins1 = [-1000000, -50, 50, 1000, 10000, 10000000]
    custom_bins2 = [0, 10, 20, 50, 100, 1000]
    # Bin the data using pandas cut function
    full_df3["buckets_total_earnings"] = pd.cut(full_df3["total_earnings"], bins=custom_bins1, labels=bin_labels)
    full_df3["buckets_number_of_items"] = pd.cut(full_df3["number_of_items"], bins=custom_bins2, labels=bin_labels)

    # Convert the Interval object to a string representation of the bin interval
    full_df3["buckets_total_earnings"] = full_df3["buckets_total_earnings"].apply(lambda x: str(x))
    full_df3["buckets_number_of_items"] = full_df3["buckets_number_of_items"].apply(lambda x: str(x))

    # Calculate the histograms of the binned columns
    histogram_amount_spent = full_df3.groupby("buckets_total_earnings").size().reset_index(name="count_total_earnings")
    histogram_amount_spent = histogram_amount_spent.sort_values("buckets_total_earnings")

    histogram_avg_duration = full_df3.groupby("buckets_number_of_items").size().reset_index(
        name="count_number_of_items")
    histogram_avg_duration = histogram_avg_duration.sort_values("buckets_number_of_items")

    # Set the color for the highlighted row
    highlight_color = 'red'

    filtered_df1 = full_df3[full_df3['buyer'].str.contains(search_text)]

    colors = ['gray', 'gray', 'gray', 'gray', 'gray']

    # Create the plots side by side
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4), gridspec_kw={'wspace': 0.8})

    # Plot the amount_spent histogram
    for i, row in histogram_amount_spent.iterrows():
        # Set the color of the bar
        if row['buckets_total_earnings'] == filtered_df1.iloc[0]['buckets_total_earnings']:
            color = highlight_color
        else:
            color = colors[i]
        # Plot the bar
        ax1.bar(row['buckets_total_earnings'], row['count_total_earnings'], width=1, edgecolor="black", color=color)
    ax1.set_xlabel("Total Earnings Buckets")
    ax1.set_ylabel("Count")
    ax1.set_title("Total Earnings Position Indicator")

    # Plot the avg_duration histogram
    for i, row in histogram_avg_duration.iterrows():
        # Set the color of the bar
        if row['buckets_number_of_items'] == filtered_df1.iloc[0]['buckets_number_of_items']:
            color = highlight_color
        else:
            color = colors[i]
        # Plot the bar
        ax2.bar(row['buckets_number_of_items'], row['count_number_of_items'], width=1, edgecolor="black", color=color)
    ax2.set_xlabel("Number of NFT items Buckets")
    ax2.set_ylabel("Count")
    ax2.set_title("Number of NFT items Position Indicator")

    # Add a text box outside each plot
    fig.subplots_adjust(right=0.85)

    ax1.text(1.12, 0.5,
             "Ranges:\n\nR1: -1000000 - -50\nR2: -50-50\nR3: 50-1,000\nR4: 1,000-$10,000\nR5: 10,000-10000000",
             transform=ax1.transAxes, fontsize=8, verticalalignment='center',
             bbox=dict(facecolor='none', edgecolor='black', pad=5.0))

    ax2.text(1.12, 0.5,
             "Ranges:\n\nR1: 0-10\nR2: 10-20\nR3: 20-50\nR4: 50-100\nR5: 100-1000",
             transform=ax2.transAxes, fontsize=8, verticalalignment='center',
             bbox=dict(facecolor='none', edgecolor='black', pad=5.0))

    # Display the plots
    st.pyplot(fig)

    ###################################################################################
    # Plot 5 and Plot 6

    bin_labels = ['Range 1', 'Range 2', 'Range 3', 'Range 4', 'Range 5']
    bin_labels2 = ['Range 1', 'Range 2', 'Range 3', 'Range 4']
    custom_bins1 = [0, 2, 5, 10, 20, 1000]
    custom_bins2 = [0, 2, 3, 4, 10]
    # Bin the data using pandas cut function
    full_df3["buckets_number_txns"] = pd.cut(full_df3["number_txns"], bins=custom_bins1, labels=bin_labels)
    full_df3["buckets_num_currency"] = pd.cut(full_df3["num_currency"], bins=custom_bins2, labels=bin_labels2)

    # Convert the Interval object to a string representation of the bin interval
    full_df3["buckets_number_txns"] = full_df3["buckets_number_txns"].apply(lambda x: str(x))
    full_df3["buckets_num_currency"] = full_df3["buckets_num_currency"].apply(lambda x: str(x))

    # Calculate the histograms of the binned columns
    histogram_amount_spent = full_df3.groupby("buckets_number_txns").size().reset_index(name="count_number_txns")
    histogram_amount_spent = histogram_amount_spent.sort_values("buckets_number_txns")

    histogram_avg_duration = full_df3.groupby("buckets_num_currency").size().reset_index(name="count_num_currency")
    histogram_avg_duration = histogram_avg_duration.sort_values("buckets_num_currency")

    # Set the color for the highlighted row
    highlight_color = 'red'

    filtered_df1 = full_df3[full_df3['buyer'].str.contains(search_text)]

    colors = ['gray', 'gray', 'gray', 'gray', 'gray']

    # Create the plots side by side
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4), gridspec_kw={'wspace': 0.8})

    # Plot the amount_spent histogram
    for i, row in histogram_amount_spent.iterrows():
        # Set the color of the bar
        if row['buckets_number_txns'] == filtered_df1.iloc[0]['buckets_number_txns']:
            color = highlight_color
        else:
            color = colors[i]
        # Plot the bar
        ax1.bar(row['buckets_number_txns'], row['count_number_txns'], width=1, edgecolor="black", color=color)
    ax1.set_xlabel("Number of Transactions Buckets")
    ax1.set_ylabel("Count")
    ax1.set_title("Number of Transactions Position Indicator")

    # Plot the avg_duration histogram
    for i, row in histogram_avg_duration.iterrows():
        # Set the color of the bar
        if row['buckets_num_currency'] == filtered_df1.iloc[0]['buckets_num_currency']:
            color = highlight_color
        else:
            color = colors[i]
        # Plot the bar
        ax2.bar(row['buckets_num_currency'], row['count_num_currency'], width=1, edgecolor="black", color=color)
    ax2.set_xlabel("Number of Currencies Buckets")
    ax2.set_ylabel("Count")
    ax2.set_title("Number of Currencies Position Indicator")

    # Add a text box outside each plot
    fig.subplots_adjust(right=0.85)

    ax1.text(1.12, 0.5,
             "Ranges:\n\nR1: 0-2\nR2: 2-5\nR3: 5-10\nR4: 10-20\nR5: 20-1000",
             transform=ax1.transAxes, fontsize=8, verticalalignment='center',
             bbox=dict(facecolor='none', edgecolor='black', pad=5.0))

    ax2.text(1.12, 0.5,
             "Ranges:\n\nR1: 0-2\nR2: 2-3\nR3: 3-4\nR4: 4-10",
             transform=ax2.transAxes, fontsize=8, verticalalignment='center',
             bbox=dict(facecolor='none', edgecolor='black', pad=5.0))

    # Display the plots
    st.pyplot(fig)

def main():
    search_text = st.text_input('Search Blockchain Address')
    search_button = st.button('Search')

    if len(search_text) != 0 and search_button:
        full_df3 = Ingestion()
        filtered_df = full_df3[full_df3['buyer'].str.contains(search_text)]

        #    No Matching Address
        if filtered_df.empty == True:
            st.write("Invalid Blockchain Address")
            return
        Tiering(full_df3, search_text)
        Radar_Chart(full_df3, search_text)
        histograms(full_df3, search_text)

if __name__ == "__main__":
    main()
