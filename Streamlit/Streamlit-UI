import streamlit as st
import pandas as pd
import pyspark
import plotly.graph_objects as go
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyecharts import options as opts
from pyecharts.charts import Pie
from PIL import Image
#import streamlit_echarts as st_echarts
from pyecharts import options as opts
from pyecharts.charts import Radar
from pyecharts.render import make_snapshot
#from pyecharts_snapshot.main import make_a_snapshot
#from snapshot_selenium import snapshot as driver
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import IntegerType
from pyspark.sql.types import ArrayType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import TimestampType
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.types import FloatType
from pyspark.sql.functions import col, expr
from pyspark.sql.functions import sequence
from pyspark.sql.functions import sum,avg,max,count
from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import desc, asc
from pyspark.sql.functions import col, unix_timestamp, from_unixtime
from functools import reduce
from pyspark.sql.functions import col,lit,array
from pyspark.ml.functions import vector_to_array
from array import array

from pyspark.sql import functions as f
from pyspark.sql.functions import when


from pathlib import Path

# header = st.container()
#
# with header:
#     st.title("NFT Tier Information")

conf = pyspark.SparkConf()
spark = SparkSession.builder.appName("Model").getOrCreate()

data_dir = Path('/Users/abhishek/Downloads/ModelResults')
full_df = pd.concat(
    pd.read_parquet(parquet_file)
    for parquet_file in data_dir.glob('*.parquet')
)



search_text = st.text_input('Search Blockchain Address')
search_button = st.button('Search')

slider1 = st.slider('Ranking1', 0.0, 1.0, 0.2)
slider2 = st.slider('Ranking2', 0.0, 1.0, 0.25)
slider3 = st.slider('Ranking3', 0.0, 1.0, 0.25)
slider4 = st.slider('Ranking4', 0.0, 1.0, 0.1)
slider5 = st.slider('Ranking5', 0.0, 1.0, 0.2)

render = st.button("Render Model")

if render:
    df2 = spark.read.parquet("/Users/abhishek/Downloads/ModelScores")
    df2 = df2.withColumn("number_txns_double", col("number_txns").cast(DoubleType()))
    df2 = df2.withColumn("num_nftcontract_double", col("num_nftcontract").cast(DoubleType()))
    df2 = df2.withColumn("num_currency_double", col("num_currency").cast(DoubleType()))

    assemble = VectorAssembler(inputCols=[
        'avg_spent',
        'number_txns_double',
        'num_nftcontract_double',
        'num_currency_double',
        'avg_duration'], outputCol='features')
    assembled_data = assemble.transform(df2)

    # Standardisation
    scaler = StandardScaler(inputCol='features', outputCol='scaled_features')
    scaler_model = scaler.fit(assembled_data)
    scalar_data = scaler_model.transform(assembled_data)

    # scores in order of features: avg_spent','number_txns','num_nftcontract','num_currency','avg_duration' add weights
    scalar_data = scalar_data.withColumn("weight_param",
                                         pyspark.sql.functions.array(lit(slider1), lit(slider2), lit(slider3), lit(slider4), lit(slider5)))
    scalar_data = scalar_data.withColumn("sfv", (vector_to_array('scaled_features', "float32")))


    def multiply_list(list1, list2):
        list3 = [a * b for a, b in zip(list1, list2)]
        return list3

    def sum_list(list1):
        res = 0.0
        for num in list1:
            res = res + num
        return res


    multiply_lists_udf = udf(multiply_list, ArrayType(DoubleType()))
    scalar_data = scalar_data.withColumn("result", multiply_lists_udf("sfv", "weight_param"))

    sum_lists_udf = udf(sum_list, DoubleType())
    scalar_data = scalar_data.withColumn("weight", sum_lists_udf("result"))

    correct_data = scalar_data.filter(col("weight") >= 0.0)

    columns_to_drop_nulls = ['avg_spent', 'number_txns', 'num_nftcontract', 'num_currency', 'avg_duration']

    # drop null values from the specified columns
    correct_data = correct_data.na.drop(subset=columns_to_drop_nulls)

    kmeans = KMeans(featuresCol='features', k=3, weightCol='weight', maxIter=20, initMode='k-means||')
    model = kmeans.fit(correct_data)
    output = model.transform(correct_data)


if search_button:
    filtered_df = full_df[full_df['buyer'].str.contains(search_text)]
    if filtered_df.iloc[0]['prediction'] == 0:
        st.metric(label="Membership Status", value="Platinum")
    elif filtered_df.iloc[0]['prediction'] == 1:
        st.metric(label="Membership Status", value="Gold")
    elif filtered_df.iloc[0]['prediction'] == 2:
        st.metric(label="Membership Status", value="Silver")
    elif filtered_df.iloc[0]['prediction'] == 3:
        st.metric(label="Membership Status", value="Bronze")

    # Load sample data
    df = pd.DataFrame({
        'Category': ['Score 1', 'Score 2', 'Score 3', 'Score 4', 'Score 5'],
        'Value': [filtered_df.iloc[0, 2], filtered_df.iloc[0, 4], filtered_df.iloc[0, 6], filtered_df.iloc[0, 8],
                  filtered_df.iloc[0, 10]],
        'Color': ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                  'rgb(148, 103, 189)']
    })

    # Create a polar subplot with 1 radial axis and 1 angular axis
    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=df['Value'],
        theta=df['Category'],
        marker=dict(
            color=df['Color'],
            line=dict(
                color='white',
                width=1.5
            )
        ),
        opacity=0.8,
        hoverinfo='text',
        hovertext=df['Category'] + ': ' + df['Value'].astype(str) + '%'
    ))

    # Set layout and plot title
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 50],
                showticklabels=False,
                ticks=''
            ),
            angularaxis=dict(
                visible=True,
                direction='clockwise'
            )
        ),
        title={
            'text': 'Strength Areas',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        plot_bgcolor='black',
        paper_bgcolor='black',
    )

    # Render the chart in Streamlit
    st.plotly_chart(fig)
