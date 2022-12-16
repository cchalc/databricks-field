# Databricks notebook source
# MAGIC %md
# MAGIC # GeoAnalytics Engine: 
# MAGIC > ## *Scalable Spatial Analysis with Big Data*
# MAGIC 
# MAGIC ### Key features
# MAGIC - **Fully integrated with Apache Spark**—Process spatial data at scale with functions developed and tested by Esri, the global leader in GIS, location intelligence, and mapping.
# MAGIC - **Easy to use**—Build spatially-enabled big data pipelines with an intuitive Python API that extends PySpark.
# MAGIC - **100+ spatial SQL functions**—Create geometries, test spatial relationships, and more using Python or SQL syntax.
# MAGIC - **Powerful analysis tools**—Run common spatiotemporal and statistical analysis workflows with only a few lines of code.
# MAGIC - **Automatic spatial indexing**—Perform optimized spatial joins and other operations immediately.
# MAGIC - **Read from and write to common** data sources—Load and save data from shapefiles, feature services, and vector tiles.
# MAGIC - **Cloud-native**—Tested and ready to install on Databricks, Amazon EMR, Azure Synapse, and Google Cloud Dataproc.

# COMMAND ----------

# MAGIC %md
# MAGIC ### GA Engine deployment requires
# MAGIC 
# MAGIC 1. An active Azure subscription
# MAGIC 2. GeoAnalytics Engine install files
# MAGIC 3. An ArcGIS Online subscription, or a license file

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import GA Engine modules and sign in

# COMMAND ----------

# Imports
import geoanalytics
from geoanalytics.sql import functions as ST
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import to_timestamp
from pyspark.sql.functions import to_date
from pyspark.sql.functions import date_trunc, col


# Sign into GeoAnalytics Engine
geoanalytics.version()
geoanalytics.auth(username="", password="")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Mobility Data Analytics

# COMMAND ----------

## Read-in dataset
df = spark.read.parquet("s3n://PQ/*.parquet") \
      .selectExpr("*", "ST_Point(client_longitude, client_latitude, 4326) as SHAPE") \
      .st.set_geometry_field("SHAPE") \
      .withColumn("date_", to_date(col("result_date"))) \
      .withColumn("time_stamp", to_timestamp('result_date', "yyyy-MM-dd HH:mm:ss")) \
      .st.set_time_fields("time_stamp")

df.cache()
df.createOrReplaceTempView("DF")

df_data = spark.sql(
"""
SELECT
  device_id,
  SHAPE,
  time_stamp,
  device_model,
  rsrp
FROM
  DF
WHERE
  #rsrp > -200 
  #AND rsrp < 0 
  #AND rsrq > -30 
  #AND rsrq < 0
  #AND 
  date_ >= '2018-01-01'
  AND date_ <= '2019-12-31'
"""
)


df_data.createOrReplaceTempView("DF_DATA")
df_data.cache()
df.unpersist()

# COMMAND ----------

## Blog Figure 2 analysis: Find hot spots 
from geoanalytics.tools import FindHotSpots
result_hot = FindHotSpots() \
            .setBins(bin_size=15000, bin_size_unit="Meters") \
            .setNeighborhood(distance=100000, distance_unit="Meters") \
            .run(dataframe=df_data)

# COMMAND ----------

url = "https://services.arcgis.com/P3ePLMYs2RVChkJx/ArcGIS/rest/services/USA_States_Generalized/FeatureServer/0"
us_states = spark.read.format("feature-service").load(url).withColumn("shape", ST.transform("shape", 4326))
#us_states.cache()

result_hot_wgs84 = result_hot.withColumn("bin_geometry", ST.transform("bin_geometry", 4326).alias("transform")) #transform to wgs84

usa_plot = us_states.st.plot(facecolor="white",
                                        edgecolors="black",
                                        figsize=(22,10),
                                        aspect="equal")
                                       
usa_plot.set_xlim(-130, -60)
usa_plot.set_ylim(22, 52)

result_plot = result_hot_wgs84.st.plot(cmap_values="Gi_Bin",
                             cmap="Wistia",
                             legend=True,
                             ax=usa_plot)

result_plot.set_title("USA cell connection hot spots")

# COMMAND ----------

## Blog Figure 3 analysis: Aggregate data into bins 
from geoanalytics.tools import AggregatePoints
result_bins = AggregatePoints() \
          .setBins(bin_size=15000, bin_size_unit="Meters",bin_type="Hexagon") \
          .addSummaryField(summary_field="rsrp",statistic="Mean").run(df_data)


## Publish pyspark dataframe to ArcGIS Online (AGOL) relational database 
#from arcgis import GIS
#gis = GIS(username="xxxx", password="yyyy")
#sdf = result_bins.st.to_pandas_sdf()
#lyr = sdf.spatial.to_featurelayer('ookla_2018_2019_bins_15km')

# COMMAND ----------

displayHTML("""<iframe src="https://tech.maps.arcgis.com/apps/View/index.html?appid=f66a5e937ef74323a45ab6a9511aceaa#mode=view" style="border:0px #ffffff none;" name="myiFrame" scrolling="no" frameborder="1" marginheight="0px" marginwidth="0px" height="650px" width="1500px" allowfullscreen></iframe>""")



# COMMAND ----------

## Blog Figure 4 analysis: Find dwell location 
from geoanalytics.tools import FindDwellLocations

# Spatial filter of data to focus analysis over Denver, Colorado 
boundingbox = df_data.selectExpr("device_id", "SHAPE", "time_stamp", "device_model","ST_EnvIntersects(SHAPE,-104.868,39.545,-104.987,39.9883) as Filter")
facility = boundingbox.filter(boundingbox['Filter'] == True)

result_dwell = FindDwellLocations() \
       .setTrackFields("device_id") \
       .setDistanceMethod(distance_method="Planar") \
       .setDwellMaxDistance(max_distance=100, max_distance_unit="Meters") \
       .setDwellMinDuration(min_duration=5, min_duration_unit="Minutes") \
       .setOutputType(output_type="Dwellpoints").run(dataframe=facility)

result_dwell = result_dwell.withColumn("DwellDuration_minutes", F.col("DwellDuration") / 60000)

result_dwell_wgs84 = result_dwell.withColumn("SHAPE", ST.transform("SHAPE", 4326))


# COMMAND ----------

displayHTML("""<style>.embed-container {position: relative; padding-bottom: 80%; height: 0; max-width: 100%;} .embed-container iframe, .embed-container object, .embed-container iframe{position: absolute; top: 0; left: 0; width: 100%; height: 100%;} small{position: absolute; z-index: 40; bottom: 0; margin-bottom: -15px;}</style><div class="embed-container"><iframe width="500" height="400" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" title="Ookla Devices Dwelling Heatmap" src="https://tech.maps.arcgis.com/apps/Embed/index.html?webmap=fbf231f8cf90465a8109a553775dfb96&extent=-105.424,39.5381,-104.4099,39.9615&zoom=true&previewImage=false&scale=true&disable_scroll=true&theme=light"></iframe></div>""")

# COMMAND ----------

import pyspark
split_col = pyspark.sql.functions.split(result_dwell_wgs84['time_stamp'], ' ')
result_dwell_wgs84 = result_dwell_wgs84.withColumn('Date', split_col.getItem(0))
dwell_df = result_dwell_wgs84.selectExpr('device_id','SHAPE',"cast(time_stamp as string) time_stamp", 'cast(Date as string) Date', 'device_model','DwellID','MeanX','MeanY','DwellDuration','MeanDistance', 'DwellDuration_minutes')

# Temporal filter to focus on dwelling on a day 
dwell_2019_05_31 = dwell_df.where("Date = '2019-05-31'")

# COMMAND ----------

display(dwell_2019_05_31)

# COMMAND ----------

from geoanalytics.tools import Overlay
safegraph_poi = spark.read.option("header", True).option("escape", "\"").csv("/mnt/core_poi-geometry/*.csv.gz") \
                .withColumn("Poly", (ST.poly_from_text("polygon_wkt", srid=4326)))

safegraph_poi_den =  safegraph_poi.where(safegraph_poi.city=="Denver").select("placekey", "parent_placekey", "safegraph_brand_ids", "location_name", "brands", "store_id", "top_category",
                                                                             "sub_category", "naics_code", "latitude", "longitude", "street_address", "city", "region", "postal_code", "polygon_wkt", "Poly").where("Poly IS NOT NULL")

overlay_result=Overlay() \
            .setOverlayType(overlay_type="Intersect") \
            .run(input_dataframe=safegraph_poi_den, overlay_dataframe=dwell_2019_05_31)

overlay_result.groupBy("top_category").mean("DwellDuration").show(truncate=False)

# COMMAND ----------

## Blog Figure 5 analysis: Total dwell duration (in milliseconds) by SafeGraph POI footprints top-category in Denver
# Try to generate the graph by yourself after running this cell. 
overlay_result_groupBy = overlay_result.groupBy("top_category").mean("DwellDuration")
display(overlay_result_groupBy)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transaction Data Analytics  

# COMMAND ----------

df_spend_2020_01 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2020/m=1/*.csv.gz").withColumn("Month", F.lit("Jan"))
df_spend_2020_02 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2020/m=2/*.csv.gz").withColumn("Month", F.lit("Feb"))
df_spend_2020_03 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2020/m=3/*.csv.gz").withColumn("Month", F.lit("Mar"))
df_spend_2020_04 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2020/m=4/*.csv.gz").withColumn("Month", F.lit("Apr"))
df_spend_2020_05 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2020/m=5/*.csv.gz").withColumn("Month", F.lit("May"))
df_spend_2020_06 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2020/m=6/*.csv.gz").withColumn("Month", F.lit("Jun"))
df_spend_2020_07 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2020/m=7/*.csv.gz").withColumn("Month", F.lit("Jul"))
df_spend_2020_08 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2020/m=8/*.csv.gz").withColumn("Month", F.lit("Aug"))
df_spend_2020_09 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2020/m=9/*.csv.gz").withColumn("Month", F.lit("Sep"))
df_spend_2020_10 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2020/m=10/*.csv.gz").withColumn("Month", F.lit("Oct"))
df_spend_2020_11 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2020/m=11/*.csv.gz").withColumn("Month", F.lit("Nov"))
df_spend_2020_12 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2020/m=12/*.csv.gz").withColumn("Month", F.lit("Dec"))

df_spend_2021_01 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2021/m=1/*.csv.gz").withColumn("Month", F.lit("Jan"))
df_spend_2021_02 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2021/m=2/*.csv.gz").withColumn("Month", F.lit("Feb"))
df_spend_2021_03 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2021/m=3/*.csv.gz").withColumn("Month", F.lit("Mar"))
df_spend_2021_04 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2021/m=4/*.csv.gz").withColumn("Month", F.lit("Apr"))
df_spend_2021_05 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2021/m=5/*.csv.gz").withColumn("Month", F.lit("May"))
df_spend_2021_06 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2021/m=6/*.csv.gz").withColumn("Month", F.lit("Jun"))
df_spend_2021_07 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2021/m=7/*.csv.gz").withColumn("Month", F.lit("Jul"))
df_spend_2021_08 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2021/m=8/*.csv.gz").withColumn("Month", F.lit("Aug"))
df_spend_2021_09 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2021/m=9/*.csv.gz").withColumn("Month", F.lit("Sep"))
df_spend_2021_10 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2021/m=10/*.csv.gz").withColumn("Month", F.lit("Oct"))
df_spend_2021_11 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2021/m=11/*.csv.gz").withColumn("Month", F.lit("Nov"))
df_spend_2021_12 = spark.read.option("header", True).option("escape", "\"").csv("/mnt/spend_patterns/y=2021/m=12/*.csv.gz").withColumn("Month", F.lit("Dec"))

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import *


import functools
def unionAll(dfs):
    return functools.reduce(lambda df1, df2: df1.union(df2.select(df1.columns)), dfs)

df_spend_2020 = unionAll([df_spend_2020_01, df_spend_2020_02, df_spend_2020_03, df_spend_2020_04, df_spend_2020_05, 
                          df_spend_2020_06, df_spend_2020_07, df_spend_2020_08, df_spend_2020_09, df_spend_2020_10,
                         df_spend_2020_11, df_spend_2020_12])
df_spend_2021 = unionAll([df_spend_2021_01, df_spend_2021_02, df_spend_2021_03, df_spend_2021_04, df_spend_2021_05, 
                          df_spend_2021_06, df_spend_2021_07, df_spend_2021_08, df_spend_2021_09, df_spend_2021_10,
                         df_spend_2021_11, df_spend_2021_12])

spend_2020_pnt = (df_spend_2020 ## assuming you have access to SafeGraph spend dataframe 
              .withColumn("point", ST.transform(ST.point("longitude", "latitude", 4326), 2263))
              .withColumn("Date_start", F.to_timestamp(date_format("spend_date_range_start","yyyy-MM-dd HH:mm:ss"), "yyyy-MM-dd HH:mm:ss"))
              .withColumn("Date_end", F.to_timestamp(date_format("spend_date_range_start","yyyy-MM-dd HH:mm:ss"), "yyyy-MM-dd HH:mm:ss"))
              .withColumn("Spend", df_spend_2020["raw_total_spend"].cast(DoubleType())) 
              .withColumn("online_trans", df_spend_2020["online_transactions"].cast(IntegerType())) 
              .withColumn("Tot_online_spend", df_spend_2020["online_spend"].cast(DoubleType()))                       
              .withColumn("customers", df_spend_2020["raw_num_customers"].cast(DoubleType()))                   
              .where("brands IS NOT NULL")
              #.select("placekey", "Date_start", "Date_end", "brands", "top_category", "sub_category", "Spend", "point")        
               )

spend_2020_pnt = spend_2020_pnt \
            .st.set_time_fields("Date_start") \
            .st.set_geometry_field("point")

spend_2021_pnt = (df_spend_2021 ## assuming you have access to SafeGraph spend dataframe 
              .withColumn("point", ST.transform(ST.point("longitude", "latitude", 4326), 2263))
              .withColumn("Date_start", F.to_timestamp(date_format("spend_date_range_start","yyyy-MM-dd HH:mm:ss"), "yyyy-MM-dd HH:mm:ss"))
              .withColumn("Date_end", F.to_timestamp(date_format("spend_date_range_start","yyyy-MM-dd HH:mm:ss"), "yyyy-MM-dd HH:mm:ss"))
              .withColumn("Spend", df_spend_2021["raw_total_spend"].cast(DoubleType())) 
              .withColumn("online_trans", df_spend_2021["online_transactions"].cast(IntegerType())) 
              .withColumn("Tot_online_spend", df_spend_2021["online_spend"].cast(DoubleType()))                       
              .withColumn("customers", df_spend_2021["raw_num_customers"].cast(DoubleType()))                   
              .where("brands IS NOT NULL")
              #.select("placekey", "Date_start", "Date_end", "brands", "top_category", "sub_category", "Spend", "point")        
               )
spend_2021_pnt = spend_2021_pnt \
            .st.set_time_fields("Date_start") \
            .st.set_geometry_field("point")

# COMMAND ----------

## Blog Figure 6 analysis: Compare annual rental car spend between 2020 and 2021
from geoanalytics.tools import AggregatePoints

# Load a polygon feature service of US county boundaries into a DataFrame
county = "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/USA_Counties_Generalized/FeatureServer/0"
df_county = spark.read.format("feature-service").load(county).withColumn("shape", ST.transform("shape", 4326))

# 2020 analysis 
ERC_spend_2020 = spend_2020_pnt.where(df_spend_2020.brands=="Enterprise Rent-A-Car")
ERC_spend_2020_point = ERC_spend_2020.withColumn("point", ST.point("longitude", "latitude", 4326))
ERC_spend_2020_county = AggregatePoints().setPolygons(df_county) \
            .addSummaryField(summary_field="raw_total_spend", statistic="Sum") \
            .run(ERC_spend_2020_point)

# 2021 analysis 
ERC_spend_2021 = spend_2021_pnt.where(df_spend_2021.brands=="Enterprise Rent-A-Car")
ERC_spend_2021_point = ERC_spend_2021.withColumn("point", ST.point("longitude", "latitude", 4326))
ERC_spend_2021_county = AggregatePoints().setPolygons(df_county) \
            .addSummaryField(summary_field="raw_total_spend", statistic="Sum") \
            .run(ERC_spend_2021_point)

# Publish dataframes to to ArcGIS Online (AGOL) and then visualize and create Fig 8 web app in AGOL
# from arcgis import GIS
# gis = GIS(username="", password="")
# sdf = ERC_spend_2020_county.st.to_pandas_sdf()
# lyr = sdf.spatial.to_featurelayer('ERC_spend_2020_county')
# sdf = ERC_spend_2021_county.st.to_pandas_sdf()
# lyr = sdf.spatial.to_featurelayer('ERC_spend_2021_county')

# COMMAND ----------

displayHTML("""<iframe src="https://amdxhqznhv7nbywn.maps.arcgis.com/apps/webappviewer/index.html?id=ca32db76023e480292d86e5975111129#mode=view" style="border:0px #ffffff none;" name="myiFrame" scrolling="no" frameborder="1" marginheight="0px" marginwidth="0px" height="650px" width="1500px" allowfullscreen></iframe>""")

## if map is not rendered then go to this url directly: https://amdxhqznhv7nbywn.maps.arcgis.com/apps/webappviewer/index.html?id=ca32db76023e480292d86e5975111129

# COMMAND ----------

## Blog Figure 7 analysis: Find similar locations 

# Load a polygon feature service of US county boundaries into a DataFrame
county = "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/USA_Counties_Generalized/FeatureServer/0"
df_county = spark.read.format("feature-service").load(county).withColumn("shape", ST.transform("shape", 4326))

# Get annual aggregated online spend at County level 
result_online_spend = AggregatePoints().setPolygons(df_county) \
        .addSummaryField(summary_field="Tot_online_spend",statistic="Sum") \
        .run(spend_2020_pnt)

# Find the County with the highest online spend at POIs
result_online_spend.orderBy(desc("SUM_Tot_online_spend")).take(1)

# Create a DataFrame with Sacramento data
Sacramento_df = result_online_spend.where("NAME = 'Sacramento'")

# Create a DataFrame without Sacramento data
without_Sacramento_df = result_online_spend.where("NAME != 'Sacramento'")


from geoanalytics.tools import FindSimilarLocations
result_similar_loc = FindSimilarLocations() \
           .setAnalysisFields("SUM_Tot_online_spend", "POPULATION", "CROP_ACR12", "AVE_SALE12") \
           .setMostOrLeastSimilar(most_or_least_similar="MostSimilar") \
           .setMatchMethod(match_method="AttributeValues") \
           .setNumberOfResults(number_of_results=5) \
           .setAppendFields("NAME", "STATE_NAME", "POPULATION",
                            "CROP_ACR12", "AVE_SALE12", "SUM_Tot_online_spend", "shape") \
           .run(reference_dataframe=Sacramento_df, search_dataframe=without_Sacramento_df)


# COMMAND ----------

# Create a continental USA states DataFrame and transform the geometry
# to WGS 1984 Web Mercator for visualization
states_subset_df = df_county.where("""STATE_NAME != 'Alaska'
                                     and STATE_NAME != 'Hawaii' and
                                     STATE_NAME != 'District of Columbia'""") \
                                .withColumn("shape", ST.transform("shape", 3857))

# Create a DataFrame that contains only the "search" locations from the analysis result
result_subset_df = result_similar_loc.where("simrank > 0") \
                                    .withColumn("shape", ST.transform("shape", 3857))

# Plot USA states that are most similar to Sacramento
states_subset_plot = states_subset_df.st.plot(facecolor="none",
                                              edgecolors="lightblue",
                                              figsize=(16,10))
result_plot = result_subset_df.st.plot(cmap_values="simrank",
                                       cmap="Oranges_r",
                                       legend=True,
                                       ax=states_subset_plot)
result_plot.set_title("Counties most similar to Sacramento in population, agriculture, and online shopping")
result_plot.set_xlabel("X (Meters)")
result_plot.set_ylabel("Y (Meters)");

# COMMAND ----------

# MAGIC %md
# MAGIC ##Public Service Data Analytics  

# COMMAND ----------

# LOAD input data from Azure blob storage
ny_311_data = spark.read.csv("/mnt/nyc311", header=True) 

# COMMAND ----------

# Custom function to clean up complaint types format
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from geoanalytics.tools import Clip

def cleanGroup(value):
  if "noise" in value.lower(): return 'noise'
  if "general construction" in value.lower(): return 'general construction'
  if "paint" in value.lower(): return 'paint/plaster'
  else: return value
  
udfCleanGroup = udf(cleanGroup, StringType())

# Data Processing: geo-enablement and cleaning
ny_311_data_cleaned = ( ny_311_data
  .withColumn("point", ST.transform(ST.point("Longitude", "Latitude", 4326), 2263))
  .withColumn("dt_created", F.to_timestamp(F.col("Created Date"), 'MM/dd/yyyy hh:mm:ss a'))
  .withColumn("dt_closed", F.to_timestamp(F.col("Closed Date"), 'MM/dd/yyyy hh:mm:ss a'))
  .withColumn("duration_hr", (F.col("dt_closed").cast("long") - F.col("dt_created").cast("long"))/3600)
  .filter(F.col("duration_hr") > 0)
  .withColumn("type", F.initcap(udfCleanGroup("Complaint Type")))
  .where("point IS NOT NULL")
  .select("Unique Key", "type", "status", "point", "dt_created", "dt_closed", "duration_hr")
)

ny_311_data_cleaned.createOrReplaceTempView("ny311")

# Spatial filtering to focus analyis over NYC 
ny_311_data_cleaned_extent = spark.sql("SELECT *, ST_EnvIntersects(point,909126.0155,110626.2880,1610215.3590,424498.0529) AS env_intersects FROM ny311")
ny_311_data_cleaned_extent.display()

# COMMAND ----------

## Blog Figure 8 analysis: spatiotemporal proximity analysis of long-response-calls complaint types 

# Calculate the sum of the mean duration and three standard deviations
ny_data_duration = ny_311_data_cleaned_extent \
        .withColumnRenamed("type", "Complaint Type") \
        .groupBy("Complaint Type").agg(
            (F.mean("duration_hr")+3*F.stddev("duration_hr")).alias("3stddevout")
      )

# Join the calculated stats to the NYC 311 call records
ny_311_stats = ny_data_duration.join(ny_311_data_cleaned, ny_311_data_cleaned["type"] == ny_data_duration["Complaint Type"], "fullouter")

# Select the records that are more than the mean duration plus three standard deviations
ny_311_calls_long_3stddev = ny_311_stats.filter("duration_hr > 3stddevout")

df = ny_311_calls_long_3stddev \
            .st.set_time_fields("dt_created") \
            .st.set_geometry_field("geometry")

# Run GroupByProximity
from geoanalytics.tools import GroupByProximity
grouper = GroupByProximity() \
           .setSpatialRelationship("NearPlanar", 500, "Feet") \
           .setTemporalRelationship("Near", 5, "Days") \
           .setAttributeRelationship("$a.type == $b.type", expression_type="Arcade")
result = grouper.run(df) 

# Filter for groups that are more than 10 records
result_group_size = result.withColumn("group_size", F.count("*").over(Window.partitionBy("group_id"))) \
      .filter("group_size > 10")

# Create the convex hull for each group
result_convex = result_group_size.groupBy("GROUP_ID") \
      .agg(ST.aggr_convex_hull("point") \
      .alias("convexhull"), F.first("Complaint type").alias("type"))

### Plot results to visualize complaint groups
# 1. Read and prepare a feature service for plotting
# 2. Plot the convex hulls


# COMMAND ----------

# Read and prepare a feature service for plotting
nyc_blocks = spark.read.format("feature-service") \
               .load("https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/NYC_Census_Blocks_for_2020_US_Census/FeatureServer/0") \
               .persist()

nyc_blocks_filtered = nyc_blocks.where((F.col("BoroName") == 'Queens') | (F.col("BoroName") == 'Brooklyn') | (F.col("BoroName") == 'Manhattan'))
top_type_lst = ["Noise", "Sidewalk Condition", "Unsanitary Condition", "Broken Muni Meter", "Homeless Person Assistance"] 

convex_hull_clip = Clip().run(input_dataframe=result_convex,
                    clip_dataframe=nyc_blocks_filtered).where(F.col("type").isin(top_type_lst))

# COMMAND ----------

# Plot 311 convex hulls groups
ax = nyc_blocks_filtered.st.plot(geometry="geometry",
                                facecolor="lightblue",
                                edgecolors="grey",
                                alpha=0.3,
                                figsize=(10,10))

result_clusters_plot = convex_hull_clip.st.plot(ax=ax, 
                                                cmap_values="type",
                                                is_categorical=True,
                                                cmap="viridis",
                                                legend=True,
                                                legend_kwds={"loc":"upper right", "fontsize":"12"}
                                                )

result_clusters_plot.set_title("Groups of NYC 311 long response calls")
result_clusters_plot.set_xlabel("X (Feet)")
result_clusters_plot.set_ylabel("Y (Feet)");
