import sys
import logging
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import avg, col, count, to_timestamp
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# ==========================================
# 0. GLUE INITIALIZATION & LOGGING SETUP
# ==========================================
# Fetch job name passed by the AWS Glue execution environment
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

logger = logging.getLogger("UserActivitySessionETL")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers if re-run
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

logger.info("Initializing User Activity Session ETL Job...")

try:
    # ==========================================
    # 1. SCHEMA DEFINITIONS
    # ==========================================
    logger.info("Defining explicit schemas...")
    sessions_schema = StructType([
        StructField("session_id", StringType(), True),
        StructField("user_id", IntegerType(), True),
        StructField("activity_type", StringType(), True),
        StructField("event_time_str", StringType(), True),
        StructField("duration_sec", IntegerType(), True)
    ])

    # ==========================================
    # 2. BRONZE LAYER (Extraction)
    # ==========================================
    logger.info("Extracting session logs into Bronze layer...")
    sessions_bronze = spark.createDataFrame([
        ("SESS-001", 101, "PAGE_VIEW", "2024-08-10 09:15:30", 45),
        ("SESS-002", 102, "PURCHASE", "2024-08-10 14:22:10", 120),
        ("SESS-003", 101, "ADD_TO_CART", "2024-08-10 18:45:00", 30),
        ("SESS-004", 103, "PAGE_VIEW", "2024-08-10 21:05:15", 15)
    ], schema=sessions_schema)

    # ==========================================
    # 3. SILVER LAYER (Transformation & Timestamp Parsing)
    # ==========================================
    logger.info("Parsing event timestamps and standardizing metrics for Silver layer...")
    sessions_silver = sessions_bronze.withColumn(
        "event_time",
        to_timestamp(col("event_time_str"), "yyyy-MM-dd hh:mm:ss")
    ).filter("duration_sec > 0")

    # ==========================================
    # 4. GOLD LAYER (User Engagement Aggregation)
    # ==========================================
    logger.info("Aggregating user session engagement for Gold layer...")
    df_gold = sessions_silver.groupBy("user_id").agg(
        count("session_id").alias("total_sessions"),
        avg("duration_sec").alias("avg_duration_sec")
    )

    logger.info("Pipeline completed successfully.")
    df_gold.show(truncate=False)

    # Commit Glue Job
    job.commit()

except Exception as e:
    logger.error("Pipeline failed during execution. Error details: %s", str(e))
    raise
