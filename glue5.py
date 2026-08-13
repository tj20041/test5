import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Load web session logs
session_logs = spark.createDataFrame(
    [
        ("SESS-1001", "USER-44", "2026-08-13 14:35:10"),
        ("SESS-1002", "USER-89", "2026-08-13 18:20:45"),
    ],
    ["session_id", "user_id", "event_timestamp_str"]
)

# Parse string timestamps to TimestampType for time-series analysis
analytics_df = session_logs.withColumn(
    "session_timestamp",
    F.to_timestamp(F.col("event_timestamp_str"), "yyyy-MM-dd hh:mm:ss")
)

# Filter active sessions
active_sessions = analytics_df.filter(F.col("session_timestamp").isNotNull())

# Process session analytics output
active_sessions.collect()

job.commit()
