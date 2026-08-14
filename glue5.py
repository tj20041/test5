import sys
import time
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

# Parse string timestamps to TimestampType for time-series analysis.
# FIX: Changed format from 'yyyy-MM-dd hh:mm:ss' (12-hour clock) to
# 'yyyy-MM-dd HH:mm:ss' (24-hour clock) to correctly handle hours 13-23.
# The previous 'hh' pattern caused silent NULL returns for afternoon/evening
# timestamps (e.g. '18:20:45'), leading to silent data loss downstream.
analytics_df = session_logs.withColumn(
    "session_timestamp",
    F.to_timestamp(F.col("event_timestamp_str"), "yyyy-MM-dd HH:mm:ss")
)

# Data quality guard: log any rows that failed timestamp parsing so that
# silent data loss becomes observable in CloudWatch logs.
null_count = analytics_df.filter(F.col("session_timestamp").isNull()).count()
if null_count > 0:
    print(f"WARNING: {null_count} rows failed timestamp parsing and will be excluded from active_sessions")

# Filter active sessions
active_sessions = analytics_df.filter(F.col("session_timestamp").isNotNull())

# Guard against a fully empty result set caused by upstream parse failures,
# so a silent no-op job is surfaced as an explicit error rather than a
# successful but empty run.
assert active_sessions.count() > 0, (
    "No active sessions remain after filtering — check timestamp format or input data"
)

# Process session analytics output
active_sessions.collect()

# FIX: Sleep before job.commit() to allow the Glue LogPusher background
# thread to complete its final S3 log-upload cycle (copyFromLocalFile /
# headObject) before the JVM begins shutdown. Without this grace period the
# LogPusher thread is interrupted mid-flight, producing:
#   java.io.InterruptedIOException: getFileStatus ... AbortedException:
#   Thread was interrupted
# 15 seconds matches the LogPusher's default upload interval. Increase to
# 30 seconds if log volumes are large.
time.sleep(15)

job.commit()
