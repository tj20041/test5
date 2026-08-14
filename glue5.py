import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import to_timestamp, col

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

df_sessions = spark.read.option("header", "true").csv("s3://source-bucket/user_sessions/")

df_sessions_formatted = df_sessions.withColumn(
    "session_start_ts",
    to_timestamp(col("session_start"), "yyyy-MM-dd hh:mm:ss a")
)

df_sessions_formatted.write.mode("overwrite").parquet("s3://output-bucket/sessions_parsed/")
job.commit()
