import sys
import logging
from datetime import datetime
from pyspark.context import SparkContext
from pyspark.sql.functions import col, lit, when, max as spark_max, coalesce
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("glue_scd2_pipeline")

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

logger.info("Initializing Dimension tables for SCD-2 processing...")

historical_schema = StructType([
    StructField("emp_id", IntegerType(), False),
    StructField("emp_name", StringType(), True),
    StructField("department", StringType(), True),
    StructField("salary", IntegerType(), True),
    StructField("start_date", StringType(), True),
    StructField("end_date", StringType(), True),
    StructField("is_current", StringType(), True)
])

history_data = [
    (101, "Alice Smith", "Engineering", 95000, "2024-01-01", "2024-12-31", "N"),
    (101, "Alice Smith", "Platform Eng", 105000, "2025-01-01", None, "Y"),
    (102, "Bob Jones", "Sales", 60000, "2024-06-01", None, "Y"),
    (103, "Charlie Brown", "Marketing", 70000, "2024-03-15", None, "Y")
]
dim_employee_df = spark.createDataFrame(history_data, schema=historical_schema)

incoming_schema = StructType([
    StructField("emp_id", IntegerType(), False),
    StructField("emp_name", StringType(), True),
    StructField("department", StringType(), True),
    StructField("salary", IntegerType(), True),
    StructField("effective_date", StringType(), True)
])

incoming_data = [
    (101, "Alice Smith", "Principal Eng", 120000, "2026-02-01"),
    (102, "Bob Jones", "Sales Lead", 75000, "2026-02-01"),
    (104, "David Miller", "Finance", 80000, "2026-02-01")
]
updates_df = spark.createDataFrame(incoming_data, schema=incoming_schema)

logger.info("Performing reconciliation join between target dimension and source batch...")

try:
    # Alias both DataFrames before the join to avoid AMBIGUOUS_REFERENCE
    # errors once the join plan merges overlapping column names
    # (emp_id, department) from both sides.
    current_dim = dim_employee_df.filter(col("is_current") == "Y").alias("cur")
    updates_aliased = updates_df.alias("upd")

    # Outer join to capture updates and new inserts
    merged_stage = current_dim.join(
        updates_aliased,
        col("cur.emp_id") == col("upd.emp_id"),
        "full_outer"
    )

    logger.info("Transforming SCD type 2 records and projecting schema...")

    # FIXED: disambiguate all overlapping columns (emp_id, department, etc.)
    # by referencing them through their aliased DataFrame names.
    final_audit = merged_stage.select(
        coalesce(col("upd.emp_id"), col("cur.emp_id")).alias("emp_id"),
        coalesce(col("upd.emp_name"), col("cur.emp_name")).alias("resolved_name"),
        when(col("upd.department") != col("cur.department"), lit("DEPT_CHANGE"))
        .otherwise(lit("NO_CHANGE")).alias("change_type"),
        coalesce(col("upd.effective_date"), col("cur.start_date")).alias("record_timestamp")
    )

    final_audit.show(10, truncate=False)
    job.commit()
except Exception as e:
    logger.exception("SCD-2 reconciliation job failed during join/select/commit: %s", str(e))
    raise
