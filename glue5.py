import sys
import logging
from datetime import datetime
from pyspark.context import SparkContext
from pyspark.sql.functions import col, lit, when, max as spark_max, coalesce
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
from pyspark.sql.utils import AnalysisException
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

# Outer join to capture updates and new inserts
current_dim = dim_employee_df.filter(col("is_current") == "Y")

try:
    # Use the string/list `on=` join syntax so Spark automatically collapses
    # the two `emp_id` columns into a single unambiguous column instead of
    # retaining separate current_dim.emp_id / updates_df.emp_id columns.
    merged_stage = current_dim.join(
        updates_df,
        on="emp_id",
        how="full_outer"
    )

    logger.info("Transforming SCD type 2 records and projecting schema...")

    # emp_id is now a single, unambiguous column thanks to the `on=` join syntax above.
    final_audit = merged_stage.select(
        col("emp_id"),
        coalesce(updates_df.emp_name, current_dim.emp_name).alias("resolved_name"),
        when(updates_df.department.isNull() | current_dim.department.isNull(), lit("NEW_OR_TERMINATED"))
        .otherwise(
            when(updates_df.department != current_dim.department, lit("DEPT_CHANGE"))
            .otherwise(lit("NO_CHANGE"))
        ).alias("change_type"),
        coalesce(updates_df.effective_date, current_dim.start_date).alias("record_timestamp")
    )

    final_audit.show(10, truncate=False)
    job.commit()
    logger.info("SCD-2 reconciliation job committed successfully.")

except AnalysisException as ae:
    logger.error("AnalysisException encountered during SCD-2 join/select transformation: %s", str(ae))
    logger.error("current_dim schema:")
    current_dim.printSchema()
    logger.error("updates_df schema:")
    updates_df.printSchema()
    # Do not call job.commit() so the Glue job is correctly marked as FAILED
    # instead of leaving it in a partially-committed state.
    raise
except Exception as e:
    logger.error("Unexpected error during SCD-2 reconciliation pipeline: %s", str(e))
    raise
