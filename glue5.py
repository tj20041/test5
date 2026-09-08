import sys
import logging
from pyspark.context import SparkContext
from pyspark.sql.functions import col, lit, when, coalesce
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
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

# Filter to current dimension records only
current_dim = dim_employee_df.filter(col("is_current") == "Y")

# Alias both sides to avoid ambiguous column references across the full outer join
current_dim_aliased = current_dim.alias("cur")
updates_df_aliased = updates_df.alias("upd")

# Use string join key 'emp_id' so Spark deduplicates the join column automatically,
# eliminating duplicate emp_name, department, and salary columns in the result schema.
merged_stage = current_dim_aliased.join(
    updates_df_aliased,
    on="emp_id",
    how="full_outer"
)

# Defensive schema assertion: ensure no duplicate columns exist after the join
merged_cols = merged_stage.columns
duplicate_cols = [c for c in set(merged_cols) if merged_cols.count(c) > 1]
if duplicate_cols:
    logger.error(
        "Unexpected duplicate columns detected after join: %s. "
        "Review source schema changes before proceeding.",
        duplicate_cols
    )
    raise ValueError(
        f"Unexpected duplicate columns detected after join: {duplicate_cols}. "
        "Review source schema changes before proceeding."
    )

logger.info("Transforming SCD type 2 records and projecting schema...")

try:
    # With a string join key, emp_id is deduplicated by Spark and can be referenced
    # unambiguously. coalesce is used as an extra safety net for full_outer null rows.
    final_audit = merged_stage.select(
        col("emp_id"),
        coalesce(
            col("upd.emp_name"),
            col("cur.emp_name")
        ).alias("resolved_name"),
        # Null-safe tri-state change_type classification for all full_outer result rows:
        # - NEW_INSERT:     row exists only in updates_df (no current_dim match)
        # - RECORD_CLOSED:  row exists only in current_dim (no incoming update)
        # - DEPT_CHANGE:    both sides present and department differs
        # - NO_CHANGE:      both sides present and department is the same
        when(col("cur.emp_id").isNull(), lit("NEW_INSERT"))
        .when(col("upd.emp_id").isNull(), lit("RECORD_CLOSED"))
        .when(col("upd.department") != col("cur.department"), lit("DEPT_CHANGE"))
        .otherwise(lit("NO_CHANGE"))
        .alias("change_type"),
        coalesce(
            col("upd.effective_date"),
            col("cur.start_date")
        ).alias("record_timestamp")
    )

    final_audit.show(10, truncate=False)
    job.commit()
    logger.info("SCD-2 pipeline completed successfully.")

except Exception as e:
    logger.error("SCD-2 pipeline failed: %s", str(e))
    raise
