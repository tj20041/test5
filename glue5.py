import sys
import logging
from pyspark.context import SparkContext
from pyspark.sql.functions import col, lit, when, max as spark_max, coalesce
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

# Alias both DataFrames before the join so all downstream column references
# are unambiguous regardless of which side of the full outer join is null.
current_dim = dim_employee_df.filter(col("is_current") == "Y").alias("cur")
updates_aliased = updates_df.alias("upd")

# Full outer join using a string key so Spark deduplicates emp_id automatically.
merged_stage = current_dim.join(
    updates_aliased,
    on="emp_id",
    how="full_outer"
)

logger.info("Transforming SCD type 2 records and projecting schema...")

try:
    # emp_id is now a single, unambiguous column because the join was expressed
    # as a string key.  The null-safe tri-state change_type expression handles:
    #   - NEW_INSERT   : row exists only in updates_df  (cur.emp_id side is null)
    #   - RECORD_CLOSED: row exists only in current_dim (upd.emp_id side is null)
    #   - DEPT_CHANGE  : both sides present and department differs
    #   - NO_CHANGE    : both sides present and department is the same
    final_audit = merged_stage.select(
        col("emp_id"),
        coalesce(col("upd.emp_name"), col("cur.emp_name")).alias("resolved_name"),
        when(col("cur.emp_id").isNull(), lit("NEW_INSERT"))
        .when(col("upd.emp_id").isNull(), lit("RECORD_CLOSED"))
        .when(col("upd.department") != col("cur.department"), lit("DEPT_CHANGE"))
        .otherwise(lit("NO_CHANGE")).alias("change_type"),
        coalesce(col("upd.effective_date"), col("cur.start_date")).alias("record_timestamp")
    )

    final_audit.show(10, truncate=False)
    job.commit()
except Exception as exc:
    logger.error("SCD-2 transformation failed: %s", exc, exc_info=True)
    raise
