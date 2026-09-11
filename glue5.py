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

try:
    # Outer join to capture updates and new inserts
    current_dim = dim_employee_df.filter(col("is_current") == "Y")

    # Defense-in-depth: proactively rename the incoming batch's join key so the
    # subsequent full outer join never produces two columns with the same name
    # (which is what caused the ambiguous 'emp_id' AnalysisException).
    updates_df_renamed = updates_df.withColumnRenamed("emp_id", "upd_emp_id")

    merged_stage = current_dim.join(
        updates_df_renamed,
        current_dim.emp_id == updates_df_renamed.upd_emp_id,
        "full_outer"
    )

    logger.info("Transforming SCD type 2 records and projecting schema...")

    # FIXED: emp_id is coalesced across both sides of the full outer join so that
    # rows originating only from the incoming batch (new hires, e.g. emp_id=104)
    # or only from the historical dimension (dropped/expired records) both retain
    # a non-null identifier in the final projection.
    final_audit = merged_stage.select(
        coalesce(current_dim.emp_id, updates_df_renamed.upd_emp_id).alias("emp_id"),
        coalesce(updates_df_renamed.emp_name, current_dim.emp_name).alias("resolved_name"),
        when(
            current_dim.department.isNull() | updates_df_renamed.department.isNull(),
            lit("NO_CHANGE")
        ).when(
            updates_df_renamed.department != current_dim.department,
            lit("DEPT_CHANGE")
        ).otherwise(lit("NO_CHANGE")).alias("change_type"),
        coalesce(updates_df_renamed.effective_date, current_dim.start_date).alias("record_timestamp")
    )

    final_audit.show(10, truncate=False)

except AnalysisException as e:
    logger.error("AnalysisException while reconciling SCD-2 dimension data: %s", str(e))
    logger.error("dim_employee_df schema:")
    dim_employee_df.printSchema()
    logger.error("updates_df schema:")
    updates_df.printSchema()
    raise
except Exception as e:
    logger.error("Unexpected error during SCD-2 reconciliation join/select: %s", str(e))
    raise

job.commit()
