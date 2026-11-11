"""Additive upgrades for scalar metrics exported by early workspace bundles."""

# Keep original coefficient fields as audit evidence while giving analytical
# consumers the same numeric column names used by selected solution groups.
OPTION_METRIC_COLUMNS = {
    "j_ch": "j_ch_contribution",
    "target_violation": "coverage_violation_count",
    "target_excess_days": "coverage_excess_days",
    "matrix_exceptions": "matrix_exception_pair_count",
    "relaxed_group": "group_size_relaxed",
    "size_excess": "group_size_excess",
    "stable_rank": "solver_tie_break_rank",
}


def migrate_option_metrics(factory):
    """Add/backfill typed retained-option metrics without altering original evidence.

    Args:
        factory: Runtime HANA connection factory for isolated schema transactions.
    Returns:
        Nothing. Existing typed values are preserved; invalid legacy scalar values
        fail explicitly rather than silently becoming null or changing a result.
    """
    table = "PRODUCTION_WHEEL_DATA_OPTION_GROUPS"
    connection = factory()
    cursor = connection.cursor()
    try:
        cursor.execute(
            "SELECT COLUMN_NAME, DATA_TYPE_NAME FROM SYS.TABLE_COLUMNS WHERE SCHEMA_NAME=CURRENT_SCHEMA AND TABLE_NAME=?",
            (table,),
        )
        columns = dict(cursor.fetchall())
        if not columns:
            return
        for source, target in OPTION_METRIC_COLUMNS.items():
            if source.upper() not in columns:
                continue
            if target.upper() not in columns:
                cursor.execute(f'ALTER TABLE "{table}" ADD ("{target.upper()}" DOUBLE)')
            elif columns[target.upper()] != "DOUBLE":
                raise RuntimeError(
                    f"Incompatible retained-option metric column: {target}"
                )
            cursor.execute(
                f'UPDATE "{table}" SET "{target.upper()}"=TO_DOUBLE("{source.upper()}") WHERE "{target.upper()}" IS NULL AND "{source.upper()}" IS NOT NULL'
            )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()
