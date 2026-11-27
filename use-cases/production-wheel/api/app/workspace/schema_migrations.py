"""Startup upgrades of stored workspace data.

- ``migrate_option_metrics``: additive typed columns for scalar metrics exported
  by early workspace bundles.
- ``migrate_runner_thresholds``: moves the high / medium / low runner cut-offs
  from SAP policy rows into each plant profile.
- ``migrate_cycle_stock_policy``: moves the runner RoC multiple settings and the
  cycle-stock cap from SAP policy rows into each plant profile.
- ``migrate_sap_settings``: moves the remaining SAP policy rows (DC lot-size
  rules, calendar profiles, recipe characters, ...) into each plant profile's
  ``sap_settings`` section.
"""

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


def migrate_runner_thresholds(repo):
    """Move runner class cut-offs from SAP policy rows into every plant profile.

    Before the three-tier runner classes, profiles held one two-tier
    ``high_runner_threshold_days`` and the SAP policy table held the
    high / medium / low cut-offs per plant. After this migration:

    - every stored profile without ``runner_thresholds`` gets them, seeded from
      its plant's former SAP policy rows (``"*"`` rows, then plant rows, then
      template defaults), and loses the two-tier key; each one is saved as a new
      profile revision;
    - the former runner rows are removed from ``sap_policy_rules`` (the policy
      no longer accepts them).

    The function is idempotent: profiles already upgraded are skipped, and with
    no legacy rows left the policy table is not rewritten. A profile changed
    concurrently by another process is skipped and picked up on the next start.

    Args:
        repo: Workspace repository (HANA or in-memory) exposing ``list``, ``cas``,
            ``rows``, ``replace_tables`` and ``transaction``.

    Returns:
        ``{"profiles": [ids upgraded], "runner_rule_profiles": [ids whose rules
        use member.runner and should be reviewed], "policy_rows_removed": n}``.
    """
    import json
    import logging

    from production_wheel.runners import RunnerThresholds
    from production_wheel.sap_settings.policy import LEGACY_RUNNER_SETTINGS, legacy_runner_thresholds

    from .sap_settings_service import POLICY_OWNER

    log = logging.getLogger(__name__)
    policy_rows = repo.rows(POLICY_OWNER, "sap_policy_rules")
    report = {"profiles": [], "runner_rule_profiles": [], "policy_rows_removed": 0}
    for profile in repo.list("profiles"):
        settings = profile.get("settings") or {}
        if "runner_thresholds" in settings:
            continue
        seeded = legacy_runner_thresholds(policy_rows, profile.get("plant") or "*")
        try:
            thresholds = RunnerThresholds(**seeded)
        except ValueError:
            log.warning("profile %s: inconsistent former runner policy %s; using defaults",
                        profile["profile_id"], seeded)
            thresholds = RunnerThresholds()
        upgraded = {k: v for k, v in settings.items() if k != "high_runner_threshold_days"}
        upgraded["runner_thresholds"] = thresholds.model_dump()
        try:
            with repo.transaction():
                repo.cas("profiles", profile["profile_id"], profile.get("revision", 1), {"settings": upgraded})
        except ValueError:
            continue
        report["profiles"].append(profile["profile_id"])
        # Rules written for two classes may change meaning now that "medium" exists.
        if "member.runner" in json.dumps(profile.get("rules") or []):
            report["runner_rule_profiles"].append(profile["profile_id"])
    kept = [row for row in policy_rows if row.get("setting") not in LEGACY_RUNNER_SETTINGS]
    if len(kept) != len(policy_rows):
        with repo.transaction():
            repo.replace_tables(POLICY_OWNER, {"sap_policy_rules": kept})
        report["policy_rows_removed"] = len(policy_rows) - len(kept)
    if report["profiles"] or report["policy_rows_removed"]:
        log.warning("runner threshold migration: %s", report)
    return report


def migrate_cycle_stock_policy(repo):
    """Move the runner RoC multiple settings from SAP policy rows into every plant profile.

    Before the managed-complexity feature the SAP policy table held, per plant,
    the minimum multiplied coverage of medium / low runners, the medium batch
    factor and the cycle-stock cap. They now belong to the plant profile's
    ``cycle_stock_policy``, so the optimizer and SAP settings read the same
    numbers. After this migration:

    - every stored profile without ``cycle_stock_policy`` gets one, seeded from
      its plant's former SAP policy rows (``"*"`` rows, then plant rows, then
      defaults) with ``mode="single"`` (today's behaviour); each one is saved as
      a new profile revision;
    - the former rows are removed from ``sap_policy_rules``.

    Idempotent: upgraded profiles are skipped and, with no legacy rows left, the
    policy table is not rewritten. A profile changed concurrently by another
    process is skipped and picked up on the next start.

    Args:
        repo: Workspace repository (HANA or in-memory) exposing ``list``, ``cas``,
            ``rows``, ``replace_tables`` and ``transaction``.

    Returns:
        ``{"profiles": [ids upgraded], "policy_rows_removed": n}``.
    """
    import logging

    from production_wheel.runners import CycleStockPolicy
    from production_wheel.sap_settings.policy import LEGACY_CYCLE_STOCK_SETTINGS, legacy_cycle_stock_policy

    from .sap_settings_service import POLICY_OWNER

    log = logging.getLogger(__name__)
    policy_rows = repo.rows(POLICY_OWNER, "sap_policy_rules")
    report = {"profiles": [], "policy_rows_removed": 0}
    for profile in repo.list("profiles"):
        settings = profile.get("settings") or {}
        if "cycle_stock_policy" in settings:
            continue
        seeded = legacy_cycle_stock_policy(policy_rows, profile.get("plant") or "*")
        try:
            policy = CycleStockPolicy(**seeded)
        except ValueError:
            log.warning("profile %s: invalid former cycle-stock policy %s; using defaults",
                        profile["profile_id"], seeded)
            policy = CycleStockPolicy()
        try:
            with repo.transaction():
                repo.cas("profiles", profile["profile_id"], profile.get("revision", 1),
                         {"settings": {**settings, "cycle_stock_policy": policy.model_dump()}})
        except ValueError:
            continue
        report["profiles"].append(profile["profile_id"])
    kept = [row for row in policy_rows if row.get("setting") not in LEGACY_CYCLE_STOCK_SETTINGS]
    if len(kept) != len(policy_rows):
        with repo.transaction():
            repo.replace_tables(POLICY_OWNER, {"sap_policy_rules": kept})
        report["policy_rows_removed"] = len(policy_rows) - len(kept)
    if report["profiles"] or report["policy_rows_removed"]:
        log.warning("cycle stock policy migration: %s", report)
    return report


def migrate_sap_settings(repo):
    """Move the SAP policy rows into the ``sap_settings`` section of every plant profile.

    SAP settings rules used to be loose rows ``(plant | "*", setting, abc_class,
    value, number)`` edited as JSON. They now belong to the plant profile, edited
    in the Settings form or through the assistant's approval card. After this
    migration:

    - every stored profile (active or removed) without ``sap_settings`` gets one,
      resolved from its plant's rows (``"*"`` rows, then plant rows, then
      defaults) and saved as a new profile revision. Rows are validated one at a
      time and fields one at a time, so one bad value falls back to its default
      without discarding the others (each fallback is logged and reported);
    - the migrated rows are removed: a plant's rows once every profile of that
      plant has ``sap_settings``, ``"*"`` rows once every profile has them.
      Rows of plants without any profile, and rows with unknown settings, are
      kept and reported (``GET /api/sap-policy`` still lists them).

    Idempotent: upgraded profiles are skipped and, when nothing is removable,
    the policy table is not rewritten. A profile changed concurrently by another
    process is skipped, and its plant's rows are kept for the next start.

    Must run after ``migrate_runner_thresholds`` and ``migrate_cycle_stock_policy``,
    which remove the rows those profile sections took over.

    Args:
        repo: Workspace repository (HANA or in-memory) exposing ``list``, ``cas``,
            ``rows``, ``replace_tables`` and ``transaction``.

    Returns:
        ``{"profiles": [ids upgraded], "dropped": {profile_id: [notes]},
        "policy_rows_removed": n, "rows_kept": n}``.
    """
    import logging

    from production_wheel.sap_settings.policy import SETTINGS, resolve_policy, validate_rows
    from production_wheel.sap_settings.profile_policy import SapSettingsPolicy

    from .sap_settings_service import POLICY_OWNER

    log = logging.getLogger(__name__)
    policy_rows = repo.rows(POLICY_OWNER, "sap_policy_rules")
    report = {"profiles": [], "dropped": {}, "policy_rows_removed": 0, "rows_kept": 0}

    # Validate one row at a time: a malformed row is skipped, the others still migrate.
    valid_rows = []
    for row in policy_rows:
        if row.get("setting") not in SETTINGS:
            continue
        try:
            valid_rows.extend(validate_rows([row]))
        except ValueError as error:
            log.warning("SAP policy row %s is invalid and was not migrated: %s", row, error)

    profiles = repo.list("profiles")
    # Profiles that hold their own sap_settings after this run (already or now).
    done = {profile["profile_id"] for profile in profiles if "sap_settings" in (profile.get("settings") or {})}
    for profile in profiles:
        settings = profile.get("settings") or {}
        if profile["profile_id"] in done:
            continue
        plant = profile.get("plant") or "*"
        policy, notes = SapSettingsPolicy.from_resolved(resolve_policy(valid_rows, plant), strict=False)
        try:
            with repo.transaction():
                repo.cas("profiles", profile["profile_id"], profile.get("revision", 1),
                         {"settings": {**settings, "sap_settings": policy.model_dump()}})
        except ValueError:
            continue
        done.add(profile["profile_id"])
        report["profiles"].append(profile["profile_id"])
        if notes:
            report["dropped"][profile["profile_id"]] = notes

    # Rows are removed only where every profile that reads them now has its own copy.
    upgraded = {}
    for profile in profiles:
        plant = profile.get("plant") or "*"
        upgraded[plant] = upgraded.get(plant, True) and profile["profile_id"] in done
    every_profile_done = bool(profiles) and all(upgraded.values())

    def removable(row):
        """True when the row's settings now live in every profile that used it."""
        if row.get("setting") not in SETTINGS:
            return False
        plant = row.get("plant") or "*"
        return every_profile_done if plant == "*" else upgraded.get(plant, False)

    kept = [row for row in policy_rows if not removable(row)]
    report["rows_kept"] = len(kept)
    if len(kept) != len(policy_rows):
        with repo.transaction():
            repo.replace_tables(POLICY_OWNER, {"sap_policy_rules": kept})
        report["policy_rows_removed"] = len(policy_rows) - len(kept)
    if report["profiles"] or report["policy_rows_removed"]:
        log.warning("SAP settings migration: %s", report)
    return report
