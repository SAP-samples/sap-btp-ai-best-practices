"""Finite expression contracts for planner-authored conditional and aggregate rules."""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from production_wheel.runners import CYCLE_STOCK_STRATEGIES, RUNNER_CLASSES


FIELDS = frozenset("group.size group.common_lines group.selected_line group.pv_id group.pv_lot_litres group.batch_litres group.batch_count group.demand_litres group.coverage_days group.wheel_roc_days group.batches_per_week group.batches_per_day group.j_ch group.equal_pck group.cycle_stock_strategy member.fini_id member.volume member.pck member.runner member.runner_multiple member.sefi_ratio member.reference_coverage_days member.demand_litres member.xyz_x_dc_count member.xyz_y_dc_count member.xyz_z_dc_count member.primary_dc_count member.secondary_dc_count member.sales_scenario member.sales_network_scenario".split())


class RuleModel(BaseModel):
    """Reject unknown properties and nonfinite numbers at the rule boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Expression(RuleModel):
    """A bounded tree of literals, known fields, logic, arithmetic and member aggregates."""

    op: Literal["literal", "field", "and", "or", "not", "if", "eq", "ne", "gt", "gte", "lt", "lte", "in", "contains", "add", "sub", "mul", "div", "count", "sum", "min", "max", "distinct", "any", "all"]
    args: tuple["Expression", ...] = Field(default=(), max_length=20)
    field: str | None = Field(default=None, json_schema_extra={"enum": [None, *sorted(FIELDS)]})
    value: str | int | float | bool | list[str] | None = None

    @model_validator(mode="after")
    def validate_tree(self):
        """Check field names, operator arity and expression complexity."""
        unary = {"not", "count", "sum", "min", "max", "distinct", "any", "all"}
        # "if" is (condition, then, else); it turns a predicate into a number for measures.
        required = 0 if self.op in {"literal", "field"} else 1 if self.op in unary else 3 if self.op == "if" else 2
        if self.op in {"and", "or"}:
            if len(self.args) < 2:
                raise ValueError(f"{self.op} requires at least two arguments")
        elif len(self.args) != required:
            raise ValueError(f"{self.op} requires {required} arguments")
        if self.op == "field" and self.field not in FIELDS:
            raise ValueError(f"unknown business rule field: {self.field}")
        if self.op != "field" and self.field is not None:
            raise ValueError("field is only valid for the field operator")
        # A comparison against an unknown class or strategy would silently never match.
        for name, allowed in (("member.runner", RUNNER_CLASSES), ("group.cycle_stock_strategy", CYCLE_STOCK_STRATEGIES)):
            if self.op in {"eq", "ne", "in"} and any(a.op == "field" and a.field == name for a in self.args):
                for arg in self.args:
                    values = arg.value if isinstance(arg.value, list) else [arg.value]
                    unknown = [v for v in values if arg.op == "literal" and v not in allowed]
                    if unknown:
                        raise ValueError(f"{name} is one of {list(allowed)}, got {unknown}")
        if self.op != "literal" and self.value is not None:
            raise ValueError("value is only valid for literal operators")
        stack = [(self, 1)]
        count = 0
        while stack:
            node, depth = stack.pop()
            count += 1
            if depth > 12 or count > 200:
                raise ValueError("business expression exceeds depth/node limit")
            stack.extend((arg, depth + 1) for arg in node.args)
        return self


class RuleScope(RuleModel):
    """Plant and optional SEFI boundaries of one business rule."""

    plant: str | None = Field(default=None, min_length=1)
    sefi: str | None = Field(default=None, min_length=1)


class BusinessRule(RuleModel):
    """Auditable interpreted rule; execution requires approved status."""

    constraint_id: str = Field(min_length=1, max_length=128)
    scope: RuleScope = Field(default_factory=RuleScope)
    source_text: str | None = None
    # Plain-language summary written by the assistant and shown in the Settings list.
    description: str | None = Field(default=None, max_length=2000)
    approval_status: Literal["draft", "approved", "rejected"] = "draft"
    enforcement: Literal["hard"] = "hard"


class GroupRule(BusinessRule):
    """Require assertion for each candidate whose when predicate is true."""

    kind: Literal["group_rule"] = "group_rule"
    when: Expression = Field(default_factory=lambda: Expression(op="literal", value=True))
    assertion: Expression


class SelectionBound(BusinessRule):
    """Bound a linear sum of per-candidate measures over selected groups in scope."""

    kind: Literal["selection_bound"] = "selection_bound"
    measure: Expression
    lower: float | None = None
    upper: float | None = None

    @model_validator(mode="after")
    def validate_bounds(self):
        """Require a nonempty finite bound interval; equality is permitted."""
        if self.lower is None and self.upper is None:
            raise ValueError("selection bound requires lower or upper")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("selection bound lower exceeds upper")
        return self


BusinessConstraint = Annotated[Union[GroupRule, SelectionBound], Field(discriminator="kind")]


class MatrixPair(RuleModel):
    """One normalized positive volume pair with explicit preference/prohibition."""

    volume_a: str
    volume_b: str
    status: Literal["Y", "AVOID", "N"]

    @model_validator(mode="after")
    def validate_volumes(self):
        """Use decimal strings so equivalent volumes have identical identities."""
        from decimal import Decimal, InvalidOperation
        for field in ("volume_a", "volume_b"):
            try:
                value = Decimal(getattr(self, field))
            except InvalidOperation as exc:
                raise ValueError("matrix volumes must be numeric") from exc
            if not value.is_finite() or value <= 0:
                raise ValueError("matrix volumes must be finite and positive")
            object.__setattr__(self, field, format(value.normalize(), "f"))
        if self.volume_a == self.volume_b and self.status != "Y":
            raise ValueError("matrix diagonal must be compatible")
        return self
