"""Lightweight schema model parsed from real CREATE TABLE DDL — used by spider_source.py
for DB-selection FK scoring and by corrupt.py's misjoined_tables corruption ("swap to a
wrong-but-plausible column/table"). Deliberately independent of BaseAdapter/Example
(rm/data/base.py) to avoid an import cycle, matching sql_exec.py's role as a shared,
data-format-agnostic primitive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sqlglot
import sqlglot.expressions as exp


@dataclass(frozen=True)
class Column:
    name: str
    data_type: str
    is_primary_key: bool


@dataclass(frozen=True)
class ForeignKey:
    table: str
    column: str
    ref_table: str
    ref_column: str


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]

    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)


@dataclass(frozen=True)
class Schema:
    tables: dict[str, Table]
    foreign_keys: tuple[ForeignKey, ...]

    def table_names(self) -> tuple[str, ...]:
        return tuple(self.tables.keys())

    def columns_of(self, table: str) -> tuple[Column, ...]:
        t = self.tables.get(table)
        return t.columns if t is not None else ()


def _table_level_pk_columns(tree: exp.Create) -> set[str]:
    names: set[str] = set()
    for pk in tree.find_all(exp.PrimaryKey):
        names.update(e.name for e in pk.expressions)
    return names


def _parse_table(tree: exp.Create) -> Table | None:
    table_expr = tree.this
    if not isinstance(table_expr, exp.Schema) or not isinstance(table_expr.this, exp.Table):
        return None
    table_name = table_expr.this.this.name

    table_level_pks = _table_level_pk_columns(tree)
    columns = []
    for col_def in table_expr.expressions:
        if not isinstance(col_def, exp.ColumnDef):
            continue
        name = col_def.this.name
        kind = col_def.args.get("kind")
        data_type = kind.this.name if kind is not None and kind.this is not None else ""
        constraints = col_def.args.get("constraints") or []
        is_pk = name in table_level_pks or any(
            isinstance(con.kind, exp.PrimaryKeyColumnConstraint) for con in constraints
        )
        columns.append(Column(name=name, data_type=str(data_type), is_primary_key=is_pk))
    return Table(name=table_name, columns=tuple(columns))


def _parse_foreign_keys(tree: exp.Create, table_name: str) -> list[ForeignKey]:
    fks = []
    for fk in tree.find_all(exp.ForeignKey):
        columns = [e.name for e in fk.expressions]
        reference = fk.args.get("reference")
        if reference is None or not isinstance(reference.this, exp.Schema):
            continue
        ref_schema = reference.this
        ref_table = ref_schema.this.name if isinstance(ref_schema.this, exp.Table) else str(ref_schema.this)
        ref_columns = [e.name for e in ref_schema.expressions]
        for i, column in enumerate(columns):
            ref_column = ref_columns[i] if i < len(ref_columns) else (ref_columns[0] if ref_columns else "")
            fks.append(ForeignKey(table=table_name, column=column, ref_table=ref_table, ref_column=ref_column))
    return fks


def build_schema(create_statements: Sequence[str]) -> Schema:
    """Parses each statement with sqlglot; statements that don't parse or don't parse to
    a CREATE TABLE are skipped (same lenient-skip convention as sql_exec._is_insert), so a
    handful of unusual DDL in a real Spider DB can't abort the whole schema build."""
    tables: dict[str, Table] = {}
    foreign_keys: list[ForeignKey] = []
    for statement in create_statements:
        try:
            tree = sqlglot.parse_one(statement, dialect="sqlite")
        except Exception:
            continue
        if not isinstance(tree, exp.Create) or tree.kind != "TABLE":
            continue
        table = _parse_table(tree)
        if table is None:
            continue
        tables[table.name] = table
        foreign_keys.extend(_parse_foreign_keys(tree, table.name))
    return Schema(tables=tables, foreign_keys=tuple(foreign_keys))
