import os
import random
from datetime import datetime
from typing import List, Tuple, Optional, Callable

from pydantic import BaseModel, Field


class FELBase(BaseModel):
    """Find Existing Location (FEL) Base class.
    """
    pass


class FEL0(FELBase):
    """
    A class representing the base route for Find Existing Location (FEL).

    Attributes
    ----------
    """
    layer1: str = Field(..., description="The first entity or layer.")
    layer2: str = Field(..., description="The second entity or layer to relate to. Can be empty")


class FEL1(FELBase):
    """
    A class representing a route for Find Existing Location (FEL) with a single layer and optional attribute filtering.

    Attributes
    ----------
    layer : str
        The entity or layer.
    where : str
        An optional SQL-like expression to perform attribute filtering.
    """
    layer: str = Field(
        ...,
        description="The entity or layer.",
    )
    where: str = Field(
        ...,
        description="An optional SQL like expression to perform attribute filtering.",
    )

    def to_fel0(self) -> FEL0:
        return FEL0(layer1=self.layer, layer2="")

    def layers(self) -> List[str]:
        """Method to return the layer in a list.
        """
        return [self.layer]


class FEL2(FELBase):
    """
    A class representing a route for Find Existing Location (FEL) with two layers and spatial relations.

    Attributes
    ----------
    layer1 : str
        The first entity or layer.
    where1 : str
        An optional SQL-like expression to perform attribute filtering on the first entity.
    layer2 : str
        The second entity or layer to relate to.
    where2 : str
        An optional SQL-like expression to perform attribute filtering on the second entity.
    relation : str
        The spatial relation between the entities.
    distance : float
        Optional spatial distance value.
    unit : str
        Optional spatial distance unit like 'meters', 'feet'.
    """
    layer1: str = Field(
        ...,
        description="The first entity or layer.",
    )
    where1: str = Field(
        ...,
        description="An optional SQL like expression to perform attribute filtering on the first entity.",
    )
    layer2: str = Field(
        ...,
        description="The second entity or layer to relate to.",
    )
    where2: str = Field(
        ...,
        description="An optional SQL like expression to perform attribute filtering on the second entity.",
    )
    relation: str = Field(
        ...,
        description="The spatial relation between the entities.",
    )
    distance: float = Field(
        ...,
        description="Optional spatial distance value.",
    )
    unit: str = Field(
        ...,
        description="Optional spatial distance unit like 'meters', 'feet'.",
    )

    def to_fel0(self) -> FEL0:
        return FEL0(layer1=self.layer1, layer2=self.layer2)

    def layers(self) -> List[str]:
        """Method to return the layers in a list.
        """
        return [self.layer1, self.layer2]


class FELLine(FELBase):
    """A class representing a line description and its corresponding FELBase instance.

    Attributes:
        line (str): The "english" description of the FEL.
        fel (FELBase): An instance of FELBase representing the FEL.
    """
    line: str = Field(..., description="The 'english' description of the FEL.")
    fel: FEL0 | FEL1 | FEL2 = Field(..., description="An instance of FELBase representing the FEL.")

    def to_fel0(self) -> "FELLine":
        """Convert the FEL1 or FEL2 instance to a FEL0 instance.
        """
        return FELLine(line=self.line, fel=self.fel.to_fel0())


class Column(BaseModel):
    """
    A class representing a column in a geospatial layer.

    Attributes
    ----------
    name : str
        The name of the column.
    alias : str
        The alias of the column.
    dtype : str
        The data type of the column.
    hints: List[str]
        A list of hints for the column.
    values : List[str]
        The list of values in the column.
    """
    name: str
    alias: str
    dtype: str  # Data type
    hints: List[str] = Field(default_factory=list)
    values: List[str] = Field(default_factory=list)

    @property
    def has_values(self) -> bool:
        """Return True if the column has values, False otherwise.
        """
        return bool(self.values)

    @property
    def attributes(self) -> str:
        """Return the attributes of the column as a string.
        """
        return f"{self.name} ({self.dtype}): {self.alias}"

    def __add__(self, that: 'Column') -> 'Column':
        """Add two Column instances together.
        Here, we combine the hints and values of both columns.

        :param that: The other Column instance to add.
        :return: A new Column instance with combined attributes.
        """
        return Column(
            name=self.name,
            alias=self.alias,  # TODO: Or that alias? for update!
            dtype=self.dtype,
            hints=list(set(self.hints + that.hints)),
            values=list(set(self.values + that.values)),
        )


class Layer(BaseModel):
    """
    A class representing a geospatial layer.

    Attributes
    ----------
    name : str
        The name of the layer.
    alias : str
        The alias of the layer.
    stype : str
        The spatial type of the layer.
    hints : List[str]
        A list of hints for the layer.
    columns : List[Column]
        The list of columns in the layer.
    uri : str
        The URI of the layer.
    """
    name: str
    alias: str
    stype: str  # Spatial type
    hints: List[str] = Field(default_factory=list)
    columns: List[Column] = Field(default_factory=list)
    uri: str

    def __add__(self, that: 'Layer') -> 'Layer':
        """Add two Layer instances together.

        :param that: The other Layer instance to add.
        :return: A new Layer instance with combined attributes.
        """
        name_column = {_.name: _ for _ in self.columns}
        for column in that.columns:
            if column.name not in name_column:
                name_column[column.name] = column
            else:
                name_column[column.name] += column

        return Layer(
            name=self.name,
            alias=that.alias,
            stype=that.stype,
            hints=list(set(self.hints + that.hints)),
            columns=list(name_column.values()),
            uri=that.uri,
        )

    @property
    def column_names(self) -> List[str]:
        """Get column names without geometry columns.

        :return: List[str]
        """
        return [_.name for _ in self.columns if _.dtype != "Geometry"]

    @property
    def columns_no_geom(self) -> List[Column]:
        """Get columns without geometry columns.

        :return: List[Column]
        """
        return [_ for _ in self.columns if _.dtype != "Geometry"]

    @property
    def attributes(self) -> str:
        """Create 'attributes' string to be used in a system prompt.
        """
        line = [
            f"Layer: {self.alias}",
            f"Geometry: {self.stype}",
            "Attributes:"
        ]
        line.extend([f"- {_.attributes}" for _ in self.columns_no_geom])
        if self.hints:
            line.append("Hints:")
            for hint in self.hints:
                line.append(f"- {hint}")
        for column in self.columns_no_geom:
            if column.hints:
                for hint in column.hints:
                    line.append(f"- {hint}")
        line.append("\n")
        return "\n".join(line)

    @property
    def has_columns(self) -> bool:
        """Property to check if the Layer object has columns.
        """
        return bool(self.columns)

    def find_column(self, name: str) -> Optional[Column]:
        """Find a column by name.

        :param name: The name of the column.
        :return: Optional[Column]
        """
        return next(filter(lambda _: _.name == name, self.columns), None)

    def prune_columns(self) -> "Layer":
        """Return a Layer instance with pruned columns with no values.
        """
        columns = [_ for _ in self.columns if _.has_values]
        return Layer(
            name=self.name,
            alias=self.alias,
            stype=self.stype,
            hints=self.hints,
            columns=columns,
            uri=self.uri
        )


class Layers(BaseModel):
    """
    A class representing a collection of geospatial layers.

    Attributes
    ----------
    layers : List[Layer]
        The list of geospatial layers.
    """
    layers: List[Layer] = Field(default_factory=list)

    def __add__(self, that: 'Layers') -> 'Layers':
        """Add two Layers instances together.

        :param that: The other Layers instance to add.
        :return: A new Layers instance with combined attributes.
        """
        name_layer = {_.name: _ for _ in self.layers}
        for layer in that.layers:
            if layer.name not in name_layer:
                name_layer[layer.name] = layer
            else:
                name_layer[layer.name] += layer

        return Layers(layers=list(name_layer.values()))

    @property
    def has_layers(self) -> bool:
        """Property to check if the Layers object has layers.
        """
        return bool(self.layers)

    @staticmethod
    def load(filename: str) -> 'Layers':
        """Static method to load a JSON file into a Layers object.
        """
        with open(
                os.path.expanduser(filename),
                mode="r",
                encoding="UTF-8",
        ) as fp:
            return Layers.model_validate_json(fp.read())

    def prune(self) -> 'Layers':
        """Return a Layers instance with pruned layers with no columns.
        """
        return Layers(layers=self.prune_layers())

    def prune_layers(self) -> List[Layer]:
        """Return list of Layer instances with pruned columns with no values.
        """
        layers = [layer.prune_columns() for layer in self.layers]
        return [_ for _ in layers if _.has_columns]

    def dump(self, filename: str, indent: int = 2) -> None:
        """Method to dump the Layers object to a JSON file.
        """
        with open(
                os.path.expanduser(filename),
                mode="w",
                encoding="UTF-8",
        ) as fp:
            fp.write(self.model_dump_json(indent=indent))


class FEL:
    def __init__(
            self,
            layers: Layers | List[Layer],
            case_insensitive: bool = False,
    ) -> None:
        """Initialize the FEL class.

        :param layers: The list of geospatial layers or Layers instance.
        :param case_insensitive: If True, treat the column values insensitively.
        """
        match layers:
            case list() if all(isinstance(_, Layer) for _ in layers):
                self.layers = layers
            case Layers():
                self.layers = layers.layers
            case _:
                self.layers = []
        self.case_insensitive = case_insensitive

    def __call__(self) -> FELLine:
        """Call create_line_0 method to generate a line.

        :return: FELLine instance.
        """
        return self.create_line_0()

    def attributes(self) -> str:
        """Method to return the attributes of the layers to be used in a system message.
        """
        return "\n".join([_.attributes for _ in self.layers])

    def create_text_eq(self, line: List[str], column: Column) -> str:
        name = column.name
        data = random.choice(column.values)
        line.append("is")
        line.append(data.lower())
        data = data.replace("'", "\\'")
        return f"lower({name}) LIKE lower('%{data}%')" if self.case_insensitive else f"{name} = '{data}'"

    def create_text_ne(self, line: List[str], column: Column) -> str:
        name = column.name
        data = random.choice(column.values)
        line.append(random.choice([
            "is not",
            "is different from",
            "is not equal to",
            "!=",
            "<>",
        ]))
        line.append(data.lower())
        data = data.replace("'", "\\'")
        return f"lower({name}) <> lower('{data}')" if self.case_insensitive else f"{name} <> '{data}'"

    def create_text_like(self, line: List[str], column: Column) -> str:
        name = column.name
        data = random.choice(column.values)
        data = random.choice(data.split(" "))
        line.append(random.choice([
            "is like",
            # "includes",
            "contains",
            "that match the pattern",
        ]))
        line.append(data.lower())
        data = data.replace("'", "\\'")
        return f"lower({name}) LIKE lower('%{data}%')" if self.case_insensitive else f"{name} LIKE '%{data}%'"

    def create_text_not_like(self, line: List[str], column: Column) -> str:
        name = column.name
        data = random.choice(column.values)
        data = random.choice(data.split(" "))
        line.append(random.choice([
            "is not like",
            "does not include",
            "does not contain",
            "excludes",
            "that does not match the pattern",
        ]))
        ## line.append("is not like")
        line.append(data.lower())
        data = data.replace("'", "\\'")
        return f"lower({name}) NOT LIKE lower('%{data}%')" if self.case_insensitive else f"{name} NOT LIKE '%{data}%'"

    def create_text_starts(self, line: List[str], column: Column) -> str:
        name = column.name
        data = random.choice(column.values)
        data = data.split(" ")[0]
        line.append(random.choice([
            "starts with",
            "starting with",
            "prefixes with",
            "beginning with",
        ]))
        line.append(data.lower())
        data = data.replace("'", "\\'")
        return f"lower({name}) LIKE lower('{data}%')" if self.case_insensitive else f"{name} LIKE '{data}%'"

    def create_text_ends(self, line: List[str], column: Column) -> str:
        name = column.name
        data = random.choice(column.values)
        data = data.split(" ")[-1]
        line.append(random.choice([
            "ends with",
            "ending with",
            "suffixes with",
            "terminating with",
            "terminated with",
        ]))
        line.append(data.lower())
        data = data.replace("'", "\\'")
        return f"lower({name}) LIKE lower('%{data}')" if self.case_insensitive else f"{name} LIKE '%{data}'"

    def create_text_blank(self, line: List[str], column: Column) -> str:
        name = column.name
        line.append(random.choice([
            "is blank",
            "is unspecified or empty",
            "is unspecified",
            "is empty",
        ]))
        return f"{name} = ''"

    def create_text_not_blank(self, line: List[str], column: Column) -> str:
        name = column.name
        line.append(random.choice([
            "is not blank",
            "is specified (not blank)",
        ]))
        return f"{name} <> ''"

    def create_text(self, line: List[str], column: Column) -> str:
        oper = random.choice([
            self.create_text_eq,
            self.create_text_ne,
            self.create_text_like,
            self.create_text_not_like,
            self.create_text_starts,
            self.create_text_ends,
            self.create_text_blank,
            self.create_text_not_blank,
        ])
        return oper(line, column)

    def create_nume_eq(self, line: List[str], column: Column) -> str:
        name = column.name
        data = random.choice(column.values)
        line.append(random.choice(["is", "="]))
        line.append(data)
        return f"{name} = {data}"

    def create_nume_ne(self, line: List[str], column: Column) -> str:
        name = column.name
        data = random.choice(column.values)
        line.append(random.choice([
            "is not",
            "<>",
            "!=",
            "excludes",
            "is different from",
        ]))
        line.append(data)
        return f"{name} <> {data}"

    def create_nume_lt(self, line: List[str], column: Column) -> str:
        name = column.name
        data = random.choice(column.values)
        line.append(random.choice([
            "<",
            "is less than",
            "is below",
            "is smaller than",
            "below",
            "not above",
            "not exceeding",
        ]))
        line.append(data)
        return f"{name} < {data}"

    def create_nume_gt(self, line: List[str], column: Column) -> str:
        name = column.name
        data = random.choice(column.values)
        line.append(random.choice([
            ">",
            "is greater than",
            "is above",
            "above",
            "exceeding",
            "larger than",
            "not below",
        ]))
        line.append(data)
        return f"{name} > {data}"

    def create_nume_bt(self, line: List[str], column: Column) -> str:
        name = column.name
        data1 = random.choice(column.values)
        data2 = random.choice(column.values)
        op1, op2 = random.choice([
            ("between", "and"),
            ("ranging from", "to"),
            ("from", "to"),
        ])
        line.append(op1)
        line.append(data1)
        line.append(op2)
        line.append(data2)
        # return f"{name} >= {data1} AND {name} < {data2}"
        return f"{name} BETWEEN {data1} AND {data2}"

    def create_nume(self, line: List[str], column: Column) -> str:
        oper = random.choice([
            self.create_nume_eq,
            self.create_nume_ne,
            self.create_nume_lt,
            self.create_nume_gt,
            self.create_nume_bt,
        ])
        return oper(line, column)

    def create_date_text(self, column: Column) -> Tuple[str, str]:
        patterns = [
            "%Y",
            "%b %Y",
            "%b %y",
            "%B %Y",
            "%B %y",
            # Add more.
        ]
        value = random.choice(column.values)
        ptime = datetime.strptime(value[:19], "%Y-%m-%d %H:%M:%S")
        choice = random.choice(patterns)
        ftime = datetime.strftime(ptime, choice)
        ptime = datetime.strptime(ftime, choice)
        value = datetime.strftime(ptime, "%Y-%m-%d")
        return value, ftime,

    def create_time_text(self, column: Column) -> Tuple[str, str]:
        patterns = [
            "%H:%M",
            "%I:%M %p",
            "%I:%M",
            "%H:%M:%S",
            # Add more.
        ]
        # Get a random time value.
        value = random.choice(column.values)
        # parse it into a datetime object.
        ptime = datetime.strptime(value, "%H:%M:%S")
        # Get a random pattern.
        choice = random.choice(patterns)
        # Format the time value base on a pattern.
        ftime = datetime.strftime(ptime, choice)
        ptime = datetime.strptime(ftime, choice)
        value = datetime.strftime(ptime, "%H:%M:%S")
        return value, ftime,

    def create_date_eq(
            self,
            line: List[str],
            column: Column,
            prefix: str = "timestamp",
    ) -> str:
        name = column.name
        data, text = self.create_date_text(column)
        line.append(random.choice(["=", "is", "on", "is on"]))
        line.append(text)
        return f"{name} = {prefix} '{data}'"

    def create_time_eq(
            self,
            line: List[str],
            column: Column,
            prefix: str = "time",
    ) -> str:
        name = column.name
        data, text = self.create_time_text(column)
        line.append(random.choice(["=", "is", "on", "is on"]))
        line.append(text)
        return f"{name} = {prefix} '{data}'"

    def create_date_ne(
            self,
            line: List[str],
            column: Column,
            prefix: str = "timestamp",
    ) -> str:
        name = column.name
        data, text = self.create_date_text(column)
        line.append(random.choice(["!=", "<>", "is not", "is not on"]))
        line.append(text)
        return f"{name} <> {prefix} '{data}'"

    def create_time_ne(
            self,
            line: List[str],
            column: Column,
            prefix: str = "time",
    ) -> str:
        name = column.name
        data, text = self.create_time_text(column)
        line.append(random.choice(["!=", "<>", "is not", "is not on"]))
        line.append(text)
        return f"{name} <> {prefix} '{data}'"

    def create_date_lt(
            self,
            line: List[str],
            column: Column,
            prefix: str = "timestamp",
    ) -> str:
        name = column.name
        data, text = self.create_date_text(column)
        line.append(random.choice(["<", "is before", "is not on or after"]))
        line.append(text)
        return f"{name} < {prefix} '{data}'"

    def create_time_lt(
            self,
            line: List[str],
            column: Column,
            prefix: str = "time",
    ) -> str:
        name = column.name
        data, text = self.create_time_text(column)
        line.append(random.choice(["<", "is before", "is not on or after"]))
        line.append(text)
        return f"{name} < {prefix} '{data}'"

    def create_date_gt(
            self,
            line: List[str],
            column: Column,
            prefix: str = "timestamp",
    ) -> str:
        name = column.name
        data, text = self.create_date_text(column)
        line.append(random.choice([">", "is after", "is not on or before"]))
        line.append(text)
        return f"{name} > {prefix} '{data}'"

    def create_time_gt(
            self,
            line: List[str],
            column: Column,
            prefix: str = "time",
    ) -> str:
        name = column.name
        data, text = self.create_time_text(column)
        line.append(random.choice([">", "is after", "is not on or before"]))
        line.append(text)
        return f"{name} > {prefix} '{data}'"

    def create_date_bt(
            self,
            line: List[str],
            column: Column,
            prefix: str = "timestamp",
    ) -> str:
        name = column.name
        data1, text1 = self.create_date_text(column)
        data2, text2 = self.create_date_text(column)
        line.append("between")
        line.append(text1)
        line.append("and")
        line.append(text2)
        # return f"{name} >= timestamp '{data1}' AND {name} < timestamp '{data2}'"
        return f"{name} BETWEEN {prefix} '{data1}' AND {prefix} '{data2}'"

    def create_time_bt(
            self,
            line: List[str],
            column: Column,
            prefix: str = "time",
    ) -> str:
        name = column.name
        data1, text1 = self.create_time_text(column)
        data2, text2 = self.create_time_text(column)
        line.append("between")
        line.append(text1)
        line.append("and")
        line.append(text2)
        # return f"{name} >= timestamp '{data1}' AND {name} < timestamp '{data2}'"
        return f"{name} BETWEEN {prefix} '{data1}' AND {prefix} '{data2}'"

    def create_date(self, line: List[str], column: Column) -> str:
        oper = random.choice([
            self.create_date_eq,
            self.create_date_ne,
            self.create_date_lt,
            self.create_date_gt,
            self.create_date_bt,
        ])
        return oper(line, column)

    def create_time_only(self, line: List[str], column: Column) -> str:
        oper = random.choice([
            self.create_time_eq,
            self.create_time_ne,
            self.create_time_lt,
            self.create_time_gt,
            self.create_time_bt,
        ])
        return oper(line, column)

    def create_line_fel(
            self,
            col_min: int = 1,
            col_max: int = 2,
            layer: Optional[Layer] = None
    ) -> FELLine:
        if layer is None:
            layer = random.choice(self.layers)
        line = [layer.alias.lower()]
        expr = []

        columns = random.choices(layer.columns_no_geom, k=random.randint(col_min, col_max))

        for index, column in enumerate(columns):
            if index == 0:
                line.append(random.choice(["where", "with"]))
            else:
                oper1, oper2 = random.choice([
                    ("and", "and"),
                    ("and where", "and"),
                    ("or", "or"),
                    ("or where", "or"),
                ])
                line.append(oper1)
                expr.append(oper2)
            line.append(column.alias)
            if column.dtype == "String":
                expr.append(self.create_text(line, column))
            elif column.dtype == "Date":
                expr.append(self.create_date(line, column))
            elif column.dtype == "TimeOnly":
                expr.append(self.create_time_only(line, column))
            else:
                expr.append(self.create_nume(line, column))

        return FELLine(line=" ".join(line), fel=FEL1(layer=layer.name, where=" ".join(expr)))

    def create_line_1(
            self,
            col_min: int = 1,
            layer: Optional[Layer] = None
    ) -> FELLine:
        """Create and return an FELLine instance with fel being an instance of FEL1.

        :param col_min: Minimum number of columns to include in the line.
        :param layer: Optional layer to use. If not provided, a random layer will be chosen.
        :return: An FELLine instance.
        """
        show = random.choice([
            "Show",
            "Show all",
            "Find",
            "Find all",
            "Locate",
            "Locate all",
            "Identify",
            "Identify all",
            "List",
            "List all",
        ])
        line_expr = self.create_line_fel(col_min=col_min, layer=layer)
        line_expr.line = f"{show} {line_expr.line}"
        return line_expr

    def create_line_0(self) -> FELLine:
        return random.choice([self.create_line_1, self.create_line_2])()

    def create_intersects(self, lhs: FELLine, rhs: FELLine) -> FELLine:
        line = " ".join([lhs.line, "that intersect", rhs.line])
        fel2 = FEL2(
            layer1=lhs.fel.layer,
            where1=lhs.fel.where,
            layer2=rhs.fel.layer,
            where2=rhs.fel.where,
            relation="intersects",
            distance=0.0,
            unit="",
        )
        return FELLine(line=line, fel=fel2)

    def create_not_intersects(self, lhs: FELLine, rhs: FELLine) -> FELLine:
        line = " ".join([lhs.line, "that do not intersect", rhs.line])
        fel2 = FEL2(
            layer1=lhs.fel.layer,
            where1=lhs.fel.where,
            layer2=rhs.fel.layer,
            where2=rhs.fel.where,
            relation="notIntersects",
            distance=0.0,
            unit="",
        )
        return FELLine(line=line, fel=fel2)

    def create_near(self, lhs: FELLine, rhs: FELLine) -> FELLine:
        pref = random.choice([
            "nearest",
            "closest",
        ])
        line = " ".join([lhs.line,
                         pref,
                         "to",
                         rhs.line])
        fel2 = FEL2(
            layer1=lhs.fel.layer,
            where1=lhs.fel.where,
            layer2=rhs.fel.layer,
            where2=rhs.fel.where,
            relation="near",
            distance=20.0,
            unit="miles",
        )
        return FELLine(line=line, fel=fel2)

    def create_within_distance(self, lhs: FELLine, rhs: FELLine) -> FELLine:
        dist = int(random.uniform(0, 100) * 10.0) / 10.0
        unit = random.choice(["meters", "kilometers", "feet", "miles"])
        pref = random.choice([
            "that are within a distance of",
            "that are within",
            "that are in the range of",
        ])
        line = " ".join([lhs.line,
                         pref,
                         str(dist),
                         unit,
                         "of",
                         rhs.line])
        fel2 = FEL2(
            layer1=lhs.fel.layer,
            where1=lhs.fel.where,
            layer2=rhs.fel.layer,
            where2=rhs.fel.where,
            relation="withinDistance",
            distance=dist,
            unit=unit,
        )
        return FELLine(line=line, fel=fel2)

    def create_contains(self, lhs: FELLine, rhs: FELLine) -> FELLine:
        line = " ".join([lhs.line, "that contain", rhs.line])
        fel2 = FEL2(
            layer1=lhs.fel.layer,
            where1=lhs.fel.where,
            layer2=rhs.fel.layer,
            where2=rhs.fel.where,
            relation="contains",
            distance=0.0,
            unit="",
        )
        return FELLine(line=line, fel=fel2)

    def create_not_contains(self, lhs: FELLine, rhs: FELLine) -> FELLine:
        line = " ".join([lhs.line, "that do not contain", rhs.line])
        fel2 = FEL2(
            layer1=lhs.fel.layer,
            where1=lhs.fel.where,
            layer2=rhs.fel.layer,
            where2=rhs.fel.where,
            relation="notContains",
            distance=0.0,
            unit="",
        )
        return FELLine(line=line, fel=fel2)

    def create_within(self, lhs: FELLine, rhs: FELLine) -> FELLine:
        line = " ".join([
            lhs.line,
            random.choice([
                "that are within",
                "that are inside",
                "that are in",
            ]),
            rhs.line
        ])
        fel2 = FEL2(
            layer1=lhs.fel.layer,
            where1=lhs.fel.where,
            layer2=rhs.fel.layer,
            where2=rhs.fel.where,
            relation="within",
            distance=0.0,
            unit="",
        )
        return FELLine(line=line, fel=fel2)

    def create_not_within(self, lhs: FELLine, rhs: FELLine) -> FELLine:
        line = " ".join([lhs.line, "that are not within", rhs.line])
        fel2 = FEL2(
            layer1=lhs.fel.layer,
            where1=lhs.fel.where,
            layer2=rhs.fel.layer,
            where2=rhs.fel.where,
            relation="notWithin",
            distance=0.0,
            unit="",
        )
        return FELLine(line=line, fel=fel2)

    def create_not_within_distance(self, lhs: FELLine, rhs: FELLine) -> FELLine:
        dist = random.uniform(0, 100)
        unit = random.choice(["meters", "kilometers", "feet", "miles"])
        pref = random.choice([
            "that are not within a distance of",
            "that are not within",
            "that are not in the range of",
        ])
        line = " ".join([lhs.line,
                         pref,
                         str(int(dist * 10.0) / 10.0),
                         unit,
                         "of",
                         rhs.line])
        fel2 = FEL2(
            layer1=lhs.fel.layer,
            where1=lhs.fel.where,
            layer2=rhs.fel.layer,
            where2=rhs.fel.where,
            relation="notWithinDistance",
            distance=dist,
            unit=unit,
        )
        return FELLine(line=line, fel=fel2)

    def select_stype(self, line_fel: FELLine) -> str:
        """Get the spatial type of the layer from the line_fel.

        :param line_fel: The FELLine instance.
        :return: The spatial type of the layer.
        """
        layer = next(filter(lambda _: _.name == line_fel.fel.layer, self.layers))
        return layer.stype

    def select_oper(
            self,
            lhs: FELLine,
            rhs: FELLine,
    ) -> Callable[[FELLine, FELLine], FELLine]:
        stype_lhs = self.select_stype(lhs)
        stype_rhs = self.select_stype(rhs)
        if stype_lhs == "Point" and stype_rhs == "Point":
            return random.choice([
                self.create_within_distance,
                # self.create_not_within_distance,
                # self.create_near,
            ])
        if stype_lhs == "Point" and stype_rhs == "Polyline":
            return random.choice([
                self.create_within_distance,
                # self.create_not_within_distance,
                # self.create_near,
            ])
        if stype_lhs == "Point" and stype_rhs == "Polygon":
            return random.choice([
                self.create_within_distance,
                # self.create_not_within_distance,
                self.create_within,
                # self.create_not_within,
                # self.create_near,
            ])
        if stype_lhs == "Polyline" and stype_rhs == "Point":
            return random.choice([
                self.create_within_distance,
                # self.create_not_within_distance,
                # self.create_near,
            ])
        if stype_lhs == "Polyline" and stype_rhs == "Polyline":
            return random.choice([
                self.create_within_distance,
                # self.create_not_within_distance,
                self.create_intersects,
                # self.create_not_intersects,
                # self.create_near,
            ])
        if stype_lhs == "Polyline" and stype_rhs == "Polygon":
            return random.choice([
                self.create_within_distance,
                # self.create_not_within_distance,
                self.create_intersects,
                # self.create_not_intersects,
                self.create_within,
                # self.create_not_within,
                # self.create_near,
            ])
        if stype_lhs == "Polygon" and stype_rhs == "Point":
            return random.choice([
                self.create_within_distance,
                # self.create_not_within_distance,
                self.create_contains,
                # self.create_not_contains,
                # self.create_near,
            ])
        if stype_lhs == "Polygon" and stype_rhs == "Polyline":
            return random.choice([
                self.create_within_distance,
                # self.create_not_within_distance,
                self.create_contains,
                # self.create_not_contains,
                # self.create_near,
            ])
        return random.choice([
            self.create_within_distance,
            # self.create_not_within_distance,
            self.create_intersects,
            # self.create_not_intersects,
            self.create_contains,
            # self.create_not_contains,
            self.create_within,
            # self.create_not_within,
            # self.create_near,
        ])

    def create_line_2(
            self,
            layer_lhs: Optional[Layer] = None,
            layer_rhs: Optional[Layer] = None,
    ) -> FELLine:
        """Create and return an FELLine instance with fel is and instance of FEL2.

        :param layer_lhs: The left-hand side layer.
        :param layer_rhs: The right-hand side layer.
        :return: FELLine instance.
        """
        fel1 = self.create_line_1(col_min=0, layer=layer_lhs)
        fel2 = self.create_line_fel(col_min=0, layer=layer_rhs)
        oper = self.select_oper(fel1, fel2)
        return oper(fel1, fel2)
