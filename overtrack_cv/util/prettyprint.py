from dataclasses import Field, dataclass, fields, is_dataclass
from pprint import PrettyPrinter, _recursion
from typing import List, Optional

try:
    from pynamodb.attributes import MapAttribute
    from pynamodb.models import Model as PynamoModel
except:

    class PynamoModel:
        pass

    MapAttribute = PynamoModel


@dataclass
class PynamoField:
    name: str


class DataclassPrettyPrinter(PrettyPrinter):
    def __init__(self, *args, **kwargs):
        if "force_use_repr" in kwargs:
            self.force_use_repr_types = kwargs.pop("force_use_repr")
        else:
            self.force_use_repr_types = None
        kwargs["width"] = kwargs.get("width", 120)
        super().__init__(*args, **kwargs)
        self.visited = {}

    def allow_use_repr(self, object):
        return True

    def force_use_repr(self, object):
        return self.force_use_repr_types and type(object) in self.force_use_repr_types

    def _format(self, object, stream, indent, allowance, context, level):
        if level == 0:
            self.visited.clear()

        objid = id(object)
        if objid in context:
            stream.write(_recursion(object))
            self._recursive = True
            self._readable = False
            return
        rep = self._repr(object, context, level)
        max_width = self._width - indent - allowance

        if isinstance(object, (PynamoModel, MapAttribute)) and type(object) not in (
            self.force_use_repr_types or []
        ):
            # Don't use default repr
            rep = (
                f"{object.__class__.__qualname__}("
                + ", ".join(
                    f"{k}={self._repr(getattr(object, k), context, level + 1)}" for k in object._attributes.keys()
                )
                + ")"
            )

        if not self.force_use_repr(object) and (not self.allow_use_repr(object) or len(rep) > max_width):
            p = self._dispatch.get(type(object).__repr__, None)
            # Modification: use custom _pprint_dict before using from _dispatch
            # Modification: add _pprint_dataclass
            if isinstance(object, list):
                context[objid] = 1
                self._pprint_list(object, stream, indent, allowance, context, level + 1)
                del context[objid]
                return
            if isinstance(object, dict):
                context[objid] = 1
                self._pprint_dict(object, stream, indent, allowance, context, level + 1)
                del context[objid]
                return
            elif is_dataclass(object):
                context[objid] = 1
                self._pprint_dataclass(object, stream, indent, allowance, context, level + 1)
                del context[objid]
                return
            elif isinstance(object, (PynamoModel, MapAttribute)):
                context[objid] = 1
                self._pprint_pynamo_model(object, stream, indent, allowance, context, level + 1)
                del context[objid]
                return
            elif p is not None:
                context[objid] = 1
                p(self, object, stream, indent, allowance, context, level + 1)
                del context[objid]
                return
        stream.write(rep)

    def _pprint_dict(self, object, stream, indent, allowance, context, level):
        write = stream.write
        write("{")
        if self._indent_per_level > 1:
            write((self._indent_per_level - 1) * " ")
        length = len(object)
        if length:
            # Modification: don't sort dictionaries
            # Modern python keeps key order to creation/insertion order, and this order often has meaning
            items = object.items()
            self._format_dict_items(items, stream, indent, allowance + 1, context, level)
        write("}")

    def _format_dict_items(self, items, stream, indent, allowance, context, level):
        write = stream.write
        indent += self._indent_per_level
        max_width = self._width - indent - allowance
        delimnl = ",\n" + " " * indent
        last_index = len(items) - 1
        for i, (key, ent) in enumerate(items):
            last = i == last_index
            rep = self._repr(key, context, level)
            if len(rep) < max_width:
                write(rep)
                write(": ")
                self._format(
                    ent,
                    stream,
                    indent + len(rep) + 2,
                    allowance if last else 1,
                    context,
                    level,
                )
            else:
                self._format(key, stream, indent, allowance if last else 1, context, level)
                write(": ")
                self._format(ent, stream, indent + 2, allowance if last else 1, context, level)
            if not last:
                write(delimnl)

    def _pprint_dataclass(self, object, stream, indent, allowance, context, level, respect_repr_hint=True):
        write = stream.write
        write(f"{object.__class__.__qualname__}(")
        if self._indent_per_level > 1:
            write((self._indent_per_level - 1) * " ")
        object_fields: List[Field] = [f for f in fields(object) if f.repr or not respect_repr_hint]
        if len(object_fields):
            self._format_dataclass_fields(object, object_fields, stream, indent, allowance + 1, context, level)
        write(")")

    def _format_dataclass_fields(
        self,
        object,
        object_fields: List[Field],
        stream,
        indent,
        allowance,
        context,
        level,
    ):
        write = stream.write
        indent += self._indent_per_level
        write("\n" + " " * indent)
        delimnl = ",\n" + " " * indent
        last_index = len(object_fields) - 1
        for i, field in enumerate(object_fields):
            last = i == last_index
            write(field.name)
            write("=")
            self._format(
                getattr(object, field.name),
                stream,
                indent + len(field.name) + 1,
                allowance if last else 1,
                context,
                level,
            )
            if not last:
                write(delimnl)
        indent -= self._indent_per_level
        write("\n" + " " * indent)

    def _pprint_pynamo_model(self, object, stream, indent, allowance, context, level):
        write = stream.write
        write(f"{object.__class__.__qualname__}(")
        if self._indent_per_level > 1:
            write((self._indent_per_level - 1) * " ")
        object_fields = [PynamoField(n) for n in object._attributes.keys()]
        if len(object_fields):
            self._format_dataclass_fields(object, object_fields, stream, indent, allowance + 1, context, level)
        write(")")


def pprint(
    object,
    stream=None,
    indent=1,
    width=80,
    depth=None,
    *,
    compact=False,
    force_use_repr: Optional[List] = None,
):
    """Pretty-print a Python object to a stream [default is sys.stdout]."""
    printer = DataclassPrettyPrinter(
        stream=stream,
        indent=indent,
        width=width,
        depth=depth,
        compact=compact,
        force_use_repr=force_use_repr,
    )
    printer.pprint(object)


def pformat(object, indent=1, width=80, depth=None, *, compact=False):
    """Format a Python object into a pretty-printed representation."""
    return DataclassPrettyPrinter(indent=indent, width=width, depth=depth, compact=compact).pformat(object)


def main() -> None:
    @dataclass
    class Foo:
        stuff: List

    stuff = ["spam", "eggs", "lumberjack", "knights", "ni"]
    stuff.insert(0, stuff)
    stuff.append({"stuff": stuff})
    stuff.append(Foo(stuff))
    pprint(stuff)


if __name__ == "__main__":
    main()
