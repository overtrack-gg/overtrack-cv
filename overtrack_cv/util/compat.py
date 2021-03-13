from typing import TYPE_CHECKING, Any, Type

try:
    from typing import Literal
except:
    if TYPE_CHECKING:
        from typing_extensions import Literal
    else:

        class _Literal:
            def __getitem__(self, typeargs: Any) -> Type[str]:
                return str

        Literal = _Literal()
