"""Install throwaway ``lib.*`` stubs, then put ``sys.modules`` back.

Tests in this directory import ``scripts/direct/run_*.py`` without the heavy
scientific stack, by faking the ``lib.*`` modules those runners import at module
scope. A fake is a plain ``types.ModuleType``, so it has no ``__path__`` and is
therefore not a package. Leaving one bound to ``"lib"`` in ``sys.modules`` makes
``lib`` unimportable as a package for the remainder of the process.

That matters because pytest imports every test module into a single process
during collection, and ``tests/direct`` sorts before ``tests/sbs``,
``tests/test_*.py`` and ``tests/unit``. The stubs leaked into five unrelated
modules, which failed to collect with::

    ModuleNotFoundError: No module named 'lib.phenotype'; 'lib' is not a package
    AttributeError: module 'lib' has no attribute '__path__'

and because collection aborts on error, the other 83 tests never ran either.
Each of those modules collects cleanly on its own -- they were casualties, not
causes.

Import the runner inside the ``with`` block. It keeps the references it bound at
import time, so these tests still exercise the stubs, while ``sys.modules`` goes
back to its previous state for everything collected afterwards::

    from _lib_stubs import stub_lib_modules

    with stub_lib_modules() as stub:
        stub("lib")
        stub("lib.shared", helper=lambda: None)
        sys.path.insert(0, str(RUNNER_DIR))
        import run_phenotype_direct as rpd

Restoring is deliberate rather than, say, giving the ``lib`` stub a real
``__path__``: that would let unstubbed submodules resolve for real but would
keep serving the fakes for stubbed ones, so a downstream test importing
``lib.shared.file_utils`` would silently get an identity ``validate_dtypes``
instead of failing loudly.
"""

import sys
import types
from contextlib import contextmanager

_MISSING = object()


@contextmanager
def stub_lib_modules():
    """Yield a ``stub(name, **attrs)`` callable; undo every binding on exit.

    Names absent from ``sys.modules`` beforehand are removed again; names that
    were already present are restored to the original module object. Stubbing
    the same name twice keeps the outermost saved value, so the restore is
    correct regardless of call order.
    """
    saved = {}

    def stub(name, **attrs):
        saved.setdefault(name, sys.modules.get(name, _MISSING))
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
        return module

    try:
        yield stub
    finally:
        for name, previous in saved.items():
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
