"""Inspect the mutmut cache for tested/surviving mutants."""
from collections import Counter
from mutmut.cache import session, get_unified_diff  # noqa

with session() as s:
    from mutmut.cache import Mutant

    count = s.query(Mutant.status).count()
    print("mutants in cache:", count)
    statuses = Counter(str(x[0]) for x in s.query(Mutant.status).all())
    print("status counts:", dict(statuses))
    if statuses:
        print("sample:", list(statuses.items())[:5])
