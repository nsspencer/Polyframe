import timeit


class TBase:
    def get(self):
        return 1


class TChild(TBase):
    def get(self):
        return 1


class TChild2(TBase):
    ...


class PropertyChild(TBase):
    @property
    def get(self):
        return 1


class AttributeChild(TBase):
    def __init__(self):
        self._get = 1

    @property
    def get(self):
        return self._get


if __name__ == "__main__":
    # Example usage
    base = TBase()
    t = TChild()
    p = TChild2()
    property_child = PropertyChild()
    attr_child = AttributeChild()

    N = 10_000_000
    print("base get", timeit.timeit(lambda: base.get(), number=N))
    print("redefined child get", timeit.timeit(
        lambda: t.get(), number=N))
    print("child inherit get", timeit.timeit(
        lambda: p.get(), number=N))
    print("property child get", timeit.timeit(
        lambda: property_child.get, number=N))
    print("attribute child get", timeit.timeit(
        lambda: attr_child.get, number=N))
    print("attribute child get", timeit.timeit(
        lambda: attr_child._get, number=N))
