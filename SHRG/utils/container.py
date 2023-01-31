# -*- coding: utf-8 -*-


class CounterItem:
    __slots__ = ('count', 'samples')

    def __init__(self, samples=[]):
        self.count = len(samples)
        self.samples = list(samples)

    def add(self, value, max_num_samples):
        if isinstance(value, CounterItem):
            self.count += value.count
            rest_space = max_num_samples - len(self.samples) - len(value.samples)
            if rest_space > 0:
                self.samples.extend(value.samples[:rest_space])
        else:
            self.count += 1
            if len(self.samples) < max_num_samples:
                self.samples.append(value)


class IndexedCounter:
    __slots__ = ('_keys', '_key2index', '_value_capacity')

    def __init__(self, value_capacity):
        self._keys = []
        self._key2index = {}
        self._value_capacity = value_capacity

    def __len__(self):
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def __getitem__(self, index):
        return self._keys[index]

    def keys(self):
        return iter(self._key2index)

    def index(self, key):
        return self._key2index.get(key)

    def add(self, key, value=None):
        index = self._key2index.get(key)
        if index is None:
            index = self._key2index[key] = len(self._keys)
            self._keys.append((key, CounterItem()))

        if value is not None:
            item = self._keys[index][1]
            item.add(value, self._value_capacity)
        return index
