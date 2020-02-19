#! /usr/bin/env python

import abc
import copy
import json
import io
import os

from itertools import product as cartesian

import numpy as np

from calib.misc import default


class IOFork(io.IOBase):
    def __init__(self, parent, forked):
        self._parent = parent
        self._forked = forked

    def close(self):
        self._foked.close()
        return self._parent.close()

    @property
    def closed(self):
        return self._parent.closed

    def fileno(self):
        return self._parent.fileno()

    def flush(self):
        self._forked.flush()
        return self._parent.flush()

    def isatty(self):
        return self._parent.isatty()

    def readable(self):
        return self._parent.readable()

    def readline(self, size=-1):
        if self._forked.readable():
            self._forked.readline(size=size)
        return self._parent.readline(size=size)

    def readlines(self, hint=-1):
        if self._forked.readable():
            self._forked.readlines(hint=hint)
        return self._parent.readlines(hint=hint)

    def seek(self, offset, whence=io.SEEK_SET):
        if self._forked.seekable():
            return self._forked.seek(offset, whence=whence)
        return self._parent.seek(offset, whence=whence)

    def seekable(self):
        return self._parent.seekable()

    def tell(self):
        return self._parent.tell()

    def truncate(self, size=None):
        _ = self._forked.truncate(size=size)
        return self._parent.truncate(size=size)

    def writable(self):
        return self._parent.writable()

    def write(self, b):
        if self._forked.writable():
            self._forked.write(b)
        return self._parent.write(b)

    def writelines(self, lines):
        if self._forked.writable():
            self._forked.writelines(lines)
        return self._parent.writelines(lines)

    def __del__(self):
        pass


class PathContext(object):
    @classmethod
    def cwd(cls):
        return cls(None, os.getcwd())

    @classmethod
    def relative(cls):
        return cls(None, ".")

    @property
    def parent(self):
        return self._parent

    @property
    def directory(self):
        return self._directory

    @property
    def computed_directory(self):
        if self.parent is None:
            return self.directory

        else:
            parent_dir = self.parent.computed_directory
            result = os.path.join(parent_dir, self.directory, "")
            return os.path.normpath(result)

    def __init__(self, parent, directory):
        self._parent = parent
        self._directory = directory

    def appended(self, directory):
        return type(self)(self, directory)

    def prepended(self, path_context):
        result = copy.deepcopy(path_context)
        root = self
        parent = root.parent
        dirs = [root.directory]

        while parent is not None:
            root = parent
            dirs.append(root.directory)
            parent = root.parent

        for directory in dirs[::-1]:
            result = result.appended(directory)

        return result

    def __copy__(self):
        return type(self)(copy.copy(self.parent), self.directory)

    def __deepcopy__(self, memo=None):
        return type(self)(copy.deepcopy(self.parent, memo), self.directory)

    def __str__(self):
        return self.computed_directory


class DataLocation(object):

    @classmethod
    def Parse(cls, path, path_context=None, None_on_fail=True):
        path_context = default(path_context, PathContext.relative())
        if isinstance(path, str):
            directory, name_ext = os.path.split(path)
            dot_idx = name_ext.rfind('.')

            if dot_idx >= 0:
                name = name_ext[:dot_idx]
                ext = name_ext[dot_idx+1:]

            else:
                name = name_ext
                ext = None

            return cls(path_context, directory, name, ext)

        else:
            if None_on_fail:
                return None
            else:
                raise ValueError("path is not a string.")

    @property
    def directory(self):
        return self._directory

    @property
    def name(self):
        return self._name

    @property
    def extension(self):
        return self._extension

    @property
    def path_context(self):
        return self._path_context

    @property
    def filename(self):
        if self.extension is not None:
            return "{}.{}".format(self.name, self.extension)

    @property
    def filepath(self):
        return os.path.normpath(os.path.join(self.directory, self.filename))

    @property
    def computed_path(self):
        base = self.path_context.computed_directory
        return os.path.normpath(os.path.join(base, self.filepath))

    def __init__(self, path_context, directory, name, extension):
        assert (extension is not None and len(extension) > 0) or len(name) > 0, (
            "Canot recieve empty name and extension." )

        assert path_context is not None
        assert directory is not None

        self._path_context = path_context
        self._directory = directory
        self._name = name
        self._extension = extension

    def as_path_context(self):
        return self.path_context.appended(self.directory)

    def prepended(self, path_context):
        return type(self)(  self._path_context.prepended(path_context),
                            self._directory,
                            self._name,
                            self._extension )

    def rebased(self, path_context):
        relpath = os.path.relpath(
            path_context.computed_directory,
            self.path_context.computed_directory)
        dirs = []
        head, tail = os.path.split(relpath)
        cont = len(tail) > 0
        while cont:
            dirs.append(tail)
            head, tail = os.path.split(head)
            cont = len(tail) > 0

        path_context = PathContext.relative()
        for dir in dirs[::-1]:
            path_context = PathContext(path_context, dir)

        directory = os.path.relpath(self._directory, relpath)

        return type(self)(  path_context,
                            directory,
                            self._name,
                            self._extension )

    def __copy__(self):
        return type(self)(  copy.copy(self.path_context),
                            self.directory,
                            self.name,
                            self.extension )

    def __deepcopy__(self, memo=None):
        return type(self)(  copy.deepcopy(self.path_context, memo),
                            self.directory,
                            self.name,
                            self.extension )


class BaseData(object, metaclass=abc.ABCMeta):

    @classmethod
    def LoadClass(cls):
        return cls

    @classmethod
    def Load(cls, path_context, obj, memo=None):
        data_location = DataLocation.Parse(obj, path_context)

        dat_loc_str = None
        if memo is not None and data_location is not None:
            dat_loc_str = os.path.abspath(data_location.computed_path)

        if dat_loc_str is not None:
            try:
                result = memo[dat_loc_str]
                return result
            except KeyError:
                pass

        cls, args, kwargs = cls._Load(path_context, data_location, obj, memo)
        result = cls(*args, **kwargs)

        if dat_loc_str is not None:
            memo[dat_loc_str] = result

        return result

    @abc.abstractclassmethod
    def _Load(cls, path_context, data_location, original_obj, memo=None):
        pass

    @property
    def data_location(self):
        return self._data_location

    @data_location.setter
    def data_location(self, value):
        self._data_location = value

    def __init__(self, data_location=None):
        self._data_location = data_location

    @abc.abstractmethod
    def save(self, path_context=None, data_location_override=None, memo=None):
        pass


class JsonData(BaseData):

    @classmethod
    def _Load(cls, path_context, data_location, original_obj, memo=None):
        if data_location is None:
            dct = original_obj

        else:
            with open(data_location.computed_path, 'r') as f:
                dct = json.load(f)

            path_context = data_location.as_path_context()

        return cls._JsonDictToArgs(path_context, data_location, dct, memo=memo)

    @abc.abstractclassmethod
    def _JsonDictToArgs(cls, path_context, data_location, dct, memo=None):
        return cls, [], {"data_location": data_location}

    def __init__(self, data_location=None):
        self.data_location = data_location

    @abc.abstractmethod
    def _to_json_dict(self, path_context, memo=None):
        pass

    def save(self, path_context=None, data_location_override=None, memo=None):
        data_location = default(data_location_override, self.data_location)
        if data_location is None:
            result = self._to_json_dict(path_context, memo=memo)

        else:
            computed_path = data_location.computed_path
            save = memo is None or computed_path not in memo

            local_path_context = data_location.as_path_context()
            if save:
                with open(computed_path, 'w') as f:
                    dct = self._to_json_dict(local_path_context, memo=memo)
                    json.dump(dct, f, indent=2)

                if memo is not None:
                    memo[computed_path] = self

            result = os.path.normpath(
                        os.path.relpath(
                            data_location.computed_path,
                            path_context.computed_directory))

        return result


class NumpyData(np.ndarray, BaseData):
    @classmethod
    def Load(cls, path_context, obj, memo=None):
        data_location = DataLocation.Parse(obj, path_context)

        dat_loc_str = None
        if memo is not None and data_location is not None:
            dat_loc_str = os.path.abspath(data_location.computed_path)

        if dat_loc_str is not None:
            try:
                return memo[dat_loc_str]
            except KeyError:
                pass

        if data_location is None:
            result = cls._FromJsonDict(data_location, obj)

        elif data_location.extension == "json":
            with open(data_location.computed_path, 'r') as f:
                obj = json.load(f)

            result = cls._FromJsonDict(data_location, obj)

        else:
            result = np.load(data_location.computed_path).view(cls)
            result.data_location = data_location

        if dat_loc_str is not None:
            memo[dat_loc_str] = result

        return result

    @classmethod
    def _FromJsonDict(cls, data_location, dct):
        dtype = dct["dtype"]
        data = dct["data"]

        if dtype is not None:
            dtype = np.dtype(dtype)
            if dtype.kind == 'V':
                def last_dim_is_tuple(lst):
                    if len(lst) > 0:
                        if not isinstance(lst[0], list):
                            lst = tuple(lst)
                        else:
                            lst = [last_dim_is_tuple(ll) for ll in lst]
                    return lst
                data = last_dim_is_tuple(data)

        result = cls(data, dtype=dtype, data_location=data_location)
        return result

    def __new__(cls, *args, **kwargs):
        data_location = kwargs.pop("data_location", None)
        result = np.asarray(*args, **kwargs).view(cls)
        BaseData.__init__(result, data_location)
        return result

    def __init__(self, *args, **kwargs):
        pass

    def __array_finalize__(self, obj):
        if obj is None:
            return

        for attr in ("_data_location",):
            setattr(self, attr, getattr(obj, attr, None))

    def save(self, path_context=None, data_location_override=None, memo=None):
        data_location = default(data_location_override, self.data_location)

        if data_location is None:
            result = self._to_json_dict()

        else:
            computed_path = data_location.computed_path
            save = memo is None or computed_path not in memo

            if save:
                if data_location.extension == "json":
                    with open(computed_path, 'w') as f:
                        json.dump(self._to_json_dict(), f)

                else:
                    np.save(data_location.computed_path, self)

                if memo is not None:
                    memo[computed_path] = self

            result = os.path.normpath(
                        os.path.relpath(
                            data_location.computed_path,
                            path_context.computed_directory))

        return result

    def _to_json_dict(self):
        dtype = self.dtype
        if dtype.kind == "V":
            dtype = dtype.descr
        else:
            dtype = dtype.str
        return {"dtype": dtype, "data": self.tolist()}


class TextFile(BaseData):

    def __init__(self, contents, data_location=None):
        super().__init__(data_location)
        self.contents = contents

    @classmethod
    def _Load(cls, path_context, data_location, original_obj, memo=None):
        if data_location is None:
            contents = cls(original_obj["data"])

        else:
            with open(data_location.computed_path, 'r') as f:
                contents = f.read()

        return cls(contents, data_location=data_location)

    def save(self, path_context=None, data_location_override=None, memo=None):
        data_location = default(data_location_override, self.data_location)
        contents = self.contents
        result = {"data" : contents}

        if data_location is not None:
            with open(data_location.computed_path, 'w') as f:
                f.write(contents)

            result = os.path.normpath(
                        os.path.relpath(
                            data_location.computed_path,
                            path_context.computed_directory))

        return result


def at_least_ndarray(obj, dtype=None):
    if not isinstance(obj, np.ndarray):
        obj = np.array(obj, dtype=dtype)
    return obj


def at_least_numpydata(obj, dtype=None):
    if not isinstance(obj, NumpyData):
        obj = at_least_ndarray(obj).view(NumpyData)
    return obj


def save(data, data_location_override=None, memo=None):
    memo = default(memo, {})
    path_context = PathContext.relative()
    return data.save(path_context, data_location_override, memo=memo)


def load(data_class, data_location, memo=None):
    memo = default(memo, {})
    return data_class.Load(data_location.path_context, data_location.filepath, memo)


# Useful for isinstance()
class IterableOfBase(object):
    pass


def iterable_of(it_class, class_):

    class JsonIterable(JsonData, it_class, IterableOfBase):
        @classmethod
        def EnsureIsInstance(cls, obj):
            if not isinstance(obj, cls):
                obj = cls(obj)

            return obj

        @classmethod
        def _JsonDictToArgs(cls, path_context, data_location, dct, memo=None):
            _, args, kwargs = super()._JsonDictToArgs(path_context, data_location, dct, memo)
            args.append(it_class(class_.Load(path_context, ee, memo) for ee in dct))
            kwargs["data_location"] = data_location
            return cls, args, kwargs

        def __new__(cls, iterable=None, data_location=None):
            iterable = tuple() if iterable is None else iterable
            return super().__new__(cls, iterable)

        def __init__(self, iterable=None, data_location=None):
            iterable = tuple() if iterable is None else iterable
            assert all(isinstance(ee, class_) for ee in iterable), (
                "A value is not instance of {}".format(class_) )

            if not issubclass(it_class, tuple):
                it_class.__init__(self, iterable)

            JsonData.__init__(self, data_location)

        def _to_json_dict(self, path_context, memo=None):
            return it_class(ee.save(path_context, memo=memo) for ee in self)

        def apply_save_pattern(self, pattern, path_context=None):
            path_context = default(path_context, PathContext.relative())
            for ii, item in enumerate(self):
                path = pattern.format(ii)
                data_location = DataLocation.Parse(path, path_context, None_on_fail=False)
                if isinstance(item, IterableOfBase):
                    item.set_children_data_location(data_location)
                else:
                    item.data_location = data_location

        def clear_save_pattern(self):
            for item in self:
                item.data_location = None

        def set_children_data_location(self, value):
            if value is None:
                self.clear_save_pattern()

            else:
                value = copy.deepcopy(value)

                name = value.name
                directory = value.directory
                extension = value.extension
                path_context = value.path_context

                def escape_formatting(string):
                    return string.replace('{', '{{').replace('}', '}}')

                name = escape_formatting(name)
                directory = escape_formatting(directory)
                extension = escape_formatting(extension)

                name = name + "[{}]"

                filtered_loc = DataLocation(path_context, directory, name, extension)
                pattern = filtered_loc.filepath

                self.apply_save_pattern(pattern, path_context)

        def __copy__(self):
            return JsonIterable(self)

        def __deepcopy__(self, memo=None):
            memo = default(memo, {})
            return JsonIterable(tuple(copy.deepcopy(ee, memo=memo) for ee in self))

    return JsonIterable


_generated_nested = dict()


# Useful for isinstance()
class NestedListsBase(object):
    pass


def nested_lists_of(class_):

    if class_ not in _generated_nested:

        class NestedListsOf(JsonData, NestedListsBase):

            @property
            def data(self):
                return self._data

            @property
            def dims(self):
                return self._dims

            @classmethod
            def EnsureIsInstance(cls, obj):
                if not isinstance(obj, cls):
                    obj = cls(obj, len(np.shape(obj)))

                return obj

            @classmethod
            def _JsonDictToArgs(cls, path_context, data_location, dct, memo=None):
                _, args, kwargs = super()._JsonDictToArgs(path_context, data_location, dct, memo=memo)
                dims = dct["dims"]
                type_ = class_
                for _ in range(dims):
                    type_ = list_of(type_)

                args.append(type_.Load(path_context, dct["data"], memo=memo))
                args.append(dims)
                return cls, args, kwargs

            def __init__(self, data, dims, data_location=None):
                super().__init__(data_location=data_location)

                assert dims >= 0

                def rec_ensure(lst, dim):
                    if dim == 0:
                        return class_, lst

                    else:
                        type_lst = [rec_ensure(llst, dim-1) for llst in lst]
                        type_, lst = tuple(zip(*type_lst))
                        type_ = list_of(type_[0])
                        lst = list(lst)

                    value = type_.EnsureIsInstance(lst)
                    return type_, value

                _, data = rec_ensure(data, dims)

                self._data = data
                self._dims = dims

            def _to_json_dict(self, path_context, memo=None):
                data = self._data.save(path_context, memo=memo)
                return {"dims": self._dims, "data": data}

            def iter_flat(self):
                if self.dims == 0:
                    yield self.data

                else:
                    for elem in self._iter(self.data, 1):
                        yield elem

            def _iter(self, lst, depth):
                if depth >= self.dims:
                    for elem in lst:
                        yield elem

                else:
                    for sub_lst in lst:
                        for elem in self._iter(sub_lst, depth+1):
                            yield elem

            def __getitem__(self, item):
                if not isinstance(item, tuple):
                    item = (item,)

                assert len(item) <= self.dims

                return self._rec_get(self.data, item)

            @staticmethod
            def _rec_get(lst, index):
                if len(index) == 0:
                    return lst

                else:
                    return NestedListsOf._rec_get(lst[index[0]], index[1:])

            def __setitem__(self, item, value):
                if not isinstance(item, tuple):
                    item = (item,)

                assert len(item) == self.dims
                assert isinstance(value, class_)

                if len(item) == 0:
                    self._data = value

                else:
                    idx, tail = (item[:-1], item[-1],)
                    lst = self._rec_get(self.data, idx)
                    lst[tail] = value

            def iter_idx(self):
                if self.dims == 0:
                    yield tuple()

                else:
                    for idx in cartesian(*tuple(range(ss) for ss in np.shape(self.data))):
                        yield idx

            def __copy__(self):
                return NestedListsOf(copy.copy(self.data), self.dims)

            def __deepcopy__(self, memo=None):
                return NestedListsOf(copy.deepcopy(self.data, memo=memo), self.dims)

            def set_children_data_location(self, data_location):
                if self.dims == 0:
                    dl = data_location
                    data_location = DataLocation(
                        dl.path_context,
                        dl.directory,
                        dl.name + "[()]",
                        dl.extension)
                    self.data.data_location = data_location

                else:
                    self.data.set_children_data_location(data_location)

        _generated_nested[class_] = NestedListsOf

    return _generated_nested[class_]


_generated_lists = dict()


def list_of(class_):
    if class_ not in _generated_lists:
        class ListOf(iterable_of(list, class_)):

            def append(self, object):
                assert isinstance(object, class_), ("Value is not "
                    "instance of {}".format(class_))
                return super().append(object)

            def __setitem__(self, key, value):
                assert isinstance(value, class_), ("Value is not "
                    "instance of {}".format(class_))
                return super().__setitem__(key, value)

        _generated_lists[class_] = ListOf

    return _generated_lists[class_]


_generated_tuples = dict()


def tuple_of(class_):
    if class_ not in _generated_tuples:
        class TupleOf(iterable_of(tuple, class_)):
            pass

        _generated_tuples[class_] = TupleOf

    return _generated_tuples[class_]


def ensure_list(arr_or_list, raise_err=True):
    if isinstance(arr_or_list, np.ndarray):
        return arr_or_list.tolist()
    elif isinstance(arr_or_list, (list, tuple,)):
        return [ensure_list(e, False) for e in arr_or_list]
    elif raise_err:
        raise ValueError("Not an numpy array nor a list.")

    return arr_or_list
