import numpy
import pandas
import scipy.sparse
import torch

import cvtda.dumping
from cvtda.dumping import UniversalDumper
from cvtda.logging import DevNullLogger


def test_universal_dumper_uses_the_shared_dumping_context(tmp_path):
    previous = cvtda.dumping.dumper()
    with UniversalDumper(tmp_path) as dumper:
        assert cvtda.dumping.dumper() is dumper
    assert cvtda.dumping.dumper() is previous


def test_arrays_tensors_tables_and_sparse_graphs_roundtrip(tmp_path):
    dumper = UniversalDumper(tmp_path)
    array = numpy.arange(6).reshape(2, 3)
    tensor = torch.tensor(array)
    table = pandas.DataFrame({"Accuracy": [0.5, 0.75]})
    graph = scipy.sparse.csr_matrix([[0, 1], [1, 0]])
    with DevNullLogger():
        for name, value in (("array", array), ("tensor", tensor), ("table", table), ("graph", graph)):
            dumper.save_dump(value, name)
        numpy.testing.assert_array_equal(dumper.get_dump("array"), array)
        torch.testing.assert_close(dumper.get_dump("tensor"), tensor)
        pandas.testing.assert_frame_equal(dumper.get_dump("table"), table)
        assert (dumper.get_dump("graph") != graph).nnz == 0


def test_model_roundtrip(tmp_path):
    model = torch.nn.Linear(2, 2)
    inputs = torch.ones(2, 2)
    dumper = UniversalDumper(tmp_path)
    with DevNullLogger():
        dumper.save_dump(model, "model")
        restored = dumper.get_dump("model")
    torch.testing.assert_close(restored(inputs), model(inputs))


def test_nested_lists_roundtrip(tmp_path):
    dumper = UniversalDumper(tmp_path)
    with DevNullLogger():
        dumper.save_dump([[numpy.array([99])], [numpy.array([index]) for index in range(12)]], "layers")
        result = dumper.get_dump("layers")
    assert len(result[0]) == 1
    numpy.testing.assert_array_equal(result[0][0], [99])
    assert len(result[1]) == 12
    for index, item in enumerate(result[1]):
        numpy.testing.assert_array_equal(item, [index])


def test_execute_reuses_cache_with_return_annotation(tmp_path):
    dumper = UniversalDumper(tmp_path)
    calls = []

    def compute(value) -> numpy.ndarray:
        calls.append(value)
        return numpy.array([value])

    with DevNullLogger():
        first = dumper.execute(compute, "cached", 3)
        second = dumper.execute(compute, "cached", 9)
    assert calls == [3]
    numpy.testing.assert_array_equal(first, [3])
    numpy.testing.assert_array_equal(second, first)
