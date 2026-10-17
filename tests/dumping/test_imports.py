import subprocess
import sys
import textwrap


def test_numpy_dumping_does_not_load_optional_serializers(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys

                for name in ("torch", "pandas", "scipy"):
                    sys.modules[name] = None

                import numpy
                import cvtda.dumping

                dumper = cvtda.dumping.NumpyDumper(sys.argv[1])
                dumper.save_dump(numpy.array([1, 2]), "values")
                assert dumper.get_dump("values").tolist() == [1, 2]
                assert "cvtda.dumping.universal" not in sys.modules
                """
            ),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
