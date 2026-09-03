"""Tests for importing XY exports as delimited text as well as Excel.

Mounters and CAD tools disagree on nearly everything about these files:
container, delimiter, column names, whether there is a Type column at
all. These check the formats that actually turn up.

Run directly:
    python tests/test_csv_import.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openpyxl import Workbook

from core.program_parser import load_mounter_xy, parse_program

tmp = Path(tempfile.mkdtemp())


def write(name, text, encoding="utf-8"):
    path = tmp / name
    path.write_text(text, encoding=encoding)
    return path


def test_plain_csv_with_title_row():
    """The same shape as the Excel export: a title line above the header,
    which is what defeats a naive CSV read."""
    path = write("basic.csv", """Mounter XY Export - Board A
X,Y,Rotation,Z,Designator,Library,Part,Skip No,Type,Pattern Type,Pattern Group
10.0,10.0,0,0,R1,RES_0402,PN-1001,0,Placement,,
20.0,15.0,90,0,C1,CAP_0603,PN-2002,0,Placement,,
0,0,0,0,----,,,0,Placement,,
5.0,5.0,0,0,,,,0,Pattern Fiducial,,
95.0,8.0,0,0,,,,0,Pattern Fiducial,,
50.0,0.0,0,0,PANEL2,,,0,Pattern Offset,,
""")
    program = parse_program(str(path), "CSV_BOARD")
    assert len(program["components"]) == 2, program["components"]
    assert len(program["fiducials"]) == 2, program["fiducials"]
    assert len(program["panel_offsets"]) == 1
    assert program["is_panel"] is True
    assert {c["designator"] for c in program["components"]} == {"R1", "C1"}
    assert program["components"][1]["rotation"] == 90.0
    print("OK test_plain_csv_with_title_row:", len(program["components"]), "components,",
          len(program["fiducials"]), "fiducials")


def test_csv_without_title_row():
    path = write("noheader.csv", """Designator,X,Y,Rotation,Part,Type
R1,10,10,0,PN-1,Placement
R2,20,10,180,PN-1,Placement
,5,5,0,,Pattern Fiducial
""")
    program = parse_program(str(path), "B")
    assert len(program["components"]) == 2
    assert len(program["fiducials"]) == 1
    print("OK test_csv_without_title_row")


def test_semicolon_delimited_with_decimal_comma():
    """European export: ';' separator and ',' as the decimal mark."""
    path = write("euro.csv", """Designator;X;Y;Rotation;Part;Type
R1;10,5;12,25;0;PN-1;Placement
R2;20,0;12,25;90;PN-1;Placement
""")
    program = parse_program(str(path), "EU")
    xs = [c["x"] for c in program["components"]]
    ys = [c["y"] for c in program["components"]]
    assert xs == [10.5, 20.0], xs
    assert ys == [12.25, 12.25], ys
    print("OK test_semicolon_delimited_with_decimal_comma:", xs, ys)


def test_tab_delimited():
    path = write("tabs.tsv", "Designator\tX\tY\tPart\tType\n"
                             "R1\t10\t10\tPN-1\tPlacement\n"
                             "R2\t20\t10\tPN-1\tPlacement\n")
    program = parse_program(str(path), "T")
    assert len(program["components"]) == 2
    print("OK test_tab_delimited")


def test_column_name_aliases():
    """RefDes/Center-X/Rot are as common as the canonical names."""
    path = write("aliases.csv", """RefDes,Center-X (mm),Center-Y (mm),Rot,Footprint,Part Number
R1,10,10,0,0402,PN-1
C2,20,15,270,0603,PN-2
""")
    program = parse_program(str(path), "A")
    assert len(program["components"]) == 2, program["components"]
    comp = program["components"][0]
    assert (comp["designator"], comp["x"], comp["y"]) == ("R1", 10.0, 10.0), comp
    assert program["components"][1]["rotation"] == 270.0
    assert comp["part"] == "PN-1" and comp["library"] == "0402"
    print("OK test_column_name_aliases:", comp)


def test_no_type_column_still_imports():
    """A plain placement list with no record-type column: every row is a
    component, and the parser says so rather than producing nothing."""
    path = write("notype.csv", """Designator,X,Y,Part
R1,10,10,PN-1
R2,20,10,PN-1
C1,30,10,PN-2
""")
    program = parse_program(str(path), "N")
    assert len(program["components"]) == 3
    assert program["fiducials"] == []
    assert any("No Type column" in n for n in program["notes"]), program["notes"]
    print("OK test_no_type_column_still_imports:", program["notes"][0][:60])


def test_no_part_column_is_reported():
    path = write("nopart.csv", """Designator,X,Y,Type
R1,10,10,Placement
""")
    program = parse_program(str(path), "P")
    assert program["components"][0]["part"] is None
    assert program["unknown_parts"] == []
    assert any("No Part column" in n for n in program["notes"]), program["notes"]
    print("OK test_no_part_column_is_reported")


def test_unrecognised_type_values_do_not_empty_the_program():
    """A Type column using this mounter's own vocabulary must not yield
    zero components -- silently importing nothing is the worst outcome."""
    path = write("othertype.csv", """Designator,X,Y,Part,Type
R1,10,10,PN-1,MOUNT
R2,20,10,PN-1,MOUNT
""")
    program = parse_program(str(path), "O")
    assert len(program["components"]) == 2, program["components"]
    assert any("treating every row" in n for n in program["notes"]), program["notes"]
    print("OK test_unrecognised_type_values_do_not_empty_the_program")


def test_ragged_rows_and_blank_lines():
    path = write("ragged.csv", """Some Title
X,Y,Rotation,Designator,Part,Type
10,10,0,R1,PN-1,Placement

20,10,0,R2
30,10,0,R3,PN-1,Placement
""")
    program = parse_program(str(path), "R")
    designators = {c["designator"] for c in program["components"]}
    # R2's row is short: it has no Type, so it is not a Placement row.
    assert "R1" in designators and "R3" in designators, designators
    print("OK test_ragged_rows_and_blank_lines:", sorted(designators))


def test_bom_and_crlf():
    """Windows exports carry a UTF-8 BOM and CRLF line endings."""
    path = write("bom.csv", "Designator,X,Y,Part,Type\r\nR1,10,10,PN-1,Placement\r\n",
                 encoding="utf-8-sig")
    program = parse_program(str(path), "W")
    assert len(program["components"]) == 1
    assert program["components"][0]["designator"] == "R1"
    print("OK test_bom_and_crlf")


def test_excel_still_works():
    """The original path must not regress."""
    wb = Workbook()
    ws = wb.active
    ws.append(["Mounter XY Export - Sample"])
    ws.append(["X", "Y", "Rotation", "Z", "Designator", "Library", "Part",
               "Skip No", "Type", "Pattern Type", "Pattern Group"])
    ws.append([10.0, 10.0, 0, 0, "R1", "RES", "PN-1001", 0, "Placement", "", ""])
    ws.append([0, 0, 0, 0, "----", None, None, 0, "Placement", "", ""])
    ws.append([5.0, 5.0, 0, 0, None, None, None, 0, "Pattern Fiducial", "", ""])
    path = tmp / "sample.xlsx"
    wb.save(path)

    program = parse_program(str(path), "XL")
    assert len(program["components"]) == 1, program["components"]
    assert len(program["fiducials"]) == 1
    assert program["components"][0]["part"] == "PN-1001"
    print("OK test_excel_still_works")


def test_missing_required_column_names_what_it_found():
    path = write("bad.csv", "Foo,Bar\n1,2\n")
    try:
        parse_program(str(path), "X")
    except ValueError as exc:
        assert "X, Y and Designator" in str(exc), str(exc)
        print("OK test_missing_required_column_names_what_it_found:", str(exc)[:70])
    else:
        raise AssertionError("expected a ValueError for a file with no usable columns")


def test_bad_coordinate_names_the_row():
    path = write("badcoord.csv", "Designator,X,Y,Type\nR1,abc,10,Placement\n")
    try:
        parse_program(str(path), "BC")
    except ValueError as exc:
        assert "R1" in str(exc) and "X" in str(exc), str(exc)
        print("OK test_bad_coordinate_names_the_row:", str(exc))
    else:
        raise AssertionError("expected a ValueError naming the offending row")


def test_empty_file():
    path = write("empty.csv", "")
    try:
        parse_program(str(path), "E")
    except ValueError as exc:
        print("OK test_empty_file:", str(exc)[:60])
    else:
        raise AssertionError("expected a ValueError for an empty file")


if __name__ == "__main__":
    test_plain_csv_with_title_row()
    test_csv_without_title_row()
    test_semicolon_delimited_with_decimal_comma()
    test_tab_delimited()
    test_column_name_aliases()
    test_no_type_column_still_imports()
    test_no_part_column_is_reported()
    test_unrecognised_type_values_do_not_empty_the_program()
    test_ragged_rows_and_blank_lines()
    test_bom_and_crlf()
    test_excel_still_works()
    test_missing_required_column_names_what_it_found()
    test_bad_coordinate_names_the_row()
    test_empty_file()
    print("\nALL CSV IMPORT TESTS PASSED")
