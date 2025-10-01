import numpy as np
import re


def nn(s):
    return re.sub(r"\d+", "", s)


def nl(s):
    return re.sub(r"\D+", "", s)


def pname(s):
    return s.split("_")[0]


def lname(s):
    if len(s.split("_")) < 2:
        lis = np.array([0.0, 0.0, 0.0])
    else:
        lis = np.asanyarray(s.split("_")[1][1:-1].split(), dtype=float)
    return lis


def arr_dimension(arr):
    if arr.ndim > 1:
        return 2
    else:
        return 1


def is_list_A_in_B(A, B):
    return all([np.allclose(a, b, atol=1e-9) for a, b in zip(A, B)])


def remove_blank_space(value):
    return re.sub(r"\s", "", value)


def remove_empty_lines(lines):
    newlines = []
    for i in range(len(lines)):
        if lines[i].strip() != "":
            newlines.append(lines[i])
    return newlines


def remove_bracket(value):
    value_float = float(re.sub(r"\(.*?\)", "", value))
    return value_float


def remove_tail_number(value):
    return re.sub(r"\d", "", value)


def add_quotes(value):
    return "'" + value + "'"


def remove_note_lines(lines):
    newlines = []
    for i in range(len(lines)):
        m = re.search(r"_", lines[i])
        if m is None:
            newlines.append(lines[i])
    return newlines


def extract_quote_lines(lines):
    newlines = []
    for i in range(len(lines)):
        if lines[i].strip()[0] == "'":
            newlines.append(lines[i])
    return newlines


def extract_xyz_lines(lines):
    newlines = []
    for i in range(len(lines)):
        if lines[i].strip()[0] != "_":
            quote_value = add_quotes(remove_blank_space(lines[i]).strip())
            newlines.append(quote_value)
    newlines = remove_empty_lines(newlines)
    return newlines


def remove_quotes(value):
    pattern = r"[\"']([^\"']+)[\"']"
    extracted_values = re.findall(pattern, value)
    return extracted_values[0]


def convert_fraction_to_decimal(expression):

    def replace_fraction(match):
        numerator, denominator = map(int, match.groups())
        return str(numerator / denominator)

    fraction_pattern = r"(-?\d+)/(\d+)"
    converted_expression = re.sub(fraction_pattern, replace_fraction,
                                  expression)
    return converted_expression


def find_keyword(keyword, s):
    m = re.search(keyword, s)
    if m:
        return True
    else:
        return False


def locate_min_idx(matrix):
    min_idx = np.unravel_index(matrix.argmin(), matrix.shape)
    return min_idx[0], min_idx[1]
