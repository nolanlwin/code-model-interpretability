"""Role extraction: variable names per role, per language.

Python uses the ast module (same logic as the probing notebooks); the other
six XLCoST languages use the regex extractors from the cross-language cells.
Labels always come from structure, never from the variable's name, so probes
must rely on hidden-state context.
"""

import ast
import builtins
import keyword
import re

from . import ROLES

PYTHON_PROTECTED = set(keyword.kwlist) | set(dir(builtins)) | {"self", "cls"}

_APPEND_METHODS = {"append", "extend", "add", "update"}


# ── Python (AST) ──────────────────────────────────────────────────────────────

def _py_index_key(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            slc = node.slice
            if isinstance(slc, ast.Index):  # Python < 3.9 AST
                slc = slc.value
            if isinstance(slc, ast.Name):
                names.add(slc.id)
    return names


def _py_loop_vars(tree):
    loop_vars = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    loop_vars.add(n.id)
    return loop_vars


def _py_accumulator(tree):
    loop_vars = _py_loop_vars(tree)
    index_names = _py_index_key(tree)
    accumulators = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.While)):
            continue
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                n = stmt.target.id
                if n not in loop_vars and n not in index_names and n != "_":
                    accumulators.add(n)
            elif (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                  and isinstance(stmt.value.func, ast.Attribute)
                  and stmt.value.func.attr in _APPEND_METHODS
                  and isinstance(stmt.value.func.value, ast.Name)):
                n = stmt.value.func.value.id
                if n not in loop_vars and n not in index_names and n != "_":
                    accumulators.add(n)
    return accumulators


def _py_boolean(tree):
    """Variables assigned a boolean literal anywhere in the program."""
    names = set()
    for node in ast.walk(tree):
        value = None
        if isinstance(node, ast.Assign):
            value = node.value
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value = node.value
            targets = [node.target]
        else:
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, bool):
            for t in targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


def _py_class_struct(tree):
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def extract_roles_python(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {role: set() for role in ROLES}
    return {
        "index_key": _py_index_key(tree),
        "accumulator": _py_accumulator(tree),
        "iterator": _py_loop_vars(tree) - {"_"},
        "boolean": _py_boolean(tree),
        "class_struct": _py_class_struct(tree),
    }


# ── Other languages (regex) ───────────────────────────────────────────────────

_C_KW = {"int", "long", "char", "short", "float", "double", "unsigned", "void",
         "size_t", "bool", "true", "false", "null", "NULL", "return", "if",
         "else", "for", "while", "break", "continue", "sizeof", "static", "const"}
_JAVA_KW = _C_KW | {"byte", "boolean", "String", "Integer", "Long", "Object",
                    "new", "public", "private", "static", "final", "class"}
_CS_KW = _C_KW | {"byte", "string", "object", "var", "new", "public", "private",
                  "static", "using", "class"}
_JS_KW = {"undefined", "null", "true", "false", "NaN", "Infinity", "length",
          "prototype", "constructor", "var", "let", "const", "function", "new",
          "return", "if", "else", "for", "while", "of", "in", "typeof"}
_PHP_KW = {"true", "false", "null", "function", "return", "if", "else", "for",
           "foreach", "while", "as", "echo", "array", "new"}

LANG_KEYWORDS = {
    "Java": _JAVA_KW, "C++": _C_KW, "C": _C_KW, "C#": _CS_KW,
    "Javascript": _JS_KW, "PHP": _PHP_KW,
}

_AUG_RE = re.compile(r"\b([a-zA-Z_]\w*)\s*(?:\+=|-=|\*=|/=|\|=|&=|\^=)")
_INC_RE = re.compile(r"\b([a-zA-Z_]\w*)\s*(?:\+\+|--)")
_APP_RE = re.compile(r"\b([a-zA-Z_]\w*)\.(?:add|push|append|extend|offer)\s*\(")
_SUBSCRIPT_RE = re.compile(r"\[\s*([a-zA-Z_]\w*)\s*\]")
_BOOL_ASSIGN_RE = re.compile(r"\b([a-zA-Z_]\w*)\s*=\s*(?:true|false)\b")
_CLASS_STRUCT_RE = re.compile(r"\b(?:class|struct)\s+([A-Za-z_]\w*)")

# For-loop header variables per language family.
_FOR_INIT_RE = re.compile(r"for\s*\(\s*(?:[a-zA-Z_][\w<>\[\],\s]*\s+)?([a-zA-Z_]\w*)\s*=")
_FOR_EACH_RE = re.compile(r"for(?:each)?\s*\(\s*[\w<>\[\],\s]+?\s([a-zA-Z_]\w*)\s*(?::|\bin\b)")
_JS_FOR_RE = re.compile(r"for\s*\(\s*(?:var|let|const)\s+([a-zA-Z_]\w*)\s*(?:=|\bof\b|\bin\b)")
# Applied after the $ sigil has been stripped from PHP code.
_PHP_FOREACH_RE = re.compile(r"foreach\s*\([^)]*?\bas\b\s*(?:&?\s*(\w+)\s*=>\s*)?&?\s*(\w+)")


def _iterator_regex(code, language):
    names = set()
    if language == "PHP":
        for m in _FOR_INIT_RE.finditer(code):
            names.add(m.group(1))
        for m in _PHP_FOREACH_RE.finditer(code):
            names.update(g for g in m.groups() if g)
    elif language == "Javascript":
        for m in _JS_FOR_RE.finditer(code):
            names.add(m.group(1))
        for m in _FOR_INIT_RE.finditer(code):
            names.add(m.group(1))
    else:
        for m in _FOR_INIT_RE.finditer(code):
            names.add(m.group(1))
        for m in _FOR_EACH_RE.finditer(code):
            names.add(m.group(1))
    return names


def extract_roles_regex(code, language):
    kw = LANG_KEYWORDS[language]
    if language == "PHP":
        # XLCoST tokenization can separate the sigil from the name ("$ spf").
        code = re.sub(r"\$\s*([a-zA-Z_]\w*)", r"\1", code)
    iterator = _iterator_regex(code, language) - kw
    accumulator = set()
    for pat in (_AUG_RE, _INC_RE, _APP_RE):
        accumulator.update(m.group(1) for m in pat.finditer(code))
    accumulator -= kw
    return {
        "index_key": {m.group(1) for m in _SUBSCRIPT_RE.finditer(code)} - kw,
        "accumulator": accumulator,
        "iterator": iterator,
        "boolean": {m.group(1) for m in _BOOL_ASSIGN_RE.finditer(code)} - kw,
        "class_struct": {m.group(1) for m in _CLASS_STRUCT_RE.finditer(code)} - kw,
    }


def extract_roles(code, language):
    """Return {role: set of variable names} for one program."""
    if language == "Python":
        return extract_roles_python(code)
    return extract_roles_regex(code, language)
