import os
import sys
from pycparser import c_parser, c_ast, parse_file
from subprocess import Popen, PIPE

LIB = "CUEW"
COPYRIGHT = """/*
 * Copyright 2011-2014 Blender Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License
 */"""
FILES = ["cuda.h"]

TYPEDEFS = []
FUNC_TYPEDEFS = []
SYMBOLS = []


class FuncDefVisitor(c_ast.NodeVisitor):
    indent = 0

    def _get_quals_string(self, node):
        if node.quals:
            return ' '.join(node.quals) + ' '
        return ''

    def _get_ident_type(self, node):
        if isinstance(node, c_ast.PtrDecl):
            return self._get_ident_type(node.type.type) + '*'
        if isinstance(node, c_ast.ArrayDecl):
            return self._get_ident_type(node.type)
        elif isinstance(node, c_ast.Struct):
            if node.name:
                return 'struct ' + node.name
            else:
                self.indent += 1
                struct = self._stringify_struct(node)
                self.indent -= 1
                return "struct {\n" + \
                       struct + ("  " * self.indent) + "}"
        elif isinstance(node, c_ast.Union):
            self.indent += 1
            union = self._stringify_struct(node)
            self.indent -= 1
            return "union {\n" + union + ("  " * self.indent) + "}"
        elif isinstance(node, c_ast.Enum):
            return 'enum ' + node.name
        elif isinstance(node, c_ast.TypeDecl):
            return self._get_ident_type(node.type)
        else:
            return node.names[0]

    def _stringify_param(self, param):
        param_type = param.type
        result = self._get_quals_string(param)
        result += self._get_ident_type(param_type)
        if param.name:
            result += ' ' + param.name
        if isinstance(param_type, c_ast.ArrayDecl):
            dim = param_type.dim.value
            result += '[' + dim + ']'
        return result

    def _stringify_params(self, params):
        result = []
        for param in params:
            result.append(self._stringify_param(param))
        return ', '.join(result)

    def _stringify_struct(self, node):
        result = ""
        children = node.children()
        for child in children:
            member = self._stringify_param(child[1])
            result += ("  " * self.indent) + member + ";\n"
        return result

    def _stringify_enum(self, node):
        result = ""
        children = node.children()
        for child in children:
            if isinstance(child[1], c_ast.EnumeratorList):
                enumerators = child[1].enumerators
                for enumerator in enumerators:
                    result += ("  " * self.indent) + enumerator.name
                    if enumerator.value:
                        result += " = " + enumerator.value.value
                    result += ",\n"
        return result

    def visit_Decl(self, node):
        if node.type.__class__.__name__ == 'FuncDecl':
            if isinstance(node.type, c_ast.FuncDecl):
                func_decl = node.type
                func_decl_type = func_decl.type

                typedef = 'typedef '
                symbol_name = None

                if isinstance(func_decl_type, c_ast.TypeDecl):
                    symbol_name = func_decl_type.declname
                    typedef += self._get_quals_string(func_decl_type)
                    typedef += self._get_ident_type(func_decl_type.type)
                    typedef += ' CUDAAPI'
                    typedef += ' (*t' + symbol_name + ') '
                elif isinstance(func_decl_type, c_ast.PtrDecl):
                    ptr_type = func_decl_type.type
                    symbol_name = ptr_type.declname
                    typedef += self._get_quals_string(ptr_type)
                    typedef += self._get_ident_type(func_decl_type)
                    typedef += ' CUDAAPI'
                    typedef += ' (*t' + symbol_name + ') '

                typedef += '(' + \
                    self._stringify_params(func_decl.args.params) + \
                    ');'

                SYMBOLS.append(symbol_name)
                FUNC_TYPEDEFS.append(typedef)

    def visit_Typedef(self, node):
        quals = self._get_quals_string(node)
        type = self._get_ident_type(node.type)
        complex = False

        if isinstance(node.type.type, c_ast.Struct):
            self.indent += 1
            struct = self._stringify_struct(node.type.type)
            self.indent -= 1
            typedef = quals + type + " {\n" + struct + "} " + node.name
            complex = True
        elif isinstance(node.type.type, c_ast.Enum):
            self.indent += 1
            enum = self._stringify_enum(node.type.type)
            self.indent -= 1
            typedef = quals + type + " {\n" + enum + "} " + node.name
            complex = True
        else:
            typedef = quals + type + " " + node.name
        if complex:
            typedef = "\ntypedef " + typedef + ";"
        else:
            typedef = "typedef " + typedef + ";"
        if typedef != "typedef long size_t;":
            TYPEDEFS.append(typedef)


def get_latest_cpp():
    path_prefix = "/usr/bin"
    for cpp_version in ["9", "8", "7", "6", "5", "4"]:
        test_cpp = os.path.join(path_prefix, "cpp-4." + cpp_version)
        if os.path.exists(test_cpp):
            return test_cpp
    return None


def preprocess_file(filename, cpp_path):
    try:
        pipe = Popen([cpp_path, "-I./", filename],
                     stdout=PIPE,
                     universal_newlines=True)
        text = pipe.communicate()[0]
    except OSError as e:
        raise RuntimeError("Unable to invoke 'cpp'.  " +
            'Make sure its path was passed correctly\n' +
            ('Original error: %s' % e))

    return text


def parse_files():
    parser = c_parser.CParser()
    cpp_path = get_latest_cpp()

    for filename in FILES:
        text = preprocess_file(filename, cpp_path)
        ast = parser.parse(text, filename)

    v = FuncDefVisitor()
    v.visit(ast)


def print_copyright():
    print(COPYRIGHT)
    print("")


def open_header_guard():
    print("#ifndef __%s_H_" % (LIB))
    print("#define __%s_H_" % (LIB))
    print("")
    print("#ifdef __cplusplus")
    print("extern \"C\" {")
    print("#endif")
    print("")


def close_header_guard():
    print("")
    print("#ifdef __cplusplus")
    print("}")
    print("#endif")
    print("")
    print("#endif  /* __%s_H_ */" % (LIB))
    print("")


def print_header():
    print_copyright()
    open_header_guard()

    # Fot size_t.
    print("#include <stdlib.h>")
    print("")

    print("/* Types. */")
    for typedef in TYPEDEFS:
        print('%s' % (typedef))

    # TDO(sergey): This is only specific to CUDA wrapper.
    print("""
#ifdef _WIN32
#  define CUDAAPI __stdcall
#else
#  define CUDAAPI
#endif""")
    print("")

    print("/* Function types. */")
    for func_typedef in FUNC_TYPEDEFS:
        print('%s' % (func_typedef))
    print("")

    print("/* Function declarations. */")
    for symbol in SYMBOLS:
        print('extern t%s %s;' % (symbol, symbol))

    close_header_guard()


def print_dl_wrapper():
    print("""#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define VC_EXTRALEAN
#  include <windows.h>

/* Utility macros. */

typedef HMODULE DynamicLibrary;

#  define dynamic_library_open(path)         LoadLibrary(path)
#  define dynamic_library_close(lib)         FreeLibrary(lib)
#  define dynamic_library_find(lib, symbol)  GetProcAddress(lib, symbol)
#else
#  include <dlfcn.h>

typedef void* DynamicLibrary;

#  define dynamic_library_open(path)         dlopen(path, RTLD_NOW)
#  define dynamic_library_close(lib)         dlclose(lib)
#  define dynamic_library_find(lib, symbol)  dlsym(lib, symbol)
#endif""")


def print_dl_helper_macro():
    print("""#define DL_LIBRARY_FIND_CHECKED(name) \\
        name = (t##name)dynamic_library_find(lib, #name);

#define DL_LIBRARY_FIND(name) \\
        name = (t##name)dynamic_library_find(lib, #name); \\
        assert(name);

static DynamicLibrary lib;""")


def print_dl_close():
    print("""static void %sExit(void) {
  if(lib != NULL) {
    /*  Ignore errors. */
    dynamic_library_close(lib);
    lib = NULL;
  }
}""" % (LIB.lower()))
    print("")


def print_lib_path():
    # TODO(sergey): get rid of hardcoded libraries.
    print("""#ifdef _WIN32
  /* Expected in c:/windows/system or similar, no path needed. */
  const char *path = "nvcuda.dll";
#elif defined(__APPLE__)
  /* Default installation path. */
  const char *path = "/usr/local/cuda/lib/libcuda.dylib";
#else
  const char *path = "libcuda.so";
#endif""")


def print_init_guard():
    print("""  static int initialized = 0;
  static int result = 0;
  int error, driver_version;

  if (initialized) {
    return result;
  }

  initialized = 1;

  error = atexit(cuewExit);
  if (error) {
    return 0;
  }

  /* Load library. */
  lib = dynamic_library_open(path);

  if (lib == NULL) {
    return 0;
  }""")
    print("")


def print_driver_version_guard():
    # TODO(sergey): Currently it's hardcoded for CUDA only.
    print("""  /* Detect driver version. */
  driver_version = 1000;

  DL_LIBRARY_FIND_CHECKED(cuDriverGetVersion);
  if (cuDriverGetVersion) {
    cuDriverGetVersion(&driver_version);
  }

  /* We require version 4.0. */
  if (driver_version < 4000) {
    return 0;
  }""")


def print_dl_init():
    print("int %sInit(void) {" % (LIB.lower()))

    print("  /* Library paths. */")
    print_lib_path()
    print_init_guard()
    print_driver_version_guard()

    print("  /* Fetch all function pointers. */")
    for symbol in SYMBOLS:
        print("  DL_LIBRARY_FIND(%s);" % (symbol))

    print("")
    print("  result = 1;")
    print("  return result;")

    print("}")


def print_implementation():
    print_copyright()

    # TODO(sergey): Get rid of hardcoded header.
    print("#include <cuew.h>")
    print("#include <assert.h>")
    print("")

    print_dl_wrapper()
    print_dl_helper_macro()

    print("/* Function definitions. */")
    for symbol in SYMBOLS:
        print('t%s %s;' % (symbol, symbol))
    print("")

    print_dl_close()

    print("/* Implementation function. */")
    print_dl_init()

    print("")
    print("const char *cuewErrorString(CUresult result) { return \"\"; }")

if __name__ == "__main__":
    parse_files()

    if len(sys.argv) != 2:
        print("Usage: %s hdr|impl" % (sys.argv[0]))
        exit(1)

    if sys.argv[1] == "hdr":
        print_header()
    elif sys.argv[1] == "impl":
        print_implementation()
    else:
        print("Unknown command %s" % (sys.argv[1]))
        exit(1)
