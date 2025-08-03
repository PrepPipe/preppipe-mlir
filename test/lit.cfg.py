import os
import sys
import platform
import re
import shutil
import subprocess
import tempfile

import lit
import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

config.name = "PREPPIPE_MLIR"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

config.suffixes = [".mlir", ".py", ".runlit"]

config.test_source_root = os.path.dirname(__file__)

config.test_exec_root = os.path.join(config.preppipe_mlir_obj_root, 'test')

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(('%preppipe-mlir-opt', os.path.join(config.llvm_tools_dir, 'preppipe-mlir-opt')))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

# llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "Inputs",
    "Examples",
    "CMakeLists.txt",
    "README.txt",
    "LICENSE.txt",
    "lit.cfg.py",
    "lit.site.cfg.py",
]

config.standalone_tools_dir = os.path.join(config.preppipe_mlir_obj_root, "bin")

llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment(
    "PATH", os.path.join(config.llvm_build_dir, "bin"), append_path=True
)

tool_dirs = [
    config.standalone_tools_dir,
    config.llvm_tools_dir,
    config.preppipe_mlir_obj_root,
]
tools = [
    "preppipe-mlir-opt",
    ToolSubst("%PYTHON", config.python_executable, unresolved="ignore"),
]

python_executable = config.python_executable
# Python configuration with sanitizer requires some magic preloading. This will only work on clang/linux/darwin.
# TODO: detect Windows situation (or mark these tests as unsupported on these platforms).
if "asan" in config.available_features:
    if "Linux" in config.host_os:
        python_executable = f"LD_PRELOAD=$({config.host_cxx} -print-file-name=libclang_rt.asan-{config.host_arch}.so) {config.python_executable}"
    if "Darwin" in config.host_os:
        # Ensure we use a non-shim Python executable, for the `DYLD_INSERT_LIBRARIES`
        # env variable to take effect
        real_python_executable = find_real_python_interpreter()
        if real_python_executable:
            python_executable = real_python_executable
            lit_config.note(
                "Using {} instead of {}".format(
                    python_executable, config.python_executable
                )
            )

        asan_rtlib = get_asan_rtlib()
        lit_config.note("Using ASan rtlib {}".format(asan_rtlib))
        config.environment["MallocNanoZone"] = "0"
        config.environment["ASAN_OPTIONS"] = "detect_stack_use_after_return=1"
        config.environment["DYLD_INSERT_LIBRARIES"] = asan_rtlib


# On Windows the path to python could contains spaces in which case it needs to be provided in quotes.
# This is the equivalent of how %python is setup in llvm/utils/lit/lit/llvm/config.py.
elif "Windows" in config.host_os:
    python_executable = '"%s"' % (python_executable)
tools.extend(
    [
        ToolSubst("%PYTHON", python_executable, unresolved="ignore"),
    ]
)

llvm_config.add_tool_substitutions(tools, tool_dirs)

# Add the python path for both the source and binary tree.
# Note that presently, the python sources come from the source tree and the
# binaries come from the build tree. This should be unified to the build tree
# by copying/linking sources to build.
if config.enable_bindings_python:
    config.environment["PYTHONPATH"] = os.getenv("MLIR_LIT_PYTHONPATH", "")
    llvm_config.with_environment(
        "PYTHONPATH",
        [
            os.path.join(config.preppipe_mlir_python_packages_dir, "preppipe_mlir"),
        ],
        append_path=True,
    )
