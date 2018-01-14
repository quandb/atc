package(default_visibility = ["//visibility:public"])

py_library(
    name = "flags",
    srcs = glob(["*.py"]),
)

py_binary(
    name = "flags_example",
    srcs = ["flags_example.py"],
    deps = [
        ":flags",
    ],
)
