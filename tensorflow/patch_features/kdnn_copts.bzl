load("//third_party/KDNN:build_defs.bzl", "if_enable_kdnn")

def kdnn_copts():
    return if_enable_kdnn(["-DENABLE_KDNN"])
