"""Provides the repository macro to import Eigen."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Eigen."""

    # Attention: tools parse and update these lines.
    # LINT.IfChange
    # KGemm kernels come from the boostkit Eigen fork. The archive is pinned to
    # an exact commit so the content behind the fixed sha256 can never change,
    # which keeps the build reproducible even if the branch keeps moving.
    # Note: gitcode does not serve archives under archive/<commit>.tar.gz or
    # archive/refs/tags/..., but it does resolve a commit id on the
    # archive/refs/heads/ path. The extracted top-level directory is still named
    # after the branch (eigen-dev-tf-2.20.0), hence the strip_prefix below.
    EIGEN_COMMIT = "a515c422db5f4ab2bda2b36ff3017f339b759969"
    EIGEN_SHA256 = "3b329f365350336a70be221454907aabe13f4f038bec7cea8bafbe1b1e7eace1"
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/eigen.cmake)

    tf_http_archive(
        name = "eigen_archive",
        build_file = "//third_party/eigen3:eigen_archive.BUILD",
        sha256 = EIGEN_SHA256,
        strip_prefix = "eigen-dev-tf-2.20.0",
        urls = tf_mirror_urls("https://raw.gitcode.com/boostkit/eigen/archive/refs/heads/{commit}.tar.gz".format(commit = EIGEN_COMMIT)),
    )
