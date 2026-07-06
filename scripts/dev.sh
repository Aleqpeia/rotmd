#!/usr/bin/env bash
# =============================================================================
# scripts/dev.sh — podman-based dev workflow for rotmd
#
# Works from either:
#   (a) a host shell where `podman` is on PATH, or
#   (b) a toolbox/distrobox container where podman lives on the host and is
#       reachable via `flatpak-spawn --host podman ...`
#
# Usage:
#   ./scripts/dev.sh build          # build the dev image
#   ./scripts/dev.sh shell          # interactive bash inside dev container
#   ./scripts/dev.sh test           # run the full test suite
#   ./scripts/dev.sh test -k foo    # run tests matching 'foo'
#   ./scripts/dev.sh test-fast      # test with JIT disabled (fast import)
#   ./scripts/dev.sh lint           # ruff check + format check
#   ./scripts/dev.sh typecheck      # mypy
#   ./scripts/dev.sh build-runtime  # build the production runtime image
#   ./scripts/dev.sh save-runtime   # export runtime image for Apptainer
#
# SELinux note (Fedora rootless podman):
#   The source directory is bind-mounted with :z (shared relabel).
#   Do NOT use :Z — it applies a private label that breaks your editor's
#   and the toolbox's access to the same directory on the host.
#   Alternative for CI or if :z causes issues: --security-opt label=disable
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_DEV="rotmd:dev"
IMAGE_RUNTIME="rotmd:latest"

# ---------------------------------------------------------------------------
# Resolve podman: prefer PATH, fall back to flatpak-spawn --host
# ---------------------------------------------------------------------------
if command -v podman &>/dev/null; then
    PODMAN="podman"
elif command -v flatpak-spawn &>/dev/null; then
    # Inside a toolbox/distrobox — bridge to the host podman
    PODMAN="flatpak-spawn --host podman"
else
    echo "ERROR: podman not found in PATH and flatpak-spawn is not available." >&2
    echo "  On Fedora Atomic hosts, install podman via:" >&2
    echo "    rpm-ostree install podman  (then reboot)" >&2
    echo "  Or run this script from a toolbox container (toolbox enter)." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Common env flags (mirror devcontainer.json containerEnv)
# ---------------------------------------------------------------------------
DEV_ENV=(
    -e ROTMD_NUMBA=0
    -e PYTHONDONTWRITEBYTECODE=1
    -e NUMBA_NUM_THREADS=2
    -e OMP_NUM_THREADS=2
    -e OPENBLAS_NUM_THREADS=2
)

# Bind-mount with :z (SELinux shared relabel) — safe with rootless podman
# on Fedora. The source goes to /workspace to match the devcontainer layout.
DEV_MOUNT="-v ${REPO_ROOT}:/workspace:z"

# ---------------------------------------------------------------------------
# Helper: run a one-shot command inside the dev container
# ---------------------------------------------------------------------------
_run_dev() {
    $PODMAN run --rm -it \
        "${DEV_ENV[@]}" \
        $DEV_MOUNT \
        -w /workspace \
        "$IMAGE_DEV" \
        "$@"
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
cmd="${1:-help}"
shift || true

case "$cmd" in
    build)
        echo "==> Building dev image (${IMAGE_DEV}) …"
        $PODMAN build \
            --target dev \
            -t "$IMAGE_DEV" \
            "$REPO_ROOT"
        ;;

    build-runtime)
        echo "==> Building runtime image (${IMAGE_RUNTIME}) …"
        $PODMAN build \
            --target runtime \
            -t "$IMAGE_RUNTIME" \
            "$REPO_ROOT"
        ;;

    save-runtime)
        # Export the runtime image to a tar archive for Apptainer conversion.
        # On the HPC login node: apptainer build rotmd.sif docker-archive://rotmd.tar
        OUT="${1:-${REPO_ROOT}/rotmd.tar}"
        echo "==> Saving runtime image to ${OUT} …"
        $PODMAN build --target runtime -t "$IMAGE_RUNTIME" "$REPO_ROOT"
        $PODMAN save "$IMAGE_RUNTIME" -o "$OUT"
        echo "Transfer rotmd.tar to HPC, then on the login node:"
        echo "  apptainer build rotmd.sif docker-archive://rotmd.tar"
        ;;

    shell)
        echo "==> Opening interactive shell in dev container …"
        echo "    Source is live-mounted at /workspace (edits visible immediately)."
        _run_dev /bin/bash
        ;;

    test)
        echo "==> Running pytest …"
        _run_dev pytest "${@}"
        ;;

    test-fast)
        # ROTMD_NUMBA=0 already set in DEV_ENV; add -n auto for parallel runs
        echo "==> Running pytest (JIT disabled, parallel) …"
        _run_dev pytest -n auto "${@}"
        ;;

    lint)
        echo "==> ruff check …"
        _run_dev ruff check src tests
        echo "==> ruff format --check …"
        _run_dev ruff format --check src tests
        ;;

    typecheck)
        echo "==> mypy …"
        _run_dev mypy src
        ;;

    help|--help|-h)
        sed -n '/^# Usage:/,/^# -----/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \?//'
        ;;

    *)
        echo "Unknown command: $cmd" >&2
        echo "Run: ./scripts/dev.sh help" >&2
        exit 1
        ;;
esac
