#!/bin/bash
# =============================================================================
# Master Workflow: PLUMED Metadynamics for Protein Orientation
# =============================================================================
#
# Usage:
#   ./workflow.sh prepare   - Prepare run from existing simulation
#   ./workflow.sh submit    - Submit metadynamics job
#   ./workflow.sh continue  - Continue from checkpoint
#   ./workflow.sh analyze   - Analyze FES
#   ./workflow.sh plot      - Generate plots
#   ./workflow.sh status    - Check job status
#   ./workflow.sh all       - Run full workflow
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$(pwd)}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# -----------------------------------------------------------------------------
cmd_prepare() {
    log_info "Preparing PLUMED run..."

    TPR="${1:-md.tpr}"
    CPT="${2:-md.cpt}"
    OUTDIR="${3:-plumed_run}"

    if [[ ! -f "$TPR" ]]; then
        log_error "TPR file not found: $TPR"
        exit 1
    fi

    bash "$SCRIPT_DIR/01_prepare_plumed.sh" "$TPR" "$CPT" "$OUTDIR"
    log_info "Preparation complete. cd $OUTDIR && ./workflow.sh submit"
}

# -----------------------------------------------------------------------------
cmd_submit() {
    log_info "Submitting metadynamics job..."

    if [[ ! -f "plumed.dat" ]]; then
        log_error "plumed.dat not found. Run 'workflow.sh prepare' first."
        exit 1
    fi

    JOB_ID=$(sbatch --parsable "$SCRIPT_DIR/02_run_metad.slurm")
    log_info "Submitted job: $JOB_ID"
    echo "$JOB_ID" >> .job_history

    log_info "Monitor with: tail -f logs/metad_${JOB_ID}.out"
}

# -----------------------------------------------------------------------------
cmd_continue() {
    log_info "Continuing simulation from checkpoint..."

    CPT=$(ls -t metad*.cpt 2>/dev/null | head -1)
    if [[ -z "$CPT" ]]; then
        log_error "No checkpoint found"
        exit 1
    fi

    log_info "Checkpoint: $CPT"
    JOB_ID=$(sbatch --parsable "$SCRIPT_DIR/03_continue_metad.slurm")
    log_info "Submitted continuation job: $JOB_ID"
    echo "$JOB_ID" >> .job_history
}

# -----------------------------------------------------------------------------
cmd_analyze() {
    log_info "Submitting analysis job..."

    if [[ ! -f "HILLS" ]]; then
        log_error "HILLS file not found. Run simulation first."
        exit 1
    fi

    JOB_ID=$(sbatch --parsable "$SCRIPT_DIR/04_analyze_fes.slurm")
    log_info "Submitted analysis job: $JOB_ID"
}

# -----------------------------------------------------------------------------
cmd_plot() {
    log_info "Generating plots..."

    module load python/3.12 2>/dev/null || true

    python3 "$SCRIPT_DIR/05_plot_results.py" analysis/
    log_info "Plots saved to analysis/"
}

# -----------------------------------------------------------------------------
cmd_status() {
    echo "============================================="
    echo "Job Status"
    echo "============================================="

    # Current jobs
    echo ""
    echo "Running jobs:"
    squeue -u "$USER" -o "%.10i %.15j %.8T %.10M %.6D %R" 2>/dev/null || echo "No jobs running"

    # Recent history
    if [[ -f .job_history ]]; then
        echo ""
        echo "Recent jobs:"
        tail -5 .job_history
    fi

    # Files
    echo ""
    echo "Files:"
    [[ -f COLVAR ]] && echo "  COLVAR: $(wc -l < COLVAR) lines, $(tail -1 COLVAR | awk '{print $1/1000}') ns"
    [[ -f HILLS ]] && echo "  HILLS: $(wc -l < HILLS) hills deposited"
    [[ -d analysis ]] && echo "  Analysis: $(ls analysis/*.dat 2>/dev/null | wc -l) FES files"
}

# -----------------------------------------------------------------------------
cmd_all() {
    log_info "Running full workflow..."

    TPR="${1:-md.tpr}"
    CPT="${2:-md.cpt}"

    cmd_prepare "$TPR" "$CPT" "plumed_run"
    cd plumed_run

    log_info "Submitting simulation..."
    JOB1=$(sbatch --parsable "$SCRIPT_DIR/02_run_metad.slurm")
    log_info "Metadynamics job: $JOB1"

    log_info "Submitting analysis (will run after metad)..."
    JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 "$SCRIPT_DIR/04_analyze_fes.slurm")
    log_info "Analysis job: $JOB2"

    echo ""
    log_info "Workflow submitted!"
    log_info "  Metadynamics: $JOB1"
    log_info "  Analysis: $JOB2 (depends on $JOB1)"
    log_info ""
    log_info "Monitor: squeue -u $USER"
}

# -----------------------------------------------------------------------------
cmd_help() {
    cat << 'EOF'
PLUMED Metadynamics Workflow for Protein Orientation

Usage: ./workflow.sh <command> [args]

Commands:
  prepare [tpr] [cpt] [dir]  Prepare run from existing simulation
  submit                      Submit metadynamics job
  continue                    Continue from checkpoint
  analyze                     Analyze FES from HILLS
  plot                        Generate publication plots
  status                      Show job status and progress
  all [tpr] [cpt]            Full workflow with dependencies

Example:
  # From directory with existing md.tpr and md.cpt:
  ./workflow.sh all md.tpr md.cpt

  # Or step by step:
  ./workflow.sh prepare md.tpr md.cpt
  cd plumed_run
  ./workflow.sh submit
  # ... wait for completion ...
  ./workflow.sh analyze
  ./workflow.sh plot

Files generated:
  COLVAR     - Time series of CVs (z, theta, phi)
  HILLS      - Deposited Gaussian hills
  WALLS      - Wall restraint energies
  analysis/  - FES reconstructions and plots

EOF
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
case "${1:-help}" in
    prepare)  shift; cmd_prepare "$@" ;;
    submit)   cmd_submit ;;
    continue) cmd_continue ;;
    analyze)  cmd_analyze ;;
    plot)     cmd_plot ;;
    status)   cmd_status ;;
    all)      shift; cmd_all "$@" ;;
    help|-h|--help) cmd_help ;;
    *)
        log_error "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac
