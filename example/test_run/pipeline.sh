#!/usr/bin/env bash
set -Eeuo pipefail

DEFAULT_CONFIG="./pipeline.config.sh"
CONFIG_FILE=""
PIPELINE_TARGET="all"
SHOW_HELP=0

declare -a POSITIONAL_ARGS=()
declare -a SET_OVERRIDES=()
declare -a ENABLE_STAGE_OVERRIDES=()
declare -a DISABLE_STAGE_OVERRIDES=()

THREADS_OVERRIDE=""
TARGET_CHROMS_OVERRIDE=""
OUTPUT_ROOT_OVERRIDE=""
PRIMARY_GROUP_OVERRIDE=""
SECONDARY_GROUP_OVERRIDE=""
SAMPLE_PETS_TOTAL_OVERRIDE=""

QC_ARGS_OVERRIDE=""
PRE_ARGS_OVERRIDE=""
COMBINE_ARGS_OVERRIDE=""
EST_RES_ARGS_OVERRIDE=""
EST_DIS_ARGS_OVERRIDE=""
EST_SIM_ARGS_OVERRIDE=""
CALL_PEAKS_ARGS_OVERRIDE=""
CALL_LOOPS_ARGS_OVERRIDE=""
CALL_DOMAINS_ARGS_OVERRIDE=""
SAMPLE_PETS_ARGS_OVERRIDE=""
CALL_DIFF_LOOPS_ARGS_OVERRIDE=""
AGG_PEAK_ARGS_OVERRIDE=""
AGG_VIEWPOINT_ARGS_OVERRIDE=""
AGG_LOOP_ARGS_OVERRIDE=""
AGG_TWO_ANCHOR_ARGS_OVERRIDE=""
AGG_DOMAIN_ARGS_OVERRIDE=""
FILTER_PETS_ARGS_OVERRIDE=""
PLOT_DOMAIN_ARGS_OVERRIDE=""
PLOT_LOOP_ARGS_OVERRIDE=""
PLOT_FILTERED_ARGS_OVERRIDE=""
PLOT_FILTERED_ARCH_ARGS_OVERRIDE=""
MONTAGE_ALL_ARGS_OVERRIDE=""
MONTAGE_VIEWPOINT_ARGS_OVERRIDE=""
DUMP_BED_ARGS_OVERRIDE=""
DUMP_BEDPE_ARGS_OVERRIDE=""
DUMP_BDG_ARGS_OVERRIDE=""
DUMP_WASHU_ARGS_OVERRIDE=""
DUMP_HIC_ARGS_OVERRIDE=""
DUMP_MATRIX_ARGS_OVERRIDE=""
QUANT_PEAKS_ARGS_OVERRIDE=""
QUANT_LOOPS_ARGS_OVERRIDE=""
QUANT_DOMAIN_ARGS_OVERRIDE=""
ANNOTATION_ARGS_OVERRIDE=""

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

split_csv() {
  local csv="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<< "${csv}"
}

join_by() {
  local delim="$1"
  shift
  local first=1
  local item
  for item in "$@"; do
    if (( first )); then
      printf '%s' "${item}"
      first=0
    else
      printf '%s%s' "${delim}" "${item}"
    fi
  done
}

set_var_value() {
  local key="$1"
  local value="$2"
  printf -v "${key}" '%s' "${value}"
}

replace_or_append_arg() {
  local value="$1"
  local flag="$2"
  local replacement="$3"
  local -a args=()
  local -a updated=()
  local i=0
  local replaced=0

  [[ -n "${value}" ]] && read -r -a args <<< "${value}"

  while (( i < ${#args[@]} )); do
    if [[ "${args[i]}" == "${flag}" ]]; then
      updated+=("${flag}" "${replacement}")
      replaced=1
      ((i += 2))
    else
      updated+=("${args[i]}")
      ((i += 1))
    fi
  done

  if (( replaced == 0 )); then
    updated+=("${flag}" "${replacement}")
  fi

  printf '%s' "$(join_by ' ' "${updated[@]}")"
}

enable_or_disable_flag() {
  local value="$1"
  local flag="$2"
  local enabled="$3"
  local -a args=()
  local -a updated=()
  local item

  [[ -n "${value}" ]] && read -r -a args <<< "${value}"

  for item in "${args[@]}"; do
    [[ "${item}" == "${flag}" ]] && continue
    updated+=("${item}")
  done

  if [[ "${enabled}" == "1" ]]; then
    updated+=("${flag}")
  fi

  printf '%s' "$(join_by ' ' "${updated[@]}")"
}

apply_set_override() {
  local kv="$1"
  [[ "${kv}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || die "Invalid --set value: ${kv}"
  local key="${kv%%=*}"
  local value="${kv#*=}"
  set_var_value "${key}" "${value}"
}

set_stage_toggle() {
  local stage="$1"
  local enabled="$2"
  local key=""

  case "${stage}" in
    clean) key="ENABLE_CLEAN" ;;
    qc) key="ENABLE_QC" ;;
    pre) key="ENABLE_PRE" ;;
    estimate) key="ENABLE_ESTIMATE" ;;
    calling) key="ENABLE_CALLING" ;;
    vis) key="ENABLE_VIS" ;;
    compare) key="ENABLE_COMPARE" ;;
    export) key="ENABLE_EXPORT" ;;
    quant) key="ENABLE_QUANT" ;;
    annotation) key="ENABLE_ANNOTATION" ;;
    extra) key="ENABLE_EXTRA_ANALYSIS" ;;
    *) die "Unknown stage: ${stage}" ;;
  esac

  set_var_value "${key}" "${enabled}"
}

apply_cli_overrides() {
  local kv

  [[ -n "${OUTPUT_ROOT_OVERRIDE}" ]] && OUTPUT_ROOT="${OUTPUT_ROOT_OVERRIDE}"
  [[ -n "${PRIMARY_GROUP_OVERRIDE}" ]] && PRIMARY_GROUP="${PRIMARY_GROUP_OVERRIDE}"
  [[ -n "${SECONDARY_GROUP_OVERRIDE}" ]] && SECONDARY_GROUP="${SECONDARY_GROUP_OVERRIDE}"
  [[ -n "${SAMPLE_PETS_TOTAL_OVERRIDE}" ]] && SAMPLE_PETS_TOTAL="${SAMPLE_PETS_TOTAL_OVERRIDE}"

  [[ -n "${QC_ARGS_OVERRIDE}" ]] && QC_ARGS="${QC_ARGS_OVERRIDE}"
  [[ -n "${PRE_ARGS_OVERRIDE}" ]] && PRE_ARGS="${PRE_ARGS_OVERRIDE}"
  [[ -n "${COMBINE_ARGS_OVERRIDE}" ]] && COMBINE_ARGS="${COMBINE_ARGS_OVERRIDE}"
  [[ -n "${EST_RES_ARGS_OVERRIDE}" ]] && EST_RES_ARGS="${EST_RES_ARGS_OVERRIDE}"
  [[ -n "${EST_DIS_ARGS_OVERRIDE}" ]] && EST_DIS_ARGS="${EST_DIS_ARGS_OVERRIDE}"
  [[ -n "${EST_SIM_ARGS_OVERRIDE}" ]] && EST_SIM_ARGS="${EST_SIM_ARGS_OVERRIDE}"
  [[ -n "${CALL_PEAKS_ARGS_OVERRIDE}" ]] && CALL_PEAKS_ARGS="${CALL_PEAKS_ARGS_OVERRIDE}"
  [[ -n "${CALL_LOOPS_ARGS_OVERRIDE}" ]] && CALL_LOOPS_ARGS="${CALL_LOOPS_ARGS_OVERRIDE}"
  [[ -n "${CALL_DOMAINS_ARGS_OVERRIDE}" ]] && CALL_DOMAINS_ARGS="${CALL_DOMAINS_ARGS_OVERRIDE}"
  [[ -n "${SAMPLE_PETS_ARGS_OVERRIDE}" ]] && SAMPLE_PETS_ARGS="${SAMPLE_PETS_ARGS_OVERRIDE}"
  [[ -n "${CALL_DIFF_LOOPS_ARGS_OVERRIDE}" ]] && CALL_DIFF_LOOPS_ARGS="${CALL_DIFF_LOOPS_ARGS_OVERRIDE}"
  [[ -n "${AGG_PEAK_ARGS_OVERRIDE}" ]] && AGG_PEAK_ARGS="${AGG_PEAK_ARGS_OVERRIDE}"
  [[ -n "${AGG_VIEWPOINT_ARGS_OVERRIDE}" ]] && AGG_VIEWPOINT_ARGS="${AGG_VIEWPOINT_ARGS_OVERRIDE}"
  [[ -n "${AGG_LOOP_ARGS_OVERRIDE}" ]] && AGG_LOOP_ARGS="${AGG_LOOP_ARGS_OVERRIDE}"
  [[ -n "${AGG_TWO_ANCHOR_ARGS_OVERRIDE}" ]] && AGG_TWO_ANCHOR_ARGS="${AGG_TWO_ANCHOR_ARGS_OVERRIDE}"
  [[ -n "${AGG_DOMAIN_ARGS_OVERRIDE}" ]] && AGG_DOMAIN_ARGS="${AGG_DOMAIN_ARGS_OVERRIDE}"
  [[ -n "${FILTER_PETS_ARGS_OVERRIDE}" ]] && FILTER_PETS_ARGS="${FILTER_PETS_ARGS_OVERRIDE}"
  [[ -n "${PLOT_DOMAIN_ARGS_OVERRIDE}" ]] && PLOT_DOMAIN_ARGS="${PLOT_DOMAIN_ARGS_OVERRIDE}"
  [[ -n "${PLOT_LOOP_ARGS_OVERRIDE}" ]] && PLOT_LOOP_ARGS="${PLOT_LOOP_ARGS_OVERRIDE}"
  [[ -n "${PLOT_FILTERED_ARGS_OVERRIDE}" ]] && PLOT_FILTERED_ARGS="${PLOT_FILTERED_ARGS_OVERRIDE}"
  [[ -n "${PLOT_FILTERED_ARCH_ARGS_OVERRIDE}" ]] && PLOT_FILTERED_ARCH_ARGS="${PLOT_FILTERED_ARCH_ARGS_OVERRIDE}"
  [[ -n "${MONTAGE_ALL_ARGS_OVERRIDE}" ]] && MONTAGE_ALL_ARGS="${MONTAGE_ALL_ARGS_OVERRIDE}"
  [[ -n "${MONTAGE_VIEWPOINT_ARGS_OVERRIDE}" ]] && MONTAGE_VIEWPOINT_ARGS="${MONTAGE_VIEWPOINT_ARGS_OVERRIDE}"
  [[ -n "${DUMP_BED_ARGS_OVERRIDE}" ]] && DUMP_BED_ARGS="${DUMP_BED_ARGS_OVERRIDE}"
  [[ -n "${DUMP_BEDPE_ARGS_OVERRIDE}" ]] && DUMP_BEDPE_ARGS="${DUMP_BEDPE_ARGS_OVERRIDE}"
  [[ -n "${DUMP_BDG_ARGS_OVERRIDE}" ]] && DUMP_BDG_ARGS="${DUMP_BDG_ARGS_OVERRIDE}"
  [[ -n "${DUMP_WASHU_ARGS_OVERRIDE}" ]] && DUMP_WASHU_ARGS="${DUMP_WASHU_ARGS_OVERRIDE}"
  [[ -n "${DUMP_HIC_ARGS_OVERRIDE}" ]] && DUMP_HIC_ARGS="${DUMP_HIC_ARGS_OVERRIDE}"
  [[ -n "${DUMP_MATRIX_ARGS_OVERRIDE}" ]] && DUMP_MATRIX_ARGS="${DUMP_MATRIX_ARGS_OVERRIDE}"
  [[ -n "${QUANT_PEAKS_ARGS_OVERRIDE}" ]] && QUANT_PEAKS_ARGS="${QUANT_PEAKS_ARGS_OVERRIDE}"
  [[ -n "${QUANT_LOOPS_ARGS_OVERRIDE}" ]] && QUANT_LOOPS_ARGS="${QUANT_LOOPS_ARGS_OVERRIDE}"
  [[ -n "${QUANT_DOMAIN_ARGS_OVERRIDE}" ]] && QUANT_DOMAIN_ARGS="${QUANT_DOMAIN_ARGS_OVERRIDE}"
  [[ -n "${ANNOTATION_ARGS_OVERRIDE}" ]] && ANNOTATION_ARGS="${ANNOTATION_ARGS_OVERRIDE}"

  if [[ -n "${THREADS_OVERRIDE}" ]]; then
    THREADS="${THREADS_OVERRIDE}"
    QC_ARGS="$(replace_or_append_arg "${QC_ARGS:-}" "-p" "${THREADS}")"
    PRE_ARGS="$(replace_or_append_arg "${PRE_ARGS:-}" "-p" "${THREADS}")"
    COMBINE_ARGS="$(replace_or_append_arg "${COMBINE_ARGS:-}" "-p" "${THREADS}")"
    EST_RES_ARGS="$(replace_or_append_arg "${EST_RES_ARGS:-}" "-p" "${THREADS}")"
    EST_DIS_ARGS="$(replace_or_append_arg "${EST_DIS_ARGS:-}" "-p" "${THREADS}")"
    EST_SIM_ARGS="$(replace_or_append_arg "${EST_SIM_ARGS:-}" "-p" "${THREADS}")"
    CALL_PEAKS_ARGS="$(replace_or_append_arg "${CALL_PEAKS_ARGS:-}" "-p" "${THREADS}")"
    CALL_LOOPS_ARGS="$(replace_or_append_arg "${CALL_LOOPS_ARGS:-}" "-p" "${THREADS}")"
    CALL_DOMAINS_ARGS="$(replace_or_append_arg "${CALL_DOMAINS_ARGS:-}" "-p" "${THREADS}")"
    SAMPLE_PETS_ARGS="$(replace_or_append_arg "${SAMPLE_PETS_ARGS:-}" "-p" "${THREADS}")"
    CALL_DIFF_LOOPS_ARGS="$(replace_or_append_arg "${CALL_DIFF_LOOPS_ARGS:-}" "-p" "${THREADS}")"
    AGG_PEAK_ARGS="$(replace_or_append_arg "${AGG_PEAK_ARGS:-}" "-p" "${THREADS}")"
    AGG_VIEWPOINT_ARGS="$(replace_or_append_arg "${AGG_VIEWPOINT_ARGS:-}" "-p" "${THREADS}")"
    AGG_LOOP_ARGS="$(replace_or_append_arg "${AGG_LOOP_ARGS:-}" "-p" "${THREADS}")"
    AGG_TWO_ANCHOR_ARGS="$(replace_or_append_arg "${AGG_TWO_ANCHOR_ARGS:-}" "-p" "${THREADS}")"
    AGG_DOMAIN_ARGS="$(replace_or_append_arg "${AGG_DOMAIN_ARGS:-}" "-p" "${THREADS}")"
    FILTER_PETS_ARGS="$(replace_or_append_arg "${FILTER_PETS_ARGS:-}" "-p" "${THREADS}")"
    PLOT_DOMAIN_ARGS="$(replace_or_append_arg "${PLOT_DOMAIN_ARGS:-}" "-p" "${THREADS}")"
    PLOT_LOOP_ARGS="$(replace_or_append_arg "${PLOT_LOOP_ARGS:-}" "-p" "${THREADS}")"
    PLOT_FILTERED_ARGS="$(replace_or_append_arg "${PLOT_FILTERED_ARGS:-}" "-p" "${THREADS}")"
    PLOT_FILTERED_ARCH_ARGS="$(replace_or_append_arg "${PLOT_FILTERED_ARCH_ARGS:-}" "-p" "${THREADS}")"
    MONTAGE_ALL_ARGS="$(replace_or_append_arg "${MONTAGE_ALL_ARGS:-}" "-p" "${THREADS}")"
    MONTAGE_VIEWPOINT_ARGS="$(replace_or_append_arg "${MONTAGE_VIEWPOINT_ARGS:-}" "-p" "${THREADS}")"
    DUMP_BED_ARGS="$(replace_or_append_arg "${DUMP_BED_ARGS:-}" "-p" "${THREADS}")"
    DUMP_BEDPE_ARGS="$(replace_or_append_arg "${DUMP_BEDPE_ARGS:-}" "-p" "${THREADS}")"
    DUMP_BDG_ARGS="$(replace_or_append_arg "${DUMP_BDG_ARGS:-}" "-p" "${THREADS}")"
    DUMP_WASHU_ARGS="$(replace_or_append_arg "${DUMP_WASHU_ARGS:-}" "-p" "${THREADS}")"
    DUMP_HIC_ARGS="$(replace_or_append_arg "${DUMP_HIC_ARGS:-}" "-p" "${THREADS}")"
    DUMP_MATRIX_ARGS="$(replace_or_append_arg "${DUMP_MATRIX_ARGS:-}" "-p" "${THREADS}")"
    QUANT_PEAKS_ARGS="$(replace_or_append_arg "${QUANT_PEAKS_ARGS:-}" "-p" "${THREADS}")"
    QUANT_LOOPS_ARGS="$(replace_or_append_arg "${QUANT_LOOPS_ARGS:-}" "-p" "${THREADS}")"
    QUANT_DOMAIN_ARGS="$(replace_or_append_arg "${QUANT_DOMAIN_ARGS:-}" "-p" "${THREADS}")"
    ANNOTATION_ARGS="$(replace_or_append_arg "${ANNOTATION_ARGS:-}" "-p" "${THREADS}")"
  fi

  if [[ -n "${TARGET_CHROMS_OVERRIDE}" ]]; then
    TARGET_CHROMS="${TARGET_CHROMS_OVERRIDE}"
    PRE_ARGS="$(replace_or_append_arg "${PRE_ARGS:-}" "-c" "${TARGET_CHROMS}")"
    EST_DIS_ARGS="$(replace_or_append_arg "${EST_DIS_ARGS:-}" "-c" "${TARGET_CHROMS}")"
  fi

  for kv in "${SET_OVERRIDES[@]}"; do
    apply_set_override "${kv}"
  done

  for kv in "${ENABLE_STAGE_OVERRIDES[@]}"; do
    set_stage_toggle "${kv}" "1"
  done

  for kv in "${DISABLE_STAGE_OVERRIDES[@]}"; do
    set_stage_toggle "${kv}" "0"
  done
}

parse_cli() {
  while (($# > 0)); do
    case "$1" in
      -h|--help)
        SHOW_HELP=1
        shift
        ;;
      -c|--config)
        CONFIG_FILE="$2"
        shift 2
        ;;
      -t|--target)
        PIPELINE_TARGET="$2"
        shift 2
        ;;
      --threads)
        THREADS_OVERRIDE="$2"
        shift 2
        ;;
      --chrom)
        TARGET_CHROMS_OVERRIDE="$2"
        shift 2
        ;;
      --output-root)
        OUTPUT_ROOT_OVERRIDE="$2"
        shift 2
        ;;
      --primary-group)
        PRIMARY_GROUP_OVERRIDE="$2"
        shift 2
        ;;
      --secondary-group)
        SECONDARY_GROUP_OVERRIDE="$2"
        shift 2
        ;;
      --sample-total)
        SAMPLE_PETS_TOTAL_OVERRIDE="$2"
        shift 2
        ;;
      --enable-stage)
        ENABLE_STAGE_OVERRIDES+=("$2")
        shift 2
        ;;
      --disable-stage)
        DISABLE_STAGE_OVERRIDES+=("$2")
        shift 2
        ;;
      --qc-args)
        QC_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --pre-args)
        PRE_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --combine-args)
        COMBINE_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --est-res-args)
        EST_RES_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --est-dis-args)
        EST_DIS_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --est-sim-args)
        EST_SIM_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --call-peaks-args)
        CALL_PEAKS_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --call-loops-args)
        CALL_LOOPS_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --call-domains-args)
        CALL_DOMAINS_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --sample-pets-args)
        SAMPLE_PETS_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --diff-loops-args)
        CALL_DIFF_LOOPS_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --agg-peak-args)
        AGG_PEAK_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --agg-viewpoint-args)
        AGG_VIEWPOINT_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --agg-loop-args)
        AGG_LOOP_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --agg-two-anchor-args)
        AGG_TWO_ANCHOR_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --agg-domain-args)
        AGG_DOMAIN_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --filter-pets-args)
        FILTER_PETS_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --plot-domain-args)
        PLOT_DOMAIN_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --plot-loop-args)
        PLOT_LOOP_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --plot-filtered-args)
        PLOT_FILTERED_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --plot-filtered-arch-args)
        PLOT_FILTERED_ARCH_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --montage-all-args)
        MONTAGE_ALL_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --montage-viewpoint-args)
        MONTAGE_VIEWPOINT_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --dump-bed-args)
        DUMP_BED_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --dump-bedpe-args)
        DUMP_BEDPE_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --dump-bdg-args)
        DUMP_BDG_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --dump-washu-args)
        DUMP_WASHU_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --dump-hic-args)
        DUMP_HIC_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --dump-matrix-args)
        DUMP_MATRIX_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --quant-peaks-args)
        QUANT_PEAKS_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --quant-loops-args)
        QUANT_LOOPS_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --quant-domain-args)
        QUANT_DOMAIN_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --annotation-args)
        ANNOTATION_ARGS_OVERRIDE="$2"
        shift 2
        ;;
      --set)
        SET_OVERRIDES+=("$2")
        shift 2
        ;;
      --)
        shift
        while (($# > 0)); do
          POSITIONAL_ARGS+=("$1")
          shift
        done
        ;;
      -*)
        die "Unknown option: $1"
        ;;
      *)
        POSITIONAL_ARGS+=("$1")
        shift
        ;;
    esac
  done

  if (( ${#POSITIONAL_ARGS[@]} >= 1 )); then
    if [[ -z "${CONFIG_FILE}" && -f "${POSITIONAL_ARGS[0]}" ]]; then
      CONFIG_FILE="${POSITIONAL_ARGS[0]}"
      if (( ${#POSITIONAL_ARGS[@]} >= 2 )) && [[ "${PIPELINE_TARGET}" == "all" ]]; then
        PIPELINE_TARGET="${POSITIONAL_ARGS[1]}"
      fi
    elif [[ "${PIPELINE_TARGET}" == "all" ]]; then
      PIPELINE_TARGET="${POSITIONAL_ARGS[0]}"
    fi
  fi

  [[ -n "${CONFIG_FILE}" ]] || CONFIG_FILE="${DEFAULT_CONFIG}"
}

run_stage_cmd() {
  local stage="$1"
  shift
  local stage_log="${LOG_DIR}/${stage}.log"
  log "$*"
  "$@" 2>&1 | tee -a "${stage_log}"
}

mark_done() {
  : > "${STATE_DIR}/$1.done"
}

is_done() {
  [[ -f "${STATE_DIR}/$1.done" ]]
}

should_run_target() {
  local stage="$1"
  [[ "${PIPELINE_TARGET}" == "all" || "${PIPELINE_TARGET}" == "${stage}" ]]
}

ensure_samples_exist() {
  local sample
  for sample in "${!SAMPLE_FILES[@]}"; do
    [[ -f "${SAMPLE_FILES[$sample]}" ]] || die "Missing input for ${sample}: ${SAMPLE_FILES[$sample]}"
  done
}

sample_dataset_dir() {
  printf '%s/%s' "${DATA_DIR}" "$1"
}

group_dataset_dir() {
  printf '%s/%s' "${DATA_DIR}" "$1"
}

group_output_prefix() {
  printf '%s/%s' "${REPORT_DIR}" "$1"
}

group_fig_prefix() {
  printf '%s/%s' "${FIG_DIR}" "$1"
}

run_hook_if_set() {
  local hook_cmd="$1"
  local stage="$2"
  if [[ -n "${hook_cmd}" ]]; then
    log "Running hook for ${stage}: ${hook_cmd}"
    bash -lc "${hook_cmd}" 2>&1 | tee -a "${LOG_DIR}/${stage}.log"
  fi
}

stage_clean() {
  [[ "${ENABLE_CLEAN:-0}" == "1" ]] || return 0
  should_run_target "clean" || return 0
  is_done "clean" && [[ "${PIPELINE_TARGET}" == "all" ]] && return 0
  log "Cleaning ${OUTPUT_ROOT}"
  rm -rf "${STATE_DIR}" "${LOG_DIR}" "${DATA_DIR}" "${REPORT_DIR}" "${FIG_DIR}" "${EXPORT_DIR}"
  mkdir -p "${STATE_DIR}" "${LOG_DIR}" "${DATA_DIR}" "${REPORT_DIR}" "${FIG_DIR}" "${EXPORT_DIR}"
  mark_done "clean"
}

stage_qc() {
  [[ "${ENABLE_QC:-1}" == "1" ]] || return 0
  should_run_target "qc" || return 0
  is_done "qc" && [[ "${PIPELINE_TARGET}" == "all" ]] && return 0

  local files=()
  local sample
  for sample in "${!SAMPLE_FILES[@]}"; do
    files+=("${SAMPLE_FILES[$sample]}")
  done
  local file_csv
  local -a qc_args=()
  file_csv="$(join_by ',' "${files[@]}")"
  [[ -n "${QC_ARGS:-}" ]] && read -r -a qc_args <<< "${QC_ARGS}"
  run_stage_cmd "qc" cLoops2 qc -f "${file_csv}" -o "${REPORT_DIR}/qc" "${qc_args[@]}"
  mark_done "qc"
}

preprocess_single_sample() {
  local sample="$1"
  local out_dir
  local -a pre_args=()
  out_dir="$(sample_dataset_dir "${sample}")"
  mkdir -p "${out_dir}"
  [[ -n "${PRE_ARGS:-}" ]] && read -r -a pre_args <<< "${PRE_ARGS}"
  run_stage_cmd "pre_${sample}" cLoops2 pre -f "${SAMPLE_FILES[$sample]}" -o "${out_dir}" "${pre_args[@]}"
}

preprocess_group_direct() {
  local group="$1"
  local reps=()
  local files=()
  local rep
  local file_csv
  local out_dir
  local -a pre_args=()
  split_csv "${GROUP_REPLICATES[$group]}" reps
  for rep in "${reps[@]}"; do
    files+=("${SAMPLE_FILES[$rep]}")
  done
  file_csv="$(join_by ',' "${files[@]}")"
  out_dir="$(group_dataset_dir "${group}")"
  mkdir -p "${out_dir}"
  [[ -n "${PRE_ARGS:-}" ]] && read -r -a pre_args <<< "${PRE_ARGS}"
  run_stage_cmd "pre_${group}" cLoops2 pre -f "${file_csv}" -o "${out_dir}" "${pre_args[@]}"
}

combine_group_from_replicates() {
  local group="$1"
  local reps=()
  local rep_dirs=()
  local rep
  local dir_csv
  local out_dir
  local -a combine_args=()
  split_csv "${GROUP_REPLICATES[$group]}" reps
  for rep in "${reps[@]}"; do
    rep_dirs+=("$(sample_dataset_dir "${rep}")")
  done
  dir_csv="$(join_by ',' "${rep_dirs[@]}")"
  out_dir="$(group_dataset_dir "${group}")"
  mkdir -p "${out_dir}"
  [[ -n "${COMBINE_ARGS:-}" ]] && read -r -a combine_args <<< "${COMBINE_ARGS}"
  run_stage_cmd "combine_${group}" cLoops2 combine -ds "${dir_csv}" -o "${out_dir}" "${combine_args[@]}"
}

stage_pre() {
  [[ "${ENABLE_PRE:-1}" == "1" ]] || return 0
  should_run_target "pre" || return 0
  is_done "pre" && [[ "${PIPELINE_TARGET}" == "all" ]] && return 0

  local group
  local reps=()
  local rep
  local mode
  for group in "${!GROUP_REPLICATES[@]}"; do
    mode="${GROUP_BUILD_MODE[$group]:-pre}"
    split_csv "${GROUP_REPLICATES[$group]}" reps
    if [[ "${mode}" == "combine" ]]; then
      for rep in "${reps[@]}"; do
        preprocess_single_sample "${rep}"
      done
      combine_group_from_replicates "${group}"
    else
      preprocess_group_direct "${group}"
    fi
  done
  mark_done "pre"
}

stage_estimate() {
  [[ "${ENABLE_ESTIMATE:-1}" == "1" ]] || return 0
  should_run_target "estimate" || return 0
  is_done "estimate" && [[ "${PIPELINE_TARGET}" == "all" ]] && return 0

  local group
  local sample
  local sim_dirs=()
  local sim_csv
  local -a est_res_args=()
  local -a est_dis_args=()
  local -a est_sim_args=()

  [[ -n "${EST_RES_ARGS:-}" ]] && read -r -a est_res_args <<< "${EST_RES_ARGS}"
  for group in "${!GROUP_REPLICATES[@]}"; do
    run_stage_cmd "estimate" cLoops2 estRes -d "$(group_dataset_dir "${group}")" -o "$(group_output_prefix "${group}")" "${est_res_args[@]}"
  done

  [[ -n "${EST_DIS_ARGS:-}" ]] && read -r -a est_dis_args <<< "${EST_DIS_ARGS}"
  run_stage_cmd "estimate" cLoops2 estDis -d "$(group_dataset_dir "${PRIMARY_GROUP}")" -o "$(group_output_prefix "${PRIMARY_GROUP}")" "${est_dis_args[@]}"

  for sample in "${!SAMPLE_FILES[@]}"; do
    [[ -d "$(sample_dataset_dir "${sample}")" ]] && sim_dirs+=("$(sample_dataset_dir "${sample}")")
  done
  for group in "${!GROUP_REPLICATES[@]}"; do
    sim_dirs+=("$(group_dataset_dir "${group}")")
  done
  sim_csv="$(join_by ',' "${sim_dirs[@]}")"
  [[ -n "${EST_SIM_ARGS:-}" ]] && read -r -a est_sim_args <<< "${EST_SIM_ARGS}"
  run_stage_cmd "estimate" cLoops2 estSim -ds "${sim_csv}" -o "${REPORT_DIR}/similarity" "${est_sim_args[@]}"
  mark_done "estimate"
}

stage_calling() {
  [[ "${ENABLE_CALLING:-1}" == "1" ]] || return 0
  should_run_target "calling" || return 0
  is_done "calling" && [[ "${PIPELINE_TARGET}" == "all" ]] && return 0

  local -a peak_args=()
  local -a loop_args=()
  local -a domain_args=()
  [[ -n "${CALL_PEAKS_ARGS:-}" ]] && read -r -a peak_args <<< "${CALL_PEAKS_ARGS}"
  [[ -n "${CALL_LOOPS_ARGS:-}" ]] && read -r -a loop_args <<< "${CALL_LOOPS_ARGS}"
  [[ -n "${CALL_DOMAINS_ARGS:-}" ]] && read -r -a domain_args <<< "${CALL_DOMAINS_ARGS}"

  run_stage_cmd "calling" cLoops2 callPeaks -d "$(group_dataset_dir "${PRIMARY_GROUP}")" -o "$(group_output_prefix "${PRIMARY_GROUP}")" "${peak_args[@]}"
  run_stage_cmd "calling" cLoops2 callLoops -d "$(group_dataset_dir "${PRIMARY_GROUP}")" -o "$(group_output_prefix "${PRIMARY_GROUP}")" "${loop_args[@]}"
  run_stage_cmd "calling" cLoops2 callDomains -d "$(group_dataset_dir "${PRIMARY_GROUP}")" -o "$(group_output_prefix "${PRIMARY_GROUP}")" "${domain_args[@]}"
  mark_done "calling"
}

stage_vis() {
  [[ "${ENABLE_VIS:-1}" == "1" ]] || return 0
  should_run_target "vis" || return 0
  is_done "vis" && [[ "${PIPELINE_TARGET}" == "all" ]] && return 0

  local primary_prefix primary_dir fig_prefix peaks_file loops_file domains_file filtered_dir
  local -a agg_peak_args=()
  local -a agg_view_args=()
  local -a agg_loop_args=()
  local -a agg_two_args=()
  local -a agg_domain_args=()
  local -a plot_domain_args=()
  local -a plot_loop_args=()
  local -a filter_args=()
  local -a plot_filtered_args=()
  local -a plot_filtered_arch_args=()

  primary_prefix="$(group_output_prefix "${PRIMARY_GROUP}")"
  primary_dir="$(group_dataset_dir "${PRIMARY_GROUP}")"
  fig_prefix="$(group_fig_prefix "${PRIMARY_GROUP}")"
  peaks_file="${primary_prefix}_peaks.bed"
  loops_file="${primary_prefix}_loops.txt"
  domains_file="${primary_prefix}_domains.bed"
  filtered_dir="${DATA_DIR}/${PRIMARY_GROUP}_filtered"

  [[ -n "${AGG_PEAK_ARGS:-}" ]] && read -r -a agg_peak_args <<< "${AGG_PEAK_ARGS}"
  [[ -n "${AGG_VIEWPOINT_ARGS:-}" ]] && read -r -a agg_view_args <<< "${AGG_VIEWPOINT_ARGS}"
  [[ -n "${AGG_LOOP_ARGS:-}" ]] && read -r -a agg_loop_args <<< "${AGG_LOOP_ARGS}"
  [[ -n "${AGG_TWO_ANCHOR_ARGS:-}" ]] && read -r -a agg_two_args <<< "${AGG_TWO_ANCHOR_ARGS}"
  [[ -n "${AGG_DOMAIN_ARGS:-}" ]] && read -r -a agg_domain_args <<< "${AGG_DOMAIN_ARGS}"
  [[ -n "${PLOT_DOMAIN_ARGS:-}" ]] && read -r -a plot_domain_args <<< "${PLOT_DOMAIN_ARGS}"
  [[ -n "${PLOT_LOOP_ARGS:-}" ]] && read -r -a plot_loop_args <<< "${PLOT_LOOP_ARGS}"
  [[ -n "${FILTER_PETS_ARGS:-}" ]] && read -r -a filter_args <<< "${FILTER_PETS_ARGS}"
  [[ -n "${PLOT_FILTERED_ARGS:-}" ]] && read -r -a plot_filtered_args <<< "${PLOT_FILTERED_ARGS}"
  [[ -n "${PLOT_FILTERED_ARCH_ARGS:-}" ]] && read -r -a plot_filtered_arch_args <<< "${PLOT_FILTERED_ARCH_ARGS}"

  run_stage_cmd "vis" cLoops2 agg -d "${primary_dir}" -peaks "${peaks_file}" -o "${fig_prefix}" "${agg_peak_args[@]}"
  run_stage_cmd "vis" cLoops2 agg -d "${primary_dir}" -viewPoints "${peaks_file}" -o "${fig_prefix}" "${agg_view_args[@]}" -bws "${BW_CTCF}"
  run_stage_cmd "vis" cLoops2 agg -d "${primary_dir}" -loops "${loops_file}" -o "${fig_prefix}" "${agg_loop_args[@]}" -bws "${BW_ATAC},${BW_CTCF}"
  run_stage_cmd "vis" cLoops2 agg -d "${primary_dir}" -twoAnchors "${loops_file}" -o "${fig_prefix}" "${agg_two_args[@]}" -bws "${BW_CTCF},${BW_ATAC}"

  if [[ -f "${primary_prefix}_${DOMAIN_BDG_100K}" && -n "${CHROM_SIZES:-}" ]]; then
    run_stage_cmd "vis" bedGraphToBigWig "${primary_prefix}_${DOMAIN_BDG_100K}" "${CHROM_SIZES}" "${FIG_DIR}/${PRIMARY_GROUP}_${DOMAIN_BW_100K}"
  fi
  if [[ -f "${primary_prefix}_${DOMAIN_BDG_250K}" && -n "${CHROM_SIZES:-}" ]]; then
    run_stage_cmd "vis" bedGraphToBigWig "${primary_prefix}_${DOMAIN_BDG_250K}" "${CHROM_SIZES}" "${FIG_DIR}/${PRIMARY_GROUP}_${DOMAIN_BW_250K}"
  fi

  run_stage_cmd "vis" cLoops2 agg -d "${primary_dir}" -domains "${domains_file}" -o "${fig_prefix}" "${agg_domain_args[@]}" -bws "${BW_CTCF},${FIG_DIR}/${PRIMARY_GROUP}_${DOMAIN_BW_100K},${FIG_DIR}/${PRIMARY_GROUP}_${DOMAIN_BW_250K}"
  run_stage_cmd "vis" cLoops2 plot -f "${primary_dir}/${TARGET_CHROMS}-${TARGET_CHROMS}.ixy" -o "${FIG_DIR}/${PRIMARY_GROUP}_domain_example" "${plot_domain_args[@]}" -domains "${domains_file}" -bws "${BW_CTCF}"
  run_stage_cmd "vis" cLoops2 plot -f "${primary_dir}/${TARGET_CHROMS}-${TARGET_CHROMS}.ixy" -o "${FIG_DIR}/${PRIMARY_GROUP}_loop_example" "${plot_loop_args[@]}" -loops "${loops_file}" -bws "${BW_ATAC},${BW_CTCF}" -beds "${BED_ENHANCER},${BED_TSS},${peaks_file}" -gtf "${GTF_FILE}"
  run_stage_cmd "vis" cLoops2 filterPETs -d "${primary_dir}" -loops "${loops_file}" -o "${filtered_dir}" "${filter_args[@]}"
  run_stage_cmd "vis" cLoops2 plot -f "${filtered_dir}/${TARGET_CHROMS}-${TARGET_CHROMS}.ixy" -o "${FIG_DIR}/${PRIMARY_GROUP}_filtered_example" "${plot_filtered_args[@]}" -loops "${loops_file}"
  run_stage_cmd "vis" cLoops2 plot -f "${filtered_dir}/${TARGET_CHROMS}-${TARGET_CHROMS}.ixy" -o "${FIG_DIR}/${PRIMARY_GROUP}_filtered_arch" "${plot_filtered_arch_args[@]}" -loops "${loops_file}"

  mark_done "vis"
}

stage_compare() {
  [[ "${ENABLE_COMPARE:-1}" == "1" ]] || return 0
  should_run_target "compare" || return 0
  is_done "compare" && [[ "${PIPELINE_TARGET}" == "all" ]] && return 0

  local primary_samp secondary_samp
  local -a sample_args=()
  local -a loop_args=()
  local -a diff_args=()
  local -a montage_all_args=()
  local -a montage_vp_args=()

  primary_samp="${DATA_DIR}/${PRIMARY_GROUP}_samp"
  secondary_samp="${DATA_DIR}/${SECONDARY_GROUP}_samp"
  [[ -n "${SAMPLE_PETS_ARGS:-}" ]] && read -r -a sample_args <<< "${SAMPLE_PETS_ARGS}"
  [[ -n "${CALL_LOOPS_ARGS:-}" ]] && read -r -a loop_args <<< "${CALL_LOOPS_ARGS}"
  [[ -n "${CALL_DIFF_LOOPS_ARGS:-}" ]] && read -r -a diff_args <<< "${CALL_DIFF_LOOPS_ARGS}"
  [[ -n "${MONTAGE_ALL_ARGS:-}" ]] && read -r -a montage_all_args <<< "${MONTAGE_ALL_ARGS}"
  [[ -n "${MONTAGE_VIEWPOINT_ARGS:-}" ]] && read -r -a montage_vp_args <<< "${MONTAGE_VIEWPOINT_ARGS}"

  run_stage_cmd "compare" cLoops2 samplePETs -d "$(group_dataset_dir "${PRIMARY_GROUP}")" -o "${primary_samp}" -tot "${SAMPLE_PETS_TOTAL}" "${sample_args[@]}"
  run_stage_cmd "compare" cLoops2 samplePETs -d "$(group_dataset_dir "${SECONDARY_GROUP}")" -o "${secondary_samp}" -tot "${SAMPLE_PETS_TOTAL}" "${sample_args[@]}"
  run_stage_cmd "compare" cLoops2 callLoops -d "${primary_samp}" -o "${REPORT_DIR}/${PRIMARY_GROUP}_samp" "${loop_args[@]}"
  run_stage_cmd "compare" cLoops2 callLoops -d "${secondary_samp}" -o "${REPORT_DIR}/${SECONDARY_GROUP}_samp" "${loop_args[@]}"
  run_stage_cmd "compare" cLoops2 callDiffLoops -tloop "${REPORT_DIR}/${PRIMARY_GROUP}_samp_loops.txt" -td "${primary_samp}" -cloop "${REPORT_DIR}/${SECONDARY_GROUP}_samp_loops.txt" -cd "${secondary_samp}" -o "${REPORT_DIR}/${PRIMARY_GROUP}_vs_${SECONDARY_GROUP}" "${diff_args[@]}"

  if [[ -n "${MONTAGE_BED:-}" ]]; then
    run_stage_cmd "compare" cLoops2 montage -f "${primary_samp}/${TARGET_CHROMS}-${TARGET_CHROMS}.ixy" -bed "${MONTAGE_BED}" -o "${FIG_DIR}/${PRIMARY_GROUP}_montage_all" "${montage_all_args[@]}"
    run_stage_cmd "compare" cLoops2 montage -f "${secondary_samp}/${TARGET_CHROMS}-${TARGET_CHROMS}.ixy" -bed "${MONTAGE_BED}" -o "${FIG_DIR}/${SECONDARY_GROUP}_montage_all" "${montage_all_args[@]}"
    run_stage_cmd "compare" cLoops2 montage -f "${primary_samp}/${TARGET_CHROMS}-${TARGET_CHROMS}.ixy" -bed "${MONTAGE_BED}" -o "${FIG_DIR}/${PRIMARY_GROUP}_montage_viewpoints" "${montage_vp_args[@]}"
    run_stage_cmd "compare" cLoops2 montage -f "${secondary_samp}/${TARGET_CHROMS}-${TARGET_CHROMS}.ixy" -bed "${MONTAGE_BED}" -o "${FIG_DIR}/${SECONDARY_GROUP}_montage_viewpoints" "${montage_vp_args[@]}"
  fi

  mark_done "compare"
}

stage_export() {
  [[ "${ENABLE_EXPORT:-1}" == "1" ]] || return 0
  should_run_target "export" || return 0
  is_done "export" && [[ "${PIPELINE_TARGET}" == "all" ]] && return 0

  local primary_dir
  local -a dump_bed_args=()
  local -a dump_bedpe_args=()
  local -a dump_bdg_args=()
  local -a dump_washu_args=()
  local -a dump_hic_args=()
  local -a dump_matrix_args=()

  primary_dir="$(group_dataset_dir "${PRIMARY_GROUP}")"
  [[ -n "${DUMP_BED_ARGS:-}" ]] && read -r -a dump_bed_args <<< "${DUMP_BED_ARGS}"
  [[ -n "${DUMP_BEDPE_ARGS:-}" ]] && read -r -a dump_bedpe_args <<< "${DUMP_BEDPE_ARGS}"
  [[ -n "${DUMP_BDG_ARGS:-}" ]] && read -r -a dump_bdg_args <<< "${DUMP_BDG_ARGS}"
  [[ -n "${DUMP_WASHU_ARGS:-}" ]] && read -r -a dump_washu_args <<< "${DUMP_WASHU_ARGS}"
  [[ -n "${DUMP_HIC_ARGS:-}" ]] && read -r -a dump_hic_args <<< "${DUMP_HIC_ARGS}"
  [[ -n "${DUMP_MATRIX_ARGS:-}" ]] && read -r -a dump_matrix_args <<< "${DUMP_MATRIX_ARGS}"

  [[ "${DUMP_BED:-0}" == "1" ]] && run_stage_cmd "export" cLoops2 dump -d "${primary_dir}" -o "${EXPORT_DIR}/${PRIMARY_GROUP}" -bed "${dump_bed_args[@]}"
  [[ "${DUMP_BEDPE:-0}" == "1" ]] && run_stage_cmd "export" cLoops2 dump -d "${primary_dir}" -o "${EXPORT_DIR}/${PRIMARY_GROUP}" -bedpe "${dump_bedpe_args[@]}"
  [[ "${DUMP_BDG:-0}" == "1" ]] && run_stage_cmd "export" cLoops2 dump -d "${primary_dir}" -o "${EXPORT_DIR}/${PRIMARY_GROUP}" -bdg "${dump_bdg_args[@]}"
  [[ "${DUMP_WASHU:-0}" == "1" ]] && run_stage_cmd "export" cLoops2 dump -d "${primary_dir}" -o "${EXPORT_DIR}/${PRIMARY_GROUP}" -washU "${dump_washu_args[@]}"
  [[ "${DUMP_HIC:-0}" == "1" ]] && run_stage_cmd "export" cLoops2 dump -d "${primary_dir}" -o "${EXPORT_DIR}/${PRIMARY_GROUP}" -hic "${dump_hic_args[@]}"
  [[ "${DUMP_MATRIX:-0}" == "1" ]] && run_stage_cmd "export" cLoops2 dump -d "${primary_dir}" -o "${EXPORT_DIR}/${PRIMARY_GROUP}" -mat "${dump_matrix_args[@]}"

  mark_done "export"
}

stage_quant() {
  [[ "${ENABLE_QUANT:-1}" == "1" ]] || return 0
  should_run_target "quant" || return 0
  is_done "quant" && [[ "${PIPELINE_TARGET}" == "all" ]] && return 0

  local secondary_dir primary_prefix
  local -a quant_peaks_args=()
  local -a quant_loops_args=()
  local -a quant_domain_args=()
  secondary_dir="$(group_dataset_dir "${SECONDARY_GROUP}")"
  primary_prefix="$(group_output_prefix "${PRIMARY_GROUP}")"
  [[ -n "${QUANT_PEAKS_ARGS:-}" ]] && read -r -a quant_peaks_args <<< "${QUANT_PEAKS_ARGS}"
  [[ -n "${QUANT_LOOPS_ARGS:-}" ]] && read -r -a quant_loops_args <<< "${QUANT_LOOPS_ARGS}"
  [[ -n "${QUANT_DOMAIN_ARGS:-}" ]] && read -r -a quant_domain_args <<< "${QUANT_DOMAIN_ARGS}"

  run_stage_cmd "quant" cLoops2 quant -d "${secondary_dir}" -peaks "${primary_prefix}_peaks.bed" -o "${REPORT_DIR}/${SECONDARY_GROUP}_${PRIMARY_GROUP}" "${quant_peaks_args[@]}"
  run_stage_cmd "quant" cLoops2 quant -d "${secondary_dir}" -loops "${primary_prefix}_loops.txt" -o "${REPORT_DIR}/${SECONDARY_GROUP}_${PRIMARY_GROUP}" "${quant_loops_args[@]}"
  run_stage_cmd "quant" cLoops2 quant -d "${secondary_dir}" -domains "${primary_prefix}_domains.txt" -o "${REPORT_DIR}/${SECONDARY_GROUP}_${PRIMARY_GROUP}" "${quant_domain_args[@]}"
  mark_done "quant"
}

stage_annotation() {
  [[ "${ENABLE_ANNOTATION:-1}" == "1" ]] || return 0
  should_run_target "annotation" || return 0
  is_done "annotation" && [[ "${PIPELINE_TARGET}" == "all" ]] && return 0

  local -a annotation_args=()
  [[ -n "${ANNOTATION_ARGS:-}" ]] && read -r -a annotation_args <<< "${ANNOTATION_ARGS}"
  run_stage_cmd "annotation" cLoops2 anaLoops -loops "$(group_output_prefix "${PRIMARY_GROUP}")_loops.txt" -o "${REPORT_DIR}/${PRIMARY_GROUP}_loops" -gtf "${GTF_FILE}" "${annotation_args[@]}"
  run_hook_if_set "${POST_ANNOTATION_HOOK:-}" "annotation"
  mark_done "annotation"
}

stage_extra_analysis() {
  [[ "${ENABLE_EXTRA_ANALYSIS:-0}" == "1" ]] || return 0
  should_run_target "extra" || return 0
  is_done "extra" && [[ "${PIPELINE_TARGET}" == "all" ]] && return 0

  run_hook_if_set "${POST_EXTRA_ANALYSIS_HOOK:-}" "extra"
  mark_done "extra"
}

print_usage() {
  cat <<'EOF'
Usage:
  ./pipeline.sh [--config FILE] [--target STAGE] [options]
  ./pipeline.sh [config-file] [target]

Targets:
  all | clean | qc | pre | estimate | calling | vis | compare | export | quant | annotation | extra

Common:
  -h, --help
  -c, --config FILE
  -t, --target STAGE
  --threads N
  --chrom CHR_LIST
  --output-root DIR
  --primary-group NAME
  --secondary-group NAME
  --sample-total N
  --enable-stage STAGE
  --disable-stage STAGE
  --set VAR=VALUE

Command arg overrides:
  --qc-args STR
  --pre-args STR
  --combine-args STR
  --est-res-args STR
  --est-dis-args STR
  --est-sim-args STR
  --call-peaks-args STR
  --call-loops-args STR
  --call-domains-args STR
  --sample-pets-args STR
  --diff-loops-args STR
  --agg-peak-args STR
  --agg-viewpoint-args STR
  --agg-loop-args STR
  --agg-two-anchor-args STR
  --agg-domain-args STR
  --filter-pets-args STR
  --plot-domain-args STR
  --plot-loop-args STR
  --plot-filtered-args STR
  --plot-filtered-arch-args STR
  --montage-all-args STR
  --montage-viewpoint-args STR
  --dump-bed-args STR
  --dump-bedpe-args STR
  --dump-bdg-args STR
  --dump-washu-args STR
  --dump-hic-args STR
  --dump-matrix-args STR
  --quant-peaks-args STR
  --quant-loops-args STR
  --quant-domain-args STR
  --annotation-args STR

Examples:
  ./pipeline.sh --config pipeline.config.sh --target calling
  ./pipeline.sh --config pipeline.config.sh --target calling --threads 8 --call-loops-args "-eps 100,200 -minPts 5 -w -j"
  ./pipeline.sh pipeline.config.sh vis --plot-loop-args "-bs 1000 -start 1 -end 100000 -triu -1D"
  ./pipeline.sh --config pipeline.config.sh --set BW_CTCF=../data/custom.bw --disable-stage compare
EOF
}

main() {
  parse_cli "$@"

  if (( SHOW_HELP )); then
    print_usage
    exit 0
  fi

  if [[ ! -f "${CONFIG_FILE}" ]]; then
    die "Config file not found: ${CONFIG_FILE}"
  fi

  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
  apply_cli_overrides

  OUTPUT_ROOT="${OUTPUT_ROOT:-./results/default}"
  STATE_DIR="${OUTPUT_ROOT}/.state"
  LOG_DIR="${OUTPUT_ROOT}/logs"
  DATA_DIR="${OUTPUT_ROOT}/datasets"
  REPORT_DIR="${OUTPUT_ROOT}/reports"
  FIG_DIR="${OUTPUT_ROOT}/figures"
  EXPORT_DIR="${OUTPUT_ROOT}/exports"

  mkdir -p "${STATE_DIR}" "${LOG_DIR}" "${DATA_DIR}" "${REPORT_DIR}" "${FIG_DIR}" "${EXPORT_DIR}"

  require_cmd cLoops2
  ensure_samples_exist
  [[ "${ENABLE_VIS:-1}" == "1" ]] && [[ -n "${CHROM_SIZES:-}" ]] && command -v bedGraphToBigWig >/dev/null 2>&1 || true

  stage_clean
  stage_qc
  stage_pre
  stage_estimate
  stage_calling
  stage_vis
  stage_compare
  stage_export
  stage_quant
  stage_annotation
  stage_extra_analysis
  log "Pipeline finished"
}

main "$@"
