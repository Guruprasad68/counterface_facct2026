#!/bin/bash

PIPELINE_RUN_ID="example_run_diffusion_plus_gan"
PATHS_FILENAME="example_attr_demo_wise_concatenated_images.json"
RESPONSE_FILENAME="example_candidate_face_pairs.json"

DATA_ROOT="/path/to/pipeline/data"

PATHS_FILE="${DATA_ROOT}/final_run/${PATHS_FILENAME}"
GPT_RESPONSES_FILE="${DATA_ROOT}/detector_run/candidate_face_responses/gpt4o/${RESPONSE_FILENAME}"
CLAUDE_RESPONSES_FILE="${DATA_ROOT}/detector_run/candidate_face_responses/claude/${RESPONSE_FILENAME}"
GEMINI_RESPONSES_FILE="${DATA_ROOT}/detector_run/candidate_face_responses/gemini/${RESPONSE_FILENAME}"
COMBINED_RESPONSE_FILE="${DATA_ROOT}/detector_run/candidate_face_responses/combined/${RESPONSE_FILENAME}"

REASSIGN_DICT_PRIMARY="${DATA_ROOT}/utils/reassign_dict.json"
REASSIGN_DICT_EXTRA=""  # optional; leave empty to skip

EXTRA_ARGS=()
if [ -n "${REASSIGN_DICT_EXTRA}" ]; then
  EXTRA_ARGS+=(--reassign_dict_extra "${REASSIGN_DICT_EXTRA}")
fi

python combining_detector_responses.py \
    --paths_file "$PATHS_FILE" \
    --gpt_responses_file "$GPT_RESPONSES_FILE" \
    --claude_responses_file "$CLAUDE_RESPONSES_FILE" \
    --gemini_responses_file "$GEMINI_RESPONSES_FILE" \
    --combined_response_file "$COMBINED_RESPONSE_FILE" \
    --reassign_dict "$REASSIGN_DICT_PRIMARY" \
    "${EXTRA_ARGS[@]}" \
    --run_id "$PIPELINE_RUN_ID"
