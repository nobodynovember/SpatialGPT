DATA_ROOT=datasets
outdir=${DATA_ROOT}/exprs_map/test/

flag="--root_dir ${DATA_ROOT}
      --img_root datasets/observations
      --split SpatialGPT_72_scenes_processed
      --end 10  
      --output_dir ${outdir}
      --max_action_len 15
      --save_pred
      --stop_after 3
      --llm gpt-4o
      --response_format json
      --max_tokens 4096
      "

python vln/main_gpt.py $flag
