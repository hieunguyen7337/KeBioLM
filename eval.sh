echo '========'
echo 'Run eval'
echo '========'
python -u -m relation_extraction.run \
  --task_name "DDI" \
  --data_dir "data/DDI/" \
  --model_name_or_path "model" \
  --output_dir "evaluation_result/DDI" \
  --do_eval --do_predict \
  --max_seq_length 256 \
  --per_device_eval_batch_size 4 \
  --overwrite_output_dir
  
echo 'Done.'
