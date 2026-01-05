1. Download tokenizers
```
curl -L https://huggingface.co/openai/whisper-small/resolve/main/tokenizer.json -o tokenizer.json
curl -L https://huggingface.co/openai/whisper-small/resolve/main/tokenizer_config.json -o tokenizer_config.json
curl -L https://huggingface.co/openai/whisper-small/resolve/main/special_tokens_map.json -o special_tokens_map.json
```

2. Install Executorch
```
CMAKE_ARGS="-DEXECUTORCH_BUILD_MPS=ON" ./install_executorch.sh
```

3. Install optimum-executorch
```
cd optimum-executorch
python install_dev.py --skip_override_torch
```

4. Export whisper with Metal backend
```
cd ..
optimum-cli export executorch \
    --model openai/whisper-small \
    --task automatic-speech-recognition \
    --recipe metal \
    --dtype bfloat16 \
    --output_dir ./whisper-small-metal
```

5. Build preprocessor
```
python -m executorch.extension.audio.mel_spectrogram \
    --feature_size 80 \
    --stack_output \
    --max_audio_len 300 \
    --output_file whisper_preprocessor_small.pte
```

6. Build Whisper runner
```
make whisper-metal
```

7. Run Whisper runner
```
cmake-out/examples/models/whisper/whisper_runner \
    --model_path whisper-small-metal/model.pte \
    --data_path whisper-small-metal/aoti_metal_blob.ptd \
    --tokenizer_path ./ \
    --audio_path obama_short20.wav \
    --processor_path whisper-small-metal/whisper_preprocessor.pte \
    --temperature 0
```
