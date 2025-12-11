# Test Fixtures

This directory contains test fixtures for WhisperSwift integration tests.

## Required Files

### Whisper Model

Download a whisper.cpp model to enable integration tests:

```bash
# Tiny English model (~75MB) - fastest, good for testing
curl -L -o ggml-tiny.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin

# Or Base English model (~142MB) - better accuracy
curl -L -o ggml-base.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
```

All models available at: https://huggingface.co/ggerganov/whisper.cpp/tree/main

### Sample Audio

The JFK sample audio is already included in `whisper.cpp/samples/jfk.wav`.

## Running Integration Tests

Once you have a model file:

```bash
swift test
```

Integration tests will automatically run if `ggml-tiny.en.bin` is present.
Without the model, only unit tests will run (integration tests are skipped).

## Model Sizes

| Model | Size | Notes |
|-------|------|-------|
| tiny.en | 75 MB | Fastest, English only |
| base.en | 142 MB | Good balance |
| small.en | 466 MB | Better accuracy |
| medium.en | 1.5 GB | High accuracy |
| large | 3.1 GB | Best accuracy, multilingual |
