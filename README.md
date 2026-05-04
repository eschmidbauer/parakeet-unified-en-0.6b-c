# Parakeet ASR — Pure C Inference Engine

A standalone speech-to-text engine for [NVIDIA Parakeet Unified EN 0.6B](https://huggingface.co/nvidia/parakeet-unified-en-0.6b), written in pure C with **zero external dependencies**. The binary links only against `libSystem` (macOS) or `libc + libm + libpthread` (Linux).

```bash
$ ./parakeet c_weights_fp32 audio.wav
The quick brown fox jumps over the lazy dog.

$ ./parakeet c_weights_fp32 a.wav b.wav c.wav
a.wav: The quick brown fox jumps over the lazy dog.
b.wav: To be or not to be that is the question.
c.wav: Hello world.
```

## Features

- **FastConformer encoder** (24 blocks, 1024-dim, 8-head relative positional attention) + **RNN-T greedy decoder** (2-layer LSTM) + **SentencePiece detokenizer** — all in C
- **Three weight formats**: fp32 (reference), int8 (per-tensor symmetric), and mixed int4/int8 (GPTQ int4 feed-forward + int8 attention — ~3.5× faster than int8 with identical transcription on short audio)
- **Concurrent transcription** — pass multiple WAV files and they are transcribed in parallel threads sharing a single loaded model
- **Chunked inference** for long audio — auto-splits sequences >30s into overlapping chunks, reducing peak memory from ~2.3 GB to ~60 MB
- **Hand-rolled BLAS** with explicit NEON (ARM) and AVX2 (x86_64) register-blocked kernels, cache-blocked B-packing, and thread-local buffer pooling
- **Parallel everything** via a built-in pthread fork-join pool: sgemm, softmax, layer norm, depthwise conv all scale across cores
- **Fused QKV projection** — Q/K/V weights concatenated at load time into a single `[D, 3D]` matmul, reducing thread pool dispatches by 2× per attention block
- **Pre-allocated workspace** — bump-allocator arena for conformer blocks eliminates ~700 malloc/free calls per encoder pass
- **SIMD elementwise ops**: vectorized `expf` (polynomial approximation), softmax, layer norm, Swish, depthwise conv, attention score combine, residual adds with NEON/AVX2 inner loops
- **Portable**: compiles on macOS ARM, macOS Intel, Linux x86_64, Linux aarch64 with the same `Makefile`

## Quick start

### 1. Build

```bash
cd csrc
make
```

Requires only a C11 compiler (`cc`, `gcc`, or `clang`). No Homebrew, no package manager, no external libraries.

### 2. Get the weights (one-time)

Two options: download the pre-extracted C weights from HuggingFace, or re-extract them from the ONNX source.

```bash
pip install -r requirements.txt
```

#### Option A — download pre-extracted weights (fastest)

Pulls `c_weights_{fp32,int8,mixed}/` directly from [eschmidbauer/parakeet-unified-en-0.6b-c](https://huggingface.co/eschmidbauer/parakeet-unified-en-0.6b-c):

```bash
python download_weights.py fp32    # 2.3 GB
python download_weights.py int8    # 873 MB
python download_weights.py mixed   # 729 MB
python download_weights.py         # all three
```

#### Option B — extract from ONNX

Downloads ONNX models from [eschmidbauer/parakeet-unified-en-0.6b-onnx](https://huggingface.co/eschmidbauer/parakeet-unified-en-0.6b-onnx) automatically if not already present, then extracts to a flat binary:

```bash
# fp32 weights (2.3 GB, highest accuracy)
python extract_weights.py fp32

# int8 weights (873 MB, ~64% smaller, same accuracy on short audio)
python extract_weights.py int8

# Mixed int4/int8 weights (729 MB, ~3.5× faster than int8)
# Pass any 16 kHz mono WAV via --calibration-audio for GPTQ calibration.
python extract_weights.py mixed --calibration-audio calibration.wav
```

### 3. Transcribe

```bash
# single file
./csrc/parakeet c_weights_fp32 audio.wav
./csrc/parakeet c_weights_int8 audio.wav
./csrc/parakeet c_weights_mixed audio.wav

# multiple files (transcribed concurrently)
./csrc/parakeet c_weights_fp32 a.wav b.wav c.wav
```

Input must be **16 kHz mono WAV** (16-bit PCM or 32-bit float).

Timing and progress go to stderr. Clean text goes to stdout, suitable for piping:

```bash
./csrc/parakeet c_weights_fp32 audio.wav > transcript.txt
```

## Configuration

### Thread count

Defaults to all online CPUs. Override with:

```bash
PK_THREADS=4 ./csrc/parakeet c_weights_fp32 audio.wav
```

### Chunked inference

Audio longer than ~30s (500 encoder frames after 8× subsampling) is automatically processed in chunks. Constants in `parakeet.h`:

```c
#define PK_CHUNK_SIZE      400   // frames per chunk (~25s of audio)
#define PK_CHUNK_OVERLAP   8     // overlap frames on each side
#define PK_CHUNK_THRESHOLD 500   // auto-chunk above this frame count
```

Short audio (<30s) uses the full-sequence path for maximum accuracy.

Override with `PK_CHUNK`:

```bash
PK_CHUNK=0 ./csrc/parakeet c_weights_fp32 long_audio.wav   # never chunk (full sequence)
PK_CHUNK=1 ./csrc/parakeet c_weights_fp32 short_audio.wav  # force chunked (benchmarking)
# unset → auto behaviour
```

## Performance

Measured on Apple M-series, all available cores:

### Short audio (2.56 s clip)

| Weight format | Size | Encoder time | RTF |
| --- | --- | --- | --- |
| fp32 | 2.3 GB | 6.66 s | 2.62× |
| int8 | 873 MB | 2.26 s | 0.90× |
| mixed | 729 MB | **0.65 s** | **0.26×** |

All three modes produce identical transcriptions on short audio. Mixed mode is ~10× faster than fp32 and ~3.5× faster than int8.

### Memory

| | Full sequence (256 s) | Chunked (256 s) |
| --- | --- | --- |
| Peak per block | ~2.3 GB | **~35 MB** |

## Concurrency

When given multiple WAV files, the CLI spawns one pthread per file. All threads share the same loaded model (weights are read-only after loading). The thread pool's dispatch mutex serialises the compute-heavy parallel operations (matmuls, layer norms, softmax), while I/O, mel spectrogram computation, and RNN-T decoding overlap freely between requests.

For a single file, no extra thread is created — behaviour is identical to the original sequential path.

Per-file timing is reported to stderr with mel/encoder/decoder breakdown and real-time factor (RTF):

```text
[0] audio.wav  mel=0.00s  enc=2.84s  dec=0.02s  total=2.86s  audio=2.56s  RTF=1.12
```

## Architecture

### Source files (~4,400 lines total)

| File | Lines | Purpose |
| --- | --- | --- |
| `blas.c` / `blas.h` | 1081 | Hand-rolled CBLAS: NEON/AVX2 4×16 register-blocked sgemm with cache-blocked B-packing; int8 per-tensor and int4 per-group dequantising pack kernels; single-threaded variant for nested use; parallel over M tiles |
| `encoder.c` | 636 | FastConformer: pre-encode (Conv2D subsampling), 24 conformer blocks (FF + MHA + conv module + FF), fused QKV matmul, bump-allocator workspace, batched per-head GEMM, SIMD elementwise ops, chunked dispatch |
| `tensor_ops.c` | 629 | SIMD layer norm (including in-place variant), softmax (with polynomial expf), Swish, depthwise conv, pointwise conv, LSTM cell. Parallel row-batched variants. `pk_wmatmul` dispatches fp32/int8/int4 paths by inspecting `Weight.bits`. |
| `threadpool.c` / `threadpool.h` | 203 | Minimal pthread fork-join pool with `__thread`-local buffer reuse, dispatch mutex for concurrent callers |
| `decoder.c` | 132 | LSTM prediction network + joint network + greedy RNN-T decode (time-first layout) |
| `mel.c` | 248 | Cooley-Tukey FFT, Slaney mel filterbank, per-channel normalization, parallel across frames |
| `sentencepiece.c` | 219 | Protobuf parser + Unigram detokenizer (reads `.model` directly) |
| `weights.c` | 564 | mmap-based loader, JSON manifest parser, fp32/int8/int4 format support, fused QKV weight construction (per-tensor int8 requantization or int4-to-fp32 dequantisation) |
| `wav.c` | 100 | WAV reader (PCM-16 and float-32) |
| `main.c` | 217 | CLI: loads model once, transcribes one or more WAV files concurrently, per-file timing with RTF |
| `parakeet.h` | 281 | Model constants, weight structs, `Weight` type (fp32/int8/int4 union with `bits` discriminator), fused QKV fields, API |

### Weight formats

| Directory | Size | Contents |
| --- | --- | --- |
| `c_weights_fp32/` | 2.3 GB | 866 fp32 tensors + tokenizer |
| `c_weights_int8/` | 873 MB | 217 int8 tensors (encoder matmul weights) + 649 fp32 tensors (biases, norms, conv weights, decoder) + tokenizer |
| `c_weights_mixed/` | 729 MB | 96 int4 tensors (feed-forward weights, GPTQ per-group) + 121 int8 tensors (attention + pre-encode, per-tensor) + 649 fp32 tensors + tokenizer |

The int8 format uses per-tensor symmetric quantization (scale stored in the JSON manifest, zero-point = 0). Dequantization happens inside the NEON/AVX2 pack routine — int8 weights stay in RAM, converted to fp32 only in a ~256 KB tile buffer during each matmul.

The int4 format uses per-group symmetric quantization (group size = 32 columns, one fp32 scale per row-group pair). Two int4 values are packed per byte (low nibble first). The NEON/AVX2 pack kernel unpacks, sign-extends, and applies the per-group scale in a single SIMD sweep.

## Mixed precision

Mixed mode quantises different weight groups to different bit-widths based on quantisation sensitivity:

- **Feed-forward layers** (`feed_forward{1,2}.linear{1,2}`) → **int4** with GPTQ per-group quantisation. These are the largest weights (~128 MB/block as fp32) and tolerate more quantisation noise. 96 int4 tensors total.
- **Attention layers** (`self_attn.linear_{q,k,v,pos,out}`) → **int8** per-tensor. Smaller and more sensitive to noise. 121 int8 tensors total.
- **`pre_encode.out`** → int8 (sensitive early projection).
- **Everything else** (biases, norms, conv weights, decoder) → fp32.

### GPTQ calibration

Int4 alone (naive round-to-nearest) is too lossy for this model — the theoretical SNR ceiling is ~20 dB per 4-bit group, while the model needs ~30 dB. GPTQ (Frantar et al. 2022) closes this gap by propagating quantisation error using an activation Hessian `H = Xᵀ X / n` collected from real audio. When quantising row k of a weight matrix, the error is distributed to subsequent rows using `H⁻¹`, so later rows absorb the earlier errors and the overall output error is minimised.

`extract_weights.py mixed` handles calibration automatically:

```bash
# Runs the fp32 ONNX encoder on the given WAV, computes H = Xᵀ X / n for each
# feed-forward MatMul input in memory, then applies GPTQ int4 quantisation.
# Any 16 kHz mono WAV works as calibration input — longer / more diverse clips
# generally yield better Hessians. Nothing is persisted to disk except the
# final c_weights_mixed/ directory.
python extract_weights.py mixed --calibration-audio calibration.wav
```

The Hessians can reach several GB in RAM (the `[4096, 4096]` Hessians for `linear2` dominate). They are freed as soon as the encoder is quantised.

## Accuracy

On short audio, the C engine produces **byte-identical tokens** to ONNX Runtime (fp32 path). The int8 and mixed paths match on short clips and have ~98% token overlap on long audio (expected for weight-only quantization).

Chunked inference produces correct, coherent transcriptions with minor cosmetic differences at chunk boundaries (e.g., numbers may appear as words instead of digits due to reduced attention context).

## License

**Code** (everything in this repository): [MIT License](LICENSE).

**Model weights** (downloaded separately via `extract_weights.py`): [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). You must comply with the NVIDIA license when using the weights.
