"""Extract weights from ONNX encoder + decoder_joint into a flat binary.

Downloads ONNX models from HuggingFace if not already present, then extracts.

Produces (little-endian):
  c_weights_{mode}/weights.bin   – tensors packed contiguously
  c_weights_{mode}/weights.json  – manifest mapping logical name -> {offset, shape, numel, dtype[, scale]}

Usage:
    python extract_weights.py fp32    # download (if needed) + extract fp32
    python extract_weights.py int8    # download (if needed) + extract int8
    python extract_weights.py mixed   # GPTQ int4 FF + int8 attention
"""
import argparse
import json, os, shutil, sys, tempfile, wave
import numpy as np
import onnx
from onnx import numpy_helper, helper
from huggingface_hub import snapshot_download

HF_REPO = "eschmidbauer/parakeet-unified-en-0.6b-onnx"


# ── Download ─────────────────────────────────────────────────────────────────

def download_onnx(input_dir):
    """Download the ONNX files for `input_dir` from HuggingFace if missing."""
    # Check if the key files already exist locally
    expected = os.listdir(input_dir) if os.path.isdir(input_dir) else []
    has_encoder = any("encoder" in f and f.endswith(".onnx") for f in expected)
    has_decoder = any("decoder" in f and f.endswith(".onnx") for f in expected)
    if has_encoder and has_decoder:
        print(f"{input_dir}/ already present, skipping download")
        return

    subdir = os.path.basename(input_dir)  # "onnx_fp32" or "onnx_int8"
    print(f"downloading {subdir}/ from {HF_REPO} …")
    local = snapshot_download(
        repo_id=HF_REPO,
        allow_patterns=f"{subdir}/*",
        local_dir=".",
    )
    print(f"download complete ({local})")


# ── Common helpers ───────────────────────────────────────────────────────────

def load_model(path):
    print(f"loading {path} …")
    return onnx.load(path, load_external_data=True)


def get_init(graph, name):
    """Get initializer as numpy array by name."""
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def find_matmul_node(graph, prefix):
    """Find a MatMul/MatMulInteger node whose output contains `prefix`."""
    for n in graph.node:
        if n.op_type in ("MatMul", "MatMulInteger"):
            for o in n.output:
                if prefix in o:
                    return n
    return None


def find_conv_node(graph, prefix):
    """Find a Conv/ConvInteger node whose output contains `prefix`."""
    for n in graph.node:
        if n.op_type in ("Conv", "ConvInteger"):
            for o in n.output:
                if prefix in o:
                    return n
    return None


def w_entry(data, scale=None, scales=None):
    """Create a weight entry dict."""
    return {"data": data, "scale": scale, "scales": scales}


# ── FP32 extraction ─────────────────────────────────────────────────────────

def extract_encoder_fp32(enc):
    g = enc.graph
    weights = {}

    def add(name):
        arr = get_init(g, name)
        assert arr is not None, f"missing {name}"
        weights[name] = w_entry(arr.astype(np.float32))

    def add_matmul(logical, prefix):
        n = find_matmul_node(g, prefix)
        assert n is not None, f"missing weight for {logical}"
        arr = get_init(g, n.input[1])
        assert arr is not None, f"missing weight for {logical}"
        weights[logical] = w_entry(arr.astype(np.float32))

    # ── pre_encode ──
    for name in [
        "pre_encode.conv.0.weight", "pre_encode.conv.0.bias",
        "pre_encode.conv.2.weight", "pre_encode.conv.2.bias",
        "pre_encode.conv.3.weight", "pre_encode.conv.3.bias",
        "pre_encode.conv.5.weight", "pre_encode.conv.5.bias",
        "pre_encode.conv.6.weight", "pre_encode.conv.6.bias",
        "pre_encode.out.bias",
    ]:
        add(name)

    add_matmul("pre_encode.out.weight", "/pre_encode/out/MatMul")

    # Positional encoding table
    for n in g.node:
        if n.op_type == "Constant" and n.output[0] == "onnx::Slice_1513":
            for a in n.attribute:
                if a.name == "value":
                    pe = numpy_helper.to_array(a.t)
                    if pe.ndim == 3:
                        pe = pe[0]
                    weights["pos_encoding.pe_table"] = w_entry(pe.astype(np.float32))
                    print(f"  PE table: {pe.shape}")
                    break
    assert "pos_encoding.pe_table" in weights, "missing PE table"

    # ── 24 conformer blocks ──
    for i in range(24):
        pfx = f"layers.{i}"

        for name in [
            f"{pfx}.norm_feed_forward1.weight",
            f"{pfx}.norm_feed_forward1.bias",
            f"{pfx}.feed_forward1.linear1.bias",
            f"{pfx}.feed_forward1.linear2.bias",
            f"{pfx}.norm_self_att.weight",
            f"{pfx}.norm_self_att.bias",
            f"{pfx}.self_attn.pos_bias_u",
            f"{pfx}.self_attn.pos_bias_v",
            f"{pfx}.self_attn.linear_q.bias",
            f"{pfx}.self_attn.linear_k.bias",
            f"{pfx}.self_attn.linear_v.bias",
            f"{pfx}.self_attn.linear_out.bias",
            f"{pfx}.norm_conv.weight",
            f"{pfx}.norm_conv.bias",
            f"{pfx}.conv.pointwise_conv1.weight",
            f"{pfx}.conv.pointwise_conv1.bias",
            f"{pfx}.conv.pointwise_conv2.weight",
            f"{pfx}.conv.pointwise_conv2.bias",
            f"{pfx}.norm_feed_forward2.weight",
            f"{pfx}.norm_feed_forward2.bias",
            f"{pfx}.feed_forward2.linear1.bias",
            f"{pfx}.feed_forward2.linear2.bias",
            f"{pfx}.norm_out.weight",
            f"{pfx}.norm_out.bias",
        ]:
            add(name)

        matmul_map = {
            f"{pfx}.feed_forward1.linear1.weight": f"/{pfx}/feed_forward1/linear1/MatMul",
            f"{pfx}.feed_forward1.linear2.weight": f"/{pfx}/feed_forward1/linear2/MatMul",
            f"{pfx}.self_attn.linear_q.weight":    f"/{pfx}/self_attn/linear_q/MatMul",
            f"{pfx}.self_attn.linear_k.weight":    f"/{pfx}/self_attn/linear_k/MatMul",
            f"{pfx}.self_attn.linear_v.weight":    f"/{pfx}/self_attn/linear_v/MatMul",
            f"{pfx}.self_attn.linear_pos.weight":  f"/{pfx}/self_attn/linear_pos/MatMul",
            f"{pfx}.self_attn.linear_out.weight":  f"/{pfx}/self_attn/linear_out/MatMul",
            f"{pfx}.feed_forward2.linear1.weight": f"/{pfx}/feed_forward2/linear1/MatMul",
            f"{pfx}.feed_forward2.linear2.weight": f"/{pfx}/feed_forward2/linear2/MatMul",
        }
        for logical, prefix in matmul_map.items():
            add_matmul(logical, prefix)

        # Depthwise conv weight
        n = find_conv_node(g, f"/{pfx}/conv/depthwise_conv/Conv")
        assert n is not None, f"missing {pfx}.conv.depthwise_conv.weight"
        w = get_init(g, n.input[1])
        assert w is not None, f"missing {pfx}.conv.depthwise_conv.weight"
        weights[f"{pfx}.conv.depthwise_conv.weight"] = w_entry(w.astype(np.float32))

        # Depthwise conv bias
        dw_bias = get_init(g, f"{pfx}.conv.depthwise_conv.bias")
        if dw_bias is None:
            for cn in g.node:
                if cn.op_type in ("Conv", "ConvInteger"):
                    for o in cn.output:
                        if f"/{pfx}/conv/depthwise_conv/Conv" in o:
                            if len(cn.input) >= 3 and cn.input[2]:
                                dw_bias = get_init(g, cn.input[2])
                                break
                    if dw_bias is not None:
                        break
        assert dw_bias is not None, f"missing {pfx}.conv.depthwise_conv.bias"
        weights[f"{pfx}.conv.depthwise_conv.bias"] = w_entry(dw_bias.astype(np.float32))

        print(f"  block {i}: {sum(w['data'].size for w in weights.values())} params so far")

    return weights


def extract_decoder_fp32(dec):
    g = dec.graph
    weights = {}

    for name in [
        "decoder.prediction.embed.weight",
        "joint.pred.bias",
        "joint.enc.bias",
        "joint.joint_net.2.bias",
    ]:
        arr = get_init(g, name)
        assert arr is not None, f"missing {name}"
        weights[name] = w_entry(arr.astype(np.float32))

    lstm_map = {
        "decoder.lstm.0.W_i": "onnx::LSTM_205",
        "decoder.lstm.0.W_r": "onnx::LSTM_206",
        "decoder.lstm.0.B":   "onnx::LSTM_207",
        "decoder.lstm.1.W_i": "onnx::LSTM_225",
        "decoder.lstm.1.W_r": "onnx::LSTM_226",
        "decoder.lstm.1.B":   "onnx::LSTM_227",
    }
    for logical, onnx_name in lstm_map.items():
        arr = get_init(g, onnx_name)
        assert arr is not None, f"missing {logical} ({onnx_name})"
        weights[logical] = w_entry(arr.astype(np.float32))

    joint_map = {
        "joint.enc.weight":  "onnx::MatMul_228",
        "joint.pred.weight": "onnx::MatMul_229",
        "joint.out.weight":  "onnx::MatMul_230",
    }
    for logical, onnx_name in joint_map.items():
        arr = get_init(g, onnx_name)
        assert arr is not None, f"missing {logical} ({onnx_name})"
        weights[logical] = w_entry(arr.astype(np.float32))

    return weights


# ── INT8 extraction ──────────────────────────────────────────────────────────

def _matmul_weight_info(graph, prefix):
    """Return (quantized_weight_name, scale_name, zero_point_name) for a
    MatMulInteger node whose output contains `prefix`."""
    n = find_matmul_node(graph, prefix)
    if n is None or n.op_type != "MatMulInteger":
        return None
    w_name = n.input[1]
    zp_name = n.input[3]
    base = w_name.replace("_quantized", "")
    return w_name, base + "_scale", zp_name


def _conv_weight_info(graph, prefix):
    """Return (quantized_weight_name, scale_name, zero_point_name) for a
    ConvInteger node whose output contains `prefix`."""
    n = find_conv_node(graph, prefix)
    if n is None or n.op_type != "ConvInteger":
        return None
    w_name = n.input[1]
    zp_name = n.input[3] if len(n.input) > 3 else None
    base = w_name.replace("_quantized", "")
    return w_name, base + "_scale", zp_name


def _dequantize_int8(q, scale, zp=0):
    return ((q.astype(np.int32) - int(zp)) * float(scale)).astype(np.float32)


def _add_q(weights, graph, logical, q_name, scale_name, zp_name=None):
    """Add an int8 quantized weight (kept as int8 with scale)."""
    q = get_init(graph, q_name)
    sc = get_init(graph, scale_name)
    assert q is not None, f"missing int8 {q_name}"
    assert sc is not None, f"missing scale {scale_name}"
    if zp_name:
        zp = get_init(graph, zp_name)
        if zp is not None:
            assert np.all(zp == 0), f"non-zero zp for {logical}: {zp}"
    assert q.dtype == np.int8
    assert sc.size == 1, f"non-scalar scale for {logical}: {sc.shape}"
    weights[logical] = w_entry(q.astype(np.int8), scale=float(sc.item()))


def _add_conv_dequant(weights, graph, logical, q_name, scale_name, zp_name):
    """Dequantize a conv weight to fp32 at extraction time."""
    q = get_init(graph, q_name)
    sc = get_init(graph, scale_name)
    assert q is not None and sc is not None
    if zp_name:
        zp = get_init(graph, zp_name)
        if zp is not None:
            assert np.all(zp == 0)
    weights[logical] = w_entry(q.astype(np.float32) * float(sc.item()))


def extract_encoder_int8(enc):
    g = enc.graph
    weights = {}

    def add_fp32(name):
        arr = get_init(g, name)
        assert arr is not None, f"missing fp32 {name}"
        weights[name] = w_entry(arr.astype(np.float32))

    # ── pre_encode biases (fp32) ──
    for nm in [
        "pre_encode.conv.0.bias",
        "pre_encode.conv.2.bias",
        "pre_encode.conv.3.bias",
        "pre_encode.conv.5.bias",
        "pre_encode.conv.6.bias",
        "pre_encode.out.bias",
    ]:
        add_fp32(nm)

    # Pre-encode conv weights – dequantise (small, <200 KB total)
    for base in [0, 2, 3, 5, 6]:
        _add_conv_dequant(weights, g,
                          f"pre_encode.conv.{base}.weight",
                          f"pre_encode.conv.{base}.weight_quantized",
                          f"pre_encode.conv.{base}.weight_scale",
                          f"pre_encode.conv.{base}.weight_zero_point")

    # pre_encode.out linear
    info = _matmul_weight_info(g, "/pre_encode/out/MatMul")
    assert info, "pre_encode.out.weight not found"
    _add_q(weights, g, "pre_encode.out.weight", *info)

    # ── positional encoding table (fp32) ──
    for n in g.node:
        if n.op_type == "Constant" and n.output[0] == "onnx::Slice_1513":
            for a in n.attribute:
                if a.name == "value":
                    pe = numpy_helper.to_array(a.t)
                    if pe.ndim == 3:
                        pe = pe[0]
                    weights["pos_encoding.pe_table"] = w_entry(pe.astype(np.float32))
                    print(f"  PE table: {pe.shape}")
                    break

    # ── 24 conformer blocks ──
    for i in range(24):
        pfx = f"layers.{i}"

        # fp32 (norms, biases, pos_bias)
        for nm in [
            f"{pfx}.norm_feed_forward1.weight",
            f"{pfx}.norm_feed_forward1.bias",
            f"{pfx}.feed_forward1.linear1.bias",
            f"{pfx}.feed_forward1.linear2.bias",
            f"{pfx}.norm_self_att.weight",
            f"{pfx}.norm_self_att.bias",
            f"{pfx}.self_attn.pos_bias_u",
            f"{pfx}.self_attn.pos_bias_v",
            f"{pfx}.self_attn.linear_q.bias",
            f"{pfx}.self_attn.linear_k.bias",
            f"{pfx}.self_attn.linear_v.bias",
            f"{pfx}.self_attn.linear_out.bias",
            f"{pfx}.norm_conv.weight",
            f"{pfx}.norm_conv.bias",
            f"{pfx}.conv.pointwise_conv1.bias",
            f"{pfx}.conv.pointwise_conv2.bias",
            f"{pfx}.norm_feed_forward2.weight",
            f"{pfx}.norm_feed_forward2.bias",
            f"{pfx}.feed_forward2.linear1.bias",
            f"{pfx}.feed_forward2.linear2.bias",
            f"{pfx}.norm_out.weight",
            f"{pfx}.norm_out.bias",
        ]:
            add_fp32(nm)

        # int8 matmul weights
        mm_paths = {
            f"{pfx}.feed_forward1.linear1.weight": f"/{pfx}/feed_forward1/linear1/MatMul",
            f"{pfx}.feed_forward1.linear2.weight": f"/{pfx}/feed_forward1/linear2/MatMul",
            f"{pfx}.self_attn.linear_q.weight":    f"/{pfx}/self_attn/linear_q/MatMul",
            f"{pfx}.self_attn.linear_k.weight":    f"/{pfx}/self_attn/linear_k/MatMul",
            f"{pfx}.self_attn.linear_v.weight":    f"/{pfx}/self_attn/linear_v/MatMul",
            f"{pfx}.self_attn.linear_pos.weight":  f"/{pfx}/self_attn/linear_pos/MatMul",
            f"{pfx}.self_attn.linear_out.weight":  f"/{pfx}/self_attn/linear_out/MatMul",
            f"{pfx}.feed_forward2.linear1.weight": f"/{pfx}/feed_forward2/linear1/MatMul",
            f"{pfx}.feed_forward2.linear2.weight": f"/{pfx}/feed_forward2/linear2/MatMul",
        }
        for logical, prefix in mm_paths.items():
            info = _matmul_weight_info(g, prefix)
            assert info, f"missing {logical} ({prefix})"
            _add_q(weights, g, logical, *info)

        # Conv weights – dequantise at extraction time (keeps C-side simple)
        conv_paths = {
            f"{pfx}.conv.pointwise_conv1.weight": f"/{pfx}/conv/pointwise_conv1/Conv",
            f"{pfx}.conv.depthwise_conv.weight":  f"/{pfx}/conv/depthwise_conv/Conv",
            f"{pfx}.conv.pointwise_conv2.weight": f"/{pfx}/conv/pointwise_conv2/Conv",
        }
        for logical, prefix in conv_paths.items():
            info = _conv_weight_info(g, prefix)
            assert info, f"missing {logical} ({prefix})"
            q_name, scale_name, zp_name = info
            q = get_init(g, q_name)
            sc = get_init(g, scale_name)
            assert q is not None and sc is not None
            if zp_name:
                zp = get_init(g, zp_name)
                if zp is not None:
                    assert np.all(zp == 0)
            weights[logical] = w_entry(q.astype(np.float32) * float(sc.item()))

        # Depthwise conv bias – trace back via the Reshape node
        dw_bias_arr = None
        target = f"/{pfx}/conv/depthwise_conv/Conv_output_0_bias_reshape_output"
        for n in g.node:
            if n.op_type == "Reshape" and n.output and n.output[0] == target:
                bias_init = get_init(g, n.input[0])
                if bias_init is not None and bias_init.size == 1024:
                    dw_bias_arr = bias_init.astype(np.float32)
                    break
        assert dw_bias_arr is not None, f"missing depthwise bias for block {i}"
        weights[f"{pfx}.conv.depthwise_conv.bias"] = w_entry(dw_bias_arr)

        if i == 0 or (i + 1) % 6 == 0:
            n_int8 = sum(1 for w in weights.values() if w["scale"] is not None)
            n_fp32 = sum(1 for w in weights.values() if w["scale"] is None)
            print(f"  block {i}: {n_int8} int8 + {n_fp32} fp32 tensors")

    return weights


def extract_decoder_int8(dec):
    """Decoder is small (~9 MB int8). Dequantize entirely to fp32 at
    extraction time so the C-side decoder stays unchanged."""
    g = dec.graph
    weights = {}

    def add_fp32(name):
        arr = get_init(g, name)
        assert arr is not None, f"missing fp32 {name}"
        weights[name] = w_entry(arr.astype(np.float32))

    # fp32 biases
    for nm in ["joint.pred.bias", "joint.enc.bias", "joint.joint_net.2.bias"]:
        add_fp32(nm)

    # Embedding: uint8 + non-zero zero_point → dequantize to fp32
    emb_q  = get_init(g, "decoder.prediction.embed.weight_quantized")
    emb_s  = get_init(g, "decoder.prediction.embed.weight_scale")
    emb_zp = get_init(g, "decoder.prediction.embed.weight_zero_point")
    assert emb_q is not None and emb_s is not None
    emb_fp32 = _dequantize_int8(emb_q, float(emb_s.item()),
                                int(emb_zp.item()) if emb_zp is not None else 0)
    weights["decoder.prediction.embed.weight"] = w_entry(emb_fp32)

    # LSTM: DynamicQuantizeLSTM stores W,R as [1, input, 4H] (transposed vs
    # fp32 [1, 4H, input]). Dequantize and transpose to match fp32 layout.
    lstm_nodes = [n for n in g.node if n.op_type == "DynamicQuantizeLSTM"]
    assert len(lstm_nodes) == 2, f"expected 2 LSTM nodes, got {len(lstm_nodes)}"

    for i, n in enumerate(lstm_nodes):
        w_q = get_init(g, n.input[1])
        r_q = get_init(g, n.input[2])
        b   = get_init(g, n.input[3])
        w_s = get_init(g, n.input[8])
        r_s = get_init(g, n.input[10])

        w_fp = _dequantize_int8(w_q, float(w_s.item()) if w_s.size == 1
                                else float(w_s.flat[0]))
        r_fp = _dequantize_int8(r_q, float(r_s.item()) if r_s.size == 1
                                else float(r_s.flat[0]))

        # Transpose last two dims: [1, input, 4H] → [1, 4H, input]
        if w_fp.shape == (1, 640, 2560):
            w_fp = w_fp.transpose(0, 2, 1).copy()
        if r_fp.shape == (1, 640, 2560):
            r_fp = r_fp.transpose(0, 2, 1).copy()

        weights[f"decoder.lstm.{i}.W_i"] = w_entry(w_fp)
        weights[f"decoder.lstm.{i}.W_r"] = w_entry(r_fp)
        weights[f"decoder.lstm.{i}.B"]   = w_entry(b.astype(np.float32))

    # Joint MatMul weights – dequantise at extraction time
    joint_paths = {
        "joint.enc.weight":  "/joint/enc/MatMul",
        "joint.pred.weight": "/joint/pred/MatMul",
        "joint.out.weight":  "/joint/joint_net/joint_net.2/MatMul",
    }
    for logical, prefix in joint_paths.items():
        info = _matmul_weight_info(g, prefix)
        assert info, f"missing {logical}"
        q_name, scale_name, zp_name = info
        q  = get_init(g, q_name)
        sc = get_init(g, scale_name)
        assert q is not None and sc is not None
        if zp_name:
            zp = get_init(g, zp_name)
            if zp is not None:
                assert np.all(zp == 0)
        weights[logical] = w_entry(q.astype(np.float32) * float(sc.item()))

    return weights


# ── INT4 quantization (per-channel, from fp32 ONNX) ─────────────────────────

INT4_GROUP_SIZE = 32  # columns per quantization group


def quantize_int4_gptq(arr, hessian=None):
    """GPTQ-style int4 quantization with per-group scales.

    Uses the GPTQ algorithm (Frantar et al. 2022) with calibration Hessian.
    The Hessian H = X^T X is [K, K] (over the input dimension).
    GPTQ iterates over the K (input) dimension of W [K, N], quantizing one
    row at a time and propagating the error to subsequent rows.

    Groups of INT4_GROUP_SIZE columns share one scale per row.

    Returns (packed_bytes, scales).
    """
    assert arr.ndim == 2
    K, N = arr.shape
    G = INT4_GROUP_SIZE
    assert N % 2 == 0 and N % G == 0

    W = arr.astype(np.float64, copy=True)
    Q = np.zeros((K, N), dtype=np.int8)
    n_groups = N // G
    scales = np.zeros((K, n_groups), dtype=np.float32)

    # Full Hessian [K, K] from calibration (or self-Hessian fallback)
    if hessian is not None:
        H_full = hessian.astype(np.float64)
    else:
        H_full = (W @ W.T).astype(np.float64)

    damp = 0.01 * np.mean(np.diag(H_full)) + 1e-8
    H_full += damp * np.eye(K)
    try:
        H_inv = np.linalg.inv(H_full)
    except np.linalg.LinAlgError:
        H_inv = np.eye(K)

    # GPTQ: iterate over rows (input dimension K) in blocks
    B = min(128, K)  # block size for row processing
    for b0 in range(0, K, B):
        b1 = min(b0 + B, K)
        Wb = W[b0:b1, :].copy()  # [B, N]

        Err = np.zeros_like(Wb)

        for i in range(b1 - b0):
            row = b0 + i
            w_row = Wb[i, :]  # [N]

            # Per-group quantization of this row
            for g_idx in range(n_groups):
                c0, c1 = g_idx * G, (g_idx + 1) * G
                absmax = np.abs(w_row[c0:c1]).max()
                if absmax == 0:
                    absmax = 1.0
                s = absmax / 7.0
                scales[row, g_idx] = float(s)

                q = np.round(w_row[c0:c1] / s).clip(-8, 7)
                Q[row, c0:c1] = q.astype(np.int8)
                w_row[c0:c1] = q * s  # update with quantized values

            # Row error
            err = (W[row, :] - w_row) # [N]
            Err[i, :] = err

            # Propagate error to remaining rows within block
            diag = max(H_inv[row, row], 1e-10)
            if i + 1 < b1 - b0:
                # Scale error by Hessian coupling and propagate
                for j in range(i + 1, b1 - b0):
                    Wb[j, :] -= err * (H_inv[row, b0 + j] / diag)

        # Block propagation to remaining rows
        if b1 < K:
            for i in range(b1 - b0):
                row = b0 + i
                diag = max(H_inv[row, row], 1e-10)
                coupling = H_inv[row, b1:] / diag  # [K - b1]
                W[b1:, :] -= np.outer(coupling, Err[i, :])

    # Pack two int4 values per byte
    lo = Q[:, 0::2] & 0x0F
    hi = (Q[:, 1::2] & 0x0F) << 4
    packed = (lo | hi).astype(np.uint8)

    return packed, scales.reshape(-1)


# ── Calibration for GPTQ (in-memory, no disk) ───────────────────────────────

def _read_wav_for_cal(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw == 2:
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr
    if sw == 4:
        return np.frombuffer(raw, dtype=np.float32).copy(), sr
    raise ValueError(f"unsupported sample width {sw}")


def _mel_for_cal(samples, sr):
    """NeMo-compatible mel spectrogram (matches csrc/mel.c)."""
    n_fft, win_len, hop_len, n_mels = 512, 400, 160, 128
    win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(win_len) / win_len))
    padded = np.pad(samples, n_fft // 2, mode="reflect")
    n_frames = (len(padded) - n_fft) // hop_len + 1

    def hz2mel(hz): return hz * 3 / 200 if hz < 1000 else 15 + 27 * np.log(hz / 1000) / np.log(6.4)
    def mel2hz(m):  return m * 200 / 3 if m < 15 else 1000 * np.exp((m - 15) * np.log(6.4) / 27)

    n_freqs = n_fft // 2 + 1
    mels_pts = np.array([
        mel2hz(hz2mel(0) + (hz2mel(sr / 2) - hz2mel(0)) * i / (n_mels + 1))
        for i in range(n_mels + 2)
    ])
    freqs = np.arange(n_freqs) * sr / n_fft
    fb = np.zeros((n_mels, n_freqs))
    for m in range(n_mels):
        lo, c, hi = mels_pts[m], mels_pts[m + 1], mels_pts[m + 2]
        norm = 2 / (hi - lo)
        for f in range(n_freqs):
            if lo <= freqs[f] <= c:
                fb[m, f] = (freqs[f] - lo) / (c - lo) * norm
            elif c < freqs[f] <= hi:
                fb[m, f] = (hi - freqs[f]) / (hi - c) * norm

    mel = np.zeros((n_mels, n_frames), dtype=np.float32)
    for f in range(n_frames):
        frame = np.zeros(n_fft)
        frame[:win_len] = padded[f * hop_len:f * hop_len + win_len] * win
        mel[:, f] = np.log(fb @ np.abs(np.fft.rfft(frame)) ** 2 + 5.96e-8)
    for m in range(n_mels):
        mu, std = mel[m].mean(), mel[m].std()
        mel[m] = (mel[m] - mu) / (std + 1e-5)
    return mel


def compute_ff_hessians(onnx_path, audio_path):
    """Run the fp32 ONNX encoder on the given audio, capture the activation
    input of every feed-forward MatMul, and return {logical_name: H = Xᵀ X / n}.

    Only FF layers are hooked (the mixed-precision path quantises only those
    to int4).  Attention weights stay int8 and don't need calibration.

    Runs entirely in memory — the Hessians are returned, never persisted.
    """
    import onnxruntime as ort

    print(f"  loading {onnx_path} for calibration …")
    model = onnx.load(onnx_path, load_external_data=True)
    graph = model.graph

    # Only FF weights need Hessians for mixed mode
    weight_to_activation = {}
    for i in range(24):
        pfx = f"layers.{i}"
        for suffix, prefix in [
            ("feed_forward1.linear1.weight", f"/{pfx}/feed_forward1/linear1/MatMul"),
            ("feed_forward1.linear2.weight", f"/{pfx}/feed_forward1/linear2/MatMul"),
            ("feed_forward2.linear1.weight", f"/{pfx}/feed_forward2/linear1/MatMul"),
            ("feed_forward2.linear2.weight", f"/{pfx}/feed_forward2/linear2/MatMul"),
        ]:
            node = find_matmul_node(graph, prefix)
            if node:
                weight_to_activation[f"{pfx}.{suffix}"] = node.input[0]

    act_names = sorted(set(weight_to_activation.values()))
    for name in act_names:
        model.graph.output.append(helper.make_tensor_value_info(name, 1, None))

    # Serialise to a temp file (the ONNX model + added outputs exceeds the 2 GB
    # protobuf-in-memory limit, so we write it with external-data format).
    tmp = tempfile.mkdtemp()
    try:
        tmp_model = os.path.join(tmp, "encoder_cal.onnx")
        onnx.save(model, tmp_model, save_as_external_data=True,
                  all_tensors_to_one_file=True, location="weights.pb")
        del model

        print(f"  reading {audio_path} …")
        samples, sr = _read_wav_for_cal(audio_path)
        mel = _mel_for_cal(samples, sr)
        print(f"  mel: {mel.shape}, running ONNX inference with {len(act_names)} hooks …")

        sess = ort.InferenceSession(tmp_model)
        out_names = [o.name for o in sess.get_outputs()]
        results = sess.run(out_names, {
            "audio_signal": mel[np.newaxis],
            "length": np.array([mel.shape[1]], dtype=np.int64),
        })
        act_tensors = {n: t for n, t in zip(out_names, results) if n in act_names}

        # Compute H = Xᵀ X / n per FF weight; discard raw activations as we go.
        hessians = {}
        for logical, act_name in weight_to_activation.items():
            t = act_tensors.get(act_name)
            if t is None:
                continue
            X = t.squeeze(0).astype(np.float64)
            if X.ndim > 2:
                X = X.reshape(-1, X.shape[-1])
            hessians[logical] = (X.T @ X / X.shape[0]).astype(np.float32)

        print(f"  computed {len(hessians)} FF Hessians in memory")
        return hessians
    finally:
        shutil.rmtree(tmp)


# ── MIXED int4/int8 extraction ──────────────────────────────────────────────
#
# Attention layers (linear_{q,k,v,pos,out}) and pre_encode.out stay int8 for
# accuracy; feed-forward layers (linear1/linear2) use int4-GPTQ for size.

def extract_encoder_mixed(enc, cal_hessians=None):
    """Mixed precision: attention+pre_encode at int8, feed-forward at int4.

    The large FF matmuls (~128 MB/block as fp32 → ~16 MB int4) dominate model
    size, while the attention matmuls (~20 MB/block) are more quantization-
    sensitive, so we keep those at int8.

    cal_hessians: optional {logical_weight_name: H} from compute_ff_hessians().
    If None, falls back to self-Hessian GPTQ (significantly less accurate).
    """
    g = enc.graph
    weights = {}

    if cal_hessians is None:
        cal_hessians = {}
        print(f"  WARNING: no calibration Hessians provided, using self-Hessian fallback")
    else:
        print(f"  using {len(cal_hessians)} calibration Hessians")

    def add_fp32(name):
        arr = get_init(g, name)
        assert arr is not None, f"missing fp32 {name}"
        weights[name] = w_entry(arr.astype(np.float32))

    # ── pre_encode: biases fp32, conv weights dequantised, linear stays int8 ──
    for nm in [
        "pre_encode.conv.0.bias",
        "pre_encode.conv.2.bias",
        "pre_encode.conv.3.bias",
        "pre_encode.conv.5.bias",
        "pre_encode.conv.6.bias",
        "pre_encode.out.bias",
    ]:
        add_fp32(nm)

    for base in [0, 2, 3, 5, 6]:
        _add_conv_dequant(weights, g,
                          f"pre_encode.conv.{base}.weight",
                          f"pre_encode.conv.{base}.weight_quantized",
                          f"pre_encode.conv.{base}.weight_scale",
                          f"pre_encode.conv.{base}.weight_zero_point")

    # pre_encode.out: int8 (sensitive first layer)
    info = _matmul_weight_info(g, "/pre_encode/out/MatMul")
    assert info, "pre_encode.out.weight not found"
    _add_q(weights, g, "pre_encode.out.weight", *info)

    # Positional encoding table
    for n in g.node:
        if n.op_type == "Constant" and n.output[0] == "onnx::Slice_1513":
            for a in n.attribute:
                if a.name == "value":
                    pe = numpy_helper.to_array(a.t)
                    if pe.ndim == 3:
                        pe = pe[0]
                    weights["pos_encoding.pe_table"] = w_entry(pe.astype(np.float32))
                    print(f"  PE table: {pe.shape}")
                    break

    # ── 24 conformer blocks ──
    for i in range(24):
        pfx = f"layers.{i}"

        # fp32 (norms, biases, pos_bias)
        for nm in [
            f"{pfx}.norm_feed_forward1.weight",
            f"{pfx}.norm_feed_forward1.bias",
            f"{pfx}.feed_forward1.linear1.bias",
            f"{pfx}.feed_forward1.linear2.bias",
            f"{pfx}.norm_self_att.weight",
            f"{pfx}.norm_self_att.bias",
            f"{pfx}.self_attn.pos_bias_u",
            f"{pfx}.self_attn.pos_bias_v",
            f"{pfx}.self_attn.linear_q.bias",
            f"{pfx}.self_attn.linear_k.bias",
            f"{pfx}.self_attn.linear_v.bias",
            f"{pfx}.self_attn.linear_out.bias",
            f"{pfx}.norm_conv.weight",
            f"{pfx}.norm_conv.bias",
            f"{pfx}.conv.pointwise_conv1.bias",
            f"{pfx}.conv.pointwise_conv2.bias",
            f"{pfx}.norm_feed_forward2.weight",
            f"{pfx}.norm_feed_forward2.bias",
            f"{pfx}.feed_forward2.linear1.bias",
            f"{pfx}.feed_forward2.linear2.bias",
            f"{pfx}.norm_out.weight",
            f"{pfx}.norm_out.bias",
        ]:
            add_fp32(nm)

        # Attention weights → int8 (sensitive)
        attn_paths = {
            f"{pfx}.self_attn.linear_q.weight":    f"/{pfx}/self_attn/linear_q/MatMul",
            f"{pfx}.self_attn.linear_k.weight":    f"/{pfx}/self_attn/linear_k/MatMul",
            f"{pfx}.self_attn.linear_v.weight":    f"/{pfx}/self_attn/linear_v/MatMul",
            f"{pfx}.self_attn.linear_pos.weight":  f"/{pfx}/self_attn/linear_pos/MatMul",
            f"{pfx}.self_attn.linear_out.weight":  f"/{pfx}/self_attn/linear_out/MatMul",
        }
        for logical, prefix in attn_paths.items():
            info = _matmul_weight_info(g, prefix)
            assert info, f"missing {logical}"
            _add_q(weights, g, logical, *info)

        # FF weights → int4 GPTQ (large, less sensitive)
        # Source from the fp32 initializer since GPTQ wants full precision
        ff_paths = {
            f"{pfx}.feed_forward1.linear1.weight": f"/{pfx}/feed_forward1/linear1/MatMul",
            f"{pfx}.feed_forward1.linear2.weight": f"/{pfx}/feed_forward1/linear2/MatMul",
            f"{pfx}.feed_forward2.linear1.weight": f"/{pfx}/feed_forward2/linear1/MatMul",
            f"{pfx}.feed_forward2.linear2.weight": f"/{pfx}/feed_forward2/linear2/MatMul",
        }
        for logical, prefix in ff_paths.items():
            # Dequantise the int8 weight back to fp32, then re-quantise with int4 GPTQ.
            # This gives us the benefit of calibration-aware quantisation.
            info = _matmul_weight_info(g, prefix)
            assert info, f"missing {logical}"
            q_name, scale_name, _ = info
            q = get_init(g, q_name)
            sc = get_init(g, scale_name)
            assert q is not None and sc is not None
            fp_arr = q.astype(np.float32) * float(sc.item())
            if fp_arr.ndim != 2:
                fp_arr = fp_arr.reshape(fp_arr.shape[0], -1)
            H = cal_hessians.get(logical)
            packed, scales = quantize_int4_gptq(fp_arr, hessian=H)
            weights[logical] = w_entry(packed, scales=scales)

        # Conv weights – dequantise at extraction time
        conv_paths = {
            f"{pfx}.conv.pointwise_conv1.weight": f"/{pfx}/conv/pointwise_conv1/Conv",
            f"{pfx}.conv.depthwise_conv.weight":  f"/{pfx}/conv/depthwise_conv/Conv",
            f"{pfx}.conv.pointwise_conv2.weight": f"/{pfx}/conv/pointwise_conv2/Conv",
        }
        for logical, prefix in conv_paths.items():
            info = _conv_weight_info(g, prefix)
            assert info, f"missing {logical}"
            q_name, scale_name, zp_name = info
            q = get_init(g, q_name)
            sc = get_init(g, scale_name)
            assert q is not None and sc is not None
            if zp_name:
                zp = get_init(g, zp_name)
                if zp is not None:
                    assert np.all(zp == 0)
            weights[logical] = w_entry(q.astype(np.float32) * float(sc.item()))

        # Depthwise bias (via Reshape)
        dw_bias_arr = None
        target = f"/{pfx}/conv/depthwise_conv/Conv_output_0_bias_reshape_output"
        for n in g.node:
            if n.op_type == "Reshape" and n.output and n.output[0] == target:
                bias_init = get_init(g, n.input[0])
                if bias_init is not None and bias_init.size == 1024:
                    dw_bias_arr = bias_init.astype(np.float32)
                    break
        assert dw_bias_arr is not None, f"missing depthwise bias for block {i}"
        weights[f"{pfx}.conv.depthwise_conv.bias"] = w_entry(dw_bias_arr)

        if i == 0 or (i + 1) % 6 == 0:
            n_q4 = sum(1 for w in weights.values() if w.get("scales") is not None)
            n_q8 = sum(1 for w in weights.values() if w.get("scale") is not None and w.get("scales") is None)
            n_f  = sum(1 for w in weights.values() if w.get("scale") is None and w.get("scales") is None)
            print(f"  block {i}: {n_q4} int4 + {n_q8} int8 + {n_f} fp32 tensors")

    return weights


# ── Pack into binary ─────────────────────────────────────────────────────────

def pack_weights(all_weights, bin_path, json_path):
    os.makedirs(os.path.dirname(bin_path) or ".", exist_ok=True)
    manifest = {}
    offset = 0

    with open(bin_path, "wb") as f:
        for name in sorted(all_weights):
            w = all_weights[name]
            arr = w["data"]
            data = arr.flatten().tobytes()
            f.write(data)

            if w.get("scales") is not None:
                # int4 per-channel: packed data + scales array
                entry = {
                    "offset": offset,
                    "shape": list(arr.shape),
                    "numel": int(arr.size),
                    "dtype": "int4",
                }
                offset += len(data)
                # Write scales immediately after the packed data
                scales_data = w["scales"].astype(np.float32).tobytes()
                entry["scales_offset"] = offset
                entry["n_scales"] = int(w["scales"].size)
                f.write(scales_data)
                offset += len(scales_data)
            else:
                entry = {
                    "offset": offset,
                    "shape": list(arr.shape),
                    "numel": int(arr.size),
                    "dtype": "int8" if arr.dtype == np.int8 else "float32",
                }
                if w["scale"] is not None:
                    entry["scale"] = w["scale"]
                offset += len(data)

            manifest[name] = entry

    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)

    n_q4 = sum(1 for e in manifest.values() if e["dtype"] == "int4")
    n_q8 = sum(1 for e in manifest.values() if e["dtype"] == "int8")
    n_f = sum(1 for e in manifest.values() if e["dtype"] == "float32")
    total_mb = offset / (1024 * 1024)
    print(f"wrote {bin_path} ({total_mb:.1f} MB), {len(manifest)} tensors")
    if n_q4:
        print(f"  {n_q4} int4 tensors + {n_f} fp32 tensors")
    elif n_q8:
        print(f"  {n_q8} int8 tensors + {n_f} fp32 tensors")


# ── Main ─────────────────────────────────────────────────────────────────────

CONFIGS = {
    "fp32": {
        "input_dir": "onnx_fp32",
        "output_dir": "c_weights_fp32",
        "encoder_file": "encoder.onnx",
        "decoder_file": "decoder_joint.onnx",
        "extract_encoder": extract_encoder_fp32,
        "extract_decoder": extract_decoder_fp32,
    },
    "int8": {
        "input_dir": "onnx_int8",
        "output_dir": "c_weights_int8",
        "encoder_file": "encoder.int8.onnx",
        "decoder_file": "decoder_joint.int8.onnx",
        "extract_encoder": extract_encoder_int8,
        "extract_decoder": extract_decoder_int8,
    },
    "mixed": {
        "input_dir": "onnx_int8",
        "output_dir": "c_weights_mixed",
        "encoder_file": "encoder.int8.onnx",
        "decoder_file": "decoder_joint.int8.onnx",
        "extract_encoder": extract_encoder_mixed,
        "extract_decoder": extract_decoder_int8,
    },
}

def run_mode(mode, calibration_audio=None):
    cfg = CONFIGS[mode]
    input_dir = cfg["input_dir"]
    output_dir = cfg["output_dir"]
    bin_path = os.path.join(output_dir, "weights.bin")
    json_path = os.path.join(output_dir, "weights.json")

    print(f"\n═══ {mode} ═══")
    download_onnx(input_dir)

    # Mixed mode: compute calibration Hessians in-memory before extracting.
    # We need the fp32 ONNX model for calibration, so download it if missing.
    cal_hessians = None
    if mode == "mixed" and calibration_audio:
        download_onnx("onnx_fp32")
        fp32_encoder = os.path.join("onnx_fp32", "encoder.onnx")
        cal_hessians = compute_ff_hessians(fp32_encoder, calibration_audio)

    enc_model = load_model(os.path.join(input_dir, cfg["encoder_file"]))
    print("extracting encoder weights …")
    if mode == "mixed":
        enc_w = cfg["extract_encoder"](enc_model, cal_hessians)
    else:
        enc_w = cfg["extract_encoder"](enc_model)
    del enc_model
    del cal_hessians  # free the Hessians (can be several GB) before packing

    dec_model = load_model(os.path.join(input_dir, cfg["decoder_file"]))
    print("extracting decoder weights …")
    dec_w = cfg["extract_decoder"](dec_model)
    del dec_model

    all_w = {**enc_w, **dec_w}
    n_total = sum(w["data"].size for w in all_w.values())
    print(f"total: {len(all_w)} tensors, {n_total:,} elements")

    pack_weights(all_w, bin_path, json_path)

    tok_src = os.path.join(input_dir, "tokenizer.model")
    if os.path.exists(tok_src):
        shutil.copy2(tok_src, os.path.join(output_dir, "tokenizer.model"))
        print(f"copied tokenizer.model → {output_dir}/")

    print("done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ONNX weights to flat binary")
    parser.add_argument("mode", nargs="?",
                        choices=["fp32", "int8", "mixed"], default=None,
                        help="extraction mode (default: all)")
    parser.add_argument("--calibration-audio", default="date.wav",
                        help="16 kHz mono WAV for GPTQ calibration (mixed mode only)")
    args = parser.parse_args()

    modes = [args.mode] if args.mode else ["fp32", "int8", "mixed"]
    for mode in modes:
        cal = args.calibration_audio if mode == "mixed" else None
        if mode == "mixed" and cal and not os.path.exists(cal):
            print(f"warning: --calibration-audio '{cal}' not found; "
                  f"mixed mode will fall back to self-Hessian GPTQ (lower accuracy)")
            cal = None
        run_mode(mode, calibration_audio=cal)
