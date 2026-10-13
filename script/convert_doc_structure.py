"""Download PaddleOCR document-structure models and convert them to MNN.

Produces the files expected by `ocr_rs::structure`:

  layout   PP-DocLayoutV3.mnn, PP-DocLayout_plus-L.mnn
  table    SLANet_plus_encoder.mnn, SLANet_plus_step.mnn, PP-LCNet_x1_0_table_cls.mnn,
           RT-DETR-L_wired_table_cell_det.mnn, RT-DETR-L_wireless_table_cell_det.mnn
  formula  PP-FormulaNet_plus-L_encoder.mnn, PP-FormulaNet_plus-L_decoder.mnn,
           PP-FormulaNet_plus-L_vocab.txt
  unwarp   UVDoc.mnn

Models whose official export contains control flow (SLANet's decode loop, FormulaNet's
generation loop) are split into an encoder and a single decode step; the loop runs in Rust.

Requirements (verified with the MNN 3.4.1 converter; models run on the MNN 3.6.1 runtime
that ocr-rs links through mnn-runtime):
  pip install "MNN==3.4.1" onnx onnxruntime onnxsim huggingface_hub
  # formula only:
  pip install torch "transformers>=5.19" safetensors

Usage:
  python script/convert_doc_structure.py --output models/structure
  python script/convert_doc_structure.py --output models/structure --components layout,table
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HF_ORG = "PaddlePaddle"
COMPONENTS = ("layout", "table", "formula", "unwarp")


def download(repo: str, cache: Path) -> Path:
    from huggingface_hub import snapshot_download

    target = cache / repo
    snapshot_download(f"{HF_ORG}/{repo}", local_dir=str(target))
    return target


def mnnconvert(onnx_path: Path, mnn_path: Path, fp16: bool) -> None:
    exe = shutil.which("mnnconvert")
    if exe is None:
        sys.exit("mnnconvert not found; install it with: pip install MNN==3.4.1")
    cmd = [exe, "-f", "ONNX", "--modelFile", str(onnx_path), "--MNNModel", str(mnn_path), "--bizCode", "mnn"]
    if fp16:
        cmd.append("--fp16")
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0 or not mnn_path.exists():
        sys.exit(f"mnnconvert failed for {onnx_path}:\n{result.stdout}\n{result.stderr}")
    print(f"  -> {mnn_path.name} ({mnn_path.stat().st_size / 1e6:.1f} MB)")


def simplify(onnx_path: Path) -> Path:
    """Constant-fold a traced graph so MNN gets static shape arithmetic."""
    import onnx
    import onnxsim

    model, ok = onnxsim.simplify(onnx.load(str(onnx_path)))
    if not ok:
        sys.exit(f"onnxsim could not validate {onnx_path}")
    out = onnx_path.with_name(onnx_path.stem + "_sim.onnx")
    onnx.save(model, str(out))
    return out


# ---------------------------------------------------------------------------
# SLANet: split the Loop into encoder + one decode step
# ---------------------------------------------------------------------------

def split_slanet(src: Path, out_dir: Path) -> tuple:
    import onnx
    from onnx import TensorProto, helper, utils

    model = onnx.load(str(src))
    graph = model.graph
    loop = next(n for n in graph.node if n.op_type == "Loop")
    feat = "p2o.pd_op.transpose.0.0"

    encoder = utils.Extractor(model).extract_model(["x"], [feat])
    enc_path = out_dir / "slanet_encoder.onnx"
    onnx.save(encoder, str(enc_path))

    body = next(a for a in loop.attribute if a.name == "body").g
    hid_in, char_in = "p2o.pd_op.full_with_tensor.0.0", "p2o.pd_op.full_with_tensor.4.0"
    hid_out, logits, loc = "p2o.sub_block.pd_op.add.7.0", "p2o.sub_block.pd_op.add.9.0", "p2o.sub_block.pd_op.sigmoid.2.0"
    onehot = "p2o.sub_block.pd_op.one_hot.0.0"

    producers = {o: n for n in body.node for o in n.output}
    needed, kept, stack = set(), set(), [hid_out, logits, loc]
    while stack:
        name = stack.pop()
        if name in needed:
            continue
        needed.add(name)
        node = producers.get(name)
        if node is not None and id(node) not in kept:
            kept.add(id(node))
            stack.extend(i for i in node.input if i)
    nodes = [n for n in body.node if id(n) in kept]
    inits = [t for t in body.initializer if t.name in needed]

    # MNN's OneHot returns [depth, B] for a [B] index: build the one-hot with Equal instead,
    # taking the previous token as float so the runtime only feeds float tensors.
    nodes = [n for n in nodes if n.op_type != "OneHot" and not (n.op_type == "Cast" and n.input[0] == char_in)]
    inits.append(helper.make_tensor("char_axes", TensorProto.INT64, [1], [1]))
    inits.append(helper.make_tensor("char_range", TensorProto.FLOAT, [1, 50], list(range(50))))
    nodes = [
        helper.make_node("Unsqueeze", ["prev_char", "char_axes"], ["prev_char_col"]),
        helper.make_node("Equal", ["prev_char_col", "char_range"], ["onehot_bool"]),
        helper.make_node("Cast", ["onehot_bool"], [onehot], to=TensorProto.FLOAT),
    ] + nodes
    # MNN squeezes every unit axis of [B, 1, C] when B == 1; a reshape keeps the batch axis.
    inits.append(helper.make_tensor("ctx_shape", TensorProto.INT64, [2], [0, -1]))
    for i, n in enumerate(nodes):
        if n.op_type == "Squeeze" and n.output[0] == "p2o.sub_block.pd_op.squeeze.0.0":
            nodes[i] = helper.make_node("Reshape", [n.input[0], "ctx_shape"], list(n.output))
    rename = {feat: "features", hid_in: "hidden", hid_out: "hidden_out", logits: "structure_logits", loc: "loc"}
    for n in nodes:
        for i, name in enumerate(n.input):
            n.input[i] = rename.get(name, name)
        for i, name in enumerate(n.output):
            n.output[i] = rename.get(name, name)

    step_graph = helper.make_graph(
        nodes,
        "slanet_step",
        [
            helper.make_tensor_value_info("features", TensorProto.FLOAT, ["B", "T", 96]),
            helper.make_tensor_value_info("hidden", TensorProto.FLOAT, ["B", 256]),
            helper.make_tensor_value_info("prev_char", TensorProto.FLOAT, ["B"]),
        ],
        [
            helper.make_tensor_value_info("hidden_out", TensorProto.FLOAT, ["B", 256]),
            helper.make_tensor_value_info("structure_logits", TensorProto.FLOAT, ["B", 50]),
            helper.make_tensor_value_info("loc", TensorProto.FLOAT, ["B", 8]),
        ],
        inits,
    )
    step = helper.make_model(step_graph, opset_imports=model.opset_import, ir_version=model.ir_version)
    onnx.checker.check_model(step)
    step_path = out_dir / "slanet_step.onnx"
    onnx.save(step, str(step_path))
    return enc_path, step_path


# ---------------------------------------------------------------------------
# PP-FormulaNet: export encoder + KV-cached decoder step from transformers weights
# ---------------------------------------------------------------------------

def export_formulanet(src: Path, out_dir: Path) -> tuple:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import PPFormulaNetForConditionalGeneration
    from transformers.models.pp_formulanet import modeling_pp_formulanet as mpf

    model = PPFormulaNetForConditionalGeneration.from_pretrained(str(src), attn_implementation="eager").eval()
    model.requires_grad_(False)
    enc_m, dec_m, lm_head = model.model.encoder, model.model.decoder, model.lm_head
    layers = len(dec_m.layers)
    heads_n = dec_m.layers[0].self_attn.num_heads
    head_dim = dec_m.layers[0].self_attn.head_dim

    def rel_table(size, rel_pos):
        idx = torch.arange(size)[:, None] - torch.arange(size)[None, :] + (size - 1)
        return rel_pos[idx].detach()

    # Static relative-position tables and MatMul instead of Einsum: MNN mis-executes the
    # dynamically indexed Einsum form of the decomposed relative position bias.
    def attention_forward(self, hidden_states, output_attentions=None):
        b, h, w, _ = hidden_states.shape
        n = self.num_attention_heads
        qkv = self.qkv(hidden_states).reshape(b, h * w, 3, n, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, b * n, h * w, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        rh, rw = rel_table(h, self.rel_pos_h), rel_table(w, self.rel_pos_w)
        bh, c = b * n, q.shape[-1]
        qr = q.reshape(bh, h, w, c)
        rel_h = torch.matmul(qr.permute(1, 0, 2, 3).reshape(h, bh * w, c), rh.transpose(1, 2))
        rel_h = rel_h.reshape(h, bh, w, h).permute(1, 0, 2, 3)
        rel_w = torch.matmul(qr.permute(2, 0, 1, 3).reshape(w, bh * h, c), rw.transpose(1, 2))
        rel_w = rel_w.reshape(w, bh, h, w).permute(1, 2, 0, 3)
        attn = attn + (rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).reshape(bh, h * w, h * w)
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).reshape(b, n, h, w, -1).permute(0, 2, 3, 1, 4).reshape(b, h, w, -1)
        return self.proj(out), attn

    with torch.no_grad():
        probe = torch.randn(1, 3, 768, 768)
        reference = enc_m(pixel_values=probe).pooler_output
        mpf.PPFormulaNetVisionAttention.forward = attention_forward
        diff = (reference - enc_m(pixel_values=probe).pooler_output).abs().max().item()
        if diff > 1e-3:
            sys.exit(f"Patched FormulaNet attention diverges from transformers (max diff {diff})")

    def split_heads(x):
        return x.view(x.shape[0], x.shape[1], heads_n, head_dim).transpose(1, 2)

    class Encoder(nn.Module):
        def forward(self, pixel_values):
            x = enc_m.patch_embed(pixel_values) + enc_m.pos_embed
            for layer in enc_m.layers:
                x = layer(x)
            e = enc_m.multi_modal_projector(enc_m.neck(x))
            ks = [split_heads(l.encoder_attn.k_proj(e))[0] for l in dec_m.layers]
            vs = [split_heads(l.encoder_attn.v_proj(e))[0] for l in dec_m.layers]
            return torch.stack(ks), torch.stack(vs)

    def attend(q, k, v, mask=None):
        s = torch.matmul(q, k.transpose(-1, -2))
        if mask is not None:
            s = s + mask
        return torch.matmul(torch.softmax(s, dim=-1), v)

    class DecoderStep(nn.Module):
        def forward(self, token, position, past_k, past_v, past_mask, cross_k, cross_v):
            tok = token.to(torch.int64).view(1, 1)
            pos = position.to(torch.int64).view(1, 1) + 2
            h = dec_m.embed_tokens(tok) + F.embedding(pos, dec_m.embed_positions.weight)
            h = dec_m.layernorm_embedding(h)
            pk, pv = past_k.permute(1, 2, 0, 3), past_v.permute(1, 2, 0, 3)
            mask = torch.cat([past_mask, torch.zeros(1, dtype=past_mask.dtype)]).view(1, 1, 1, -1)
            new_k, new_v = [], []
            for i, layer in enumerate(dec_m.layers):
                sa = layer.self_attn
                x = layer.self_attn_layer_norm(h)
                q = split_heads(sa.q_proj(x)) * sa.scaling
                k_new, v_new = split_heads(sa.k_proj(x)), split_heads(sa.v_proj(x))
                new_k.append(k_new[0, :, 0])
                new_v.append(v_new[0, :, 0])
                k = torch.cat([pk[i:i + 1], k_new], dim=2)
                v = torch.cat([pv[i:i + 1], v_new], dim=2)
                h = h + sa.out_proj(attend(q, k, v, mask).transpose(1, 2).reshape(1, 1, heads_n * head_dim))
                ca = layer.encoder_attn
                x = layer.encoder_attn_layer_norm(h)
                q = split_heads(ca.q_proj(x)) * ca.scaling
                a = attend(q, cross_k[i:i + 1], cross_v[i:i + 1]).transpose(1, 2).reshape(1, 1, heads_n * head_dim)
                h = h + ca.out_proj(a)
                h = h + layer.fc2(layer.activation_fn(layer.fc1(layer.final_layer_norm(h))))
            logits = lm_head(dec_m.layer_norm(h))[0, 0]
            next_token = torch.argmax(logits).to(torch.float32).view(1)
            return next_token, torch.stack(new_k).unsqueeze(0), torch.stack(new_v).unsqueeze(0)

    enc_path, dec_path = out_dir / "formula_encoder.onnx", out_dir / "formula_decoder.onnx"
    with torch.no_grad():
        torch.onnx.export(Encoder(), (torch.randn(1, 3, 768, 768),), str(enc_path),
                          input_names=["pixel_values"], output_names=["cross_k", "cross_v"],
                          opset_version=17, dynamo=False)
        p = 3
        args = (torch.tensor([2.0]), torch.tensor([0.0]),
                torch.randn(p, layers, heads_n, head_dim), torch.randn(p, layers, heads_n, head_dim),
                torch.zeros(p), torch.randn(layers, heads_n, 144, head_dim), torch.randn(layers, heads_n, 144, head_dim))
        torch.onnx.export(DecoderStep(), args, str(dec_path),
                          input_names=["token", "position", "past_k", "past_v", "past_mask", "cross_k", "cross_v"],
                          output_names=["next_token", "new_k", "new_v"],
                          dynamic_axes={"past_k": {0: "P"}, "past_v": {0: "P"}, "past_mask": {0: "P"}},
                          opset_version=17, dynamo=False)
    return enc_path, dec_path


def write_vocab(tokenizer_json: Path, out_path: Path) -> None:
    vocab = json.loads(tokenizer_json.read_text(encoding="utf-8"))["model"]["vocab"]
    tokens = [None] * len(vocab)
    for token, idx in vocab.items():
        tokens[idx] = token
    if any(t is None or "\n" in t or "\r" in t for t in tokens):
        sys.exit("Unexpected vocabulary layout in tokenizer.json")
    out_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    print(f"  -> {out_path.name} ({len(tokens)} tokens)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("models/structure"))
    parser.add_argument("--cache", type=Path, default=None, help="download/work directory (default: <output>/_work)")
    parser.add_argument("--components", default=",".join(COMPONENTS))
    parser.add_argument("--no-fp16", action="store_true", help="keep float32 weights")
    args = parser.parse_args()

    components = [c.strip() for c in args.components.split(",") if c.strip()]
    unknown = set(components) - set(COMPONENTS)
    if unknown:
        sys.exit(f"Unknown components: {', '.join(sorted(unknown))}")
    out = args.output
    work = args.cache or out / "_work"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    fp16 = not args.no_fp16

    def convert_official(repo: str, name: str) -> None:
        print(f"[{name}]")
        src = download(repo, work)
        mnnconvert(src / "inference.onnx", out / f"{name}.mnn", fp16)

    if "layout" in components:
        convert_official("PP-DocLayoutV3_onnx", "PP-DocLayoutV3")
        convert_official("PP-DocLayout_plus-L_onnx", "PP-DocLayout_plus-L")

    if "table" in components:
        print("[SLANet_plus]")
        src = download("SLANet_plus_onnx", work)
        enc, step = split_slanet(src / "inference.onnx", work)
        mnnconvert(enc, out / "SLANet_plus_encoder.mnn", fp16)
        mnnconvert(step, out / "SLANet_plus_step.mnn", fp16)
        convert_official("PP-LCNet_x1_0_table_cls_onnx", "PP-LCNet_x1_0_table_cls")
        convert_official("RT-DETR-L_wired_table_cell_det_onnx", "RT-DETR-L_wired_table_cell_det")
        convert_official("RT-DETR-L_wireless_table_cell_det_onnx", "RT-DETR-L_wireless_table_cell_det")

    if "formula" in components:
        print("[PP-FormulaNet_plus-L]")
        src = download("PP-FormulaNet_plus-L_safetensors", work)
        enc, dec = export_formulanet(src, work)
        mnnconvert(simplify(enc), out / "PP-FormulaNet_plus-L_encoder.mnn", fp16)
        mnnconvert(dec, out / "PP-FormulaNet_plus-L_decoder.mnn", fp16)
        write_vocab(src / "tokenizer.json", out / "PP-FormulaNet_plus-L_vocab.txt")

    if "unwarp" in components:
        convert_official("UVDoc_onnx", "UVDoc")

    print(f"Done. Models are in {out}. Remove {work} to free the downloaded sources.")


if __name__ == "__main__":
    main()
