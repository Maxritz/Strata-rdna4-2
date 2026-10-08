#!/usr/bin/env python3
"""Quick GGUF metadata reader for Strata compatibility checks."""
import struct
import sys
import os

def read_gguf(path, max_keys=60):
    with open(path, 'rb') as f:
        data = f.read(65536)
    if data[:4] not in (b'gguf', b'GGUF'):
        print(f"  NOT GGUF (magic={data[:4]})")
        return
    ver = struct.unpack_from('<I', data, 4)[0]
    nt = struct.unpack_from('<Q', data, 8)[0]
    nk = struct.unpack_from('<Q', data, 16)[0]
    print(f"  GGUF v{ver} | tensors={nt} | kv_pairs={nk} | size={os.path.getsize(path)/1e9:.1f}GB")
    off = 24
    wanted = {
        'general.architecture','general.name','llama.context_length','llama.rope.freq_base',
        'llama.block_count','llama.embedding_length','llama.feed_forward_length',
        'llama.rope.dimension_count','llama.attention.head_count','llama.attention.head_count_kv',
        'llama.vocab_size','llama.expert_count','llama.expert_used','llama.expert_model',
        'llama.expert_gates','llama.expert_weights',
        'metadata.tokenizer.ggml.pretokenizer','llama.rope_scaling.factor',
        'llama.rope_scaling.attn_factor','general.quantized_by',
    }
    for i in range(min(nk, max_keys)):
        if off >= len(data):
            break
        klen = struct.unpack_from('<Q', data, off)[0]
        off += 8
        key = data[off:off+klen].decode('utf-8', 'replace')
        off += klen
        vt = struct.unpack_from('<I', data, off)[0]
        off += 4
        if vt == 1:
            slen = struct.unpack_from('<Q', data, off)[0]
            off += 8
            val = data[off:off+slen].decode('utf-8', 'replace')
            off += slen
        elif vt == 2:
            val = struct.unpack_from('<I', data, off)[0]
            off += 4
        elif vt == 3:
            val = struct.unpack_from('<Q', data, off)[0]
            off += 8
        elif vt == 4:
            val = struct.unpack_from('<q', data, off)[0]
            off += 8
        elif vt == 5:
            val = struct.unpack_from('<d', data, off)[0]
            off += 8
        elif vt == 6:
            val = bool(data[off])
            off += 1
        elif vt == 7:
            alen = struct.unpack_from('<Q', data, off)[0]
            off += 8
            atype = struct.unpack_from('<I', data, off)[0]
            off += 4
            val = f"ARRAY[{alen}] type={atype}"
            for j in range(alen):
                if atype == 1:
                    sl = struct.unpack_from('<Q', data, off)[0]
                    off += 8 + sl
                elif atype == 2:
                    off += 4
                elif atype in (3, 4, 5):
                    off += 8
                elif atype == 6:
                    off += 1
                else:
                    off += 4
        else:
            val = f"type{vt}"
        if key in wanted:
            print(f"  {key} = {val}")

if __name__ == '__main__':
    base = r'H:\OLLAMA-Models\GGUF'
    files = [
        'Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf',
        'Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf',
        'qwen3.8-flash-next-reap-288-Q4_K_M.gguf',
        'Qwen3.8-Flash-Next-ngram-embeddings-Q4_0.gguf',
    ]
    for name in files:
        p = os.path.join(base, name)
        if os.path.exists(p):
            print(f"\n=== {name} ===")
            read_gguf(p)
        else:
            print(f"\n=== {name}: NOT FOUND ===")
