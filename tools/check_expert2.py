import struct

def read_kv_at(data, off):
    """Read key+type+value at given offset, return (key, val, next_off)."""
    key_len = struct.unpack_from('<Q', data, off)[0]
    off += 8
    key = data[off:off + key_len].decode('utf-8', 'replace')
    off += key_len
    vtype = struct.unpack_from('<I', data, off)[0]
    off += 4

    type_names = {0:'UNDEFINED',1:'STRING',2:'UINT32',3:'UINT64',4:'INT64',5:'FLOAT64',6:'BOOL',7:'ARRAY',8:'??',9:'??',20:'??'}

    if vtype == 1:  # STRING
        slen = struct.unpack_from('<Q', data, off)[0]
        off += 8
        val = data[off:off+slen].decode('utf-8', 'replace')
        off += slen
    elif vtype == 2:  # UINT32
        val = struct.unpack_from('<I', data, off)[0]
        off += 4
    elif vtype == 3:  # UINT64
        val = struct.unpack_from('<Q', data, off)[0]
        off += 8
    elif vtype == 4:  # INT64 - but let's check both 4 and 8 byte interpretations
        val = struct.unpack_from('<q', data, off)[0]
        off += 8
    elif vtype == 5:  # FLOAT64
        val = struct.unpack_from('<d', data, off)[0]
        off += 8
    elif vtype == 6:  # BOOL
        val = bool(data[off])
        off += 1
    elif vtype == 7:  # ARRAY
        alen = struct.unpack_from('<Q', data, off)[0]
        off += 8
        atype = struct.unpack_from('<I', data, off)[0]
        off += 4
        val = f"ARRAY[{alen}] type={atype}({type_names.get(atype, '?')})"
        for j in range(alen):
            if atype == 1:
                sl = struct.unpack_from('<Q', data, off)[0]
                off += 8 + sl
            elif atype == 2: off += 4
            elif atype == 3: off += 8
            elif atype == 4: off += 8
            elif atype == 5: off += 8
            elif atype == 6: off += 1
            else: off += 4
    elif vtype == 9:  # Maybe uint8 array or similar
        alen = struct.unpack_from('<Q', data, off)[0]
        off += 8
        atype = struct.unpack_from('<I', data, off)[0]
        off += 4
        val = f"ARRAY[{alen}] type={atype}({type_names.get(atype, '?')})"  # type 9 = MMAP or similar?
        for j in range(alen):
            if atype == 1:
                sl = struct.unpack_from('<Q', data, off)[0]
                off += 8 + sl
            elif atype == 2: off += 4
            elif atype == 3: off += 8
            elif atype == 4: off += 8
            elif atype == 5: off += 8
            elif atype == 6: off += 1
            else: off += 4
    else:
        val = f"UNSUPPORTED type={vtype}"

    return key, val, off, vtype

# Read Swift IQ2_XS
p = r'H:\OLLAMA-Models\GGUF\Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf'
with open(p, 'rb') as f:
    data = f.read(262144)  # 256 KB

print("=== Swift IQ2_XS ===")
off = 24  # skip header (magic, version, tensor_count, kv_count)
nk = struct.unpack_from('<Q', data, 16)[0]
print(f"KV pairs: {nk}")
for i in range(min(nk, 80)):
    try:
        key, val, off, vtype = read_kv_at(data, off)
        tn = {0:'UNDEFINED',1:'STRING',2:'UINT32',3:'UINT64',4:'INT64',5:'FLOAT64',6:'BOOL',7:'ARRAY',9:'ARRAY9'}.get(vtype, f'type{vtype}')
        if any(k in key for k in ['arch', 'expert', 'block_count', 'head_count', 'context_length', 'embed', 'rope.freq', 'ff', 'vocab']):
            print(f"  {key} ({tn}) = {val}")
    except Exception as e:
        print(f"  ERROR at offset {off}: {e}")
        break

# Read Q4_K_M reap-288
p2 = r'H:\OLLAMA-Models\GGUF\qwen3.8-flash-next-reap-288-Q4_K_M.gguf'
print("\n=== Q4_K_M reap-288 ===")
with open(p2, 'rb') as f:
    data2 = f.read(262144)

off = 24
nk = struct.unpack_from('<Q', data2, 16)[0]
print(f"KV pairs: {nk}")
for i in range(min(nk, 80)):
    try:
        key, val, off, vtype = read_kv_at(data2, off)
        tn = {0:'UNDEFINED',1:'STRING',2:'UINT32',3:'UINT64',4:'INT64',5:'FLOAT64',6:'BOOL',7:'ARRAY',9:'ARRAY9'}.get(vtype, f'type{vtype}')
        if any(k in key for k in ['arch', 'expert', 'block_count', 'head_count', 'context_length', 'embed', 'rope.freq', 'quant', 'reap', 'name']):
            print(f"  {key} ({tn}) = {val}")
    except Exception as e:
        print(f"  ERROR at offset {off}: {e}")
        break
