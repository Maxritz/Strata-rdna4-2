import struct, sys

p = r'H:\OLLAMA-Models\GGUF\Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf'
with open(p, 'rb') as f:
    data = f.read(131072)

print("Searching for expert_count key...")
key = b'qwen4exp.expert_count'
pos = data.find(key)
print(f"  find result: {pos}")

if pos >= 0 and pos < len(data):
    print(f"  Key at offset {pos}")
    vtype_off = pos + len(key)
    if vtype_off + 4 <= len(data):
        vt = struct.unpack_from('<I', data, vtype_off)[0]
        print(f"  Value type: {vt}")
        val_off = vtype_off + 4
        if val_off + 8 <= len(data):
            if vt == 2:
                val = struct.unpack_from('<I', data, val_off)[0]
                print(f"  expert_count (UINT32): {val}")
            elif vt == 3:
                val = struct.unpack_from('<Q', data, val_off)[0]
                print(f"  expert_count (UINT64): {val}")
            elif vt == 8:
                val = struct.unpack_from('<Q', data, val_off)[0]
                print(f"  expert_count (type8-as-uint64): {val}")
            else:
                print(f"  Unknown type, raw 8 bytes: {data[val_off:val_off+8].hex()}")

# Also search for all occurrences of "expert"
print("\nSearching for 'expert' occurrences...")
idx = 0
while True:
    idx = data.find(b'expert', idx)
    if idx < 0: break
    print(f"  offset {idx}: {data[idx:idx+40]}")
    idx += 1

# Search for architecture string
print("\nSearching for 'qwen4exp' occurrences...")
idx = 0
while True:
    idx = data.find(b'qwen4exp', idx)
    if idx < 0: break
    print(f"  offset {idx}: {data[idx:idx+50]}")
    idx += 1
