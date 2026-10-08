import struct

# Search for expert_count value in Q4_K_M file using hex patterns
p = r'H:\OLLAMA-Models\GGUF\qwen3.8-flash-next-reap-288-Q4_K_M.gguf'
with open(p, 'rb') as f:
    data = f.read(262144)

print("=== Q4_K_M reap-288 ===")

# Search for qwen4exp.expert_count
key = b'qwen4exp.expert_count'
pos = data.find(key)
if pos >= 0:
    print(f"  expert_count key at offset {pos}")
    vt = struct.unpack_from('<I', data, pos + len(key))[0]
    print(f"  value_type: {vt}")
    # In this GGUF v3, type 4 = 4-byte uint32
    if vt == 4:
        val = struct.unpack_from('<I', data, pos + len(key) + 4)[0]
        print(f"  expert_count = {val}")
        next_bytes = data[pos+len(key)+8:pos+len(key)+16]
        print(f"  next bytes (next key_length?): {next_bytes.hex()}")

# Search for reap
for search_term in [b'reap', b'288', b'pruned', b'merged']:
    positions = []
    idx = 0
    while True:
        idx = data.find(search_term, idx)
        if idx < 0: break
        positions.append(idx)
        idx += 1
    if positions:
        print(f"  '{search_term.decode()}' found at offsets: {positions[:10]}")
        for p in positions[:3]:
            print(f"    context: {data[max(0,p-20):p+40]}")
    else:
        print(f"  '{search_term.decode()}': not found in header")

# Search for qwen4exp
idx = 0
while True:
    idx = data.find(b'qwen4exp', idx)
    if idx < 0: break
    # Get surrounding context
    ctx_start = max(0, idx - 10)
    ctx = data[ctx_start:idx+60]
    # Try to extract the key name (look backwards for the start of the key)
    print(f"  offset {idx}: 'qwen4exp' in context: {ctx}")
    idx += 1
    if idx > 5000: break

# Also search for architecture name
arch = data.find(b'qwen4exp')
if arch >= 0:
    print(f"\n  general.architecture = qwen4exp (confirmed)")

# Check for the "reap" concept — maybe it's in a different metadata field
for term in [b'description', b'purpose', b'reap', b'prune', b'merged']:
    pos = data.find(term)
    if pos >= 0:
        klen = struct.unpack_from('<Q', data, pos - 8)[0] if pos >= 8 else 0
        print(f"  Found '{term.decode()}' at offset {pos}, preceding key_len={klen}")
