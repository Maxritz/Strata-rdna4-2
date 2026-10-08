import re
p = r'H:\OLLAMA-Models\GGUF\Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf'
with open(p, 'rb') as f:
    data = f.read(8192)
strings = re.findall(rb'[\x20-\x7e]{4,}', data)
for s in strings:
    print(s.decode('ascii', 'replace'))
