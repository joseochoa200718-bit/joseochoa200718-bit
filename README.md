# Hi, I'm Jose 👋

I'm learning programming with Scrimba.

I'm passionate about coding and excited to grow my skills.

Thanks for visiting my GitHub profile!

## Reducir tokens en OpenAI/Codex

Este repositorio incluye `token_optimizer.py`, una capa intermedia que:

- conserva bloques marcados como `CRITICAL:`, `REQUIREMENTS:`, `FILES:`, `ERRORS:` o `INSTRUCTIONS:`;
- comprime el resto de las consultas largas con LLMLingua;
- cuenta los tokens antes y después con `tiktoken`;
- usa GPTCache para evitar llamadas repetidas cuando se configura un servidor de caché.

### Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="tu-clave"
```

Para activar GPTCache, inicia un servidor GPTCache y configura:

```bash
export GPTCACHE_URI="http://localhost:8000"
```

### Uso

```python
from token_optimizer import TokenOptimizer

optimizer = TokenOptimizer(compression_rate=0.5)
answer = optimizer.chat([
    {"role": "system", "content": "Responde con cambios concretos."},
    {"role": "user", "content": """
CRITICAL: No cambies la API pública.
FILES: src/app.py, tests/test_app.py
REQUIREMENTS: Mantén compatibilidad con Python 3.11.

Aquí va el contexto largo de la consulta o del repositorio...
"""},
])
print(answer)
print(optimizer.last_stats)
```

Empieza con `compression_rate=0.7` y bájalo gradualmente. Compara la calidad de
las respuestas; si se pierde información, aumenta la tasa y añade esa información
a un bloque protegido. Para usar Codex, envía el resultado de `compress()` a tu
cliente Codex en lugar de enviar la consulta original.
