# 🤝 Contributing Guidelines

¡Gracias por tu interés en contribuir a este proyecto! Este documento proporciona las pautas para contribuir al proyecto **RAG Multimodal con Gemini Embeddings 2**.

## 📋 Tabla de Contenidos

- [Código de Conducta](#-código-de-conducta)
- [¿Cómo Contribuir?](#-cómo-contribuir)
- [Reportar Bugs](#-reportar-bugs)
- [Solicitar Características](#-solicitar-características)
- [Proceso de Pull Request](#-proceso-de-pull-request)
- [Estilo de Código](#-estilo-de-código)
- [Configuración del Entorno](#-configuración-del-entorno)

## 🎯 Código de Conducta

- Sé respetuoso con todos los contribuyentes
- Mantén un tono profesional y constructivo
- Acepta críticas constructivas de buena manera
- Enfócate en lo que es mejor para la comunidad

## 🚀 ¿Cómo Contribuir?

### 1. Fork el Repositorio
Haz clic en el botón "Fork" en GitHub para crear tu propia copia del proyecto.

### 2. Clona tu Fork
```bash
git clone https://github.com/tu-usuario/Gemini_Embeddings_2.git
cd Gemini_Embeddings_2
```

### 3. Crea una Rama
```bash
git checkout -b feature/tu-caracteristica
# o para bugs:
git checkout -b fix/tu-bug
```

### 4. Realiza tus Cambios
- Sigue el estilo de código existente
- Documenta tus cambios
- Añade pruebas si es necesario

### 5. Commit
```bash
git commit -m "feat: añade nueva característica X"
# o para bugs:
git commit -m "fix: corrige error Y en la búsqueda semántica"
```

### 6. Push
```bash
git push origin feature/tu-caracteristica
```

### 7. Abre un Pull Request
Ve a GitHub y abre un Pull Request describiendo tus cambios.

## 🐛 Reportar Bugs

### Antes de Reportar
- [ ] Verifica que el bug no haya sido reportado previamente
- [ ] Prueba con la última versión del código
- [ ] Revisa la documentación y troubleshooting

### Plantilla de Bug Report
```markdown
**Descripción:**
Descripción clara y concisa del bug.

**Pasos para Reproducir:**
1. Ir a '...'
2. Click en '....'
3. Scroll down to '....'
4. Ver error

**Comportamiento Esperado:**
Descripción de lo que esperabas que sucediera.

**Capturas de Pantalla:**
Si aplica, añade capturas para explicar el problema.

**Entorno:**
- OS: [e.g. Windows 10, macOS, Ubuntu]
- Python: [e.g. 3.9, 3.10]
- Versión del proyecto: [e.g. 1.0.0]

**Información Adicional:**
Cualquier otro contexto relevante.
```

## ✨ Solicitar Características

### Plantilla de Feature Request
```markdown
**¿Tu solicitud está relacionada con un problema?**
Descripción clara del problema. Ej: "Siempre me frustro cuando..."

**Describe la solución que te gustaría**
Descripción clara de lo que quieres que suceda.

**Describe alternativas que has considerado**
Descripción de soluciones alternativas o características existentes.

**¿Cómo te gustaría implementar esto?**
Si tienes ideas sobre cómo implementar la característica.

**¿Por qué es importante?**
Explica por qué esta característica sería útil para la comunidad.
```

## 📝 Proceso de Pull Request

1. **Revisión Automática:** GitHub Actions ejecutará pruebas automáticas
2. **Revisión del Mantenedor:** Al menos un mantenedor revisará tu código
3. **Feedback:** Se proporcionará feedback constructivo
4. **Aprobación:** Una vez aprobado, tu PR será mergeado
5. **Cierre:** El PR se cerrará después del merge

### Criterios de Aceptación
- [ ] El código sigue el estilo existente
- [ ] Las pruebas pasan exitosamente
- [ ] La documentación está actualizada
- [ ] Los commits son descriptivos
- [ ] No hay conflictos de merge

## 💻 Estilo de Código

### Python
- Sigue [PEP 8](https://pep8.org/)
- Usa type hints cuando sea posible
- Documenta funciones complejas
- Máximo 100 caracteres por línea

### Nombres de Variables
```python
# ✅ Correcto
def get_file_content(file_path: Path) -> str:
    total_documents = len(documents)
    
# ❌ Incorrecto
def getfilecontent(fp):
    totaldocs = len(docs)
```

### Comentarios y Documentación
```python
def analyze_video_with_llm(client, model: str, file_path: Path, query: str = None) -> str:
    """
    Analiza un video con Gemini LLM multimodal para extraer información relevante.
    
    Args:
        client: Cliente de Gemini
        model: Modelo LLM a usar (ej: gemini-2.5-flash)
        file_path: Ruta al archivo de video
        query: Pregunta específica para enfocar el análisis (opcional)
    
    Returns:
        Descripción/transcripción del video
    """
```

## 🛠️ Configuración del Entorno

### Requisitos
- Python 3.9 o superior
- pip
- Git

### Instalación
```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/Gemini_Embeddings_2.git
cd Gemini_Embeddings_2

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Edita .env con tu API Key

# Ejecutar pruebas
python test_rag_multimodal_completo.py
```

## 📚 Recursos Adicionales

- [Documentación del Proyecto](README.md)
- [Guía de Chat RAG](GUÍA_CHAT_RAG.md)
- [Código de Conducta](CODE_OF_CONDUCT.md)
- [Licencia](LICENSE)

## 🙏 Reconocimientos

Todos los contribuyentes serán reconocidos en el README.md del proyecto.

## 📞 Contacto

Si tienes preguntas, no dudes en abrir un issue o contactar al mantenedor principal.

---

**¡Gracias por contribuir! 🎉**
