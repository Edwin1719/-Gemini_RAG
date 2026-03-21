# 🧠 RAG Multimodal con Gemini Embeddings 2 + LLM

Aplicación Streamlit que permite cargar archivos de diferentes formatos y realizar **búsquedas semánticas con Chat RAG** usando **Gemini Embeddings 2** y **Gemini LLM**.

![texto del vínculo](https://i.ytimg.com/vi/zUkKvWBJ_0I/maxresdefault.jpg)

## ✨ Características Principales

- **📤 Carga Multimodal:** Soporta texto, imágenes, PDF, audio y video
- **🔍 Búsqueda Semántica:** Encuentra archivos relevantes usando consultas en lenguaje natural
- **🖼️ Búsqueda por Imagen:** Sube una imagen para encontrar archivos visualmente similares
- **💬 Chat RAG:** Haz preguntas en lenguaje natural y recibe respuestas basadas en tus documentos
- **💾 Almacenamiento Persistente:** Los embeddings se guardan en ChromaDB
- **🎯 Umbrales Dinámicos:** Ajuste automático según el tipo de archivo para mejorar la recuperación
- **🎬 Transcripción de Videos:** Análisis automático de videos con LLM bajo demanda
- **⚡ Caché Inteligente:** Evita reprocesar archivos que no han cambiado (ahorro de 80-90% en costos)
- **📚 Citas Interactivas:** Visualiza los fragmentos exactos usados para generar cada respuesta

## 🎯 Novedades - Mejoras Implementadas

### 🔹 Umbrales Dinámicos por Tipo de Archivo

El sistema ahora utiliza **umbrales de similitud dinámicos** que se ajustan automáticamente según el tipo de archivo recuperado. Esto resuelve el problema de que diferentes modalidades (texto, imagen, video) tienen diferentes rangos de similitud cuando se comparan con consultas de texto.

| Tipo de Archivo | Umbral | Justificación |
|-----------------|--------|---------------|
| 📄 **Texto/PDF** | **0.50** | Texto↔texto tiene alta similitud semántica |
| 🖼️ **Imagen** | **0.45** | Texto↔imagen tiene similitud media-alta |
| 🎵 **Audio** | **0.35** | Texto↔audio tiene similitud media-baja |
| 🎬 **Video** | **0.30** | Texto↔video tiene la similitud más baja |

**¿Por qué es importante?**

Los embeddings multimodales de Gemini Embedding 2 mapean diferentes modalidades (texto, imagen, audio, video) al mismo espacio semántico. Sin embargo, las similitudes de coseno resultantes varían significativamente:

- **Texto↔Texto:** 0.60 - 0.80 (alta similitud)
- **Texto↔Imagen:** 0.45 - 0.65 (similitud media)
- **Texto↔Video:** 0.30 - 0.45 (similitud baja)

Con un umbral fijo de 0.50, los videos nunca se recuperaban. Con umbrales dinámicos, cada tipo de archivo tiene el umbral apropiado para su rango de similitud natural.

### 🔹 Análisis de Videos con LLM

El Chat RAG ahora puede **analizar videos automáticamente** cuando se necesita responder preguntas:

1. **Detección Automática:** Cuando un video se recupera como contexto, el sistema verifica si ya tiene transcripción
2. **Análisis con LLM:** Si no hay transcripción, usa `gemini-2.5-flash` para analizar el video
3. **Caché Persistente:** La transcripción se guarda en un archivo `.transcripcion.txt` junto al video
4. **Reutilización:** Próximas consultas usan la transcripción guardada (sin reprocesar)

**Ventajas:**
- ✅ No necesita reprocesar videos ya analizados
- ✅ Extrae información específica relevante para la pregunta
- ✅ Permite responder preguntas sobre el contenido del video
- ✅ Se integra naturalmente con el flujo de RAG

### 🔹 Búsqueda Semántica Mejorada

La pestaña de **Búsqueda Semántica** ahora incluye:

- **Checkbox "Umbral automático":** Activa/desactiva umbrales dinámicos
- **Indicadores visuales:** Colores según similitud (🟢🟡🟠🔴)
- **Información de umbral:** Muestra el umbral aplicado a cada resultado
- **Mensajes de ayuda:** Explica por qué algunos resultados se filtran

### 🔹 Caché Inteligente de Archivos ⚡ **NUEVO**

El sistema ahora implementa un **sistema de caché basado en hash SHA256** que detecta automáticamente si un archivo ha cambiado antes de generar embeddings.

**¿Cómo funciona?**

```
┌─────────────────────────────────────────────────────────┐
│ 1. Usuario sube archivo                                 │
│ 2. Calcular hash SHA256 del archivo                     │
│ 3. ¿Existe en ChromaDB?                                 │
│    ├─ NO → Generar embedding y guardar con hash ✅      │
│    └─ SÍ → ¿Hash igual?                                 │
│         ├─ SÍ → Saltar (⏭️ Sin costo de API)            │
│         └─ NO → Regenerar embedding (🔄 Archivo cambió) │
└─────────────────────────────────────────────────────────┘
```

**Beneficios:**
- ⚡ **80-90% menos tiempo de carga** si los archivos no cambian
- 💰 **Ahorro significativo** en costos de API de Gemini
- 🔄 **Detección automática** de archivos modificados
- 📦 **Metadata enriquecida** con `file_hash` en ChromaDB

**Ejemplo de uso:**
```
Primera carga:
  ✅ DATABiQ.txt → Generando embedding... (2.3s)

Segunda carga (mismo archivo):
  ⏭️ DATABiQ.txt no ha cambiado. Saltando... (0.1s)

Tercera carga (archivo modificado):
  🔄 DATABiQ.txt ha cambiado. Regenerando embedding... (2.3s)
```

### 🔹 Citas Interactivas en Chat RAG 📚 **NUEVO**

El Chat RAG ahora muestra **los fragmentos exactos de documentos** usados para generar cada respuesta, proporcionando transparencia total y capacidad de verificación.

**¿Qué muestra?**

- 📖 **Fragmento de contexto:** El texto exacto usado por el LLM
- 📊 **Información de similitud:** Qué tan relevante fue cada documento
- 🔗 **Referencia completa:** Nombre del archivo y tipo
- 🎯 **Múltiples fuentes:** Todas las citas en expandibles separados

**Ejemplo de visualización:**
```
🤖 Asistente:
De cara al futuro, DATABiQ planea ofrecer los siguientes servicios:
* Servicios de gestión de bases de datos con SQL Server.
* Proyectos innovadores de automatización utilizando Integration Services.

─────────────────────────────────────────────────────
📚 Fuentes utilizadas:
  
  📄 Fuente 1: DATABiQ.txt (📝 Texto, Similitud: 0.62)
  ┌─────────────────────────────────────────────────┐
  │ Fragmento usado como contexto:                  │
  │ DATABiQ es una empresa especializada en...      │
  │ Servicios: SQL Server, Integration Services...  │
  └─────────────────────────────────────────────────┘
  
  🎬 Fuente 2: DATABiQ_...mp4 (🎬 Video, Similitud: 0.42)
  ┌─────────────────────────────────────────────────┐
  │ Fragmento usado como contexto:                  │
  │ En el video se mencionan los servicios futuros  │
  │ de gestión de bases de datos y automatización  │
  └─────────────────────────────────────────────────┘
```

**Ventajas:**
- ✅ **Transparencia total:** Sabés exactamente qué información usó el LLM
- ✅ **Verificación rápida:** Podés confirmar la precisión de las respuestas
- ✅ **Contexto enriquecido:** Accedés a más detalles si lo necesitás
- ✅ **Trazabilidad:** Seguimiento completo de las fuentes usadas

## 📋 Formatos Soportados

| Tipo | Extensiones | Umbral | Embedding |
|------|-------------|--------|-----------|
| 📄 Documentos | PDF, TXT, MD, JSON, CSV | 0.50 | Texto directo |
| 🖼️ Imágenes | PNG, JPG, JPEG, GIF, WEBP | 0.45 | Vision model |
| 🎵 Audio | MP3, WAV, OGG | 0.35 | Audio model |
| 🎬 Video | MP4, AVI, MOV, WEBM | 0.30 | Video + Audio |

**Nota:** Los videos tienen límite de **80 segundos con audio** o **120 segundos sin audio** según la documentación oficial de Gemini Embedding 2.

## 🚀 Instalación

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar API Key

Asegúrate de que tu API Key de Gemini esté configurada en el archivo `.env`:

```
GEMINI_API_KEY=tu_api_key_aqui
```

## ▶️ Ejecutar la Aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📖 Cómo Usar

### 1️⃣ Cargar Archivos

1. Ve a la pestaña **"📤 Cargar Archivos"**
2. Selecciona uno o múltiples archivos (hasta 4 tipos diferentes)
3. Espera a que se generen los embeddings
4. Los archivos quedarán indexados en ChromaDB

**Proceso de carga:**
```
Archivo → Leer bytes → Gemini Embeddings 2 → Vector (3072d) → ChromaDB
```

**💡 Caché Inteligente:**
- La **primera vez** que cargás un archivo, se genera el embedding
- Si **volvés a cargar el mismo archivo**, el sistema lo detecta y lo saltea
- Si el archivo **cambió**, el sistema lo detecta y regenera el embedding automáticamente

**Ejemplo:**
```
✅ Primer carga:   DATABiQ.txt → Procesando... (2.3s)
⏭️ Segunda carga:  DATABiQ.txt → No ha cambiado. Saltando... (0.1s)
🔄 Archivo nuevo:  DATABiQ_v2.txt → Procesando... (2.3s)
```

### 2️⃣ Búsqueda Semántica

1. Ve a la pestaña **"🔍 Búsqueda Semántica"**
2. Elige el tipo de búsqueda:
   - **🔤 Por Texto:** Escribe una consulta en lenguaje natural
   - **🖼️ Por Imagen:** Sube una imagen para encontrar similares
3. Activa **"Umbral automático"** (recomendado para RAG multimodal)
4. Ajusta el número de resultados
5. ¡Explora los resultados!

**Ejemplo de consulta:**
```
Consulta: "¿Cuáles son los servicios que brindara DATABiQ de cara al futuro?"
Resultados:
  ✅ 🎬 DATABiQ_...mp4 (Similitud: 0.3990, Umbral: 0.30) ✅ RECUPERADO
```

### 3️⃣ Chat RAG

1. Ve a la pestaña **"💬 Chat RAG"**
2. Escribe tu pregunta en lenguaje natural
3. El sistema:
   - Busca documentos relevantes con umbrales dinámicos
   - Analiza videos automáticamente si es necesario
   - Genera una respuesta con el LLM citando las fuentes
4. Revisa el historial de conversación
5. **📚 Ver citas:** Expandí las fuentes para ver los fragmentos exactos usados

**Ejemplo de respuesta:**
```
Pregunta: "¿Cuáles son los servicios que brindara DATABiQ de cara al futuro?"

Respuesta:
De cara al futuro, DATABiQ planea ofrecer los siguientes servicios:
* Servicios de gestión de bases de datos con SQL Server.
* Proyectos innovadores de automatización utilizando Integration Services.
(DATABiQ_ Tu Aliado en Ciencia d 2024-10-24.mp4)

─────────────────────────────────────────────────────
📚 Fuentes utilizadas:
  
  📄 Fuente 1: DATABiQ.txt (Similitud: 0.62)
  [Expandir para ver fragmento usado]
  
  🎬 Fuente 2: DATABiQ_...mp4 (Similitud: 0.42)
  [Expandir para ver fragmento usado]
```

### 4️⃣ Gestionar Archivos

1. Ve a la pestaña **"📚 Archivos Indexados"**
2. Visualiza todos los archivos cargados con metadata
3. Elimina archivos individuales si es necesario

## 🎯 Ejemplos de Uso

### Búsqueda Cruzada Multimodal

| Consulta | Tipo Esperado | Resultado | Similitud | Umbral |
|----------|---------------|-----------|-----------|--------|
| "¿Email de contacto?" | 📝 Texto | DATABiQ.txt | 0.6216 | 0.50 ✅ |
| "¿Quién es el fundador?" | 📝 Texto | DATABiQ.txt | 0.6029 | 0.50 ✅ |
| "¿Servicios futuros?" | 🎬 Video | Video + Texto | 0.39-0.52 | 0.30-0.50 ✅ |
| "¿Qué es DATABiQ?" | Mixto | Texto + Imagen + Video | 0.35-0.56 | 0.30-0.50 ✅ |

### Preguntas que Responden Diferentes Fuentes

| Pregunta | Fuente | Respuesta |
|----------|--------|-----------|
| ¿Email corporativo? | 📝 DATABiQ.txt | databiq29@gmail.com |
| ¿Formación del fundador? | 📝 DATABiQ.txt | Maestría en Big Data, Ingeniería Industrial |
| ¿Servicios futuros? | 🎬 Video | SQL Server + Integration Services |
| ¿Ubicación sede? | 📝 DATABiQ.txt | Pereira, Risaralda, Colombia |
| ¿Tecnologías para ML? | 📝+🎬 Mixto | Python, Scikit-learn, TensorFlow |

## 🏗️ Arquitectura

### Flujo de Indexación (Carga de Archivos)
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Archivos      │────▶│  Gemini         │────▶│  ChromaDB       │
│   (multiformato)│     │  Embeddings 2   │     │  (vectores)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Flujo de RAG (Chat con Documentos) - Con Umbrales Dinámicos
```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Pregunta   │────▶│  Gemini         │────▶│  ChromaDB       │
│   (usuario)  │     │  Embeddings 2   │     │  (búsqueda)     │
└──────────────┘     └──────────────────┘     └─────────────────┘
                            │                        │
                            │                        ▼
                            │              ┌─────────────────┐
                            │              │  Documentos     │
                            │              │  relevantes     │
                            │              └─────────────────┘
                            │                        │
                            │                        ▼
                            │              ┌─────────────────┐
                            │              │ Umbral Dinámico │
                            │              │ según tipo      │
                            │              └─────────────────┘
                            │                        │
                            ▼                        ▼
                     ┌─────────────────────────────────┐
                     │      Gemini LLM (Respuesta)     │
                     │  Contexto + Pregunta → Respuesta│
                     └─────────────────────────────────┘
                                    │
                                    ▼
                          ┌─────────────────┐
                          │   Respuesta     │
                          │   al Usuario    │
                          └─────────────────┘
```

### Flujo de Análisis de Video
```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Video      │────▶│  ¿Existe         │────▶│  Usar caché     │
│   recuperado │     │  transcripción?  │     │  (.txt)         │
└──────────────┘     └──────────────────┘     └─────────────────┘
                            │ NO
                            ▼
                     ┌──────────────────┐
                     │  Gemini LLM      │
                     │  (analizar video)│
                     └──────────────────┘
                            │
                            ▼
                     ┌──────────────────┐
                     │  Guardar caché   │
                     │  (.transcripcion)│
                     └──────────────────┘
```

## ⚙️ Configuración Avanzada

### Dimensiones del Embedding

| Dimensión | Uso Recomendado |
|-----------|-----------------|
| **3072** | Máxima calidad (recomendado para producción) |
| **1536** | Balance calidad/almacenamiento |
| **768** | Mínimo almacenamiento (rápido) |

### Modelos LLM Disponibles

| Modelo | Características | Uso Recomendado |
|--------|-----------------|-----------------|
| **gemini-2.5-flash** | Mejor balance costo/rendimiento | Uso general (recomendado) |
| **gemini-2.5-pro** | Más preciso | Respuestas complejas |
| **gemini-2.5-flash-lite** | Más rápido y económico | Prototipado rápido |

### Longitud del Contexto

| Rango | Características |
|-------|-----------------|
| **2000-4000** | Respuestas rápidas, menos contexto |
| **8000** | Balance recomendado |
| **16000** | Máximo contexto, más preciso pero más lento |

### Umbrales de Similitud

**Con Umbrales Dinámicos (Recomendado):**
- El sistema ajusta automáticamente según el tipo de archivo
- No necesitas configurar manualmente

**Con Umbral Fijo (No recomendado para multimodal):**
- Texto: 0.50-0.60
- Imagen: 0.45-0.50
- Video: 0.30-0.35

## 📁 Estructura del Proyecto

```
Gemini_Embeddings_2/
├── app.py                              # Aplicación Streamlit
├── requirements.txt                    # Dependencias
├── .env                                # API Key (no compartir)
├── README.md                           # Documentación
├── GUÍA_CHAT_RAG.md                    # Guía de uso
├── test_rag_multimodal_completo.py    # Script de prueba completo
├── Data/                               # Archivos de ejemplo
│   ├── DATABiQ.txt
│   ├── DATABiQ.pdf
│   ├── LOGO-DATABiQ.png
│   ├── DATABiQ_...mp4
│   └── DATABiQ_...mp4.transcripcion.txt  # Caché de video
└── chroma_db/                          # Base de datos vectorial
    └── multimodal_rag/
```

## 🧪 Pruebas y Validación

El proyecto incluye un script de prueba completo que valida todo el flujo:

```bash
python test_rag_multimodal_completo.py
```

**El script realiza:**
1. Carga de 3 archivos (video, texto, imagen)
2. 8 consultas de búsqueda semántica
3. 7 preguntas de Chat RAG
4. Validación de umbrales dinámicos
5. Verificación de análisis de video

## 🔒 Seguridad

⚠️ **Importante:** Nunca compartas el archivo `.env` ni lo subas a repositorios públicos.

## 📊 Rendimiento

**Tiempos promedio de procesamiento:**
- Texto (< 50KB): < 1 segundo
- Imagen (< 5MB): 2-5 segundos
- Video (< 150MB): 10-30 segundos (primera vez)
- Video (caché): < 1 segundo

## 🐛 Solución de Problemas

### Los videos no se recuperan en la búsqueda
**Causa:** Umbral de similitud demasiado alto (0.50 por defecto)
**Solución:** Activa "Umbral automático" o baja el umbral fijo a 0.30-0.35

### El Chat RAG no responde preguntas sobre videos
**Causa:** El video no tiene transcripción
**Solución:** El sistema analiza automáticamente el video la primera vez que se consulta

### Error "Invalid dimension" en ChromaDB
**Causa:** Mezcla de embeddings con diferentes dimensiones
**Solución:** Vacía la colección y recarga todos los archivos con la misma configuración

### Los archivos se procesan cada vez que los cargo (Caché Inteligente)
**Causa:** El archivo cambió o la metadata de hash no existe
**Solución:** 
- Si es la **primera carga**, es normal (se genera el hash)
- Si **ya lo cargaste antes**, verificá que el archivo no haya sido modificado
- Los archivos cargados antes de esta versión no tienen hash (se procesarán una vez más)

### No veo las citas en el Chat RAG (Citas Interactivas)
**Causa:** Las citas solo se muestran cuando el LLM usa documentos como contexto
**Solución:**
- Asegurate de que **"Usar LLM para respuesta"** esté activado
- Verificá que haya documentos recuperados (si no hay contexto, no hay citas)
- Las citas aparecen **después de la respuesta**, buscá "📚 Fuentes utilizadas"

## 📄 Licencia

Este proyecto es de uso educativo y demostrativo.

## 🙏 Créditos

- **Embeddings:** Google Gemini Embeddings 2 (`gemini-embedding-2-preview`)
- **LLM:** Google Gemini (`gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.5-flash-lite`)
- **Framework:** Streamlit
- **Vector DB:** ChromaDB
- **PDF Processing:** PyPDF2

## 📚 Recursos Adicionales

- [Documentación oficial de Gemini Embedding 2](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/embedding-2)
- [Documentación de ChromaDB](https://docs.trychroma.com/)
- [Documentación de Streamlit](https://docs.streamlit.io/)
