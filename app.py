# -*- coding: utf-8 -*-
"""
Aplicación Streamlit para RAG Multimodal con Gemini Embeddings 2
================================================================
Permite cargar archivos de diferentes formatos y realizar búsquedas semánticas
usando embeddings multimodales y Chat RAG.
"""

import os
from pathlib import Path
from datetime import datetime

import streamlit as st
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="RAG Multimodal - Gemini Embeddings 2",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONSTANTES Y CONFIGURACIÓN GLOBAL
# =============================================================================
EMBEDDING_MODEL = "gemini-embedding-2-preview"
DEFAULT_LLM_MODEL = "gemini-2.5-flash"
DEFAULT_COLLECTION = "multimodal_rag"
DEFAULT_CONTEXT_LENGTH = 8000
DEFAULT_DIMENSIONS = 3072

# Mapeos de tipos de archivos
MIME_TYPES = {
    '.pdf': 'application/pdf', '.png': 'image/png', '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg', '.gif': 'image/gif', '.webp': 'image/webp',
    '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.ogg': 'audio/ogg',
    '.mp4': 'video/mp4', '.avi': 'video/avi', '.mov': 'video/quicktime',
    '.webm': 'video/webm', '.txt': 'text/plain', '.md': 'text/markdown',
    '.json': 'application/json', '.csv': 'text/csv',
}

FILE_ICONS = {
    '.pdf': '📄', '.png': '🖼️', '.jpg': '🖼️', '.jpeg': '🖼️', '.gif': '🖼️', '.webp': '🖼️',
    '.mp3': '🎵', '.wav': '🎵', '.ogg': '🎵',
    '.mp4': '🎬', '.avi': '🎬', '.mov': '🎬', '.webm': '🎬',
    '.txt': '📝', '.md': '📝', '.json': '📋', '.csv': '📊',
}

# Umbrales dinámicos por tipo MIME
THRESHOLDS = {
    'text': 0.50,
    'pdf': 0.50,
    'image': 0.45,
    'audio': 0.35,
    'video': 0.30,
    'default': 0.35
}

# =============================================================================
# ESTILOS CSS
# =============================================================================
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E88E5; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .result-card { background-color: #f0f2f6; border-radius: 10px; padding: 15px; margin: 10px 0; border-left: 4px solid #1E88E5; }
    .distance-score { font-size: 0.9rem; color: #666; font-style: italic; }
    </style>
""", unsafe_allow_html=True)


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def get_mime_type(file_name: str) -> str:
    """Determina el tipo MIME basado en la extensión del archivo."""
    ext = Path(file_name).suffix.lower()
    return MIME_TYPES.get(ext, 'application/octet-stream')


def get_file_icon(file_name: str) -> str:
    """Retorna un ícono según el tipo de archivo."""
    ext = Path(file_name).suffix.lower()
    return FILE_ICONS.get(ext, '📁')


def get_dynamic_threshold(mime_type: str) -> float:
    """Retorna el umbral de similitud dinámico según el tipo de archivo."""
    if not mime_type:
        return THRESHOLDS['default']
    if any(t in mime_type for t in ['text', 'pdf']):
        return THRESHOLDS['text']
    if 'image' in mime_type:
        return THRESHOLDS['image']
    if 'audio' in mime_type:
        return THRESHOLDS['audio']
    if 'video' in mime_type:
        return THRESHOLDS['video']
    return THRESHOLDS['default']


def format_distance(distance: float) -> float:
    """Calcula la similitud desde la distancia de coseno."""
    return 1 - (distance / 2) if distance <= 2 else 0


def get_file_type_label(mime_type: str) -> str:
    """Retorna una etiqueta legible para el tipo de archivo."""
    if 'video' in mime_type:
        return "🎬 Video"
    if 'image' in mime_type:
        return "🖼️ Imagen"
    if 'audio' in mime_type:
        return "🎵 Audio"
    if 'pdf' in mime_type:
        return "📄 PDF"
    if 'text' in mime_type:
        return "📝 Texto"
    return "📁 Archivo"


def calculate_file_hash(file_data: bytes) -> str:
    """Calcula hash SHA256 de los datos del archivo para caché inteligente."""
    import hashlib
    return hashlib.sha256(file_data).hexdigest()


# =============================================================================
# FUNCIONES DE EMBEDDING
# =============================================================================

def embed_file(client, model: str, file_data: bytes, mime_type: str) -> list | None:
    """Genera el embedding para un archivo."""
    try:
        text_mime_types = ['text/plain', 'text/markdown', 'text/csv', 'application/json']
        
        if mime_type in text_mime_types:
            text_content = file_data.decode('utf-8')
            result = client.models.embed_content(model=model, contents=[text_content])
        else:
            result = client.models.embed_content(
                model=model,
                contents=[types.Part.from_bytes(data=file_data, mime_type=mime_type)]
            )
        return result.embeddings[0].values
    except Exception as e:
        st.error(f"Error al generar embedding: {str(e)}")
        return None


def embed_text(client, model: str, text: str, task_type: str = "RETRIEVAL_QUERY") -> list | None:
    """Genera el embedding para un texto de consulta."""
    try:
        result = client.models.embed_content(
            model=model,
            contents=[text],
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=DEFAULT_DIMENSIONS)
        )
        return result.embeddings[0].values
    except Exception as e:
        st.error(f"Error al generar embedding del texto: {str(e)}")
        return None


# =============================================================================
# FUNCIONES DE PROCESAMIENTO DE ARCHIVOS
# =============================================================================

def get_file_content(file_path: Path) -> str:
    """Extrae el contenido de un archivo para usar como contexto en el LLM."""
    if not file_path.exists():
        return ""

    mime_type = get_mime_type(file_path.name)

    try:
        # Archivos de texto
        if mime_type in ['text/plain', 'text/markdown', 'text/csv', 'application/json']:
            return file_path.read_text(encoding='utf-8')

        # PDFs
        if mime_type == 'application/pdf':
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(open(file_path, 'rb'))
                return "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
            except ImportError:
                return f"[PDF: {file_path.name}] - PyPDF2 no instalado"
            except Exception as e:
                return f"[PDF: {file_path.name}] - Error: {str(e)}"

        # Video con transcripción caché
        if 'video' in mime_type:
            transcripcion_path = file_path.with_suffix(file_path.suffix + '.transcripcion.txt')
            if transcripcion_path.exists():
                return transcripcion_path.read_text(encoding='utf-8')
            return f"[VIDEO: {file_path.name}]"

        # Imágenes y audio
        if 'image' in mime_type:
            return f"[IMAGEN: {file_path.name}]"
        if 'audio' in mime_type:
            return f"[AUDIO: {file_path.name}]"

        return ""
    except Exception as e:
        return f"[Error leyendo archivo: {str(e)}]"


def analyze_video_with_llm(client, model: str, file_path: Path, query: str = None) -> str:
    """Analiza un video con Gemini LLM multimodal para extraer información relevante."""
    transcripcion_path = file_path.with_suffix(file_path.suffix + '.transcripcion.txt')
    
    # Verificar caché
    if transcripcion_path.exists():
        return transcripcion_path.read_text(encoding='utf-8')

    try:
        prompt = """Analiza este video y extrae información detallada sobre:
1. Resumen del contenido
2. Puntos clave, servicios, productos o temas mencionados
3. Información específica que pueda ser útil para responder preguntas

Responde en español de manera clara y estructurada."""

        if query:
            prompt += f"\n\nPRESTA ESPECIAL ATENCIÓN A: {query}"

        video_data = file_path.read_bytes()
        response = client.models.generate_content(
            model=model,
            contents=[types.Part.from_bytes(data=video_data, mime_type='video/mp4'), prompt]
        )

        if response.text:
            transcripcion_path.write_text(response.text, encoding='utf-8')
            return response.text

        return f"[VIDEO: {file_path.name}]"
    except Exception as e:
        return f"[VIDEO: {file_path.name}] - Error: {str(e)}"


# =============================================================================
# FUNCIONES DE GENERACIÓN DE RESPUESTAS
# =============================================================================

def generate_rag_response(client, model: str, query: str, context_docs: list, max_context_length: int = 8000) -> dict:
    """Genera una respuesta usando el LLM con el contexto recuperado.
    
    Returns:
        dict: {"respuesta": str, "citas": list}
    """
    if not context_docs:
        return {"respuesta": "❌ No hay contexto suficiente para responder.", "citas": []}

    try:
        # Construir contexto eficientemente
        context_parts = []
        total_length = 0
        max_per_doc = max_context_length // len(context_docs)
        
        # Guardar fragmentos usados para las citas
        citas = []

        for doc in context_docs:
            content = doc.get('content', '')[:max_per_doc]
            if content:
                context_parts.append(f"=== DOCUMENTO: {doc.get('filename', 'Desconocido')} ===\n{content}\n")
                # Guardar cita con el fragmento usado
                citas.append({
                    "archivo": doc.get('filename', 'Desconocido'),
                    "fragmento": content[:500] + "..." if len(content) > 500 else content,
                    "mime_type": doc.get('mime_type', ''),
                    "similarity": doc.get('similarity', 0)
                })
                total_length += len(content)
                if total_length >= max_context_length:
                    break

        context_text = "\n".join(context_parts) or "No hay contexto disponible."

        system_prompt = f"""Eres un asistente experto en responder preguntas basadas en documentos proporcionados.

**Instrucciones:**
1. Responde ÚNICAMENTE basándote en la información de los documentos proporcionados
2. Si la respuesta no está en los documentos, di claramente "No encuentro esta información en los documentos proporcionados"
3. Cita los nombres de los archivos cuando uses información específica
4. Sé preciso y conciso
5. Si hay información contradictoria entre documentos, menciónalo

**Documentos de referencia:**
{context_text}

**Pregunta del usuario:**
{query}

**Respuesta:**
"""

        response = client.models.generate_content(model=model, contents=[system_prompt])
        respuesta_texto = response.text if response.text else "❌ No se pudo generar una respuesta."
        
        return {"respuesta": respuesta_texto, "citas": citas}
    except Exception as e:
        return {"respuesta": f"❌ Error al generar respuesta: {str(e)}", "citas": []}


# =============================================================================
# INICIALIZACIÓN DE CLIENTES
# =============================================================================

@st.cache_resource
def init_clients():
    """Inicializa los clientes de Gemini y ChromaDB."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ No se encontró la GEMINI_API_KEY. Por favor configúrala en el archivo .env")
        st.stop()

    return genai.Client(api_key=api_key), chromadb.PersistentClient(path="./chroma_db")


# =============================================================================
# COMPONENTES DE INTERFAZ
# =============================================================================

def render_file_preview(file_path: Path, mime_type: str, doc_id: str):
    """Muestra vista previa del archivo según su tipo."""
    if not file_path.exists():
        return

    if 'image' in mime_type:
        st.image(str(file_path), caption=doc_id, width=200)
    elif 'video' in mime_type:
        st.video(str(file_path))
    elif 'audio' in mime_type:
        st.audio(str(file_path))
    elif 'text' in mime_type or mime_type == 'application/json':
        content = file_path.read_text(encoding='utf-8')
        with st.expander(f"📝 Ver contenido de {doc_id}"):
            st.text(content[:1000] + "..." if len(content) > 1000 else content)


def render_search_result(doc_id: str, metadata: dict, distance: float, use_dynamic_threshold: bool, threshold: float = None):
    """Muestra un resultado de búsqueda."""
    similarity = format_distance(distance)
    mime_type = metadata.get('mime_type', '')
    file_threshold = get_dynamic_threshold(mime_type) if use_dynamic_threshold else threshold

    if similarity < file_threshold:
        return False

    # Determinar color
    if similarity >= 0.7:
        color = "🟢"
    elif similarity >= 0.5:
        color = "🟡"
    elif similarity >= 0.35:
        color = "🟠"
    else:
        color = "🔴"

    threshold_info = f"(umbral: {file_threshold:.2f})" if use_dynamic_threshold else ""

    st.markdown(f"""
    <div class="result-card">
        <h4>{color} {get_file_icon(doc_id)} {doc_id}</h4>
        <p class="distance-score">
            Similitud: <strong>{similarity:.4f}</strong> {threshold_info} |
            Distancia: {distance:.4f} |
            Tamaño: {metadata.get('size', 'N/A')} bytes |
            Tipo: {get_file_type_label(mime_type)}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Vista previa
    file_path = Path("./Data") / doc_id
    render_file_preview(file_path, mime_type, doc_id)
    st.divider()

    return True


def render_citations(citas: list):
    """Muestra citas expandibles con los fragmentos de documentos usados."""
    if not citas:
        return
    
    st.divider()
    st.markdown("**📚 Fuentes utilizadas:**")
    
    for i, cita in enumerate(citas, 1):
        archivo = cita.get('archivo', 'Desconocido')
        fragmento = cita.get('fragmento', '')
        mime_type = cita.get('mime_type', '')
        similarity = cita.get('similarity', 0)
        
        icon = get_file_icon(archivo)
        tipo = get_file_type_label(mime_type)
        
        with st.expander(f"{icon} Fuente {i}: {archivo} ({tipo}, Similitud: {similarity:.2f})"):
            st.markdown("**Fragmento usado como contexto:**")
            st.text(fragmento)
            st.caption(f"Este fragmento fue usado para generar la respuesta del asistente.")


# =============================================================================
# PESTAÑAS PRINCIPALES
# =============================================================================

def tab_cargar_archivos(chroma_client, collection_name: str, gemini_client):
    """Pestaña de carga de archivos."""
    st.header("Carga de Archivos Multimodales")
    st.info("""
    **Formatos soportados:**
    - 📄 **Documentos:** PDF, TXT, MD, JSON, CSV
    - 🖼️ **Imágenes:** PNG, JPG, JPEG, GIF, WEBP
    - 🎵 **Audio:** MP3, WAV, OGG
    - 🎬 **Video:** MP4, AVI, MOV, WEBM
    """)

    uploaded_files = st.file_uploader(
        "Selecciona los archivos a cargar",
        type=['pdf', 'txt', 'md', 'json', 'csv', 'png', 'jpg', 'jpeg', 'gif', 'webp',
              'mp3', 'wav', 'ogg', 'mp4', 'avi', 'mov', 'webm'],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"📦 **{len(uploaded_files)} archivo(s) seleccionado(s)**")
        progress_bar = st.progress(0)
        status_text = st.empty()
        collection = chroma_client.get_or_create_collection(name=collection_name)

        success_count, error_count = 0, 0

        for i, uploaded_file in enumerate(uploaded_files):
            try:
                progress_bar.progress((i + 1) / len(uploaded_files))
                status_text.text(f"Procesando: {uploaded_file.name}")

                file_data = uploaded_file.read()
                mime_type = get_mime_type(uploaded_file.name)
                file_hash = calculate_file_hash(file_data)

                # Verificar existencia y cambios (caché inteligente)
                existing = collection.get(ids=[uploaded_file.name])
                if existing['ids']:
                    existing_metadata = existing['metadatas'][0]
                    existing_hash = existing_metadata.get('file_hash', None)
                    
                    if existing_hash == file_hash:
                        st.info(f"⏭️ '{uploaded_file.name}' no ha cambiado. Saltando...")
                        error_count += 1
                        continue
                    else:
                        st.warning(f"🔄 '{uploaded_file.name}' ha cambiado. Regenerando embedding...")
                        # Eliminar versión anterior para actualizar
                        collection.delete(ids=[uploaded_file.name])

                # Generar embedding y guardar
                embedding = embed_file(gemini_client, EMBEDDING_MODEL, file_data, mime_type)
                if embedding:
                    collection.add(
                        ids=[uploaded_file.name],
                        embeddings=[embedding],
                        documents=[uploaded_file.name],
                        metadatas=[{
                            "source": uploaded_file.name,
                            "mime_type": mime_type,
                            "size": len(file_data),
                            "file_hash": file_hash,
                            "uploaded_at": datetime.now().isoformat()
                        }]
                    )
                    success_count += 1
                    st.success(f"✅ {get_file_icon(uploaded_file.name)} {uploaded_file.name}")
            except Exception as e:
                error_count += 1
                st.error(f"❌ Error con {uploaded_file.name}: {str(e)}")

        progress_bar.progress(1.0)
        status_text.text("Proceso completado")

        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Exitosos", success_count)
        col2.metric("Errores/Saltados", error_count)
        col3.metric("Total en colección", collection.count())


def tab_busqueda_semantica(chroma_client, collection_name: str, gemini_client):
    """Pestaña de búsqueda semántica."""
    st.header("Búsqueda Semántica Multimodal")
    st.info("""
    **Tipos de búsqueda disponibles:**
    - 🔤 **Texto:** Escribe una consulta en lenguaje natural
    - 🖼️ **Imagen:** Sube una imagen para encontrar archivos similares
    """)

    search_type = st.radio("Tipo de búsqueda:", ["🔤 Búsqueda por Texto", "🖼️ Búsqueda por Imagen"], horizontal=True)

    collection = chroma_client.get_or_create_collection(name=collection_name)
    total_docs = collection.count()

    if total_docs == 0:
        st.warning("⚠️ No hay archivos indexados. Por favor carga archivos en la pestaña 'Cargar Archivos'.")
        return

    # Configuración
    col1, col2 = st.columns([3, 1])
    with col1:
        n_results = st.slider("Número de resultados:", 1, min(10, total_docs), 5)
    with col2:
        use_dynamic = st.checkbox("Umbral automático", value=True, help="Usar umbrales dinámicos según tipo de archivo")
        if use_dynamic:
            st.info("💡 Umbral automático activado")
            threshold = None
        else:
            threshold = st.slider("Umbral fijo:", 0.0, 1.0, 0.35, 0.05)

    query_embedding, query_description = None, ""

    # Búsqueda por texto
    if "🔤 Búsqueda por Texto" in search_type:
        query_text = st.text_area("Escribe tu consulta:", placeholder="Ej: ¿Qué información hay sobre ciencia de datos?", height=100)
        if query_text:
            query_embedding = embed_text(gemini_client, EMBEDDING_MODEL, query_text)
            query_description = f"🔤 Consulta: **{query_text}**"

    # Búsqueda por imagen
    elif "🖼️ Búsqueda por Imagen" in search_type:
        query_image = st.file_uploader("Sube una imagen para buscar archivos similares:", type=['png', 'jpg', 'jpeg', 'gif', 'webp'])
        if query_image:
            st.image(query_image, caption="Imagen de consulta", use_container_width=True)
            if st.button("🔍 Buscar archivos similares"):
                query_embedding = embed_file(gemini_client, EMBEDDING_MODEL, query_image.read(), get_mime_type(query_image.name))
                query_description = f"🖼️ Imagen: **{query_image.name}**"

    # Ejecutar búsqueda
    if query_embedding:
        st.divider()
        st.subheader("Resultados")

        results = collection.query(query_embeddings=[query_embedding], n_results=n_results, include=["metadatas", "documents", "distances"])

        if results['ids'] and results['ids'][0]:
            st.markdown(f"**{query_description}**")
            st.write(f"Encontrados **{len(results['ids'][0])}** resultado(s)")
            st.divider()

            results_shown = 0
            for doc_id, metadata, distance in zip(results['ids'][0], results['metadatas'][0], results['distances'][0]):
                if render_search_result(doc_id, metadata, distance, use_dynamic, threshold):
                    results_shown += 1

            if results_shown == 0:
                st.warning("""
                ⚠️ **No se mostraron resultados** con el umbral actual.

                Esto es común en búsquedas **texto↔video** o **texto↔audio** porque las similitudes son más bajas (~0.30-0.45).

                **Recomendaciones:**
                - El umbral automático ya está ajustado para cada tipo de archivo
                - Si usas umbral fijo, prueba bajándolo a 0.30-0.35 para videos
                """)
        else:
            st.warning("No se encontraron resultados.")


def tab_chat_rag(chroma_client, collection_name: str, gemini_client, llm_model: str, max_context: int):
    """Pestaña de Chat RAG."""
    st.header("💬 Chat RAG - Preguntas sobre tus Documentos")
    st.info("""
    **Cómo usar:**
    1. Escribe tu pregunta en lenguaje natural
    2. El sistema buscará los documentos relevantes
    3. El LLM generará una respuesta basada en el contexto recuperado
    """)

    collection = chroma_client.get_or_create_collection(name=collection_name)
    if collection.count() == 0:
        st.warning("⚠️ No hay archivos indexados.")
        return

    # Configuración
    col1, col2 = st.columns([3, 1])
    with col1:
        n_results = st.slider("Documentos a recuperar:", 1, min(10, collection.count()), 5, key="chat_n_results")
    with col2:
        use_llm = st.checkbox("Usar LLM para respuesta", value=True)

    # Historial
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Inicializar pregunta en session state
    if "query_chat" not in st.session_state:
        st.session_state.query_chat = ""

    # Input del usuario
    query_chat = st.text_area(
        "Haz tu pregunta:",
        value=st.session_state.query_chat,
        placeholder="Ej: ¿Cuál es el email de contacto de DATABiQ?",
        height=80,
        key="chat_input"
    )

    # Botones
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    with col_btn1:
        btn_preguntar = st.button("💬 Preguntar", type="primary", use_container_width=True, key="btn_preguntar")
    with col_btn2:
        btn_limpiar = st.button("🗑️ Limpiar chat", use_container_width=True, key="btn_limpiar")
    with col_btn3:
        btn_reset = st.button("🧹 Resetear", use_container_width=True, key="btn_reset")

    # Limpiar chat
    if btn_limpiar:
        st.session_state.chat_history = []
        st.session_state.query_chat = ""
        st.rerun()
    
    # Resetear todo
    if btn_reset:
        st.session_state.chat_history = []
        st.session_state.query_chat = ""
        st.rerun()

    # Procesar pregunta (solo si se presionó el botón)
    if btn_preguntar and query_chat:
        # Agregar pregunta al historial
        st.session_state.chat_history.append({"role": "user", "content": query_chat})

        with st.spinner("🔍 Buscando documentos relevantes..."):
            query_embedding = embed_text(gemini_client, EMBEDDING_MODEL, query_chat)
            
            if query_embedding:
                results = collection.query(query_embeddings=[query_embedding], n_results=n_results, include=["metadatas", "distances"])

                context_docs = []
                data_folder = Path("./Data")

                if results['ids'] and results['ids'][0]:
                    for doc_id, metadata, distance in zip(results['ids'][0], results['metadatas'][0], results['distances'][0]):
                        similarity = format_distance(distance)
                        mime_type = metadata.get('mime_type', '')

                        if similarity < get_dynamic_threshold(mime_type):
                            continue

                        content = get_file_content(data_folder / doc_id)

                        # Analizar video si es necesario
                        if 'video' in mime_type and content.startswith("[VIDEO:"):
                            with st.spinner(f"🎬 Analizando video: {doc_id}..."):
                                content = analyze_video_with_llm(gemini_client, llm_model, data_folder / doc_id, query_chat)

                        if content:
                            context_docs.append({"filename": doc_id, "content": content, "similarity": similarity, "mime_type": mime_type})

                # Generar respuesta
                if use_llm and context_docs:
                    with st.spinner("🤖 Generando respuesta..."):
                        resultado = generate_rag_response(gemini_client, llm_model, query_chat, context_docs, max_context)
                        respuesta = resultado.get("respuesta", "")
                        citas = resultado.get("citas", [])
                else:
                    respuesta = "📋 **Documentos encontrados:**\n\n" + "\n".join(f"- **{doc['filename']}** (Similitud: {doc['similarity']:.2f})" for doc in context_docs)
                    respuesta += "\n*Activa 'Usar LLM para respuesta' para obtener una respuesta completa.*"
                    citas = []

                # Guardar respuesta con citas en el historial
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": respuesta,
                    "citas": citas
                })
                
                # Limpiar el input después de procesar
                st.session_state.query_chat = ""
                st.rerun()
            else:
                st.error("Error al procesar tu pregunta. Intenta de nuevo.")
                st.session_state.chat_history.pop()  # Remover pregunta fallida

    # Mostrar historial de chat
    for msg in st.session_state.chat_history:
        bg_color = "#e3f2fd" if msg["role"] == "user" else "#f1f8e9"
        icon = "🧑‍💼 Tú" if msg["role"] == "user" else "🤖 Asistente"
        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 10px; border-radius: 10px; margin: 10px 0;">
            <strong>{icon}:</strong> {msg["content"]}
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar citas si es mensaje del asistente y tiene citas
        if msg["role"] == "assistant" and msg.get("citas"):
            render_citations(msg["citas"])


def tab_archivos_indexados(chroma_client, collection_name: str):
    """Pestaña de archivos indexados."""
    st.header("Archivos Indexados")
    collection = chroma_client.get_or_create_collection(name=collection_name)

    if collection.count() == 0:
        st.info("No hay archivos indexados.")
        return

    all_docs = collection.get(include=["metadatas"])
    st.write(f"**Total de archivos:** {len(all_docs['ids'])}")

    for doc_id, metadata in zip(all_docs['ids'], all_docs['metadatas']):
        with st.expander(f"{get_file_icon(doc_id)} {doc_id}"):
            st.write(f"- **Tipo MIME:** {metadata.get('mime_type', 'N/A')}")
            st.write(f"- **Tamaño:** {metadata.get('size', 'N/A')} bytes")
            st.write(f"- **Fecha de carga:** {metadata.get('uploaded_at', 'N/A')}")

            if st.button("🗑️ Eliminar", key=f"delete_{doc_id}"):
                collection.delete(ids=[doc_id])
                st.success(f"Eliminado: {doc_id}")
                st.rerun()


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    # Encabezado
    st.markdown('<p class="main-header">🧠 RAG Multimodal con Gemini Embeddings 2</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Carga archivos de diferentes formatos y realiza búsquedas semánticas inteligentes</p>', unsafe_allow_html=True)

    # Inicializar clientes
    gemini_client, chroma_client = init_clients()

    # Barra lateral
    with st.sidebar:
        st.header("⚙️ Configuración")
        collection_name = st.text_input("Nombre de la colección:", value=DEFAULT_COLLECTION)

        st.divider()
        st.header("🤖 Modelo LLM")
        llm_model = st.selectbox("Modelo para generar respuestas:", ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"], index=0)

        max_context = st.slider("Longitud máxima del contexto (caracteres):", 2000, 16000, DEFAULT_CONTEXT_LENGTH, 1000)

        st.divider()
        st.header("📊 Estadísticas")
        try:
            collection = chroma_client.get_or_create_collection(name=collection_name)
            st.metric("Archivos indexados", collection.count())
        except Exception as e:
            st.error(f"Error: {e}")

        st.divider()
        st.header("🗑️ Gestión")
        if st.button("🗑️ Vaciar colección", use_container_width=True):
            try:
                chroma_client.delete_collection(name=collection_name)
                st.success("Colección vaciada exitosamente")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
        if st.button("🔄 Recargar página", use_container_width=True):
            st.rerun()

    # Pestañas principales
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Cargar Archivos", "🔍 Búsqueda Semántica", "💬 Chat RAG", "📚 Archivos Indexados"])

    with tab1:
        tab_cargar_archivos(chroma_client, collection_name, gemini_client)
    with tab2:
        tab_busqueda_semantica(chroma_client, collection_name, gemini_client)
    with tab3:
        tab_chat_rag(chroma_client, collection_name, gemini_client, llm_model, max_context)
    with tab4:
        tab_archivos_indexados(chroma_client, collection_name)

    # Pie de página
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>Desarrollado con Streamlit + Gemini Embeddings 2 + ChromaDB + Gemini LLM</p>
        <p>Embedding: <code>gemini-embedding-2-preview</code> | LLM: <code>gemini-2.5-flash</code></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
