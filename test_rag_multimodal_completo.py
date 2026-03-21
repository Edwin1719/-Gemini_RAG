# -*- coding: utf-8 -*-
"""
Prueba Completa: RAG Multimodal con 3 archivos (Video, Texto, Imagen)
======================================================================
Este script prueba el RAG multimodal completo cargando:
1. Video: DATABiQ_ Tu Aliado en Ciencia d 2024-10-24.mp4
2. Texto: DATABiQ.txt
3. Imagen: LOGO-DATABiQ.png

Y realiza múltiples preguntas tanto en Búsqueda Semántica como en Chat RAG.
"""

import os
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv
import chromadb
import numpy as np

load_dotenv()

# Configuración
API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "gemini-embedding-2-preview"
LLM_MODEL = "gemini-2.5-flash"
COLLECTION_NAME = "multimodal_rag_test"  # Colección separada para pruebas

# Archivos a cargar
DATA_FOLDER = Path("./Data")
FILES_TO_LOAD = [
    "DATABiQ_ Tu Aliado en Ciencia d 2024-10-24.mp4",
    "DATABiQ.txt",
    "LOGO-DATABiQ.png"
]

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def get_mime_type(file_name: str) -> str:
    ext = Path(file_name).suffix.lower()
    mime_types = {
        '.pdf': 'application/pdf',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.mp4': 'video/mp4',
        '.txt': 'text/plain',
    }
    return mime_types.get(ext, 'application/octet-stream')


def get_file_icon(file_name: str) -> str:
    ext = Path(file_name).suffix.lower()
    icons = {
        '.pdf': '📄', '.png': '🖼️', '.jpg': '🖼️',
        '.mp4': '🎬', '.txt': '📝',
    }
    return icons.get(ext, '📁')


def format_distance(distance: float) -> str:
    similarity = 1 - (distance / 2) if distance <= 2 else 0
    return f"{similarity:.4f}"


def get_dynamic_threshold(mime_type: str) -> float:
    if not mime_type:
        return 0.35
    if any(x in mime_type for x in ['text/plain', 'application/pdf']):
        return 0.50
    if 'image' in mime_type:
        return 0.45
    if 'audio' in mime_type:
        return 0.35
    if 'video' in mime_type:
        return 0.30
    return 0.35


def embed_file(client, model: str, file_data: bytes, mime_type: str) -> list:
    """Genera el embedding para un archivo."""
    text_mime_types = ['text/plain', 'text/markdown', 'text/csv', 'application/json']

    if mime_type in text_mime_types:
        text_content = file_data.decode('utf-8')
        result = client.models.embed_content(
            model=model,
            contents=[text_content],
        )
    else:
        result = client.models.embed_content(
            model=model,
            contents=[types.Part.from_bytes(data=file_data, mime_type=mime_type)],
        )
    return result.embeddings[0].values


def get_file_content(file_path: Path) -> str:
    """Extrae el contenido de un archivo para usar como contexto en el LLM."""
    try:
        if not file_path.exists():
            return ""

        mime_type = get_mime_type(file_path.name)

        if mime_type in ['text/plain', 'text/markdown', 'text/csv', 'application/json']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        elif mime_type == 'application/pdf':
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(open(file_path, 'rb'))
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
            except Exception as e:
                return f"[PDF: {file_path.name}] - Error: {str(e)}"

        elif 'image' in mime_type:
            return f"[IMAGEN: {file_path.name}]"

        elif 'audio' in mime_type:
            return f"[AUDIO: {file_path.name}]"

        elif 'video' in mime_type:
            transcripcion_path = file_path.with_suffix(file_path.suffix + '.transcripcion.txt')
            if transcripcion_path.exists():
                return transcripcion_path.read_text(encoding='utf-8')
            return f"[VIDEO: {file_path.name}]"

        return ""
    except Exception as e:
        return f"[Error leyendo archivo: {str(e)}]"


def analyze_video_with_llm(client, model: str, file_path: Path, query: str = None) -> str:
    """Analiza un video con Gemini LLM multimodal."""
    try:
        transcripcion_path = file_path.with_suffix(file_path.suffix + '.transcripcion.txt')
        if transcripcion_path.exists():
            return transcripcion_path.read_text(encoding='utf-8')
        
        prompt = f"""Analiza este video y extrae información detallada sobre:
1. Resumen del contenido
2. Puntos clave, servicios, productos o temas mencionados
3. Información específica que pueda ser útil para responder preguntas

Responde en español de manera clara y estructurada."""
        
        if query:
            prompt += f"\n\nPRESTA ESPECIAL ATENCIÓN A: {query}"
        
        with open(file_path, 'rb') as f:
            video_data = f.read()
        
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=video_data, mime_type='video/mp4'),
                prompt
            ]
        )
        
        if response.text:
            transcripcion_path.write_text(response.text, encoding='utf-8')
            return response.text
        
        return f"[VIDEO: {file_path.name}]"
    except Exception as e:
        return f"[VIDEO: {file_path.name}] - Error: {str(e)}"


def generate_rag_response(client, model: str, query: str, context_docs: list, max_context_length: int = 8000) -> str:
    """Genera una respuesta usando el LLM con el contexto recuperado."""
    try:
        context_parts = []
        total_length = 0

        for doc in context_docs:
            filename = doc.get('filename', 'Desconocido')
            content = doc.get('content', '')

            if len(content) > 0:
                max_per_doc = max_context_length // len(context_docs)
                truncated_content = content[:max_per_doc] if len(content) > max_per_doc else content
                context_parts.append(f"=== DOCUMENTO: {filename} ===\n{truncated_content}\n")
                total_length += len(truncated_content)
                if total_length >= max_context_length:
                    break

        context_text = "\n".join(context_parts) if context_parts else "No hay contexto disponible."

        system_prompt = f"""Eres un asistente experto en responder preguntas basadas en documentos proporcionados.

**Instrucciones:**
1. Responde ÚNICAMENTE basándote en la información de los documentos proporcionados
2. Si la respuesta no está en los documentos, di claramente "No encuentro esta información en los documentos proporcionados"
3. Cita los nombres de los archivos cuando uses información específica
4. Sé preciso y conciso

**Documentos de referencia:**
{context_text}

**Pregunta del usuario:**
{query}

**Respuesta:**
"""

        response = client.models.generate_content(
            model=model,
            contents=[system_prompt]
        )

        return response.text if response.text else "❌ No se pudo generar una respuesta."
    except Exception as e:
        return f"❌ Error al generar respuesta: {str(e)}"


# =============================================================================
# CARGA DE ARCHIVOS
# =============================================================================

def load_files():
    """Carga los 3 archivos en la colección."""
    print("=" * 80)
    print("FASE 1: CARGA DE ARCHIVOS MULTIMODALES")
    print("=" * 80)

    client = genai.Client(api_key=API_KEY)
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # Limpiar colección si existe
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except:
        pass
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    print(f"\n📁 Archivos a cargar:")
    for file_name in FILES_TO_LOAD:
        file_path = DATA_FOLDER / file_name
        exists = "✅" if file_path.exists() else "❌"
        print(f"   {exists} {file_name}")

    print("\n⏳ Procesando archivos...")
    print("-" * 80)

    success_count = 0
    for file_name in FILES_TO_LOAD:
        file_path = DATA_FOLDER / file_name
        if not file_path.exists():
            print(f"❌ {file_name} - No encontrado")
            continue

        try:
            mime_type = get_mime_type(file_name)
            
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            file_size_mb = len(file_data) / (1024 * 1024)
            print(f"\n📄 {get_file_icon(file_name)} {file_name}")
            print(f"   Tipo MIME: {mime_type}")
            print(f"   Tamaño: {file_size_mb:.2f} MB")

            # Generar embedding
            print(f"   ⏳ Generando embedding...", end=" ")
            embedding = embed_file(client, EMBEDDING_MODEL, file_data, mime_type)
            print(f"✅ ({len(embedding)} dimensiones)")

            # Guardar en ChromaDB
            collection.add(
                ids=[file_name],
                embeddings=[embedding],
                documents=[file_name],
                metadatas=[{
                    "source": file_name,
                    "mime_type": mime_type,
                    "size": len(file_data)
                }]
            )
            print(f"   ✅ Indexado en ChromaDB")
            success_count += 1

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

    print("-" * 80)
    print(f"\n✅ {success_count}/{len(FILES_TO_LOAD)} archivos cargados exitosamente")
    print(f"📊 Total en colección: {collection.count()}")

    return collection


# =============================================================================
# PRUEBAS DE BÚSQUEDA SEMÁNTICA
# =============================================================================

def test_semantic_search(collection):
    """Prueba la búsqueda semántica con múltiples preguntas."""
    print("\n" + "=" * 80)
    print("FASE 2: BÚSQUEDA SEMÁNTICA MULTIMODAL")
    print("=" * 80)

    embed_client = genai.Client(api_key=API_KEY)

    # Preguntas diseñadas para recuperar diferentes tipos de archivos
    test_queries = [
        # Para recuperar TEXTO
        ("¿Cuál es el email de contacto de DATABiQ?", "texto"),
        ("¿Quién es el fundador de DATABiQ?", "texto"),
        
        # Para recuperar VIDEO
        ("¿Qué servicios futuros menciona el video?", "video"),
        ("¿De qué trata el video de DATABiQ?", "video"),
        
        # Para recuperar IMAGEN
        ("¿Cómo es el logo de la empresa?", "imagen"),
        ("Muéstrame la imagen corporativa", "imagen"),
        
        # Preguntas mixtas
        ("¿Cuáles son los servicios de Machine Learning?", "texto"),
        ("¿Qué es DATABiQ?", "mixto"),
    ]

    for query_text, expected_type in test_queries:
        print(f"\n{'─' * 80}")
        print(f"🔍 Consulta: '{query_text}'")
        print(f"   Tipo esperado: {expected_type}")
        print(f"{'─' * 80}")

        # Generar embedding
        query_result = embed_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[query_text],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        query_embedding = query_result.embeddings[0].values

        # Buscar
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["metadatas", "distances"]
        )

        if results['ids'] and results['ids'][0]:
            for doc_id, distance, metadata in zip(
                results['ids'][0],
                results['distances'][0],
                results['metadatas'][0]
            ):
                similarity = float(format_distance(distance))
                mime_type = metadata.get('mime_type', '')
                file_threshold = get_dynamic_threshold(mime_type)
                passes = similarity >= file_threshold
                
                status = "✅" if passes else "❌"
                print(f"   {status} {get_file_icon(doc_id)} {doc_id}")
                print(f"      Similitud: {similarity:.4f} (umbral: {file_threshold:.2f})")
        else:
            print("   ❌ No se encontraron resultados")


# =============================================================================
# PRUEBAS DE CHAT RAG
# =============================================================================

def test_chat_rag(collection):
    """Prueba el Chat RAG con múltiples preguntas."""
    print("\n" + "=" * 80)
    print("FASE 3: CHAT RAG MULTIMODAL")
    print("=" * 80)

    embed_client = genai.Client(api_key=API_KEY)
    llm_client = genai.Client(api_key=API_KEY)

    # Preguntas para Chat RAG
    rag_queries = [
        # Preguntas que debe responder el TEXTO
        {
            "question": "¿Cuál es el email corporativo de DATABiQ?",
            "expected_source": "DATABiQ.txt",
            "category": "📝 Texto"
        },
        {
            "question": "¿Qué formación académica tiene el fundador de DATABiQ?",
            "expected_source": "DATABiQ.txt",
            "category": "📝 Texto"
        },
        {
            "question": "¿Cuáles son los 6 servicios principales que ofrece DATABiQ?",
            "expected_source": "DATABiQ.txt",
            "category": "📝 Texto"
        },
        
        # Preguntas que debe responder el VIDEO
        {
            "question": "¿Qué información hay en el video sobre DATABiQ?",
            "expected_source": "Video",
            "category": "🎬 Video"
        },
        {
            "question": "¿Cuáles son los servicios que brindara DATABiQ de cara al futuro?",
            "expected_source": "Video",
            "category": "🎬 Video"
        },
        
        # Preguntas mixtas
        {
            "question": "¿Dónde está ubicada la sede principal de DATABiQ?",
            "expected_source": "DATABiQ.txt",
            "category": "📝 Texto"
        },
        {
            "question": "¿Qué tecnologías usa DATABiQ para Machine Learning?",
            "expected_source": "DATABiQ.txt",
            "category": "📝 Texto"
        },
    ]

    for i, query in enumerate(rag_queries, 1):
        question = query["question"]
        expected = query["expected_source"]
        category = query["category"]
        
        print(f"\n{'=' * 80}")
        print(f"PRUEBA RAG #{i} {category}")
        print(f"{'=' * 80}")
        print(f"\n🤔 Pregunta: {question}")
        print(f"📁 Fuente esperada: {expected}")
        print(f"{'-' * 80}")

        # Generar embedding
        query_result = embed_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[question],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        query_embedding = query_result.embeddings[0].values

        # Buscar documentos
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["metadatas", "distances"]
        )

        # Preparar contexto
        context_docs = []
        data_folder = DATA_FOLDER

        if results['ids'] and results['ids'][0]:
            print(f"\n📚 Documentos recuperados:")
            
            for doc_id, metadata, distance in zip(
                results['ids'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                similarity = float(format_distance(distance))
                mime_type = metadata.get('mime_type', '')
                file_threshold = get_dynamic_threshold(mime_type)

                if similarity < file_threshold:
                    continue

                print(f"   ✅ {get_file_icon(doc_id)} {doc_id} (similitud: {similarity:.4f})")

                file_path = data_folder / doc_id
                content = get_file_content(file_path)

                # Si es video y no hay transcripción, analizar con LLM
                if 'video' in mime_type and content.startswith("[VIDEO:"):
                    print(f"      ⏳ Analizando video con LLM...")
                    content = analyze_video_with_llm(llm_client, LLM_MODEL, file_path, question)
                    print(f"      ✅ Video analizado: {len(content)} caracteres")

                if content:
                    context_docs.append({
                        "filename": doc_id,
                        "content": content,
                        "similarity": similarity,
                        "mime_type": mime_type
                    })

        # Generar respuesta
        if context_docs:
            print(f"\n🤖 Generando respuesta con LLM...")
            respuesta = generate_rag_response(
                llm_client,
                LLM_MODEL,
                question,
                context_docs,
                max_context_length=8000
            )
            
            print(f"\n📝 RESPUESTA:")
            print(f"{'-' * 80}")
            print(respuesta)
            print(f"{'-' * 80}")
        else:
            print(f"\n❌ No se encontró contexto suficiente para responder")

        # Pausa entre preguntas
        if i < len(rag_queries):
            print(f"\n⏸️  Continuando con siguiente pregunta...")


# =============================================================================
# PRINCIPAL
# =============================================================================

def main():
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  PRUEBA COMPLETA: RAG MULTIMODAL (Video + Texto + Imagen)".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")

    # Fase 1: Cargar archivos
    collection = load_files()

    # Fase 2: Búsqueda Semántica
    test_semantic_search(collection)

    # Fase 3: Chat RAG
    test_chat_rag(collection)

    # Resumen final
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    print(f"\n✅ Pruebas completadas exitosamente")
    print(f"📊 Colección: {COLLECTION_NAME}")
    print(f"📁 Archivos indexados: {collection.count()}")
    print(f"\n📝 Archivos de transcripción creados:")
    for file_name in FILES_TO_LOAD:
        if 'video' in file_name or 'mp4' in file_name:
            transcripcion_path = (DATA_FOLDER / file_name).with_suffix('.mp4.transcripcion.txt')
            if transcripcion_path.exists():
                print(f"   ✅ {transcripcion_path.name} ({transcripcion_path.stat().st_size} bytes)")
    
    print("\n" + "=" * 80)
    print("¡PRUEBA FINALIZADA!")
    print("=" * 80)


if __name__ == "__main__":
    main()
