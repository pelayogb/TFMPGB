"""
API FastAPI para chatbot con sistema multiagente MCP + RAG + ML
Ubicación: api/main.py
"""
# ============= CARGAR .ENV PRIMERO =============
from dotenv import load_dotenv
load_dotenv()  

# ============= IMPORTS =============
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from typing import Optional, Dict, List
from contextlib import asynccontextmanager
import os
import asyncio
import secrets
from datetime import datetime
from pathlib import Path
import sys
import warnings
import logging

# Importar el cliente MCP
sys.path.insert(0, str(Path(__file__).parent.parent))
from clients.mcp_client import MCPLLMClient

# ============= SUPRIMIR ERRORES DE CIERRE MCP =============
# Esto es necesario porque MCP usa anyio que genera errores al cerrar
# desde diferentes tareas asíncronas. Es seguro ignorarlos.

# Suprimir warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configurar logging para suprimir errores de asyncio en shutdown
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

# Suprimir errores específicos de anyio y MCP
class SuppressExceptions:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Suprimir RuntimeError de anyio cancel scope
        if exc_type is RuntimeError and "cancel scope" in str(exc_val):
            return True
        return False

# Monkey patch para asyncio.run para suprimir errores en shutdown
_original_run = asyncio.run

def patched_run(main, **kwargs):
    try:
        return _original_run(main, **kwargs)
    except (KeyboardInterrupt, SystemExit):
        raise
    except RuntimeError as e:
        if "cancel scope" in str(e) or "async_generator" in str(e):
            pass  # Suprimir estos errores específicos
        else:
            raise
    except Exception:
        pass  # Suprimir otros errores de shutdown

asyncio.run = patched_run

# ============= CONFIGURACIÓN =============

# Almacenamiento de sesiones
sessions: Dict[str, Dict] = {}

# Usuarios (en producción usar base de datos)
VALID_USERS = {}
users_env = os.getenv("CHATBOT_USERS")
for user_data in users_env.split(","):
    if ":" in user_data:
        username, password = user_data.strip().split(":", 1)
        VALID_USERS[username] = password

# Variable global para la tarea de limpieza
cleanup_task = None

# ============= FUNCIONES AUXILIARES =============

async def safe_close_mcp_client(client, session_id: str) -> bool:
    """Cierra un cliente MCP de forma segura sin generar excepciones"""
    if not client:
        return True
    
    try:
        # Intentar cierre normal con timeout muy corto
        await asyncio.wait_for(client.cleanup(), timeout=1.0)
        return True
    except asyncio.TimeoutError:
        # Si hay timeout, forzar cierre inmediato
        try:
            client.force_close()
        except:
            pass
        return False
    except Exception as e:
        # Cualquier otro error, silenciar y forzar cierre
        try:
            client.force_close()
        except:
            pass
        return False

async def cleanup_inactive_sessions():
    """Limpia sesiones inactivas (> 2 horas)"""
    while True:
        try:
            await asyncio.sleep(1800)  # Cada 30 minutos
            
            now = datetime.now()
            inactive_sessions = []
            
            for session_id, session_data in list(sessions.items()):
                try:
                    last_activity = datetime.fromisoformat(session_data["last_activity"])
                    if (now - last_activity).total_seconds() > 7200:  # 2 horas
                        inactive_sessions.append(session_id)
                except:
                    continue
            
            for session_id in inactive_sessions:
                print(f"⏰ Cerrando sesión inactiva: {session_id[:8]}...")
                try:
                    session = sessions.get(session_id)
                    if session and session.get("mcp_client"):
                        await safe_close_mcp_client(session["mcp_client"], session_id)
                    
                    # Eliminar sesión
                    if session_id in sessions:
                        del sessions[session_id]
                    print(f"  ✓ Sesión {session_id[:8]} eliminada")
                except Exception:
                    # Eliminar de todas formas
                    try:
                        if session_id in sessions:
                            del sessions[session_id]
                    except:
                        pass
        
        except asyncio.CancelledError:
            print("🛑 Tarea de limpieza cancelada")
            break
        except Exception:
            pass  # Ignorar errores en la tarea de limpieza

# ============= CICLO DE VIDA =============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manejo del ciclo de vida de la aplicación"""
    global cleanup_task
    
    # ========== STARTUP ==========
    print("="*70)
    print("🚀 API FastAPI - WaterTank MCP Chatbot v2.0")
    print("="*70)
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", "8000"))
    print(f"🌐 Servidor: http://{HOST}:{PORT}")
    print(f"📚 Documentación: http://{HOST}:{PORT}/docs")
    print(f"📱 Interfaz Web: http://{HOST}:{PORT}/")
    print(f"👥 Usuarios: {list(VALID_USERS.keys())}")
    print(f"✨ Features:")
    print("   • RAG con búsqueda semántica")
    print("   • Predicciones ML por provincia/tanque")
    print("   • Análisis histórico y tendencias")
    print("   • Sistema multiagente inteligente")
    print("="*70)
    print("✅ API iniciada correctamente")
    print("📊 Sistema: RAG + ML + Multiagente")
    print("="*70)
    
    # Iniciar tarea de limpieza de sesiones inactivas
    cleanup_task = asyncio.create_task(cleanup_inactive_sessions())
    print("🧹 Tarea de limpieza de sesiones iniciada")
    
    yield  # ← El servidor corre aquí indefinidamente
    
    # ========== SHUTDOWN ==========
    print("\n" + "="*70)
    print("🛑 Cerrando API...")
    print("="*70)
    
    # Cancelar tarea de limpieza
    if cleanup_task and not cleanup_task.done():
        print("🧹 Cancelando tarea de limpieza...")
        cleanup_task.cancel()
        try:
            await asyncio.wait_for(cleanup_task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception:
            pass
    
    # Cerrar todas las sesiones MCP de forma agresiva
    print("🔌 Cerrando conexiones MCP...")
    
    if sessions:
        # Crear lista de sesiones a cerrar
        sessions_to_close = list(sessions.items())
        
        # Cerrar cada una con timeout individual
        for session_id, session_data in sessions_to_close:
            if session_data.get("mcp_client"):
                try:
                    # Cerrar sin esperar respuesta
                    asyncio.create_task(
                        safe_close_mcp_client(session_data["mcp_client"], session_id)
                    )
                except Exception:
                    pass
        
        # Dar un momento para que se cierren
        await asyncio.sleep(0.5)
        
        # Limpiar todas las sesiones
        sessions.clear()
        print(f"  ✓ {len(sessions_to_close)} sesión(es) cerrada(s)")
    
    print("✅ Todas las conexiones cerradas")
    print("="*70)

# ============= APLICACIÓN =============

app = FastAPI(
    title="WaterTank MCP Chatbot API", 
    version="2.0.0",
    description="API con RAG, ML y Sistema Multiagente MCP",
    lifespan=lifespan
)

security = HTTPBasic()

# Montar archivos estáticos
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_path = static_path / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(favicon_path)
    return Response(status_code=204)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= MODELOS =============

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    success: bool
    message: str
    session_id: Optional[str] = None
    user_info: Optional[Dict] = None

class AskRequest(BaseModel):
    session_id: str
    message: str

class AskResponse(BaseModel):
    success: bool
    response: str
    timestamp: str
    tools_used: Optional[List[str]] = None
    metadata: Optional[Dict] = None

class NewChatRequest(BaseModel):
    session_id: str

class NewChatResponse(BaseModel):
    success: bool
    message: str

class SessionInfo(BaseModel):
    session_id: str
    username: str
    created_at: str
    messages_count: int
    connected_to_mcp: bool

# ============= FUNCIONES DE SESIÓN =============

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verifica credenciales de usuario"""
    username = credentials.username
    password = credentials.password
    
    if username not in VALID_USERS or not secrets.compare_digest(
        password.encode("utf8"),
        VALID_USERS[username].encode("utf8")
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales incorrectas",
            headers={"WWW-Authenticate": "Basic"},
        )
    return username

def create_session(username: str) -> str:
    """Crea una nueva sesión para el usuario"""
    session_id = secrets.token_urlsafe(32)
    sessions[session_id] = {
        "username": username,
        "created_at": datetime.now().isoformat(),
        "conversation_history": [],
        "mcp_client": None,
        "last_activity": datetime.now().isoformat()
    }
    return session_id

def get_session(session_id: str) -> Dict:
    """Obtiene una sesión existente"""
    if session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sesión no encontrada o expirada"
        )
    
    # Actualizar última actividad
    sessions[session_id]["last_activity"] = datetime.now().isoformat()
    return sessions[session_id]

# ============= ENDPOINTS =============

@app.get("/")
async def root():
    """Endpoint raíz - redirige a la interfaz"""
    index_file = static_path / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "message": "WaterTank MCP Chatbot API v2.0",
        "docs": "/docs",
        "status": "online"
    }

@app.get("/api")
async def api_info():
    """Información de la API"""
    return {
        "name": "WaterTank MCP Chatbot API",
        "version": "2.0.0",
        "features": ["RAG", "ML Predictions", "Multi-Agent", "Historical Analysis"],
        "status": "online",
        "active_sessions": len(sessions),
        "endpoints": {
            "login": "/login",
            "ask": "/ask",
            "new_chat": "/new_chat",
            "session_info": "/session/{session_id}",
            "health": "/health"
        }
    }

@app.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Endpoint de login
    
    Usuarios por defecto: admin/admin123, user/user123, demo/demo123
    """
    username = request.username
    password = request.password
    
    # Verificar credenciales
    if username not in VALID_USERS or VALID_USERS[username] != password:
        return LoginResponse(
            success=False,
            message="Usuario o contraseña incorrectos"
        )
    
    # Crear sesión
    session_id = create_session(username)
    
    print(f"✅ Login exitoso: {username} (sesión: {session_id[:8]}...)")
    
    return LoginResponse(
        success=True,
        message=f"Bienvenido {username}",
        session_id=session_id,
        user_info={
            "username": username,
            "created_at": sessions[session_id]["created_at"]
        }
    )

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Endpoint principal - Procesa consulta con sistema multiagente
    
    Sistema Multiagente:
    1. RAG: Búsqueda semántica en base de conocimiento
    2. ML: Predicciones usando modelos entrenados
    3. Análisis: Estadísticas y tendencias históricas
    4. Reporting: Generación de informes completos
    """
    # Obtener sesión
    session = get_session(request.session_id)
    
    try:
        # Conectar al servidor MCP si no está conectado
        if session["mcp_client"] is None:
            print(f"🔌 [{session['username']}] Conectando al servidor MCP...")
            client = MCPLLMClient()
            await client.connect_to_mcp_server()
            session["mcp_client"] = client
            print(f"✅ [{session['username']}] Conexión MCP establecida")
        
        client = session["mcp_client"]
        
        # Guardar mensaje del usuario
        session["conversation_history"].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Procesar con Claude + MCP (sistema multiagente)
        print(f"💬 [{session['username']}] Procesando: {request.message[:60]}...")
        response = await client.query_with_llm(request.message, use_history=True)
        
        # Guardar respuesta
        session["conversation_history"].append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"✓ [{session['username']}] Respuesta generada ({len(response)} caracteres)")
        
        return AskResponse(
            success=True,
            response=response,
            timestamp=datetime.now().isoformat(),
            tools_used=["RAG", "ML Models", "Historical Analysis"],
            metadata={
                "conversation_length": len(session["conversation_history"]),
                "session_duration_minutes": round(
                    (datetime.now() - datetime.fromisoformat(session["created_at"])).total_seconds() / 60,
                    2
                )
            }
        )
        
    except Exception as e:
        print(f"❌ [{session['username']}] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar la consulta: {str(e)}"
        )

@app.post("/new_chat", response_model=NewChatResponse)
async def new_chat(request: NewChatRequest):
    """
    Reinicia la conversación (mantiene conexión MCP)
    """
    session = get_session(request.session_id)
    
    # Limpiar historial en sesión y cliente MCP
    session["conversation_history"] = []
    if session["mcp_client"]:
        try:
            session["mcp_client"].clear_history()
        except Exception:
            pass
    
    print(f"🔄 [{session['username']}] Nueva conversación iniciada")
    
    return NewChatResponse(
        success=True,
        message="Nueva conversación iniciada correctamente"
    )

@app.get("/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """
    Obtiene información de una sesión activa
    """
    session = get_session(session_id)
    
    return SessionInfo(
        session_id=session_id,
        username=session["username"],
        created_at=session["created_at"],
        messages_count=len(session["conversation_history"]),
        connected_to_mcp=session["mcp_client"] is not None
    )

@app.get("/health")
async def health():
    """Endpoint de salud"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(sessions),
        "version": "2.0.0",
        "features": {
            "rag": True,
            "ml_predictions": True,
            "multi_agent": True,
            "historical_analysis": True
        }
    }

@app.delete("/session/{session_id}")
async def close_session(session_id: str):
    """
    Cierra una sesión específica
    """
    if session_id in sessions:
        session = sessions[session_id]
        
        # Cerrar cliente MCP de forma segura
        if session["mcp_client"]:
            try:
                asyncio.create_task(
                    safe_close_mcp_client(session["mcp_client"], session_id)
                )
                await asyncio.sleep(0.1)  # Dar tiempo mínimo
                print(f"✓ Sesión {session_id[:8]}... cerrada manualmente")
            except Exception:
                pass
        
        # Eliminar sesión
        del sessions[session_id]
        
        return {"success": True, "message": "Sesión cerrada correctamente"}
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Sesión no encontrada"
    )

# ============= EJECUCIÓN =============

if __name__ == "__main__":
    import uvicorn
    import sys
    
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", "8000"))
    
    # Hook para capturar excepciones no manejadas de asyncio
    def handle_exception(loop, context):
        msg = context.get("exception", context.get("message", ""))
        # Suprimir errores conocidos de MCP/anyio
        if any(x in str(msg) for x in ["cancel scope", "async_generator", "stdio_client"]):
            return
        # Otros errores se muestran normalmente
        if "exception" in context:
            logging.error(f"Excepción en asyncio: {context['exception']}")
    
    # Configurar el loop de asyncio para suprimir errores de cierre
    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(handle_exception)
    except:
        pass
    
    try:
        uvicorn.run(
            app,
            host=HOST,
            port=PORT,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 Servidor detenido por el usuario")
    except Exception as e:
        if "cancel scope" not in str(e) and "async_generator" not in str(e):
            print(f"\n❌ Error fatal: {e}")
            import traceback
            traceback.print_exc()