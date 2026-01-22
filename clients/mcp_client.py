"""
Cliente LLM para servidor MCP WaterTank con RAG y ML
Ubicación: clients/mcp_client.py
"""
import asyncio
import json
import sys
import io
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from anthropic import Anthropic
import os
from typing import Optional

# Configurar UTF-8 para Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# API Key de Anthropic (mejor usar variable de entorno)
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

if not ANTHROPIC_API_KEY:
    raise ValueError("❌ ANTHROPIC_API_KEY no está configurada. Crea un archivo .env con tu API key")

class MCPLLMClient:
    def __init__(self):
        self.session = None
        self.available_tools = []
        self.project_root = Path(__file__).parent.parent
        self.conversation_history = []
        
    async def connect_to_mcp_server(self):
        """Conecta con el servidor MCP"""
        server_path = self.project_root / "servers" / "mcp_server.py"
        
        if not server_path.exists():
            raise FileNotFoundError(f"❌ Servidor no encontrado: {server_path}")
        
        # Usar el mismo Python del entorno virtual
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(server_path)],
            env=None
        )
        
        print(f"🔌 Conectando al servidor MCP...")
        
        # Inicializar conexión
        self.stdio_context = stdio_client(server_params)
        self.stdio_transport = await self.stdio_context.__aenter__()
        self.session = ClientSession(self.stdio_transport[0], self.stdio_transport[1])
        await self.session.__aenter__()
        await self.session.initialize()
        
        # Obtener herramientas
        tools_response = await self.session.list_tools()
        self.available_tools = tools_response.tools
        
        print(f"✅ Conectado! {len(self.available_tools)} herramientas disponibles")
        print("\n📋 Herramientas MCP:")
        for tool in self.available_tools:
            print(f"   • {tool.name}: {tool.description[:60]}...")
        print()
        
        return self.available_tools
    
    async def call_tool(self, tool_name: str, arguments: dict):
        """Ejecuta una herramienta del servidor"""
        result = await self.session.call_tool(tool_name, arguments)
        return result.content[0].text if result.content else None
    
    async def query_with_llm(self, user_question: str, use_history: bool = True):
        """Procesa pregunta usando Claude + MCP con sistema multiagente"""
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        
        # Convertir herramientas a formato Claude
        claude_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            for tool in self.available_tools
        ]
        
        # System prompt con contexto multiagente
        system_prompt = """Eres un asistente experto en análisis de sistemas de tanques de agua con capacidades de:

1. **PREDICCIÓN ML**: Usa modelos entrenados para predecir magnitudes (nivel, cloro, pH, temperatura, etc.)
2. **RAG (Búsqueda Semántica)**: Busca información relevante en la base de conocimiento antes de responder
3. **ANÁLISIS HISTÓRICO**: Analiza tendencias y patrones en datos históricos
4. **REPORTING**: Genera reportes completos con predicciones y recomendaciones

**FLUJO DE TRABAJO MULTIAGENTE:**

Para cada consulta, sigue este proceso:

1. **BUSCAR CONOCIMIENTO** (search_knowledge): Siempre comienza buscando información relevante en la base de conocimiento
2. **ANALIZAR DATOS** (analyze_historical_data): Si la pregunta involucra tendencias o histórico
3. **PREDECIR** (predict_magnitude): Si necesitas hacer predicciones específicas
4. **GENERAR REPORTE** (generate_comprehensive_report): Para análisis completos
5. **RECOMENDAR** (get_recommendations): Para sugerencias operativas

**PROVINCIAS Y TARGETS DISPONIBLES:**
- Provincias: Usa list_provinces_and_tanks para obtener la lista actualizada
- Targets principales: watertanklevel, clexitlevel, phexitlevel, temperature, pressurelevel, flowexitlevel

**IMPORTANTE:**
- SIEMPRE usa search_knowledge PRIMERO para obtener contexto
- Combina múltiples herramientas para respuestas completas
- Presenta resultados de forma clara y estructurada
- Incluye valores numéricos con sus unidades
- Da recomendaciones accionables

Responde en español de forma profesional y concisa."""

        print(f"💭 Pregunta: {user_question}\n")
        print("🤖 Claude procesando con sistema multiagente...\n")
        
        # Preparar mensajes (incluir historial si se desea)
        messages = []
        if use_history and self.conversation_history:
            messages.extend(self.conversation_history[-6:])  # Últimos 3 intercambios
        
        messages.append({"role": "user", "content": user_question})
        
        # Llamada inicial a Claude
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            tools=claude_tools,
            messages=messages
        )
        
        # Loop de herramientas (sistema multiagente)
        iteration = 0
        max_iterations = 10
        
        while response.stop_reason == "tool_use" and iteration < max_iterations:
            iteration += 1
            print(f"🔄 Iteración {iteration}: Claude usando herramientas...\n")
            
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"   🔧 {block.name}")
                    if block.input:
                        for k, v in list(block.input.items())[:3]:  # Mostrar primeros 3 args
                            print(f"      • {k}: {v}")
                    
                    # Ejecutar herramienta
                    try:
                        result = await self.call_tool(block.name, block.input)
                        print(f"   ✅ Resultado obtenido ({len(result)} caracteres)\n")
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })
                    except Exception as e:
                        print(f"   ❌ Error: {str(e)}\n")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Error: {str(e)}",
                            "is_error": True
                        })
            
            # Actualizar conversación
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            
            # Nueva llamada a Claude
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system_prompt,
                tools=claude_tools,
                messages=messages
            )
        
        # Extraer respuesta final
        final_response = "".join(block.text for block in response.content if hasattr(block, "text"))
        
        # Guardar en historial
        self.conversation_history.append({"role": "user", "content": user_question})
        self.conversation_history.append({"role": "assistant", "content": final_response})
        
        return final_response
    
    def clear_history(self):
        """Limpia el historial de conversación"""
        self.conversation_history = []
    
    async def close(self):
        """Cierra conexión"""
        if self.session:
            await self.session.__aexit__(None, None, None)
        if hasattr(self, 'stdio_context'):
            await self.stdio_context.__aexit__(None, None, None)

async def main():
    """Función principal para testing"""
    client = MCPLLMClient()
    
    print("="*70)
    print("🎯 CLIENTE LLM + MCP WATERTANK CON RAG Y ML")
    print("="*70)
    
    try:
        # Conectar al servidor
        await client.connect_to_mcp_server()
        
        # Preguntas de ejemplo mejoradas
        questions = [
            "¿Qué provincias y tanques están disponibles en el sistema?",
            "Busca información sobre la provincia de Burgos y sus tanques",
            "Predice el nivel de agua para un tanque en Burgos con temperatura 22°C, presión 2.5 bar y flujo 4.2 m³/h",
            "Analiza los datos históricos de las últimas 48 horas para Burgos",
            "Genera un reporte completo para la provincia de Burgos",
            "Dame recomendaciones para un tanque con nivel actual de 35%"
        ]
        
        # Ejecutar pregunta (cambia el índice)
        question = questions[0]
        answer = await client.query_with_llm(question)
        
        print("="*70)
        print("📝 RESPUESTA DE CLAUDE:")
        print("="*70)
        print(f"\n{answer}\n")
        print("="*70)
        
        # MODO INTERACTIVO
        print("\n💡 Modo interactivo (escribe 'salir' para terminar, 'limpiar' para nuevo chat)\n")
        while True:
            q = input("💬 Tu pregunta: ")
            
            if q.lower() in ['salir', 'exit', 'q']:
                break
            
            if q.lower() in ['limpiar', 'clear', 'nuevo']:
                client.clear_history()
                print("✅ Historial limpiado - Nueva conversación\n")
                continue
            
            if q.strip():
                ans = await client.query_with_llm(q)
                print(f"\n📝 {ans}\n" + "-"*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.close()
        print("\n👋 Conexión cerrada\n")

if __name__ == "__main__":
    asyncio.run(main())