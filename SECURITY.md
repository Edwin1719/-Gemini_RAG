# 🔒 Política de Seguridad

## Reportar una Vulnerabilidad

Toma muy en serio la seguridad de este proyecto. Si descubres una vulnerabilidad de seguridad, por favor repórtala de manera responsable.

### ¿Cómo Reportar?

**NO** crees un issue público para reportar vulnerabilidades de seguridad.

En su lugar, envía un correo electrónico a:

📧 **Email:** [databiq29@gmail.com](mailto:databiq29@gmail.com)

### Información a Incluir

Por favor incluye la mayor cantidad de información posible:

- Descripción clara de la vulnerabilidad
- Pasos para reproducir el problema
- Versión del software afectada
- Cualquier detalle relevante del entorno
- Posibles impactos de la vulnerabilidad
- Sugerencias para mitigar el problema (si las tienes)

### Qué Esperar

1. **Confirmación de Recepción:** Recibirás una confirmación dentro de 48 horas hábiles
2. **Evaluación:** El equipo evaluará el reporte dentro de 5 días hábiles
3. **Comunicación:** Mantendremos comunicación constante sobre el progreso
4. **Resolución:** Trabajaremos para resolver el problema lo antes posible
5. **Divulgación:** Coordinaremos la divulgación pública contigo

### Divulgación Responsable

Por favor permite un tiempo razonable para resolver la vulnerabilidad antes de divulgarla públicamente. Nos comprometemos a:

- Mantener informados a los reportadores
- Resolver vulnerabilidades críticas en menos de 30 días
- Publicar un advisory de seguridad cuando sea apropiado
- Reconocer tu contribución (si lo deseas)

## Buenas Prácticas de Seguridad

### Para Usuarios

1. **API Keys:**
   - ⚠️ **NUNCA** compartas tu archivo `.env`
   - ⚠️ **NUNCA** subas tu API Key a repositorios públicos
   - ✅ Usa variables de entorno para credenciales
   - ✅ Rota tus API Keys periódicamente

2. **Dependencias:**
   - ✅ Mantén las dependencias actualizadas
   - ✅ Revisa los advisories de seguridad de las librerías
   - ✅ Usa `pip audit` o herramientas similares

3. **Datos:**
   - ✅ No cargues datos sensibles o confidenciales
   - ✅ Encripta datos sensibles en reposo
   - ✅ Usa conexiones seguras (HTTPS)

### Para Contribuyentes

1. **Código:**
   - ✅ Sigue las prácticas de codificación segura
   - ✅ Valida todas las entradas de usuario
   - ✅ No hardcodees credenciales
   - ✅ Usa librerías de confianza y mantenidas

2. **Pull Requests:**
   - ✅ Describe claramente los cambios de seguridad
   - ✅ No incluyas información sensible en el PR
   - ✅ Prueba tus cambios localmente

3. **Issues:**
   - ✅ No reportes vulnerabilidades en issues públicos
   - ✅ Usa el canal privado para reportes de seguridad
   - ✅ Proporciona información detallada pero responsable

## Vulnerabilidades Conocidas

Actualmente no hay vulnerabilidades conocidas reportadas.

## Actualizaciones de Seguridad

Las actualizaciones de seguridad se publicarán como:

- **Releases** en GitHub con la etiqueta `security`
- **Advisories** en la sección de Security del repositorio
- **Notificaciones** a los usuarios registrados (si aplica)

## Herramientas de Seguridad Recomendadas

```bash
# Auditar dependencias de Python
pip install pip-audit
pip-audit

# Verificar seguridad del código
pip install bandit
bandit -r app.py

# Analizar secrets en el repositorio
pip install detect-secrets
detect-secrets scan > .secrets.baseline
```

## Contacto de Seguridad

- **Email:** [databiq29@gmail.com](mailto:databiq29@gmail.com)
- **Tiempo de Respuesta:** 48 horas hábiles
- **Idiomas:** Español, Inglés

---

**Gracias por ayudar a mantener seguro este proyecto.**
