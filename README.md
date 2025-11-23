# Teoinfo-Proyect

Use UV (un packet manager de python) para crear el proyecto y pues tener las dependencias organizadas. Para poder ejecutar los comandos, obviamente toca tener descargado uv... así que aquí está el comando:

``` bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

ya una vez instalado solo se ejecuta:
``` bash
uv sync
.\.venv\Scripts\Activate.ps1
```

Si no funciona el segundo comando toca ejecutar este comando: 
``` bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Y listo. Esto activará el entorno virtual donde están las dependencias y no estarán instaladas localmente. Para ejecutar el proyecto solo es:
``` bash
uv run main.py
```