"""
get_token.py (raíz del proyecto)
---------------------------------
Script de utilidad para obtener y mostrar el access token OAuth2
de SAP Document AI.

Uso (desde la raíz del proyecto):
    python get_token.py

NO ejecutar desde subdirectorios.
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from modules.auth.get_token import AuthenticationError, TokenManager
from utils.config_loader import load_config


def main() -> None:
    try:
        config = load_config()
        manager = TokenManager(config)
        token = manager.get_token()

        print("\n" + "=" * 60)
        print("  ✅ Access Token obtenido correctamente")
        print("=" * 60)
        print(f"\n  Token (primeros 80 chars):\n  {token[:80]}...")
        print(f"\n  Token completo:\n  {token}")
        print("\n" + "=" * 60)

    except FileNotFoundError as exc:
        logging.error("Archivo de credenciales no encontrado: %s", exc)
        sys.exit(1)
    except AuthenticationError as exc:
        logging.error("Error de autenticación: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logging.exception("Error inesperado: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()