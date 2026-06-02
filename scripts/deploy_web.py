"""
scripts/deploy_web.py
Copia los archivos web a docs/ y publica en GitHub Pages.

Uso:
    python scripts/deploy_web.py                    # solo copia a docs/
    python scripts/deploy_web.py --simulate 1000    # re-simula antes de copiar
    python scripts/deploy_web.py --push             # copia + git push
    python scripts/deploy_web.py --simulate 1000 --push  # todo en uno
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd, check=True):
    print(f"    $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=ROOT)
    if check and result.returncode != 0:
        print(f"  [ERROR] Falló: {cmd}")
        sys.exit(1)
    return result


def main():
    parser = argparse.ArgumentParser(description="Deploy WC2026 Predictor a GitHub Pages")
    parser.add_argument("--simulate", type=int, default=0, metavar="N",
                        help="Re-ejecutar N simulaciones antes de exportar (0 = omitir)")
    parser.add_argument("--push", action="store_true",
                        help="Hacer git add + commit + push al final")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  WC2026 Predictor — Deploy a GitHub Pages")
    print("=" * 55 + "\n")

    # ── 1. Simular (opcional) ──────────────────────────────────────────────────
    if args.simulate > 0:
        print(f"[1/4] Ejecutando {args.simulate} simulaciones Monte Carlo...")
        run(f"python main.py --step simulate --simulations {args.simulate}")
    else:
        print("[1/4] Simulaciones omitidas (usa --simulate N para re-ejecutar)")

    # ── 2. Exportar JSON ──────────────────────────────────────────────────────
    print("\n[2/4] Generando web_data.json...")
    run("python src/export_web.py")

    # ── 3. Copiar a docs/ ─────────────────────────────────────────────────────
    print("\n[3/4] Copiando archivos a docs/...")
    docs         = ROOT / "docs"
    docs_results = docs / "results"
    docs.mkdir(exist_ok=True)
    docs_results.mkdir(exist_ok=True)

    # Buscar index.html en web/ o en root
    html_copied = False
    for src in [ROOT / "web" / "index.html", ROOT / "index.html"]:
        if src.exists():
            shutil.copy(src, docs / "index.html")
            print(f"    Copiado: {src.relative_to(ROOT)} → docs/index.html")
            html_copied = True
            break
    if not html_copied:
        print("  [ERROR] No se encontró index.html.")
        print("          Colócalo en world-cup-2026-predictor/web/index.html")
        sys.exit(1)

    json_src = ROOT / "results" / "web_data.json"
    if not json_src.exists():
        print("  [ERROR] results/web_data.json no existe.")
        print("          Ejecuta primero: python src/export_web.py")
        sys.exit(1)

    shutil.copy(json_src, docs_results / "web_data.json")
    size_kb = json_src.stat().st_size / 1024
    print(f"    Copiado: results/web_data.json ({size_kb:.0f} KB) → docs/results/")

    print(f"\n  docs/ listo:")
    for f in sorted(docs.rglob("*")):
        if f.is_file():
            print(f"    {f.relative_to(docs)}")

    # ── 4. Git push (opcional) ────────────────────────────────────────────────
    if args.push:
        print("\n[4/4] Publicando en GitHub...")
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        run("git add docs/")
        run(f'git commit -m "🏆 Update WC2026 predictions {date_str}"')
        run("git push")

        # Obtener URL de GitHub Pages
        result = subprocess.run(
            "git remote get-url origin",
            shell=True, capture_output=True, text=True, cwd=ROOT
        )
        url = result.stdout.strip()
        if "github.com" in url:
            url = url.replace("git@github.com:", "https://github.com/")
            url = url.replace("https://github.com/", "")
            url = url.replace(".git", "")
            parts = url.split("/")
            if len(parts) >= 2:
                pages_url = f"https://{parts[0]}.github.io/{parts[1]}/"
                print(f"\n  ✓ Publicado. Tu web estará disponible en ~1 minuto:")
                print(f"    {pages_url}")
    else:
        print("\n[4/4] Git push omitido. Cuando quieras publicar:")
        print('    git add docs/ && git commit -m "Update predictions" && git push')
        print("\n  Para activar GitHub Pages:")
        print("    GitHub repo → Settings → Pages → Branch: main / Folder: /docs")


if __name__ == "__main__":
    main()
