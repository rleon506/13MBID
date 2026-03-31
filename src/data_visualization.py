import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def visualize_data(
    datos_creditos: Path | None = None,
    datos_tarjetas: Path | None = None,
    output_dir: Path | None = None
) -> None:
    """
    Genera visualizaciones de los datos del escenario
    mediante gráficos de Seaborn y Matplotlib.
    """

    # Raíz del proyecto: .../13MBID
    project_root = Path(__file__).resolve().parent.parent

    # Rutas por defecto dentro del proyecto
    if datos_creditos is None:
        datos_creditos = project_root / "data" / "raw" / "datos_creditos.csv"

    if datos_tarjetas is None:
        datos_tarjetas = project_root / "data" / "raw" / "datos_tarjetas.csv"

    if output_dir is None:
        output_dir = project_root / "docs" / "figures"

    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lectura de datos
    df_creditos = pd.read_csv(datos_creditos, sep=";")
    df_tarjetas = pd.read_csv(datos_tarjetas, sep=";")

    sns.set_style("whitegrid")

    # Distribución de la variable target
    plt.figure(figsize=(10, 6))
    sns.countplot(x="falta_pago", data=df_creditos)
    plt.title("Distribución de la variable target")
    plt.xlabel("¿Presentó mora el cliente?")
    plt.ylabel("Cantidad de clientes")
    plt.tight_layout()
    plt.savefig(output_dir / "target_distribution.png")
    plt.close()

    # Correlación créditos
    num_df = df_creditos.select_dtypes(include=["float64", "int64"])
    corr = num_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de correlaciones - Créditos")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap_creditos.png")
    plt.close()

    # Correlación tarjetas
    num_df = df_tarjetas.select_dtypes(include=["float64", "int64"])
    corr = num_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de correlaciones - Tarjetas")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap_tarjetas.png")
    plt.close()

    print(f"Figuras guardadas en: {output_dir}")


    ##################################################################################s
    # TODO: Agregar al menos dos (2) gráficos adicionales que consideren variables.
    # OPCIÓN EXTRA (ejemplo):  agregar la generación del reporte con ydata-profiling.
    ##################################################################################

if __name__ == "__main__":
    visualize_data()
