import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ydata_profiling import ProfileReport


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

    # Crear directorio de salida para los reportes
    output_dir_report = project_root / "docs"
    output_dir_report.mkdir(parents=True, exist_ok=True)

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

    # Importe solicitado por falta de pago
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df_creditos, x="falta_pago", y="importe_solicitado")
    plt.title("Importe solicitado según falta de pago")
    plt.xlabel("Falta de pago")
    plt.ylabel("Importe solicitado")
    plt.tight_layout()
    plt.savefig(output_dir / "boxplot_importe_falta_pago.png")
    plt.close()

    # Límite de crédito por nivel de tarjeta
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df_tarjetas, x="nivel_tarjeta", y="limite_credito_tc")
    plt.title("Límite de crédito según nivel de tarjeta")
    plt.xlabel("Nivel de tarjeta")
    plt.ylabel("Límite de crédito")
    plt.tight_layout()
    plt.savefig(output_dir / "boxplot_limite_nivel_tarjeta.png")
    plt.close()

    #Generación de los reportes

    print("Generando reportes automáticos...")

    # Reporte créditos
    profile_creditos = ProfileReport(
        df_creditos,
        title="Reporte Perfilado - Créditos",
        explorative=True
    )

    profile_creditos.to_file(output_dir_report / "reporte_creditos.html")


    # Reporte tarjetas
    profile_tarjetas = ProfileReport(
        df_tarjetas,
        title="Reporte Perfilado - Tarjetas",
        explorative=True
    )

    profile_tarjetas.to_file(output_dir_report / "reporte_tarjetas.html")

    print(f"Reportes guardados en: {output_dir_report}")
    print(f"Figuras guardadas en: {output_dir}")


if __name__ == "__main__":
    visualize_data()
