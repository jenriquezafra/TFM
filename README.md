# TFM

Repositorio del Trabajo Fin de Máster sobre métodos neuronales para la valoración de opciones bajo el modelo de Heston, con dos ramas principales de trabajo:

- una rama supervisada basada en redes neuronales sobre volatilidad implícita, usada como surrogate rápido y para el análisis de Greeks;
- una rama PINN no supervisada, entrenada sobre la ecuación diferencial parcial de Heston para aproximar superficies de precios.

La memoria final puede consultarse directamente en [`thesis/main.pdf`](thesis/main.pdf). La fuente LaTeX está en [`thesis/main.tex`](thesis/main.tex), con capítulos, apéndices, bibliografía y figuras dentro de `thesis/`.

## Instalación

El proyecto se ha desarrollado con Python y un entorno virtual local. Desde la raíz del repositorio:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Los scripts están pensados para ejecutarse desde la raíz del repositorio, de forma que las rutas relativas a `configs/`, `data/`, `outputs/` y `src/` se resuelvan correctamente.

Como comprobación rápida del entorno:

```bash
.venv/bin/python -m pytest
```

Para compilar la memoria, desde la raíz:

```bash
latexmk -pdf thesis/main.tex
```

La configuración de `latexmk` escribe los archivos auxiliares en `thesis/build/`. La copia principal de lectura está en `thesis/main.pdf`.

## Artefactos finales usados en la memoria

Los resultados finales de la memoria se apoyan en los siguientes runs y evaluaciones. Las carpetas contienen configuraciones copiadas, checkpoints, métricas, diagnósticos y figuras generadas durante la experimentación.

| Rama | Modelo / evaluación | Ruta |
| --- | --- | --- |
| ANN supervisada | ANN baseline | `outputs/runs/Liu_like_tanh_1M_v01` |
| ANN supervisada | ANN Sobolev | `outputs/runs/ANN_IV_Sobolev_v01` |
| ANN supervisada | Evaluación de Greeks del baseline ANN con parámetros de la PINN | `outputs/ann_iv_greeks/Liu_like_tanh_1M_v01_pinn_params_grid161` |
| ANN supervisada | Evaluación de Greeks de la ANN Sobolev con parámetros de la PINN | `outputs/ann_iv_greeks/ANN_IV_Sobolev_v01_pinn_params_grid161` |
| PINN no supervisada | PINN baseline | `outputs/pinn/PINN_mix_scaled_fixed_theta` |
| PINN no supervisada | PINN Sobolev | `outputs/protos/sobolev_mix_fixed_theta/PINN_mix_scaled_fixed_theta_20260420_225517` |
| PINN no supervisada | PINN Sobolev + ACV | `outputs/pinn/acv_hard_patch_sobolev_control_variate_best_gate_tau_floor_5e4` |

En los runs ANN, los checkpoints principales están en `checkpoints/model_best.pt` y las métricas en `metrics/`. En los runs PINN, los checkpoints y diagnósticos dependen de la variante: la PINN baseline guarda el entrenamiento en `train/`, mientras que las variantes Sobolev y Sobolev+ACV incluyen sus métricas y diagnósticos en `benchmark/`, `diagnostics*/` o `metrics/` según el caso.

## Estructura útil del repositorio

```text
.
├── configs/      # Configuraciones de generación, entrenamiento y evaluación
├── data/         # Datos sintéticos y datos auxiliares
├── outputs/      # Runs, checkpoints, métricas, diagnósticos y figuras
├── scripts/      # Entradas ejecutables para entrenar, evaluar y generar figuras
├── src/          # Código reutilizable del proyecto
├── tests/        # Pruebas y comprobaciones básicas
└── thesis/       # Memoria LaTeX, figuras, bibliografía y PDF final
```

Módulos principales en `src/`:

- `src/solvers/`: pricers de referencia, en particular COS para Heston.
- `src/models/`: arquitecturas ANN y utilidades de normalización.
- `src/greeks/`: cálculo y comparación de Greeks.
- `src/pinn/`: construcción de datos, modelo, pérdidas y entrenamiento PINN.
- `src/sobolev/`: utilidades de entrenamiento y evaluación Sobolev.
- `src/calibration/`: calibración inversa y evaluación rápida mediante surrogate.

## Scripts principales

Los scripts de `scripts/` incluyen tanto ejecuciones finales como herramientas usadas durante la experimentación. Los más relevantes para auditar el flujo principal son:

- `scripts/gen_synth.py`: generación de datos sintéticos.
- `scripts/train_pricer.py`: entrenamiento de la ANN supervisada.
- `scripts/eval_pricer.py`: evaluación de una ANN entrenada.
- `scripts/eval_ann_iv_greeks.py`: evaluación de Greeks para la rama ANN en IV.
- `scripts/calibrate_cann.py`: calibración inversa usando el surrogate ANN.
- `scripts/run_pinn_pipeline.py`: entrenamiento y evaluación de la PINN baseline.
- `scripts/run_pinn_sobolev_mix.py`: ajuste Sobolev de la PINN.
- `scripts/run_acv_hard_patch.py`: variante PINN Sobolev+ACV.
- `scripts/run_pinn_greeks_benchmark.py`: benchmark de Greeks PINN frente a referencia semi-analítica.
- `scripts/build_requested_thesis_figures.py`: generación de figuras finales de la memoria.

## Notas de lectura

La memoria distingue explícitamente dos líneas metodológicas. La ANN baseline, la ANN Sobolev y la CaNN pertenecen a la rama supervisada sobre volatilidad implícita. La PINN baseline, la PINN Sobolev y la PINN Sobolev+ACV forman una rama distinta, no supervisada, basada en la PDE de Heston y orientada principalmente al pricing, con análisis posterior de Greeks.

Los directorios `outputs/` contienen también pruebas intermedias y experimentos descartados. Para reproducir o revisar los resultados discutidos en la versión final, conviene usar las rutas listadas en la tabla de artefactos finales.
