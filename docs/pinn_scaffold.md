# PINN Scaffold

Este documento fija la estructura inicial para implementar un PINN que use parametros calibrados por CaNN y aprenda precio.

## 1) Objetivo de la base

- Definir contratos de entrada/salida.
- Separar responsabilidades por etapa.
- Permitir validacion de wiring con `dry-run`.
- Evitar acoplar implementacion numerica prematuramente.

## 2) Estructura propuesta

```text
src/pinn/
├── __init__.py
├── contracts.py      # dataclasses del contrato de pipeline
├── config.py         # carga y resolucion de config YAML
├── cann_bridge.py    # validacion y lectura de artefactos CaNN
├── data_builder.py   # TODO: datasets supervisado/colocacion/frontera
├── model.py          # TODO: arquitectura PINN (forward)
├── losses.py         # TODO: perdida total (data + fisica + frontera)
├── trainer.py        # TODO: loop de entrenamiento
├── evaluator.py      # TODO: evaluacion y metricas
└── pipeline.py       # orquestador de etapas (scaffold)
```

## 3) Contrato CaNN -> PINN

Entrada esperada desde calibracion:

- `summary.yaml`:
  - `theta_star` (vector de parametros calibrados)
  - `parameter_order` (orden de parametros)
- `quotes_comparison.parquet`:
  - `moneyness`, `tau`, `r`
  - `iv_market`, `iv_pred` (actualmente disponibles)

Notas:
- El target de precio esta declarado como placeholder (`price_market`) hasta definir fuente final.
- `src/pinn/cann_bridge.py` valida que artefactos existan antes de arrancar etapas.

## 4) Pipeline declarado

Etapas (controladas por `configs/pinn_pipeline.yaml`):

1. `prepare_dataset`
2. `train`
3. `evaluate`

Comportamiento actual:
- `--dry-run`: valida rutas y genera plan YAML.
- Ejecucion real:
  - `prepare_dataset`: implementada
  - `train`: implementada (supervisado + MSE)
  - `evaluate`: pendiente

## 5) Configs

- `configs/pinn_pipeline.yaml`: orquestacion e inputs.
- `configs/pinn_model_architecture.yaml`: estructura de red.
- `configs/pinn_training.yaml`: optimizacion, sampling, perdida, callbacks.

## 6) Comandos

```bash
.venv/bin/python scripts/run_pinn_pipeline.py \
  --config configs/pinn_pipeline.yaml \
  --stage all \
  --dry-run \
  --dump-plan
```

Salida esperada:
- resumen de plan en terminal
- `outputs/pinn/PINN_v01/pipeline_plan.yaml`

Ejecucion real:
```bash
.venv/bin/python scripts/run_pinn_pipeline.py \
  --config configs/pinn_pipeline.yaml \
  --stage all \
  --dump-plan
```

Salida adicional esperada:
- `outputs/pinn/PINN_v01/pipeline_execution.yaml`
- `outputs/pinn/PINN_v01/data/supervised_dataset.npz`
- `outputs/pinn/PINN_v01/train/checkpoints/model_best.pt`
