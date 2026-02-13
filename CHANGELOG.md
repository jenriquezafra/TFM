## 2026-01-17
- Inicio de CHANGELOG.md
- Inicio de TODO.md
- Añadida info a TODO.md

## 2026-01-24
- Implementación de Levenberg-Marquardt en 'solvers/implied_vol.py'
    - NOTE: no puedo poner bounds para la IV (peligroso?)
    - Añadiendo elección de root-finder en 'synth.yaml' y en 'gen_synth.py' (WIP)

## 2026-01-25
- Creación de 'configs/experiments/' para poner .yaml de experimentos (como el de ver sensibilidad)

## 2026-01-26
- Terminada la implementación de LM en 'gen_synth.py'.
- Testeado LM en 'some_checks.ipynb' (comparado con Brent)
- Cambios en la estructura de 'gen_synth.py' para los parámetros que se fijen.
- Añadidos presets y methods en 'synth.yaml'.
- Generado dataset con solo 2 params de Heston libres (10k rows).

## 2026-01-27
- Añadido parte de outputs en 'train_pricer.py'
- Añadida parte de outputs en 'model_training.yaml'
- Creado 'scripts/sensitivity_pricer.py' para estudiar la sensibilidad ahí. 
- Rellenada toda la estructura y el I/O en "sensitivity_pricer.py"
- Creado 'sensitivity_config.yaml" para variar los parámetros importantes ahí

## 2026-01-29
- Añadido el shutil para guardar metadatos de los datos synth en el entrenamiento.

- Añadidos los pares de parámetros en 'sensitivity.yaml'

## 2026-02-01
- Añadidos valores de los fixed params en "sensitivity_config.yaml"
- Creado el grid para interpolar/extrapolar en "sensitivity_pricer.py"
- Cargada NN en sensitivity_pricer.py

## 2026-02-02
- Creado 'predict.py' en 'src/inference/'
- Creado pero sin hacer 'load.py' en 'src/inference/'
- Cambiada la lógica de 'sensitivity_pricer.py' para que sea una clase. Creo que es mucho mejor, ya que quiero hacer loos plots para diferentes pares de parámetros.

## 2026-02-03
- Generated synth dataset with 1M points using the LM root-finder to get the IV.
- Introduced a manual seed for the shuffling during training in 'train_pricing.py'
- Defined all the logic for the LR scheduler on 'train_pricer.py'
- Añadido step_lr callback en 'callbacks.py'
- Added logging for ETA in the training on 'train_pricer.py'

# 2026-02-04
- Cambiado el batch_size de la validación durante el training. Ajustado para que sea todo.

# 2026-02-13
- Implementado L-BFGS optmizer puro en el training (sin tests)

- Implementada la lógica del mix de optimizers (sin tests)

- 