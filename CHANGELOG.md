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