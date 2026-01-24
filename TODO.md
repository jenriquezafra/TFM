## Short-term
- [ ] Cambiar loss function a RMSE (root MSE)

- [ ] Cambiar Brent por Levenberg-Marquardt (ver si L-BFGS también)
    - [X] implementar la función en 'implied_vol.py'
    - [ ] cambiar en 'synth.yaml' para elección del rootfinder
    - [ ] cambiar en 'gen_synth.py' la elección del rootfinder (cada uno tiene sus params)
    - [ ] hacer tests

- [ ] Training: en cada epoch (o cada 10), generar un nuevo batch para entrenar (re-sampling, no shuffle)

- [ ] Generar nuevos datasets con todos los parámetros fijos menos 2 (uno de ellos gamma). Hacer LHS en d=2

- [ ] Hacer un script para estudiar la sensibilidad (que genere plots también)
    - [ ] esto es un entrenamiento diferente (2 params variando solo), así que debería poner en alguna config cómo elegir el tipo de entrenamiento (corto / largo / estudio de sensibilidad/ etc)

- [ ] Probar la red en el nuevo datasets con params fijos y hacer surfaces plots par ver el error en esos 2 params (usar MSE/RMSE/MAE, ...)

## Medium-term


## Low priority


## Optionals
- [ ] Añadir L-BFGS como root finder